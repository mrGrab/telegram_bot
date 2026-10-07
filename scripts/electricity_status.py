import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock

import click
from rich import print

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.sensu_client import SensuClient

logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 15
POLL_INTERVAL_SECONDS = 1
RECENT_RESULT_SECONDS = 1


class ElectricityStatus(IntEnum):
    OK = 0
    WARNING = 1
    OUTAGE = 2
    UNKNOWN = 3


@dataclass(frozen=True)
class ElectricityOutcome:
    status: ElectricityStatus
    reason: str

    @property
    def fresh(self) -> bool:
        return self.reason in ("completed", "check_unknown")


class ElectricityChecker:
    """Get one correlated Sensu result, shared by concurrent callers."""

    def __init__(
        self,
        sensu: SensuClient,
        check_name: str,
        entity_name: str,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.sensu = sensu
        self.check_name = check_name
        self.entity_name = entity_name
        self.clock = clock
        self.sleep = sleep
        self._lock = Lock()
        self._inflight: Future[ElectricityOutcome] | None = None
        self._recent: tuple[float, ElectricityOutcome] | None = None

    def check(self) -> ElectricityOutcome:
        """Share an in-progress check and briefly reuse its fresh result."""
        with self._lock:
            if self._inflight is not None:
                future = self._inflight
                owner = False
            elif (
                self._recent and self.clock() - self._recent[0] < RECENT_RESULT_SECONDS
            ):
                return self._recent[1]
            else:
                future = Future()
                self._inflight = future
                owner = True

        if not owner:
            return future.result()

        try:
            outcome = self._run_check()
        except Exception:
            logger.exception("Unexpected electricity check failure")
            outcome = ElectricityOutcome(ElectricityStatus.UNKNOWN, "invalid_response")

        with self._lock:
            self._recent = (self.clock(), outcome) if outcome.fresh else None
            self._inflight = None
            future.set_result(outcome)
        return outcome

    def _run_check(self) -> ElectricityOutcome:
        try:
            issued = self.sensu.execute_check(self.check_name)
        except ValueError as e:
            logger.warning("Invalid Sensu execute response: %s", e)
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "invalid_response")
        except Exception:
            logger.exception("Failed to start Sensu check %s", self.check_name)
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "trigger_failed")
        if issued is None:
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "trigger_failed")
        if isinstance(issued, bool) or not isinstance(issued, int) or issued <= 0:
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "invalid_response")

        deadline = self.clock() + MAX_WAIT_SECONDS
        saw_invalid = False
        saw_response = False
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            try:
                event = self.sensu.get_event_check(
                    self.check_name, self.entity_name, timeout=min(10, remaining)
                )
            except Exception:
                logger.exception("Failed to fetch Sensu event")
                event = None
            if self.clock() > deadline:
                break
            if event is not None:
                saw_response = True
                if not isinstance(event, dict) or not isinstance(
                    event.get("check"), dict
                ):
                    saw_invalid = True
                else:
                    check = event["check"]
                    event_issued = check.get("issued")
                    if isinstance(event_issued, bool) or not isinstance(
                        event_issued, int
                    ):
                        saw_invalid = True
                    elif event_issued == issued:
                        status = check.get("status")
                        if isinstance(status, bool) or not isinstance(status, int):
                            saw_invalid = True
                        elif status in (0, 1, 2):
                            return ElectricityOutcome(
                                ElectricityStatus(status), "completed"
                            )
                        else:
                            return ElectricityOutcome(
                                ElectricityStatus.UNKNOWN, "check_unknown"
                            )
            remaining = deadline - self.clock()
            if remaining > 0:
                self.sleep(min(POLL_INTERVAL_SECONDS, remaining))

        reason = (
            "invalid_response"
            if saw_invalid
            else "timed_out"
            if saw_response
            else "fetch_failed"
        )
        logger.warning("No fresh electricity result: %s", reason)
        return ElectricityOutcome(ElectricityStatus.UNKNOWN, reason)


@click.command()
@click.option(
    "--sensu-api-url",
    default=lambda: os.environ.get("SENSU_API_URL"),
    required=True,
    help="Sensu API URL",
)
@click.option(
    "--sensu-api-key",
    default=lambda: os.environ.get("SENSU_API_KEY"),
    required=True,
    help="Sensu API Key",
)
@click.option(
    "--sensu-namespace",
    default=lambda: os.environ.get("SENSU_NAMESPACE"),
    required=True,
    help="Sensu API Namespace",
)
@click.option("--check-name", required=True, help="Name of the Sensu check to execute")
@click.option(
    "--entity-name", required=True, help="Name of the Sensu entity running the check"
)
def main(sensu_api_url, sensu_api_key, sensu_namespace, check_name, entity_name):
    """
    Monitors electricity status by triggering a Sensu check
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sensu_client = SensuClient(
        url=sensu_api_url, api_key=sensu_api_key, namespace=sensu_namespace
    )
    result = ElectricityChecker(sensu_client, check_name, entity_name).check()

    if result.status == ElectricityStatus.OK:
        print("OK: Electricity check passed")
        sys.exit(0)
    elif result.status == ElectricityStatus.WARNING:
        print("WARNING: Electricity check warning")
        sys.exit(1)
    elif result.status == ElectricityStatus.OUTAGE:
        print("CRITICAL: Electricity outage detected!")
        sys.exit(2)
    else:
        reasons = {
            "trigger_failed": "Could not start Sensu check",
            "timed_out": "Fresh result did not arrive before the deadline",
            "fetch_failed": "Could not retrieve Sensu event",
            "invalid_response": "Sensu returned an unusable response",
            "check_unknown": "Sensu check reported unknown status",
        }
        print(f"UNKNOWN: {reasons.get(result.reason, 'Could not verify check status')}")
        sys.exit(3)


if __name__ == "__main__":
    main()
