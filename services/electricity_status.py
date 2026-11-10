"""Read the current apartment power state from Home Assistant."""

import logging
from dataclasses import dataclass
from enum import Enum

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 5


class ElectricityStatus(str, Enum):
    OK = "on"
    OUTAGE = "off"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ElectricityOutcome:
    status: ElectricityStatus
    reason: str


class ElectricityChecker:
    """Return the latest state of one Home Assistant binary sensor."""

    def __init__(self, url: str, token: str, entity_id: str):
        self.url = url.rstrip("/")
        self.token = token
        self.entity_id = entity_id

    def check(self) -> ElectricityOutcome:
        try:
            response = requests.get(
                f"{self.url}/api/states/{self.entity_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Failed to read Home Assistant power state")
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "fetch_failed")

        try:
            data = response.json()
        except ValueError:
            logger.warning("Home Assistant returned invalid JSON for power state")
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "invalid_response")

        if not isinstance(data, dict) or data.get("entity_id") != self.entity_id:
            logger.warning("Home Assistant returned an invalid power state response")
            return ElectricityOutcome(ElectricityStatus.UNKNOWN, "invalid_response")

        state = data.get("state")
        if state == "on":
            return ElectricityOutcome(ElectricityStatus.OK, "current")
        if state == "off":
            return ElectricityOutcome(ElectricityStatus.OUTAGE, "current")
        return ElectricityOutcome(ElectricityStatus.UNKNOWN, "unavailable")
