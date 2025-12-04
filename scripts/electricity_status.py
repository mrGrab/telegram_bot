#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import logging
import click
import time
from rich import print

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.sensu_client import SensuClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 5
POLL_INTERVAL_SECONDS = 1


def get_electricity_status(sensu: SensuClient, check_name: str,
                           entity_name: str):

    issued = sensu.execute_check(check_name)
    if issued is None:
        logger.error(f"Failed to issue Sensu check '{check_name}'")
        return 3

    logger.debug(f"Check '{check_name}' issued on {issued}")

    # Sensu event fetching loop
    for i in range(MAX_WAIT_SECONDS):
        time.sleep(POLL_INTERVAL_SECONDS)

        event = sensu.get_event_check(check_name, entity_name)
        if not event:
            logger.warning(f"Event not found for '{check_name}' after {i+1}s")
            continue

        # Check the current status
        check = event.get("check", {})
        executed = check.get("executed")
        result_status = check.get("status")

        # Check if the latest event matches the issued timestamp
        if executed == issued:
            logger.info(f"Check result received. Status: {result_status}")
            return result_status

        # Check the history (in case a new check superseded the latest)
        history = check.get("history", [])
        for past_event in history:
            if past_event.get("executed") == issued:
                result_status = past_event.get("status")
                logger.info(
                    f"Check result found in history. Status: {result_status}")
                return result_status

        logger.debug(
            f"Latest event ({executed}) does not match issued ({issued}). Continuing to wait..."
        )

    # Loop finished without finding the expected timestamp
    last_status = event.get("check", {}).get("status", 3) if event else 3
    logger.warning(
        f"Timed out waiting for new result. Returning last known status: {last_status}"
    )
    return last_status


@click.command()
@click.option('--sensu-api-url',
              default=lambda: os.environ.get("SENSU_API_URL"),
              required=True,
              help='Sensu API URL')
@click.option('--sensu-api-key',
              default=lambda: os.environ.get("SENSU_API_KEY"),
              required=True,
              help='Sensu API Key')
@click.option('--sensu-namespace',
              default=lambda: os.environ.get("SENSU_NAMESPACE"),
              required=True,
              help='Sensu API Namespace')
@click.option('--check-name',
              required=True,
              help='Name of the Sensu check to execute')
@click.option('--entity-name',
              required=True,
              help='Name of the Sensu entity running the check')
def main(sensu_api_url, sensu_api_key, sensu_namespace, check_name,
         entity_name):
    """
    Monitors electricity status by triggering a Sensu check
    """
    sensu_client = SensuClient(url=sensu_api_url,
                               api_key=sensu_api_key,
                               namespace=sensu_namespace)

    result = get_electricity_status(sensu=sensu_client,
                                    check_name=check_name,
                                    entity_name=entity_name)

    if result == 0:
        print("OK: Electricity check passed")
        sys.exit(0)
    elif result == 1:
        print("WARNING: Electricity check warning")
        sys.exit(1)
    elif result == 2:
        print("CRITICAL: Electricity outage detected!")
        sys.exit(2)
    else:
        print("UNKNOWN: Could not verify check status")
        sys.exit(3)


if __name__ == "__main__":
    main()
