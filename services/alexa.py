"""Check TCP connectivity to the configured Alexa/1C endpoint."""

import socket

from core.config import settings
from core.logger import logger

TIMEOUT_SECONDS = 5


def check_connection() -> int:
    logger.info(
        "Attempting socket connection to %s:%s",
        settings.ALEXA_HOST,
        settings.ALEXA_PORT,
    )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(TIMEOUT_SECONDS)
            connection.connect((settings.ALEXA_HOST, settings.ALEXA_PORT))
    except OSError:
        logger.exception("Connection failed")
        return 1

    logger.info("Connection successful")
    return 0
