import logging
import socket
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict


class AlexaTarget(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=True,
        extra="ignore",
    )

    ALEXA_HOST: str = "127.0.0.1"
    ALEXA_PORT: int = 80


target = AlexaTarget()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def check_connection(host: str, port: int, timeout: int) -> int:
    """Check only the configured Alexa endpoint."""

    if (host, port) != (target.ALEXA_HOST, target.ALEXA_PORT):
        logger.error("Connection target is not the configured Alexa endpoint")
        return 1

    logger.info(
        f"Attempting socket connection to {target.ALEXA_HOST}:{target.ALEXA_PORT}"
    )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((target.ALEXA_HOST, target.ALEXA_PORT))

        logger.info("Connection successful")
        return 0

    except OSError as e:
        logger.error(f"Connection failed. Error: {e}")
        return 1


def main() -> int:
    """CLI entry point"""

    if len(sys.argv) < 3:
        logger.error("Usage: python check_alexa.py <host> <port> [timeout]")
        return 1

    host = sys.argv[1]

    try:
        port = int(sys.argv[2])
    except ValueError:
        logger.error("Port must be an integer")
        return 1

    # Optional timeout argument, defaults to 3 if not provided via CLI
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    # Execute check and return status directly
    return check_connection(host, port, timeout)


if __name__ == "__main__":
    sys.exit(main())
