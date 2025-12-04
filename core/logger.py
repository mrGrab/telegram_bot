import sys
import logging
from logging import Logger


def configure_logger() -> Logger:
    logger = logging.getLogger("telegram_bot")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False

    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    return logger


logger = configure_logger()
