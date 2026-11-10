"""Run change-based DTEK notifications from cron."""

from core.config import settings
from core.logger import logger
from services.electricity_status import ElectricityChecker
from telegram.bot import TelegramBot
from telegram.commands import build_dtek_service


def main() -> None:
    chat_ids = settings.notification_chat_ids("dtek")
    if not chat_ids:
        logger.info("DTEK monitoring skipped: no subscribed chats")
        return

    electricity = ElectricityChecker(
        url=settings.HASS_URL,
        token=settings.HASS_TOKEN,
        entity_id=settings.HASS_POWER_ENTITY_ID,
    )
    bot = TelegramBot(token=settings.BOT_TOKEN)
    result = build_dtek_service(electricity).notify(chat_ids, bot.send_to_chat)
    if result is None:
        logger.info("DTEK monitoring skipped: another run is active")


if __name__ == "__main__":
    main()
