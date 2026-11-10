from collections.abc import Callable
from pathlib import Path

from core.config import settings
from core.logger import logger
from services.alexa import check_connection
from services.dtek_monitor import MonitorContext, MonitorService
from services.electricity_status import ElectricityChecker, ElectricityStatus
from telegram.bot import TelegramBot


def cmd_help(bot: TelegramBot, **kwargs) -> None:
    """Send help/command list"""
    logger.info(f"{bot.chat_id}: Showing help")
    bot.send_message(
        "<b>Available Commands:</b>\n"
        "/start – Show this message\n"
        "/check_electricity – Check if there's power\n"
        "/check_alexa – Check connection to Alexa/1C\n"
        "/dtek_report – Show the current status and DTEK schedule\n"
    )


def cmd_check_electricity(bot: TelegramBot, **kwargs) -> None:
    """Reply with the current Home Assistant electricity status."""
    logger.info(f"{bot.chat_id}: Checking electricity status")
    bot.send_typing()

    outcome = kwargs["electricity"].check()

    logger.debug(f"Electricity status: {outcome.status.name}, reason: {outcome.reason}")

    messages = {
        ElectricityStatus.OK: "🟢 Все добре",
        ElectricityStatus.OUTAGE: "🔴 Відключення електроенергії",
    }

    bot.send_message(messages.get(outcome.status, "⚫️ Не вдалося перевірити статус"))


def cmd_check_alexa(bot: TelegramBot, **kwargs) -> None:
    """Checks Alexa/1C connection"""
    logger.info(f"{bot.chat_id}: Checking Alexa server status")
    bot.send_typing()

    result = check_connection()
    logger.debug(f"Alexa connectivity result: {result}")

    if result == 0:
        bot.send_message("🟢 Alexa/1C доступна")
    else:
        bot.send_message("🔴 Alexa/1C недоступна")


def build_dtek_service(electricity: ElectricityChecker) -> MonitorService:
    ctx = MonitorContext(
        city=settings.DTEK_CITY,
        street=settings.DTEK_STREET,
        building=settings.DTEK_BUILDING,
        state_file=Path(settings.DTEK_STATE_FILE),
        forced_group=None,
    )
    return MonitorService(ctx, electricity=electricity)


def cmd_dtek_report(bot: TelegramBot, **kwargs) -> None:
    """Send the latest power and DTEK report to the requesting chat."""
    logger.info(f"{bot.chat_id}: Fetching DTEK report")
    bot.send_typing()

    message = build_dtek_service(kwargs["electricity"]).report()
    bot.send_message(message)


# --- Command Registry ---
COMMANDS: dict[str, Callable] = {
    "/help": cmd_help,
    "/start": cmd_help,
    "/check_electricity": cmd_check_electricity,
    "/check_alexa": cmd_check_alexa,
    "/dtek_report": cmd_dtek_report,
}
