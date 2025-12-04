from pathlib import Path
from typing import Dict, Callable
from telegram.bot import TelegramBot
from core.logger import logger
from core.config import settings

from scripts.check_alexa import check_connection
from scripts.dtek_monitor import MonitorContext, MonitorService
from scripts.electricity_status import get_electricity_status


def cmd_help(bot: TelegramBot, **kwargs) -> None:
    """Send help/command list"""
    logger.info(f"{bot.chat_id}: Showing help")
    bot.send_message("<b>Available Commands:</b>\n"
                     "/start – Show this message\n"
                     "/check_electricity – Check if there's power\n"
                     "/check_alexa – Check connection to Alexa/1C\n"
                     "/dtek_schedule – Show DTEK outage schedule\n")


def cmd_check_electricity(bot: TelegramBot, **kwargs) -> None:
    """Checks electricity status via Sensu"""
    logger.info(f"{bot.chat_id}: Checking electricity status")
    bot.send_typing()

    status_code = get_electricity_status(sensu=kwargs["sensu"],
                                         check_name="home-electricity",
                                         entity_name="adjutant")

    logger.debug(f"Electricity status code: {status_code}")

    messages = {
        0: "🟢 Все добре",
        1: "🟡 Щось не зрозуміло",
        2: "🔴 Відключення електроенергії",
    }

    bot.send_message(
        messages.get(status_code, "⚫️ Не вдалося перевірити статус"))


def cmd_check_alexa(bot: TelegramBot, **kwargs) -> None:
    """Checks Alexa/1C connection"""
    logger.info(f"{bot.chat_id}: Checking Alexa server status")
    bot.send_typing()

    result = check_connection(host=settings.ALEXA_HOST,
                              port=settings.ALEXA_PORT,
                              timeout=5)
    logger.debug(f"Alexa connectivity result: {result}")

    if result == 0:
        bot.send_message("🟢 Everything's okay 👌")
    else:
        bot.send_message("🔴 No connection to 1C 😡")


def cmd_dtek_schedule(bot: TelegramBot, **kwargs) -> None:
    """Fetch and send DTEK outage schedule"""
    logger.info(f"{bot.chat_id}: Fetching DTEK schedule")
    bot.send_typing()

    ctx = MonitorContext(city=settings.DTEK_CITY,
                         street=settings.DTEK_STREET,
                         building=settings.DTEK_BUILDING,
                         state_file=Path(settings.DTEK_STATE_FILE),
                         forced_group=None)

    message, _ = MonitorService(ctx).run()

    bot.send_message(message)


# --- Command Registry ---
COMMANDS: Dict[str, Callable] = {
    "/help": cmd_help,
    "/start": cmd_help,
    "/check_electricity": cmd_check_electricity,
    "/check_alexa": cmd_check_alexa,
    "/dtek_schedule": cmd_dtek_schedule,
}
