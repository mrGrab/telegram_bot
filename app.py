import hmac

from flask import Flask, jsonify, request

from core.config import settings
from core.logger import logger
from scripts.electricity_status import ElectricityChecker
from services.sensu_client import SensuClient
from telegram.bot import TelegramBot
from telegram.commands import COMMANDS

app = Flask(__name__)
sensu_client = SensuClient(
    url=settings.SENSU_API_URL,
    api_key=settings.SENSU_API_KEY,
    namespace=settings.SENSU_NAMESPACE,
)
electricity_checker = ElectricityChecker(
    sensu=sensu_client,
    check_name="home-electricity",
    entity_name="adjutant",
)


@app.post("/")
def webhook():
    """Handles incoming Telegram webhook messages"""

    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(secret, settings.TELEGRAM_WEBHOOK_SECRET):
        return jsonify({"status": "forbidden"}), 403

    context = request.get_json(cache=False, silent=True)

    if not context or "message" not in context:
        logger.warning("Received invalid Telegram update format")
    else:
        message = context["message"]
        chat = message.get("chat", {})
        if (
            chat.get("type") != "private"
            or chat.get("id") not in settings.TELEGRAM_ALLOWED_CHAT_IDS
        ):
            return jsonify({"status": "received"}), 200
        text = message.get("text", "").strip()
        bot = TelegramBot(token=settings.BOT_TOKEN)
        bot.chat_id = chat.get("id")

        handler = COMMANDS.get(text)
        if handler:
            logger.info(f"Command '{text}' received from '{chat.get('first_name')}'")
            handler(bot, electricity=electricity_checker)
        else:
            if text.startswith("/"):
                bot.send_message("Unknown command. Try /help")

    return jsonify({"status": "received"}), 200


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG)
