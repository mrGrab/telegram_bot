from flask import Flask, jsonify, request

from core.logger import logger
from core.config import settings
from telegram.bot import TelegramBot
from telegram.commands import COMMANDS
from services.sensu_client import SensuClient

app = Flask(__name__)
bot = TelegramBot(token=settings.BOT_TOKEN)
sensu_client = SensuClient(url=settings.SENSU_API_URL,
                           api_key=settings.SENSU_API_KEY,
                           namespace=settings.SENSU_NAMESPACE)


@app.post("/")
def webhook():
    """Handles incoming Telegram webhook messages"""

    context = request.get_json(cache=False, silent=True)
    if not context or 'message' not in context:
        logger.warning("Received invalid Telegram update format")
        return '', 200

    message = context['message']
    chat = message.get("chat", {})
    text = message.get('text', '').strip()
    bot.chat_id = chat.get("id")

    handler = COMMANDS.get(text)
    if handler:
        logger.info(
            f"Command '{text}' received from '{chat.get('first_name')}'")
        handler(bot, sensu=sensu_client)
    else:
        if text.startswith("/"):
            bot.send_message("Unknown command. Try /help")

    return '', 200


@app.get('/health')
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG)
