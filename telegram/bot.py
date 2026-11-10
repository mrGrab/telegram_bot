import requests

from core.logger import logger


class TelegramBot:
    """
    Handles core Telegram API communication (sending messages, actions)
    """

    def __init__(self, token: str):
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.chat_id: int | None = None
        logger.debug("TelegramBot initialized")

    def send_to_chat(self, chat_id: int | str, message: str) -> bool:
        """Return whether Telegram accepted the message for this chat."""
        url = f"{self.api_url}/sendMessage"
        data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

        try:
            response = requests.post(url, data=data, timeout=5)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise ValueError("Telegram did not accept the message")
            logger.info("%s: Message has been sent", chat_id)
            return True
        except requests.RequestException, ValueError:
            logger.exception("%s: Failed to send message", chat_id)
            return False

    def send_message(self, message: str) -> None:
        """Sends a text message to the current chat_id"""
        if self.chat_id is None:
            logger.error("Attempted to send message but chat_id was not set")
            return
        self.send_to_chat(self.chat_id, message)

    def send_typing(self):
        """Sends the 'typing' chat action to the current chat_id"""

        if not self.chat_id:
            return

        url = f"{self.api_url}/sendChatAction"
        data = {"chat_id": self.chat_id, "action": "typing"}

        try:
            r = requests.post(url, data=data, timeout=5)
            r.raise_for_status()
            logger.debug(f"{self.chat_id}: Typing action has been sent")
        except requests.exceptions.RequestException as e:
            logger.error(f"{self.chat_id}: Failed to send typing action. Error: {e}")
