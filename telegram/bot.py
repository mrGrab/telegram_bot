import requests
from typing import Optional
from core.logger import logger


class TelegramBot:
    """
    Handles core Telegram API communication (sending messages, actions)
    """

    def __init__(self, token: str):
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.chat_id: Optional[int] = None
        logger.debug("TelegramBot initialized")

    def send_message(self, message: str):
        """Sends a text message to the current chat_id"""
        if not self.chat_id:
            logger.error("Attempted to send message but chat_id was not set")
            return

        url = f"{self.api_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}

        try:
            r = requests.post(url, data=data, timeout=5)
            r.raise_for_status()
            logger.info(
                f"{self.chat_id}: Message has been sent: {r.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"{self.chat_id}: Failed to send message: {e}")

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
            logger.error(
                f"{self.chat_id}: Failed to send typing action. Error: {e}")
