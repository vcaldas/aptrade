import requests
from aptrade.ports.notification import NotificationService


class TelegramNotifier(NotificationService):
    def __init__(self, bot_key: str, chat_id: str):
        self.bot_key = bot_key
        self.chat_id = chat_id

    def send(self, message: str):
        url = f"https://api.telegram.org/bot{self.bot_key}/sendMessage"
        print(f"Sending Telegram message: {message}")
        requests.post(
            url,
            json={
                "chat_id": self.chat_id,
                "text": message,
            },
        )
