import asyncio
import logging
import threading
import time
from collections import defaultdict

from telegram import Bot, Update
from telegram.ext import ContextTypes
from telegram.request import HTTPXRequest

from aptrade.core.config import settings

BOTKEY = settings.TELEGRAM_BOTKEY

CHAT_ID = settings.CHAT_ID

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(
        self, bot_key: str = BOTKEY, chat_id: str = CHAT_ID, timeout: int = 30
    ):
        request = HTTPXRequest(connect_timeout=timeout, read_timeout=timeout)
        self.bot = Bot(token=bot_key, request=request)
        self.chat_id = chat_id
        self._loop = None
        self._thread = None
        self._start_background_loop()

        # Rate limiting: track last message time per ticker
        self._last_notification = defaultdict(float)
        self._min_notification_interval = (
            300  # 5 minutes between notifications for same ticker
        )
        self._notification_lock = threading.Lock()

        # Global rate limit to avoid hitting Telegram API limits
        self._last_message_time = 0
        self._min_message_interval = 1.0  # At least 1 second between any messages

    def _start_background_loop(self):
        """Start a background thread with a persistent event loop."""

        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=run_loop, args=(self._loop,), daemon=True
        )
        self._thread.start()

    def _run_async(self, coro):
        """Schedule a coroutine in the background event loop."""
        if self._loop is None or not self._loop.is_running():
            logger.warning("Background event loop not running, skipping notification")
            return

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Don't wait for result to avoid blocking
        return future

    async def hello(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:  # noqa: ARG002
        await update.message.reply_text(f"Hello {update.effective_user.first_name}")

    async def send_startup_message(self) -> None:
        await self.bot.send_message(
            chat_id=self.chat_id,
            message_thread_id=4,
            text="This is an automated message. Bot has started!",
        )

    async def _async_message(self, ticker: str, message: str) -> None:
        """Internal async method to send notification."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                message_thread_id=4,
                text=f"{ticker} : {message}",
            )
            logger.info(f"Telegram notification sent for {ticker}")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def _async_notify_new_ticker(self, ticker: str, pct_change: float) -> None:
        """Internal async method to send notification with rate limiting."""
        # Check if we should send notification for this ticker
        with self._notification_lock:
            current_time = time.time()
            last_time = self._last_notification.get(ticker, 0)

            # Skip if notified recently for this ticker
            if current_time - last_time < self._min_notification_interval:
                logger.debug(f"Skipping notification for {ticker} (rate limited)")
                return

            # Global rate limit: wait if needed
            time_since_last_message = current_time - self._last_message_time
            if time_since_last_message < self._min_message_interval:
                sleep_time = self._min_message_interval - time_since_last_message
                await asyncio.sleep(sleep_time)

            # Update tracking
            self._last_notification[ticker] = current_time
            self._last_message_time = time.time()

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                message_thread_id=4,
                text=f"New ticker detected: {ticker} with change {pct_change:.2f}%",
            )
            logger.info(f"Telegram notification sent for {ticker}")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def notify_new_ticker(self, ticker: str, pct_change: float) -> None:
        """Synchronous wrapper that schedules the async notification.

        This can be called from non-async code (like strategies).
        """
        self._run_async(self._async_notify_new_ticker(ticker, pct_change))

    def send_message(self, ticker: str, message: str) -> None:
        self._run_async(self._async_message(ticker, message))


telegram_bot = TelegramBot()
