import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from telegram import Update
from telegram.ext import ContextTypes

from aptrade.telegram_bot import TelegramBot


@pytest.fixture
def mock_bot(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    bot_stub = SimpleNamespace(send_message=AsyncMock())

    def fake_bot(token: str, **kwargs: Any) -> Any:  # noqa: ARG001 - token and kwargs not used in stub
        return bot_stub

    monkeypatch.setattr("aptrade.telegram_bot.Bot", fake_bot)
    return cast(AsyncMock, bot_stub.send_message)


def test_send_startup_message(mock_bot: AsyncMock) -> None:
    telegram_bot = TelegramBot(bot_key="dummy-token", chat_id="chat-id")

    asyncio.run(telegram_bot.send_startup_message())

    mock_bot.assert_awaited_once_with(
        chat_id="chat-id",
        message_thread_id=4,
        text="This is an automated message. Bot has started!",
    )


def test_notify_new_ticker(mock_bot: AsyncMock) -> None:
    telegram_bot = TelegramBot(bot_key="dummy-token", chat_id="chat-id")

    telegram_bot.notify_new_ticker("AAPL", 3.5)

    # Give the background loop time to process
    import time

    time.sleep(0.1)

    mock_bot.assert_awaited_once_with(
        chat_id="chat-id",
        message_thread_id=4,
        text="New ticker detected: AAPL with change 3.50%",
    )


def test_hello_replies_with_user_name(mock_bot: AsyncMock) -> None:
    telegram_bot = TelegramBot(bot_key="dummy-token", chat_id="chat-id")

    reply = AsyncMock()
    update = SimpleNamespace(
        message=SimpleNamespace(reply_text=reply),
        effective_user=SimpleNamespace(first_name="Alice"),
    )

    asyncio.run(
        telegram_bot.hello(
            cast(Update, update),
            cast(ContextTypes.DEFAULT_TYPE, object()),
        )
    )

    reply.assert_awaited_once_with("Hello Alice")
