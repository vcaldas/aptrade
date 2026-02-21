import datetime as dt

import typer

from aptrade.core.config import settings
from aptrade.scanner import run_live_scanner
from aptrade.version import VERSION as APP_VERSION
import os
from aptrade import ResearchEngine
from aptrade.infrastructure.telegram_notifier import TelegramNotifier

APP_AUTHOR = "Victor Caldas"
APP_COPYRIGHT = "2025-2026"

CLI_HELP_TEXT = (
    "APTrade Command Line Interface\n\n"
    f"Author: {APP_AUTHOR}\n"
    f"Years: {APP_COPYRIGHT}\n"
    f"Version: {APP_VERSION}"
)

cli = typer.Typer(help=CLI_HELP_TEXT, no_args_is_help=True)


@cli.command()
def scanner():
    """Start the APTrade Scanner Service."""
    # Run with conservative limits for testing
    # Runs daily from 4 AM to 8 PM ET
    run_live_scanner(
        api_key=settings.MASSIVE_API_KEY,
        max_symbols=30,  # Start small
        min_volume=500_000,
        min_price=None,
        max_price=50.0,
        start_time=dt.time(4, 0),  # 4 AM ET
        stop_time=dt.time(20, 0),  # 8 PM ET
    )


@cli.command()
def test():
    """Run a test command to verify setup."""
    bot_key = os.getenv("APTRADE_TELEGRAM_BOTKEY")

    notifier = None
    print(f"APTRADE_TELEGRAM_BOTKEY: {bot_key}")
    if bot_key:
        notifier = TelegramNotifier(bot_key=bot_key, chat_id="...")

    engine = ResearchEngine(notifier=notifier)
    engine.run_backtest(config=None)
    print("APTrade CLI is working! Hello from the test command.")


if __name__ == "__main__":
    cli()
