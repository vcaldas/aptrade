from __future__ import annotations

import datetime as dt
import logging

import typer

from aptrade import __version__ as APP_VERSION
from aptrade.core.config import settings
from aptrade.core.logger import get_logger
from aptrade.telegram_bot import (
    disable_telegram_notifications,
    enable_telegram_notifications,
)

APP_AUTHOR = "Victor Caldas"
APP_COPYRIGHT = "2025-2026"

CLI_HELP_TEXT = (
    "APTrade Command Line Interface\n\n"
    f"Author: {APP_AUTHOR}\n"
    f"Years: {APP_COPYRIGHT}\n"
    f"Version: {APP_VERSION}"
)

logger = get_logger()


app = typer.Typer(
    help=CLI_HELP_TEXT,
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)


def run_live_scanner(
    api_key: str,
    max_symbols: int = 100,
    min_volume: float = 1_000_000,
    min_price: float | None = 1.0,
    max_price: float | None = 100.0,
    start_time: dt.time = dt.time(4, 0),
    stop_time: dt.time = dt.time(20, 0),
    scanner_timezone: str = settings.SCANNER_TIMEZONE,
) -> None:
    from .scanner import run_live_scanner as scanner_runner

    scanner_runner(
        api_key=api_key,
        max_symbols=max_symbols,
        min_volume=min_volume,
        min_price=min_price,
        max_price=max_price,
        start_time=start_time,
        stop_time=stop_time,
        scanner_timezone=scanner_timezone,
    )


@app.callback()
def main(
    ctx: typer.Context,
    api: bool = typer.Option(
        True, "--api/--no-api", help="Enable or disable API routes."
    ),
    loglevel: str = typer.Option("info", "--loglevel", help="Set the log level."),
    telegram: bool = typer.Option(
        False, "--telegram", help="Enable Telegram messaging."
    ),
) -> None:
    resolved_level = loglevel.upper()
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    if telegram:
        enable_telegram_notifications()
    else:
        disable_telegram_notifications()

    ctx.obj = {
        "api": api,
        "loglevel": resolved_level,
        "telegram": telegram,
    }


@app.command()
def scanner(
    dev: bool = typer.Option(
        False,
        "--dev",
        help="Disable Telegram notifications for local development.",
    ),
):
    if dev:
        disable_telegram_notifications()
        logger.info("Running in development mode; Telegram notifications disabled")

    run_live_scanner(
        api_key=settings.MASSIVE_API_KEY,
        max_symbols=500,  # Start small
        min_volume=0.0,
        min_price=1.0,
        max_price=50.0,
        start_time=dt.time(4, 0),  # 4 AM ET
        stop_time=dt.time(20, 0),  # 8 PM ET
    )
