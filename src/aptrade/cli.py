from __future__ import annotations

import datetime as dt

import typer

from aptrade.core.config import settings
from aptrade.telegram_bot import disable_telegram_notifications
from aptrade.version import VERSION as APP_VERSION
from aptrade.core.logger import get_logger

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
    help=CLI_HELP_TEXT, no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main(
    api: bool = typer.Option(
        True, "--api/--no-api", help="Enable or disable API routes."
    ),
    scanner: bool = typer.Option(
        True, "--scanner/--no-scanner", help="Enable or disable the stock scanner."
    ),
    loglevel: str = typer.Option("info", "--loglevel", help="Set the log level."),
    telegram: bool = typer.Option(
        False, "--telegram", help="Enable Telegram messaging."
    ),
) -> None:
    typer.echo("Hello Aptrade")
    typer.echo(f"api={api}")
    typer.echo(f"scanner={scanner}")
    typer.echo(f"loglevel={loglevel}")
    typer.echo(f"telegram={telegram}")