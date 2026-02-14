import typer

from aptrade.version import VERSION as APP_VERSION

APP_AUTHOR = "Victor Caldas"
APP_COPYRIGHT = "2025-2026"

CLI_HELP_TEXT = (
    "APTrade Command Line Interface\n\n"
    f"Author: {APP_AUTHOR}\n"
    f"Years: {APP_COPYRIGHT}\n"
    f"Version: {APP_VERSION}"
)

cli = typer.Typer(help=CLI_HELP_TEXT, no_args_is_help=True)


@cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
):
    """APTrade CLI entrypoint. Use `--version` to print version and exit."""
    if version:
        print(APP_VERSION)
        raise typer.Exit()


@cli.command()
def server(
    dev: bool = typer.Option(
        False, "--dev", "-d", help="Run server in development mode (with reload)"
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    """Start the APTrade Server."""
    print("Starting APTrade Server...")


@cli.command()
def scanner():
    """Start the APTrade Scanner Service."""
    print("Starting APTrade Scanner Service...")

@cli.command()
def backtest(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable verbose output and keep running after individual ticker failures",
    ),
):
    """Run backtests described in the given YAML configuration."""
    print("Running backtests...")

if __name__ == "__main__":
    cli()
