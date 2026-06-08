from __future__ import annotations

import typer


app = typer.Typer(
    add_completion=False,
    help="Aptrade command line interface.",
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