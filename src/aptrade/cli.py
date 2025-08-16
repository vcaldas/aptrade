# Create a cli entry to start the fastapi server
import click
import uvicorn


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--dev", is_flag=True, help="Run server in development mode (with reload)"
)
def server(dev):
    """Start the FastAPI server."""
    if dev:
        uvicorn.run("aptrade.server:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run("aptrade.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    cli()
