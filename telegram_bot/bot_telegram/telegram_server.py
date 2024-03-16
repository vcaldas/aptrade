# Substitua 'YOUR_TELEGRAM_BOT_TOKEN' pelo token do seu bot fornecido pelo @BotFather
from typing import Dict, List
import aiohttp
import os

TOKEN = os.environ.get("TELEGRAM_TOKEN")
URL = os.environ.get("WEB_API_URL")

import logging

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackContext
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Define a few command handlers. These usually take the two arguments update and


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"testando bot? {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_html("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    user = update.effective_user
    print("echo ", user.mention_html())
    await update.message.reply_text(update.message.text)


async def get_news() -> List[Dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{URL}/items") as response:
            return await response.json()


async def noticia(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    news = await get_news()
    await update.message.reply_text(news[0]["title"])
    await update.message.reply_text(news[0]["link"])
    await update.message.reply_text(
        'A primeira notícia eh \n-------------------'
    )


async def noticia_t(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:

    news = await get_news()
    for i in range(3):
        await update.message.reply_text(news[i]["title"])
    await update.message.reply_text(
        "Todas as noticias da API \n-------------------"
    )


async def noticias(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    news = await get_news()
    for i in range(len(news)):
        await update.message.reply_text(news[i]["title"])
        await update.message.reply_text(news[i]["link"])
    await update.message.reply_text(
        "Todas as noticias da API \n-------------------"
    )


async def menu(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /menu is issued."""
    menu_message = (
        "Comandos disponíveis:\n"
        "/start - Iniciar o bot\n"
        "/help - Mostrar ajuda\n"
        "/noticia - Obter uma notícia\n"
        "/noticias - Listar notícias\n"
        "/noticia_t - Notícia específica\n"
    )
    await update.message.reply_text(menu_message)



def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    handlers = {
        "start": start,
        "help": help_command,
        "menu": menu,  # Adicione esta linha
        "noticia": noticia,
        "noticias": noticias,
        "noticia_t": noticia_t
    }
    for command, handler in handlers.items():
        application.add_handler(CommandHandler(command, handler))


    # on non command i.e message - echo the message on Telegram
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, echo)
    )

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
