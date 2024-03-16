from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes
from telegram import ForceReply
import pytest
import sys
sys.path.append("/app/bot_telegram")
from bot_telegram.telegram_server import start

import pytest
from bot_telegram.telegram_server import *
from telegram.ext import ContextTypes
from telegram import Update
from unittest.mock import Mock


# Mock the Update object to be passed to the test functions

# Additional tests can be added as needed for other functions
