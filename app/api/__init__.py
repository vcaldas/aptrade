from flask import Blueprint

bp = Blueprint("api", __name__)

from app.api import time  # noqa: F401, E402
# This import is necessary to register the route defined in time.py
