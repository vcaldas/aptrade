import logging
import os, csv
import pandas
import yaml
import talib
import yfinance as yf

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from aptrade.patterns import candlestick_patterns
from app.api.assets.data import DATA_STOCK
dir_path = os.path.dirname(os.path.realpath(__file__))

# Automatically determine all api routes dynamically if they are placed
# under the ./app/api folder and have the correct information the
# corresponding __init__.py file
from .api.api_routes import get_dynamic_api_details

# Import the get_settings from the config module
from .config.config import get_settings

# Set the loglevel of the root-logger based on environmnet variable if set
app_log_level = os.getenv("APP_LOG_LEVEL")
if app_log_level:
    logging.getLogger().setLevel(app_log_level.upper())


# Get a reference to the settings (needed for rendering the templates and providing
# FastAPI with the current application version)
settings = get_settings()


# Dynamically search for API_ENDPOINT_ROUTER information in packages
# under folder /app/api
# This will give us both the routers as well as the tags descriptions
dynamic_routing_details, dynamic_metadata_info = get_dynamic_api_details()


# Initialize tags_metadata as empty list
tags_metadata = []

# Extend this with our dynamically obtained tags from the packages
# tags_metadata.extend(dynamic_metadata_info)

# This is when everything is hardcoded
#
# If the developer provided the information in the file api/tags_metadata.yml, supply
# this data to the FastAPI object
metadata_file = f"{os.path.dirname(__file__)}/api/tags_metadata.yml"
if os.path.exists(metadata_file):
    tags_metadata.extend(yaml.safe_load(open(metadata_file)).get("tags_metadata", []))

# Create the main FastAPI object, using the settings and the metadata
app = FastAPI(
    title="APTrade",
    version=settings.app_version,
    openapi_tags=tags_metadata,
)


# Make all files like css/js/etc in folder static available directly under /static url
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# All files under app/templates can be rendered as jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Include everythig that was automatically found under the app/api folder under the
# endpoint of /api
app.include_router(dynamic_routing_details, prefix="/api")


# If end-user goes to main page, automatically redirect to the index.html
@app.get("/", include_in_schema=False)
@app.get("/index", include_in_schema=False)
def root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request, "settings": settings})

# Any path accessed in url starting with /ui/ should be rendered as jinja2 template
@app.get("/screener", response_class=HTMLResponse, include_in_schema=False)
def serve_screener_html_page(request: Request, pattern = None):
    stocks = {}
    for entry in DATA_STOCK:
        _asset = DATA_STOCK[entry]
        stocks[_asset.symbol] = {"company": _asset.name}

    if pattern:
        for filename in os.listdir(os.path.join(dir_path,'./datasets/daily')):
            df = pandas.read_csv(os.path.join(dir_path,'./datasets/daily/{}'.format(filename)))
            pattern_function = getattr(talib, pattern)
            symbol = filename.split('.')[0]
            
            try:
                results = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
                last = results.tail(1).values[0]

                if last > 0:
                    stocks[symbol][pattern] = 'bullish'
                elif last < 0:
                    stocks[symbol][pattern] = 'bearish'
                else:
                    stocks[symbol][pattern] = None
            except Exception as e:
                print('failed on filename: ', filename, e)

    return templates.TemplateResponse('screener.html', {"request": request, "settings": settings, "candlestick_patterns":candlestick_patterns, 'pattern': pattern, 'stocks': stocks})

# Add health endpoint
@app.get("/health", status_code=status.HTTP_200_OK, include_in_schema=False)
def perform_healthcheck():
    return {"healthcheck": "Everything OK!"}
