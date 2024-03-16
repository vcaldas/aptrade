from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import xmltodict
import os
import feedparser


ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN")


app = FastAPI()

# Defina as origens permitidas
origins = [
    ALLOWED_ORIGIN
]

# Adicione o middleware CORS com as origens especificadas
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use a lista de origens definida acima
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml"

def get_news_from_feed():
    """
    Retrieves news items from a RSS feed URL and returns a list of news items.

    This function sends a GET request to the RSS feed URL and parses the
    response content into an ordered dictionary.
    The function assumes that the RSS feed URL is in a specific format
    and structure.

    Returns:
    A list of news items, where each item is a dictionary containing the
    title, source URL, and link of the news article.
    """
    request = requests.get(url)
    news_data = feedparser.parse(url)

    return news_data.get("entries", {})


@app.get("/items")
def get_items_with_urls():
    news_items = get_news_from_feed()
    items_with_urls = [
        {
            "title": item["title"],
            "source_url": item.get("source", {}).get("@url"),
            "link": item["link"]
        } for item in news_items
    ]

    return items_with_urls


@app.get("/")
def main():
    """
    This is the main function that returns news from a feed.
    """
    try:
        return get_news_from_feed()
    except Exception as e:
        return {"error": str(e)}
