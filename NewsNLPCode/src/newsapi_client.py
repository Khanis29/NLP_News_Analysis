# src/newsapi_client.py

import os
import time
from typing import List, Dict

import requests

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"


def newsapi_search(
    query: str,
    sources: List[str],
    from_date: str,
    to_date: str,
    page_size: int = 50,
    max_pages: int = 2,
    sleep_between: float = 1.0,
) -> List[Dict]:
    """
    Call NewsAPI /v2/everything and return a list of article dicts.

    from_date, to_date: 'YYYY-MM-DD'
    sources: list of NewsAPI source IDs (e.g. ['cnn', 'fox-news']).
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY environment variable is not set")

    all_articles: List[Dict] = []
    sources_param = ",".join(sources)
    page = 1

    while page <= max_pages:
        params = {
            "q": query,
            "sources": sources_param,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "pageSize": page_size,
            "page": page,
            "apiKey": NEWSAPI_KEY,
        }

        r = requests.get(NEWSAPI_URL, params=params, timeout=30)
        if r.status_code == 429:
            wait = 5 * page
            print(f"[NewsAPI] 429 Too Many Requests. Sleeping {wait}s...")
            time.sleep(wait)
            page += 1
            continue

        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            break

        all_articles.extend(articles)

        total = data.get("totalResults", 0)
        if len(all_articles) >= total:
            break

        page += 1
        time.sleep(sleep_between)

    return all_articles
