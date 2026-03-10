# src/nyt_client.py

import os
import time
from typing import List, Dict

import requests

NYT_KEY = os.getenv("NYT_API_KEY")
NYT_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"


def nyt_search(
    query: str,
    begin_date: str,   # 'YYYYMMDD'
    end_date: str,     # 'YYYYMMDD'
    page_size: int = 10,
    max_pages: int = 5,
    sleep_between: float = 1.0,
) -> List[Dict]:
    """
    Lightweight wrapper around the NYT Article Search API.

    Parameters
    ----------
    query : str
        Search query string (e.g., 'Afghanistan AND surge').
    begin_date : str
        Start date in 'YYYYMMDD' format.
    end_date : str
        End date in 'YYYYMMDD' format.
    page_size : int
        Number of results per page (NYT uses `page` as an index).
    max_pages : int
        Maximum number of pages to request (NYT hard limit is 100).
    sleep_between : float
        Seconds to sleep between calls to avoid 429s.

    Returns
    -------
    List[dict]
        List of NYT article docs from the API.
    """
    if not NYT_KEY:
        raise RuntimeError("NYT_API_KEY environment variable is not set.")

    all_docs: List[Dict] = []
    page = 0

    while page < max_pages:
        params = {
            "q": query,
            "api-key": NYT_KEY,
            "begin_date": begin_date,
            "end_date": end_date,
            "page": page,
        }

        try:
            resp = requests.get(NYT_URL, params=params, timeout=20)
        except Exception as e:
            print(f"[NYT] Request failed on page {page}: {e}")
            break

        # Handle rate limiting
        if resp.status_code == 429:
            print("[NYT] 429 Too Many Requests. Sleeping 5s...")
            time.sleep(5)
            continue

        resp.raise_for_status()
        data = resp.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break

        all_docs.extend(docs)
        page += 1
        time.sleep(sleep_between)

    return all_docs
