# src/nyt_client.py
from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import requests
from requests import Response

BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"


def _get_key() -> str:
    # Fetch dynamically so changes to env vars take effect without restarting kernel
    key = os.getenv("NYT_API_KEY")
    if not key:
        raise RuntimeError(
            "NYT_API_KEY environment variable is not set. "
            "Please export your NYT Article Search API key before running."
        )
    return key


def _request_with_retries(
    params: dict,
    timeout: int = 30,
    max_attempts: int = 8,
    base_sleep: float = 1.5,
) -> Response:
    """
    Robust GET with exponential backoff.
    Handles:
      - 429 rate limit
      - transient connection/DNS failures
      - 5xx server errors
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=timeout)

            # Rate limit
            if r.status_code == 429:
                sleep_s = base_sleep * (2 ** (attempt - 1))
                print(f"[NYT] 429 Too Many Requests. Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue

            # Server hiccups
            if 500 <= r.status_code < 600:
                sleep_s = base_sleep * (2 ** (attempt - 1))
                print(f"[NYT] {r.status_code} server error. Sleeping {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** (attempt - 1))
            print(f"[NYT] Network error ({type(e).__name__}). Sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            continue
        except requests.exceptions.HTTPError as e:
            # Non-retryable most of the time (e.g., 401, 403, 400)
            raise

    raise RuntimeError(f"NYT request failed after {max_attempts} attempts: {last_err}")


def search_articles(
    query: str,
    begin_date: str,
    end_date: str,
    page_limit: int = 5,
    sleep_secs: float = 1.2,
) -> List[Dict]:
    """
    Query the NYT Article Search API and return list of doc dicts.

    begin_date/end_date can be 'YYYY-MM-DD' or 'YYYYMMDD'.
    page_limit controls max pages (10 articles/page).
    """
    api_key = _get_key()

    b = begin_date.replace("-", "")
    e = end_date.replace("-", "")

    all_docs: List[Dict] = []
    page = 0

    while page < page_limit:
        params = {
            "q": query,
            "begin_date": b,
            "end_date": e,
            "page": page,
            "api-key": api_key,
            # Optional: keep it deterministic-ish
            "sort": "oldest",
        }

        r = _request_with_retries(params=params, timeout=30)
        payload = r.json()

        docs = (payload.get("response") or {}).get("docs") or []
        if not docs:
            break

        all_docs.extend(docs)
        page += 1
        time.sleep(sleep_secs)

    return all_docs
