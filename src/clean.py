# src/clean.py
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            timeout=25,
            headers={"User-Agent": "Mozilla/5.0 (ds385-nyt-project)"},
        )
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # NYT pages often keep article text in <p>
    ps = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in ps)

    # Basic cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_articles(articles_df: pd.DataFrame, min_words: int = 60) -> pd.DataFrame:
    """
    Takes the NYT index (URLs + metadata), scrapes HTML, extracts text into `text_clean`.
    Keeps rows with >= min_words.
    """
    rows = []
    total = 0

    for _, a in articles_df.iterrows():
        total += 1
        url = a.get("url")
        if not isinstance(url, str) or not url.strip():
            continue

        html = fetch_html(url)
        if not html:
            continue

        text = extract_main_text(html)
        n_words = len(text.split())
        if n_words < min_words:
            continue

        row = a.to_dict()
        row["text_clean"] = text
        row["word_count"] = int(n_words)
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"[clean] Kept {len(out)} of {total} NYT articles with >= {min_words} words.")
    return out
