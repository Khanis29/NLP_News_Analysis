# src/fetch_newsapi.py

import pandas as pd
from urllib.parse import urlparse

from src.newsapi_client import newsapi_search

INTERVENTION_TAGS = [
    "intervention",
    "air strike",
    "airstrike",
    "coalition forces",
    "no-fly zone",
    "regime change",
    "deployment",
    "missile strike",
]

# NewsAPI source IDs (you can edit this list)
NEWSAPI_SOURCES = [
    "cnn",
    "fox-news",
    "the-new-york-times",
    "the-wall-street-journal",
    "associated-press",
]


def _articles_to_df(articles, event_id: str, phase: str) -> pd.DataFrame:
    rows = []
    for a in articles:
        url = a.get("url")
        if not url:
            continue

        parsed = urlparse(url)
        domain = parsed.netloc

        published = a.get("publishedAt") or ""
        seendate = published[:10] if len(published) >= 10 else None

        rows.append(
            {
                "title": a.get("title"),
                "url": url,
                "domain": domain,
                "seendate": seendate,
                "language": "en",
                "sourcecountry": None,
                "event_id": event_id,
                "phase": phase,
            }
        )
    return pd.DataFrame(rows)


def collect_for_event_row_newsapi(row) -> pd.DataFrame:
    query = " OR ".join(INTERVENTION_TAGS)

    pre_articles = newsapi_search(
        query=query,
        sources=NEWSAPI_SOURCES,
        from_date=row["start"],
        to_date=row["event_date"],
        page_size=50,
        max_pages=1,
    )

    post_articles = newsapi_search(
        query=query,
        sources=NEWSAPI_SOURCES,
        from_date=row["event_date"],
        to_date=row["end"],
        page_size=50,
        max_pages=1,
    )

    df_pre = _articles_to_df(pre_articles, row["event_id"], "pre")
    df_post = _articles_to_df(post_articles, row["event_id"], "post")

    return pd.concat([df_pre, df_post], ignore_index=True)


def build_articles_index_newsapi(events_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, r in events_df.iterrows():
        print(f"[NewsAPI] Collecting for {r['event_id']} ({r['start']} → {r['end']})")
        try:
            df_ev = collect_for_event_row_newsapi(r)
            if not df_ev.empty:
                frames.append(df_ev)
            else:
                print(f"[NewsAPI] No articles for event {r['event_id']}")
        except Exception as e:
            print(f"[NewsAPI] Error on event {r['event_id']}: {e}")
            continue

    if not frames:
        print("[NewsAPI] No data collected at all.")
        return pd.DataFrame(
            columns=[
                "title", "url", "domain", "seendate",
                "language", "sourcecountry", "event_id", "phase",
            ]
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
