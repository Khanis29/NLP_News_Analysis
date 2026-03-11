# src/fetch_nyt.py
from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

from src.nyt_client import search_articles

EVENT_QUERIES = {
    "AFG2009": "Afghanistan AND surge",
    "PAK2011": '"Osama bin Laden" OR Abbottabad',
    "LBY2011": "Libya AND (air strikes OR NATO OR Gaddafi)",
    "SYR2013": "Syria AND (chemical weapons OR sarin)",
    "IRQ2014": 'ISIS AND (airstrikes OR "air strikes")',
    "IRQ2016": "Iraq AND Mosul AND offensive",
    "SYR2017": 'Syria AND (missile strike OR "Tomahawk" OR Shayrat)',
    "SYR2018": "Syria AND Douma AND chemical",
}


def _to_datestr(x) -> str:
    """Return YYYY-MM-DD string from str/Timestamp/date."""
    if isinstance(x, str):
        return x[:10]
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse date: {x}")
    return ts.date().isoformat()


def _docs_to_df(docs: list[dict], event_row: pd.Series) -> pd.DataFrame:
    """
    Convert NYT docs to a DataFrame with event metadata.
    Note: raw_text is only snippet-like metadata; full text comes from scraping.
    """
    rows = []

    event_date = pd.to_datetime(event_row["event_date"]).date()

    for d in docs:
        pub_str = d.get("pub_date")
        if not pub_str:
            continue
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            pub_date = pub_dt.date()
        except Exception:
            continue

        phase = "pre" if pub_date < event_date else "post"

        web_url = d.get("web_url")
        if not web_url:
            continue

        domain = urlparse(web_url).netloc or ""

        headline_data = d.get("headline") or {}
        headline = headline_data.get("main", "")

        # Metadata “text” (not full article). Keep for debugging/logging only.
        raw_text = (d.get("lead_paragraph") or d.get("abstract") or d.get("snippet") or "").strip()

        rows.append(
            {
                "event_id": event_row["event_id"],
                "event_name": event_row.get("name"),
                "support_level": event_row.get("support_level"),
                "event_date": _to_datestr(event_row["event_date"]),
                "start": _to_datestr(event_row["start"]),
                "end": _to_datestr(event_row["end"]),
                "pub_date": pub_date.isoformat(),
                "phase": phase,
                "url": web_url,
                "domain": domain,
                "source": d.get("source"),
                "headline": headline,
                "raw_text": raw_text,
            }
        )

    return pd.DataFrame(rows)


def collect_for_event_row_nyt(event_row: pd.Series, per_event_cap: int = 50) -> pd.DataFrame:
    """
    Fetch up to per_event_cap docs for one event.
    Since NYT returns 10 docs/page, page_limit = ceil(per_event_cap/10).
    """
    event_id = event_row["event_id"]
    query = EVENT_QUERIES.get(event_id, event_row.get("name", event_id))

    start = _to_datestr(event_row["start"])
    end = _to_datestr(event_row["end"])

    page_limit = max(1, int((per_event_cap + 9) // 10))

    print(f"[NYT] Collecting {event_id} with query: '{query}'")
    docs = search_articles(query=query, begin_date=start, end_date=end, page_limit=page_limit)

    if not docs:
        print(f"[NYT] No docs for event {event_id}")
        return pd.DataFrame()

    df = _docs_to_df(docs, event_row)

    # Cap and de-dupe by URL
    if not df.empty:
        df = df.drop_duplicates(subset=["url"]).head(per_event_cap)

    if df.empty:
        print(f"[NYT] No usable docs for event {event_id}")
    else:
        print(f"[NYT] {len(df)} articles for event {event_id}")

    return df


def build_articles_index_nyt(events_df: pd.DataFrame, per_event_cap: int = 50) -> pd.DataFrame:
    """
    Loop over events, query NYT, and build combined article index.
    """
    frames: list[pd.DataFrame] = []

    for _, row in events_df.iterrows():
        try:
            df_ev = collect_for_event_row_nyt(row, per_event_cap=per_event_cap)
            if not df_ev.empty:
                frames.append(df_ev)
        except Exception as e:
            print(f"[NYT] Error on event {row.get('event_id')}: {e}")

    if not frames:
        print("[NYT] No data collected at all.")
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_name",
                "support_level",
                "event_date",
                "start",
                "end",
                "pub_date",
                "phase",
                "url",
                "domain",
                "source",
                "headline",
                "raw_text",
            ]
        )

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
    return df
