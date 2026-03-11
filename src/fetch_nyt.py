# src/fetch_nyt.py

from datetime import datetime
from urllib.parse import urlparse
from typing import List

import pandas as pd

from src.nyt_client import nyt_search
from src.events import historical_events_df


# Mapping from event_id -> NYT query string
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


def _docs_to_df(docs: List[dict], event_row: pd.Series) -> pd.DataFrame:
    """
    Convert a list of NYT docs into a tidy DataFrame with event metadata.

    Expected columns in output:
      - title
      - url
      - domain
      - pub_date (ISO string)
      - source
      - event_id
      - phase ('pre' or 'post')
    """
    if not docs:
        return pd.DataFrame(
            columns=[
                "title",
                "url",
                "domain",
                "pub_date",
                "source",
                "event_id",
                "phase",
            ]
        )

    rows = []
    # center_date, start, end *already computed* and stored in event_row
    center_date = datetime.fromisoformat(event_row["event_date"]).date()
    start_date = datetime.fromisoformat(event_row["start"]).date()

    for d in docs:
        # Fallbacks to avoid KeyError if fields are missing
        web_url = d.get("web_url", "")
        headline = d.get("headline", {})
        title = headline.get("main") or d.get("snippet") or ""

        pub_date_raw = d.get("pub_date")
        try:
            pub_dt = datetime.fromisoformat(pub_date_raw.replace("Z", "+00:00"))
            pub_date = pub_dt.date()
            pub_iso = pub_dt.date().isoformat()
        except Exception:
            pub_date = None
            pub_iso = None

        phase = None
        if pub_date is not None:
            phase = "pre" if pub_date < center_date else "post"

        domain = urlparse(web_url).netloc

        rows.append(
            {
                "title": title,
                "url": web_url,
                "domain": domain,
                "pub_date": pub_iso,
                "source": d.get("source"),
                "event_id": event_row["event_id"],
                "phase": phase,
            }
        )

    return pd.DataFrame(rows)


def collect_for_event_row_nyt(event_row: pd.Series) -> pd.DataFrame:
    """
    For a single event row, call NYT API and build a DataFrame of articles.

    `event_row` must at least contain:
      - event_id
      - event_date (ISO str)
      - start (ISO str)
      - end (ISO str)
    """
    event_id = event_row["event_id"]
    query = EVENT_QUERIES.get(event_id)

    if query is None:
        print(f"[NYT] No query configured for event {event_id}, skipping.")
        return pd.DataFrame()

    print(f"[NYT] Collecting {event_id} with query: '{query}'")

    # NYT Article Search expects dates in YYYYMMDD
    start_iso = event_row["start"].replace("-", "")
    end_iso = event_row["end"].replace("-", "")

    docs = nyt_search(
        query=query,
        begin_date=start_iso,
        end_date=end_iso,
        page_size=10,
        max_pages=5,
        sleep_between=1.0,
    )

    if not docs:
        return pd.DataFrame()

    df_ev = _docs_to_df(docs, event_row)
    print(f"[NYT] {len(df_ev)} articles for event {event_id}")
    return df_ev


def build_articles_index_nyt(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over all historical events and stack the article-level DataFrames.
    """
    frames = []
    for _, r in events_df.iterrows():
        try:
            df_ev = collect_for_event_row_nyt(r)
            if df_ev.empty:
                print(f"[NYT] No articles for event {r['event_id']}")
            else:
                frames.append(df_ev)
        except Exception as e:
            print(f"[NYT] Error on event {r['event_id']}: {e}")
            continue

    if not frames:
        print("[NYT] No NYT data collected at all.")
        return pd.DataFrame(
            columns=[
                "title",
                "url",
                "domain",
                "pub_date",
                "source",
                "event_id",
                "phase",
            ]
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
