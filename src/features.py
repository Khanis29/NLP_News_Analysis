# src/features.py
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from src.metrics import readability_scores
from src.lingual import rarity_features


def build_features(
    articles_with_text: pd.DataFrame,
    use_lingua: bool = True,
    lingua_sample_cap: Optional[int] = 1500,
) -> pd.DataFrame:
    """
    Event-focused framework:
      outcome ~ post_event + event fixed effects (+ optional domain FE)

    Requires:
      - event_id
      - event_name (optional)
      - event_date
      - pub_date
      - domain
      - phase ('pre'/'post') OR enough info to compute post_event
      - text_clean (scraped article text)
    """
    required = ["event_id", "domain", "text_clean", "pub_date", "event_date"]
    missing = [c for c in required if c not in articles_with_text.columns]
    if missing:
        raise KeyError(f"build_features missing columns: {missing}")

    df = articles_with_text.copy()
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    # If phase not present, compute it from pub_date vs event_date
    if "phase" not in df.columns:
        df["phase"] = np.where(df["pub_date"] >= df["event_date"], "post", "pre")

    # Readability
    metrics_rows: list[Dict[str, Any]] = []
    for i, row in df.iterrows():
        txt = row.get("text_clean", "")
        if not isinstance(txt, str) or not txt.strip():
            continue
        r = readability_scores(txt)
        metrics_rows.append(
            {
                "idx": i,
                "fk_grade": r["fk_grade"],
                "dale_chall": r["dale_chall"],
                "avg_sentence_len": r["avg_sentence_len"],
                "ttr": r["ttr"],
            }
        )
    mdf = pd.DataFrame(metrics_rows).set_index("idx")
    df = df.join(mdf)

    # Rarity (corpus-based within sample)
    if use_lingua:
        sub = df[["text_clean"]].rename(columns={"text_clean": "content"}).copy()
        rar = rarity_features(sub, sample_cap=lingua_sample_cap)
        df = df.join(rar)

    # Dummies
    df["post_event"] = (df["phase"] == "post").astype(int)

    # Simple domain bucket to reduce overfitting / sparse FE
    df["domain_simple"] = np.where(
        df["domain"].astype(str).str.contains("nytimes.com", case=False, na=False),
        "nytimes",
        "other",
    )

    return df
