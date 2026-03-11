# src/features.py

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.metrics import readability_scores
from src.lingual import rarity_features


def build_features(
    df: pd.DataFrame,
    use_lingua: bool = True,
    lingua_sample_cap: Optional[int] = 1500,
) -> pd.DataFrame:
    """
    Build readability + lexical rarity features and add DiD dummies.

    Requires columns:
      - clean_text  (from clean_articles)
      - event_id
      - phase       ('pre' or 'post')
      - support_level
      - domain

    Returns a DataFrame that includes:
      - fk_grade, dale_chall, avg_sentence_len, ttr
      - avg_rarity, rare_share (if use_lingua and key present, else None)
      - post_event, low_support, interaction
    """
    if df.empty:
        raise ValueError(
            "build_features received an empty DataFrame. "
            "Most likely the cleaning step filtered out all articles. "
            "Try lowering min_words in clean_articles or inspect the raw NYT data."
        )

    if "clean_text" not in df.columns:
        raise KeyError(
            f"build_features expected a 'clean_text' column, but got columns: {list(df.columns)}"
        )

    rows = []
    for _, r in df.iterrows():
        base = r.to_dict()

        text = r["clean_text"]
        rd = readability_scores(text)

        if use_lingua:
            lex = rarity_features(text, sample_cap=lingua_sample_cap or 1500)
        else:
            lex = {"avg_rarity": None, "rare_share": None}

        rows.append({**base, **rd, **lex})

    feats = pd.DataFrame(rows)

    # DiD helpers
    feats["post_event"] = (feats["phase"] == "post").astype(int)
    feats["low_support"] = (feats["support_level"] == "low").astype(int)
    feats["interaction"] = feats["post_event"] * feats["low_support"]

    return feats
