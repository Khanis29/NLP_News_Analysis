# src/clean.py

from __future__ import annotations

import re
import pandas as pd


def clean_articles(df: pd.DataFrame, min_words: int = 40) -> pd.DataFrame:
    """
    Generic cleaning for NYT articles.

    Expects a column 'text' containing the short article text
    (abstract/lead_paragraph/snippet concatenation from the NYT API).

    Steps:
      - strip HTML tags (defensive)
      - squash whitespace
      - filter out very short pieces (< min_words)
      - add 'clean_text' and 'word_count'
    """
    if "text" not in df.columns:
        raise KeyError(
            f"clean_articles expected a 'text' column, but got columns: {list(df.columns)}"
        )

    df = df.copy()

    # Basic cleanup
    cleaned = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.replace(r"<[^>]+>", " ", regex=True)   # remove HTML tags
        .str.replace(r"\s+", " ", regex=True)       # collapse whitespace
        .str.strip()
    )

    df["clean_text"] = cleaned
    df["word_count"] = df["clean_text"].str.split().str.len()

    before = len(df)
    df = df[df["word_count"] >= min_words].copy()
    after = len(df)

    print(
        f"[CLEAN] Kept {after}/{before} articles "
        f"(min_words={min_words})."
    )

    return df
