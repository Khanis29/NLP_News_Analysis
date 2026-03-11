# src/lingual.py
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 2]


def rarity_features(
    text_or_df,
    sample_cap: Optional[int] = None,
) -> pd.DataFrame | Dict[str, float]:
    """
    Robust rarity proxy that does NOT depend on external NLP models.

    Two modes:
    - If given a string: returns dict {"avg_rarity": ..., "rare_share": ...}
      using a trivial fallback (cannot compute corpus rarity from one doc),
      so it returns NaNs.
    - If given a DataFrame with column 'content': returns a DataFrame with
      avg_rarity and rare_share computed against the corpus frequency
      of the provided sample (up to sample_cap rows).

    This is enough to keep your pipeline running and yields interpretable
    rarity metrics: rarer words = lower corpus frequency.
    """
    if isinstance(text_or_df, str):
        return {"avg_rarity": np.nan, "rare_share": np.nan}

    df = text_or_df.copy()
    if "content" not in df.columns:
        raise KeyError("rarity_features expects a DataFrame with a 'content' column.")

    if sample_cap is not None and sample_cap > 0:
        df = df.iloc[:sample_cap].copy()

    toks_list = [_tokenize(t) for t in df["content"].astype(str).tolist()]
    corpus_counts = Counter(t for toks in toks_list for t in toks)
    total = sum(corpus_counts.values()) or 1

    def doc_stats(toks: List[str]) -> Dict[str, float]:
        if not toks:
            return {"avg_rarity": np.nan, "rare_share": np.nan}
        freqs = np.array([corpus_counts[t] / total for t in toks], dtype=float)
        # "rarity" = -log(freq); higher = rarer
        rarity = -np.log(freqs + 1e-12)
        avg_rarity = float(np.mean(rarity))
        # rare_share: share of tokens in the bottom 10% by corpus frequency
        thresh = np.quantile(freqs, 0.10)
        rare_share = float(np.mean(freqs <= thresh))
        return {"avg_rarity": avg_rarity, "rare_share": rare_share}

    rows = [doc_stats(toks) for toks in toks_list]
    out = pd.DataFrame(rows, index=df.index)
    return out
