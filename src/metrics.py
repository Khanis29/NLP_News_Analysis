from __future__ import annotations

import numpy as np
import textstat


def readability_scores(text: str, min_words: int = 15):
    """
    Compute readability + simple lexical metrics.

    IMPORTANT CHANGE:
    The NYT Article Search API does not provide full article bodies, so our
    cleaned text is often short. The earlier version required >=120 words,
    which caused nearly all rows to become None and regressions to collapse.

    We now compute metrics for shorter texts (default min_words=15) and return
    NaN only if text is empty / extremely short.
    """
    if not isinstance(text, str):
        return {"fk_grade": np.nan, "dale_chall": np.nan, "avg_sentence_len": np.nan, "ttr": np.nan}

    words = text.split()
    if len(words) < min_words:
        return {"fk_grade": np.nan, "dale_chall": np.nan, "avg_sentence_len": np.nan, "ttr": np.nan}

    # FK grade + Dale-Chall
    try:
        fk = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        fk = np.nan

    try:
        dc = float(textstat.dale_chall_readability_score(text))
    except Exception:
        dc = np.nan

    # Avg sentence length (very rough)
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) == 0:
        avg_sent_len = np.nan
    else:
        avg_sent_len = float(len(words) / len(sentences))

    # Type-token ratio
    uniq = len(set(w.lower() for w in words))
    ttr = float(uniq / max(len(words), 1))

    return {
        "fk_grade": fk,
        "dale_chall": dc,
        "avg_sentence_len": avg_sent_len,
        "ttr": ttr,
    }
