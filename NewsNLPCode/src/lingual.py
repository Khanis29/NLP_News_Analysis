# src/lingual.py
"""
Lexical rarity metrics.

Right now this is a *self-contained* rarity measure that does NOT call
external APIs (like Lingua Robot), so you won't hit extra rate limits.

Idea:
  - Tokenize text with spaCy.
  - Work with lemmas of alphabetic tokens.
  - Define a "surprisal"-style rarity score from within-article frequencies.
  - Summarize:
        avg_rarity  = average surprisal across tokens
        rare_share  = share of tokens above a rarity threshold
"""

import math
from collections import Counter

import spacy

_nlp = None


def _get_nlp():
    """Lazy-load a lightweight spaCy model for tokenization/lemmatization."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
    return _nlp


def compute_rarity_scores(text: str, sample_cap: int = 3000) -> dict:
    """
    Compute simple lexical rarity metrics for an article.

    Parameters
    ----------
    text : str
        Clean article text.
    sample_cap : int
        Maximum number of tokens to use (truncate to avoid huge docs).

    Returns
    -------
    dict with keys:
        - avg_rarity: average -log(freq) across tokens
        - rare_share: share of tokens with rarity above threshold
    """
    if text is None:
        text = ""
    text = text.strip()
    if not text:
        return {"avg_rarity": None, "rare_share": None}

    nlp = _get_nlp()
    doc = nlp(text)

    # Use lemmas of alphabetic tokens
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha]
    if not tokens:
        return {"avg_rarity": None, "rare_share": None}

    if len(tokens) > sample_cap:
        tokens = tokens[:sample_cap]

    counts = Counter(tokens)
    total = sum(counts.values())

    # frequency p(w) = count / total; rarity = -log(p(w))
    rarity_by_type = {
        w: -math.log(c / total) for w, c in counts.items()
    }

    rarities = [rarity_by_type[w] for w in tokens]

    if not rarities:
        return {"avg_rarity": None, "rare_share": None}

    avg_rarity = float(sum(rarities) / len(rarities))

    # Define "rare" as rarer than the 75th percentile within the article
    sorted_r = sorted(rarities)
    idx = int(0.75 * (len(sorted_r) - 1))
    threshold = sorted_r[idx]

    rare_tokens = sum(r > threshold for r in rarities)
    rare_share = float(rare_tokens / len(rarities))

    return {"avg_rarity": avg_rarity, "rare_share": rare_share}
