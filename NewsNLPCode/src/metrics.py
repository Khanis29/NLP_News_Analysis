# src/metrics.py
"""
Low-level text metrics:
- Readability scores (Flesch-Kincaid, Dale–Chall)
- Basic tokenization helpers

Depends on:
  - spacy
  - textstat
Make sure you've run (in your environment):
  pip install spacy textstat
  python -m spacy download en_core_web_sm
"""

import spacy
from textstat import textstat

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model with sentencizer."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
        # ensure we have sentence boundaries
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def readability_scores(text: str):
    """
    Compute readability metrics for a single article.

    Returns a dict with:
      - fk_grade: Flesch-Kincaid grade level
      - dale_chall: Dale–Chall readability score
      - avg_sentence_len: average sentence length (tokens per sentence)
      - ttr: type-token ratio (unique/total tokens)
    """
    if text is None:
        text = ""
    text = text.strip()

    # Require at least some length; otherwise return NAs
    if len(text.split()) < 80:
        return {
            "fk_grade": None,
            "dale_chall": None,
            "avg_sentence_len": None,
            "ttr": None,
        }

    nlp = _get_nlp()
    doc = nlp(text)

    sents = list(doc.sents)
    tokens_alpha = [t for t in doc if t.is_alpha]
    words_clean = [t.text for t in tokens_alpha]
    types = set(w.lower() for w in words_clean)

    avg_sent_len = len(words_clean) / max(1, len(sents))
    ttr = len(types) / max(1, len(words_clean))

    fk_grade = float(textstat.flesch_kincaid_grade(text))
    dale = float(textstat.dale_chall_readability_score(text))

    return {
        "fk_grade": fk_grade,
        "dale_chall": dale,
        "avg_sentence_len": avg_sent_len,
        "ttr": ttr,
    }
