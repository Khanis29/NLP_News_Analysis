# run_pipeline.py
"""
Pipeline for the NYT readability project (NYT only, no NewsAPI).

Steps:
1. Load event definitions
2. Download NYT articles
3. Clean text
4. Build features (readability + lexical rarity)
5. Run DiD regressions + placebo tests
6. Make plots and save everything to disk
"""

from pathlib import Path
import pandas as pd

from src.events import historical_events_df
from src.fetch_nyt import build_articles_index_nyt
from src.clean import clean_articles
from src.features import build_features
from src.analyze import (
    run_did_regression,
    describe_pre_post,
    run_fk_rarity_interaction,
    run_placebo_tests,
)
from src import plots



# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = Path("figures")

for d in [INTERIM_DIR, PROCESSED_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Main NYT pipeline
# ---------------------------------------------------------------------
DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = Path("figures")


def run_pipeline_nyt(n_placebo: int = 200) -> None:
    """End-to-end pipeline for historical NYT events."""

    events = historical_events_df()

    #Fetch NYT articles
    articles = build_articles_index_nyt(events)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    articles.to_parquet(INTERIM_DIR / "nyt_raw.parquet", index=False)

    #clean text
    articles_text = clean_articles(articles, min_words=150)
    articles_text.to_parquet(
        INTERIM_DIR / "articles_text_nyt.parquet", index=False
    )
    
    #Build features (FK + rarity)
    feats = build_features(
        articles_text,
        use_lingua=True,
        lingua_sample_cap=1500,
    )
    feats["dataset"] = "nyt_historical"
    feats.to_parquet(PROCESSED_DIR / "features_nyt.parquet", index=False)

    #Run DiD + interaction + placebo tests
    print("\n================ NYT regressions ================")
    run_did_regression(feats, outcome="fk_grade")
    run_did_regression(feats, outcome="avg_rarity")
    run_did_regression(feats, outcome="rare_share")

    run_fk_rarity_interaction(feats)

    print("\n================ Placebo tests ==================")
    run_placebo_tests(feats, outcome="fk_grade", n_placebo=n_placebo)

    #Descriptive stats + plots
    print("\n================ Descriptive stats ==============")
    describe_pre_post(feats)

    plots.plot_fk_pre_post_by_domain(
        feats, savepath=FIG_DIR / "nyt_fk_pre_post_by_domain.png"
    )
    plots.plot_fk_pre_post_by_support(
        feats, savepath=FIG_DIR / "nyt_fk_pre_post_by_support.png"
    )
    plots.plot_fk_vs_rarity_scatter(
        feats, savepath=FIG_DIR / "nyt_fk_vs_rarity_scatter.png"
    )
    plots.plot_rarity_pre_post_by_support(
        feats, savepath=FIG_DIR / "nyt_rarity_pre_post_by_support.png"
    )

    print("\n[NYT] Pipeline complete. Figures saved in 'figures/' directory.")


__all__ = ["run_pipeline_nyt"]
