# run_pipeline.py
from __future__ import annotations

from pathlib import Path

from src.events import load_events_nyt
from src.fetch_nyt import build_articles_index_nyt
from src.clean import clean_articles
from src.features import build_features
from src.analyze import (
    run_event_prepost_regression,
    describe_pre_post_by_event,
    run_placebo_tests,
)
from src import plots


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path("figures")

for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_pipeline_nyt(
    min_words: int = 60,
    per_event_cap: int = 50,
    use_lingua: bool = True,
    lingua_sample_cap: int = 1500,
    n_placebo: int = 200,
):
    # 1) Events
    events = load_events_nyt()
    print("[NYT] Events:")
    print(events)

    # 2) Fetch NYT index (URLs + metadata)
    articles_raw = build_articles_index_nyt(events, per_event_cap=per_event_cap)
    if articles_raw is None or articles_raw.empty:
        print("[NYT] No NYT articles collected. Check API key / queries.")
        return

    articles_raw.to_parquet(INTERIM_DIR / "articles_raw_nyt.parquet", index=False)

    # 3) Scrape + clean text
    articles_text = clean_articles(articles_raw, min_words=min_words)
    if articles_text is None or articles_text.empty:
        print("[NYT] After cleaning, no articles remain. Lower min_words or inspect scraping.")
        return

    articles_text.to_parquet(INTERIM_DIR / "articles_text_nyt.parquet", index=False)

    # 4) Features
    feats = build_features(
        articles_text,
        use_lingua=use_lingua,
        lingua_sample_cap=lingua_sample_cap,
    )

    # Save features (canonical location)
    feats_path = PROCESSED_DIR / "features_nyt.parquet"
    feats.to_parquet(feats_path, index=False)

    # Optional convenience copy at data/ root
    feats_root_path = DATA_DIR / "features_nyt.parquet"
    feats.to_parquet(feats_root_path, index=False)

    print(f"[save] Wrote features to: {feats_path}")
    print(f"[save] Also wrote features to: {feats_root_path}")

    print("\n================ NYT regressions (event pre/post) ================\n")

    for outcome in ["fk_grade", "avg_rarity", "rare_share"]:
        print(f"\n--- Pre/Post FE regression for {outcome} ---")
        try:
            model = run_event_prepost_regression(feats, outcome=outcome, use_domain_fe=True)
            print(model.summary())
            print("\nPre/Post means by event:")
            print(describe_pre_post_by_event(feats, outcome=outcome).head(20).to_string(index=False))
        except Exception as e:
            print(f"Skipping {outcome}: {e}")

    print("\n--- Placebo tests on FK grade ---")
    try:
        placebo = run_placebo_tests(feats, outcome="fk_grade", n_placebo=n_placebo)
        print(placebo.describe().to_string())
    except Exception as e:
        print(f"Placebo tests failed: {e}")

    # Plots
    try:
        plots.plot_outcome_pre_post_by_event(feats, outcome="fk_grade", figures_dir=FIGURES_DIR)
    except Exception as e:
        print(f"Plotting failed (pre/post by event): {e}")

    try:
        plots.plot_fk_vs_rarity_scatter(feats, figures_dir=FIGURES_DIR)
    except Exception as e:
        print(f"Plotting failed (scatter): {e}")

    print("\n[NYT] Pipeline complete. Figures saved in 'figures/' directory.")
