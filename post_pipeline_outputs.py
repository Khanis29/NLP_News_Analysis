from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If your project is structured as:
#   FinalTest/
#     run_pipeline.py
#     src/
#       analyze.py, plots.py, ...
#
# Then these imports should work after you add src to path as you already do in notebook.
from src import analyze, plots


# =========================
# Paths / helpers
# =========================
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def _read_latest_features() -> pd.DataFrame:
    """
    Looks for the features output from the pipeline.
    Tries data/ first (your current setup), then common fallbacks.
    """
    candidates = [
        # ---- your actual save location FIRST ----
        Path("data/features_nyt.parquet"),
        Path("data/nyt_features.parquet"),
        Path("data/features_nyt.csv"),
        Path("data/nyt_features.csv"),

        # ---- fallbacks (older conventions) ----
        Path("processed/nyt_features.parquet"),
        Path("processed/features_nyt.parquet"),
        Path("processed/nyt_features.csv"),
        Path("processed/features_nyt.csv"),
        Path("nyt_features.parquet"),
        Path("nyt_features.csv"),
    ]

    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)

    raise FileNotFoundError(
        "Could not find features file. Update _read_latest_features() with your saved path."
    )

def _standardize_cols(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()

    # Ensure post_event exists
    if "post_event" not in feats.columns and "phase" in feats.columns:
        feats["post_event"] = (feats["phase"] == "post").astype(int)

    # If domain_simple isn't present, create a stable one
    if "domain_simple" not in feats.columns and "domain" in feats.columns:
        feats["domain_simple"] = (
            feats["domain"]
            .astype(str)
            .str.replace("^www\\.", "", regex=True)
            .str.replace("^mobile\\.", "", regex=True)
        )

    # Force types
    if "post_event" in feats.columns:
        feats["post_event"] = feats["post_event"].astype(int)

    return feats


def save_table(df: pd.DataFrame, name: str) -> None:
    """Save as CSV + LaTeX."""
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)

    # Reasonable LaTeX export
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    print(f"[tables] Saved {csv_path}")
    print(f"[tables] Saved {tex_path}")


# =========================
# Core descriptive tables
# =========================
def make_descriptive_tables(feats: pd.DataFrame) -> None:
    # 1) Overall counts
    counts = pd.DataFrame(
        {
            "N_total": [len(feats)],
            "N_pre": [(feats["post_event"] == 0).sum()],
            "N_post": [(feats["post_event"] == 1).sum()],
            "N_events": [feats["event_id"].nunique() if "event_id" in feats.columns else np.nan],
            "N_domains": [feats["domain_simple"].nunique() if "domain_simple" in feats.columns else np.nan],
        }
    )
    save_table(counts, "nyt_counts_summary")

    # 2) Pre/post means by event (for all outcomes you have)
    outcomes = [c for c in ["fk_grade", "avg_rarity", "rare_share", "dale_chall", "avg_sentence_len", "ttr"] if c in feats.columns]
    if "event_id" in feats.columns and outcomes:
        rows = []
        for y in outcomes:
            tmp = (
                feats.groupby(["event_id", "post_event"])[y]
                .agg(["mean", "count", "std"])
                .reset_index()
                .rename(columns={"count": "n"})
            )
            tmp.insert(0, "outcome", y)
            rows.append(tmp)
        by_event = pd.concat(rows, ignore_index=True)
        save_table(by_event, "nyt_prepost_means_by_event")

    # 3) Pre/post means by domain (top domains only)
    if "domain_simple" in feats.columns and outcomes:
        rows = []
        top_domains = feats["domain_simple"].value_counts().head(10).index
        sub = feats[feats["domain_simple"].isin(top_domains)].copy()

        for y in outcomes:
            tmp = (
                sub.groupby(["domain_simple", "post_event"])[y]
                .agg(["mean", "count", "std"])
                .reset_index()
                .rename(columns={"count": "n"})
            )
            tmp.insert(0, "outcome", y)
            rows.append(tmp)

        by_domain = pd.concat(rows, ignore_index=True)
        save_table(by_domain, "nyt_prepost_means_by_domain_top10")


# =========================
# Regression tables
# =========================
def run_and_save_regressions(feats: pd.DataFrame) -> None:
    """
    Uses your analyze module if available. If not, it falls back to statsmodels locally.
    """
    outcomes = [c for c in ["fk_grade", "avg_rarity", "rare_share"] if c in feats.columns]
    reg_rows = []

    for y in outcomes:
        try:
            # Your analyze module likely has something like:
            # analyze.run_prepost_fe_regression(feats, outcome=y)
            # If not, update this function based on your analyze.py.
            res = analyze.run_prepost_fe_regression(feats, outcome=y)

            # Expecting statsmodels result object
            beta = float(res.params.get("post_event", np.nan))
            se = float(res.bse.get("post_event", np.nan))
            p = float(res.pvalues.get("post_event", np.nan))

            reg_rows.append(
                {
                    "outcome": y,
                    "beta_post_event": beta,
                    "se_post_event": se,
                    "p_post_event": p,
                    "N": int(res.nobs),
                    "R2": float(getattr(res, "rsquared", np.nan)),
                }
            )

            # Also save the full text summary
            summary_path = TABLES_DIR / f"reg_summary_{y}.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(str(res.summary()))
            print(f"[tables] Saved {summary_path}")

        except Exception as e:
            print(f"[warn] Regression for {y} failed: {e}")

    if reg_rows:
        save_table(pd.DataFrame(reg_rows), "nyt_prepost_fe_regression_keycoeffs")


# =========================
# Plots (safe, no missing funcs)
# =========================
def plot_prepost_by_event(feats: pd.DataFrame, outcome: str) -> None:
    if "event_id" not in feats.columns:
        return

    g = (
        feats.groupby(["event_id", "post_event"])[outcome]
        .mean()
        .reset_index()
        .pivot(index="event_id", columns="post_event", values=outcome)
        .rename(columns={0: "pre", 1: "post"})
    )

    # Only keep events with at least one pre or post
    g = g.sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g.index, g["pre"], marker="o", label="Pre")
    ax.plot(g.index, g["post"], marker="o", label="Post")
    ax.set_title(f"Pre vs Post mean of {outcome} by event")
    ax.set_xlabel("Event")
    ax.set_ylabel(outcome)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out = FIGURES_DIR / f"nyt_{outcome}_prepost_by_event_manual.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] Saved {out}")


def plot_prepost_by_domain(feats: pd.DataFrame, outcome: str, top_k: int = 8) -> None:
    if "domain_simple" not in feats.columns:
        return

    top_domains = feats["domain_simple"].value_counts().head(top_k).index
    sub = feats[feats["domain_simple"].isin(top_domains)].copy()

    means = (
        sub.groupby(["domain_simple", "post_event"])[outcome]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for d in top_domains:
        dm = means[means["domain_simple"] == d].sort_values("post_event")
        ax.plot(dm["post_event"], dm[outcome], marker="o", label=d)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_title(f"Pre vs Post mean of {outcome} by domain (top {top_k})")
    ax.set_xlabel("Phase")
    ax.set_ylabel(outcome)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = FIGURES_DIR / f"nyt_{outcome}_prepost_by_domain_top{top_k}_manual.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] Saved {out}")


def plot_fk_vs_rarity(feats: pd.DataFrame) -> None:
    if "fk_grade" not in feats.columns or "avg_rarity" not in feats.columns:
        return

    sub = feats.dropna(subset=["fk_grade", "avg_rarity"]).copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["avg_rarity"], sub["fk_grade"], alpha=0.7)
    ax.set_title("FK Grade vs Average Rarity")
    ax.set_xlabel("avg_rarity")
    ax.set_ylabel("fk_grade")
    fig.tight_layout()

    out = FIGURES_DIR / "nyt_fk_vs_avg_rarity_scatter_manual.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] Saved {out}")


def run_all_outputs():
    feats = _read_latest_features()
    feats = _standardize_cols(feats)

    # --- Tables ---
    make_descriptive_tables(feats)
    run_and_save_regressions(feats)

    # --- Plots (use your plots.py if available, otherwise manual fallbacks) ---
    # Your pipeline already saved:
    #   figures/nyt_fk_grade_pre_post_by_event.png
    #   figures/nyt_fk_vs_rarity_scatter.png
    # But these manual ones give you extra variants and always work.

    for outcome in ["fk_grade", "avg_rarity", "rare_share"]:
        if outcome in feats.columns:
            plot_prepost_by_event(feats, outcome)
            plot_prepost_by_domain(feats, outcome, top_k=8)

    plot_fk_vs_rarity(feats)

    print("\n[done] Post-pipeline outputs generated.")
    print(f"Figures: {FIGURES_DIR.resolve()}")
    print(f"Tables : {TABLES_DIR.resolve()}")


if __name__ == "__main__":
    run_all_outputs()
