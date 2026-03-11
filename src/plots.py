# src/plots.py
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_outcome_pre_post_by_event(
    df: pd.DataFrame,
    outcome: str = "fk_grade",
    figures_dir: str | Path = "figures",
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sub = df.dropna(subset=[outcome, "post_event", "event_id"]).copy()
    if sub.empty:
        print(f"[plots] No data to plot {outcome} pre/post by event.")
        return

    agg = (
        sub.groupby(["event_id", "post_event"])[outcome]
        .mean()
        .unstack("post_event")
        .rename(columns={0: "pre", 1: "post"})
    )

    ax = agg.plot(kind="bar", rot=45)
    ax.set_title(f"{outcome}: pre vs post by event")
    ax.set_xlabel("event_id")
    ax.set_ylabel(outcome)
    plt.tight_layout()

    outpath = figures_dir / f"nyt_{outcome}_pre_post_by_event.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plots] Saved {outpath}")


def plot_fk_vs_rarity_scatter(
    df: pd.DataFrame,
    figures_dir: str | Path = "figures",
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    need = ["fk_grade", "avg_rarity"]
    sub = df.dropna(subset=need).copy()
    if sub.empty:
        print("[plots] No data to plot fk_grade vs avg_rarity scatter.")
        return

    plt.figure()
    plt.scatter(sub["avg_rarity"], sub["fk_grade"], alpha=0.5)
    plt.xlabel("avg_rarity")
    plt.ylabel("fk_grade")
    plt.title("FK grade vs lexical rarity")
    plt.tight_layout()

    outpath = figures_dir / "nyt_fk_vs_rarity_scatter.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plots] Saved {outpath}")
