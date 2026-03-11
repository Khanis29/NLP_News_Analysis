# plots.py
"""
Plotting helpers for the NYT readability project.

All plotting functions take a feature DataFrame and optional
`savepath` and `show` arguments.

New in this version:
- Bar plots show 95% confidence intervals (via seaborn)
- FK–rarity scatter uses a regression line with 95% CI
- A histogram plot for placebo DiD coefficients
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


def _maybe_save_show(savepath: str | None, show: bool):
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Core descriptive plots
# ---------------------------------------------------------------------------

def plot_fk_pre_post_by_domain(
    feats: pd.DataFrame,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Bar plot of mean FK grade by domain and phase with 95% CIs.
    """
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=feats,
        x="domain",
        y="fk_grade",
        hue="phase",
        estimator=np.mean,
        ci=95,
        capsize=0.1,
    )
    ax.set_title("Mean Flesch–Kincaid grade by domain and phase")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Mean Flesch–Kincaid grade")
    plt.xticks(rotation=35, ha="right")
    _maybe_save_show(savepath, show)


def plot_fk_pre_post_by_support(
    feats: pd.DataFrame,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Boxplot of FK grade by support level × phase.
    Boxplots already encode variability (IQR & whiskers).
    """
    df = feats.copy()
    df["support_phase"] = np.where(df["low_support"] == 1, "Low", "High") + " / " + df[
        "phase"
    ]

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x="support_phase", y="fk_grade")
    ax.set_title("Flesch–Kincaid grade by support level × phase")
    ax.set_xlabel("Support / Phase")
    ax.set_ylabel("Flesch–Kincaid grade")
    _maybe_save_show(savepath, show)


def plot_fk_vs_rarity_scatter(
    feats: pd.DataFrame,
    savepath: str | None = None,
    show: bool = True,
):
    """
    Scatter (colored by phase) plus regression line with 95% CI.
    """
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        data=feats,
        x="fk_grade",
        y="avg_rarity",
        hue="phase",
        alpha=0.8,
        s=60,
    )
    # Overall regression line with CI
    sns.regplot(
        data=feats,
        x="fk_grade",
        y="avg_rarity",
        scatter=False,
        ci=95,
        ax=ax,
        color="k",
        line_kws={"linewidth": 2, "alpha": 0.7},
    )
    ax.set_title("FK grade vs lexical rarity")
    ax.set_xlabel("Flesch–Kincaid grade")
    ax.set_ylabel("Average rarity (−log prob)")
    _maybe_save_show(savepath, show)


def plot_rarity_pre_post_by_support(
    feats: pd.DataFrame,
    savepath: str | None = None,
    show: bool = True,
    metric: str = "rare_share",
):
    """
    Boxplot of lexical rarity metric by support level × phase.

    Parameters
    ----------
    metric : {"rare_share", "avg_rarity"}
        Which rarity metric to plot on the y-axis.
    """
    df = feats.copy()
    df["support_phase"] = np.where(df["low_support"] == 1, "Low", "High") + " / " + df[
        "phase"
    ]

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x="support_phase", y=metric)
    if metric == "rare_share":
        ylabel = "Share of rare/long words"
        title = "Lexical rarity (rare_share) by support level × phase"
    else:
        ylabel = "Average rarity (−log prob)"
        title = "Lexical rarity (avg_rarity) by support level × phase"
    ax.set_title(title)
    ax.set_xlabel("Support / Phase")
    ax.set_ylabel(ylabel)
    _maybe_save_show(savepath, show)


# ---------------------------------------------------------------------------
# Placebo histogram
# ---------------------------------------------------------------------------

def plot_placebo_hist(
    betas: np.ndarray,
    true_beta: float,
    outcome: str = "fk_grade",
    savepath: str | None = None,
    show: bool = True,
):
    """
    Plot the distribution of placebo interaction coefficients with the
    true DiD estimate overlaid as a vertical line.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(betas, bins=30, kde=True)
    ax.axvline(true_beta, color="red", linestyle="--", linewidth=2, label="True DiD β")
    ax.set_title(f"Placebo distribution of DiD interaction (outcome = {outcome})")
    ax.set_xlabel("Placebo interaction coefficient")
    ax.set_ylabel("Count")
    ax.legend()
    _maybe_save_show(savepath, show)
