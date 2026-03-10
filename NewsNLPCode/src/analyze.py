# src/analyze.py

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResults


# ---------------------------------------------------------------------
# Core DiD helper
# ---------------------------------------------------------------------

def _prepare_dd_data(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Drop rows with missing data for the DiD regression."""
    needed = [outcome, "post_event", "low_support", "interaction", "domain"]
    sub = df.dropna(subset=needed).copy()
    if sub.empty:
        raise ValueError(
            f"No valid rows left after dropping NAs for {needed}. "
            "Cannot run regression."
        )
    return sub


# ---------------------------------------------------------------------
# Main DiD regression (used by pipeline)
# ---------------------------------------------------------------------

def run_did_regression(
    df: pd.DataFrame,
    outcome: str = "fk_grade",
    **kwargs,
) -> RegressionResults:
    """
    Run a difference-in-differences regression:

        outcome ~ post_event + low_support + interaction + C(domain)

    Returns the fitted statsmodels results object.
    """
    sub = _prepare_dd_data(df, outcome)
    formula = f"{outcome} ~ post_event + low_support + interaction + C(domain)"
    model = smf.ols(formula, data=sub).fit(cov_type="HC1")
    return model


# Backwards-compatible name (if anything still calls run_dd_reg)
def run_dd_reg(df: pd.DataFrame, y: str = "fk_grade") -> RegressionResults:
    return run_did_regression(df, outcome=y)


# ---------------------------------------------------------------------
# Simple descriptive stats
# ---------------------------------------------------------------------

def describe_pre_post(df: pd.DataFrame, outcome: str = "fk_grade") -> pd.DataFrame:
    """
    Mean outcome by domain × phase (pre vs post).
    """
    return (
        df.groupby(["domain", "phase"], dropna=False)[outcome]
          .mean()
          .unstack()
    )


# ---------------------------------------------------------------------
# Event-specific DiD models
# ---------------------------------------------------------------------

def event_specific_did(
    df: pd.DataFrame,
    outcome: str = "fk_grade",
    min_cells: int = 4,
) -> dict[str, RegressionResults]:
    """
    Run the DiD regression separately for each event_id.

    Returns
    -------
    dict
        Mapping event_id -> fitted RegressionResults.
        Events that lack variation in post_event or low_support are skipped.
    """
    results: dict[str, RegressionResults] = {}

    for ev, sub in df.groupby("event_id"):
        # Need variation in treatment and timing
        if sub["post_event"].nunique() < 2 or sub["low_support"].nunique() < 2:
            continue
        if len(sub) < min_cells:
            continue

        try:
            res = run_did_regression(sub, outcome=outcome)
            results[ev] = res
        except Exception:
            # If a particular event blows up (perfect collinearity, etc.),
            # just skip it instead of killing the whole pipeline.
            continue

    return results


# ---------------------------------------------------------------------
# FK grade ~ lexical rarity interaction
# ---------------------------------------------------------------------

def run_fk_rarity_interaction(
    df: pd.DataFrame,
    fk_col: str = "fk_grade",
    rarity_col: str = "avg_rarity",
    **kwargs,
) -> RegressionResults:
    """
    Regress Flesch–Kincaid grade on lexical rarity with an interaction
    with post_event:

        fk_grade ~ avg_rarity * post_event + low_support + C(domain)

    This tells you whether the rarity–readability slope changes after the event.
    """
    needed = [fk_col, rarity_col, "post_event", "low_support", "domain"]
    sub = df.dropna(subset=needed).copy()
    if sub.empty:
        raise ValueError(
            f"No valid rows left after dropping NAs for {needed}. "
            "Cannot run FK×rarity regression."
        )

    formula = f"{fk_col} ~ {rarity_col} * post_event + low_support + C(domain)"
    model = smf.ols(formula, data=sub).fit(cov_type="HC1")
    return model


# ---------------------------------------------------------------------
# Placebo tests: randomize treatment timing
# ---------------------------------------------------------------------

def run_placebo_tests(
    df: pd.DataFrame,
    outcome: str = "fk_grade",
    n_rep: int = 500,
    seed: int | None = 123,
    **kwargs,
) -> pd.DataFrame:
    """
    Simple placebo test by randomly permuting post_event across articles
    and re-estimating the DiD coefficient each time.

    Returns
    -------
    DataFrame
        Columns:
            iteration          – placebo run index
            interaction_coef   – DiD coefficient from that placebo
    """
    rng = np.random.default_rng(seed)

    sub = _prepare_dd_data(df, outcome)
    formula = f"{outcome} ~ post_event + low_support + interaction + C(domain)"

    coefs: list[float] = []

    for i in range(n_rep):
        shuffled = sub.copy()
        # Permute post_event labels
        shuffled["post_event"] = rng.permutation(shuffled["post_event"].values)
        shuffled["interaction"] = shuffled["post_event"] * shuffled["low_support"]

        model = smf.ols(formula, data=shuffled).fit(cov_type="HC1")
        coefs.append(model.params.get("interaction", np.nan))

    placebo_df = pd.DataFrame(
        {"iteration": np.arange(n_rep), "interaction_coef": coefs}
    )
    return placebo_df
