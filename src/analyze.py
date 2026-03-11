# src/analyze.py
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _prep(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    needed = [outcome, "post_event", "event_id"]
    sub = df.dropna(subset=needed).copy()

    if sub["post_event"].nunique() < 2:
        raise ValueError("post_event has no pre/post variation after filtering.")

    if sub["event_id"].nunique() < 2:
        # still can run, but FE doesn't help much
        pass

    return sub


def run_event_prepost_regression(
    df: pd.DataFrame,
    outcome: str = "fk_grade",
    use_domain_fe: bool = True,
):
    """
    Main model (event-focused):
        outcome = β * post_event + event fixed effects (+ optional domain bucket FE)

    Identification: within-event pre vs post shift in average complexity.
    """
    sub = _prep(df, outcome)

    if use_domain_fe and "domain_simple" in sub.columns:
        formula = f"{outcome} ~ post_event + C(event_id) + C(domain_simple)"
    else:
        formula = f"{outcome} ~ post_event + C(event_id)"

    print(f"[analyze] Using {len(sub)} rows for outcome '{outcome}'.")
    print("[analyze] Fitting OLS with formula:\n    " + formula)

    model = smf.ols(formula, data=sub).fit(cov_type="HC1")
    return model


def describe_pre_post_by_event(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    sub = df.dropna(subset=[outcome, "post_event", "event_id"]).copy()
    g = (
        sub.groupby(["event_id", "post_event"])[outcome]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return g


def run_placebo_tests(
    df: pd.DataFrame,
    outcome: str = "fk_grade",
    n_placebo: int = 200,
    seed: int = 0,
):
    """
    Placebo: within each event window, randomly pick a cutoff date and define a fake post.
    Then estimate beta on fake_post with event FE. Returns placebo betas.
    Requires pub_date + start/end per event.
    """
    if "pub_date" not in df.columns or "start" not in df.columns or "end" not in df.columns:
        raise ValueError("Placebos require columns: pub_date, start, end.")

    sub = df.dropna(subset=[outcome, "event_id", "pub_date", "start", "end"]).copy()
    sub["pub_date"] = pd.to_datetime(sub["pub_date"], errors="coerce")
    sub["start"] = pd.to_datetime(sub["start"], errors="coerce")
    sub["end"] = pd.to_datetime(sub["end"], errors="coerce")

    rng = np.random.default_rng(seed)
    betas = []

    for _ in range(n_placebo):
        tmp = sub.copy()

        # draw a fake cutoff per event within [start,end]
        fake_cut = {}
        for eid, g in tmp.groupby("event_id"):
            s = g["start"].iloc[0]
            e = g["end"].iloc[0]
            if pd.isna(s) or pd.isna(e) or e <= s:
                continue
            days = (e - s).days
            if days < 3:
                continue
            k = int(rng.integers(1, days))  # avoid edges
            fake_cut[eid] = s + pd.Timedelta(days=k)

        if len(fake_cut) < 2:
            continue

        tmp["fake_post"] = tmp.apply(
            lambda r: 1 if r["pub_date"] >= fake_cut.get(r["event_id"], r["end"]) else 0,
            axis=1,
        )

        if tmp["fake_post"].nunique() < 2:
            continue

        formula = f"{outcome} ~ fake_post + C(event_id)"
        m = smf.ols(formula, data=tmp).fit(cov_type="HC1")
        betas.append(float(m.params.get("fake_post", np.nan)))

    if not betas:
        raise ValueError("run_placebo_tests: no valid placebo draws produced.")

    return pd.DataFrame({"placebo_beta": betas})
