"""The data-understanding dossier (outcome-blind, read before hypotheses are registered) and the signal map
(read after): every value checked against a direct computation, and nothing that reveals a small count."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from codas.core import eda, preprocess


def _frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    age = rng.integers(20, 85, n)
    frame = pd.DataFrame({
        "age": age, "female": rng.integers(0, 2, n), "days_worn": rng.integers(30, 180, n),
        "steps": np.exp(8.5 + 0.4 * latent + rng.normal(0, 0.3, n)),
        "active": 30 + 10 * latent + rng.normal(0, 3, n),
        "hr": 70 - 3 * latent + rng.normal(0, 5, n),
        "flat": np.ones(n), "cov": 27 + 3 * rng.normal(size=n),
    })
    frame["worn_share"] = frame["days_worn"] / 180 + rng.normal(0, 0.05, n)  # tracks wear, not physiology
    frame.loc[rng.random(n) < 0.3 * (age > 60), "hr"] = np.nan  # missing more often in older people
    risk = np.exp(-0.3 * latent + 0.03 * (age - 50))
    t = rng.exponential(1500 / risk)
    frame["dx"], frame["followup"] = (t < 1800).astype(int), np.minimum(t, 1800)
    frame["lab"] = 5 + 0.2 * latent + rng.normal(0, 1, n)
    return frame


FEATURES = ["steps", "active", "hr", "flat", "worn_share"]


def test_each_feature_is_described_as_numpy_computes_it():
    frame = _frame()
    d = eda.describe(frame["steps"])
    x = frame["steps"].to_numpy()
    assert d["mean"] == pytest.approx(x.mean(), rel=1e-3) and d["sd"] == pytest.approx(x.std(ddof=1), rel=1e-3)
    assert d["skew"] == pytest.approx(stats.skew(x), rel=1e-3)
    assert d["quantiles"]["p99"] == pytest.approx(np.quantile(x, 0.99), rel=1e-3)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    assert d["natural_unit"] == float(f"{q3 - q1:.1g}") and d["near_constant"] is False
    assert eda.describe(frame["flat"])["near_constant"] is True


def test_nothing_reveals_a_small_count():
    frame = _frame(n=300)
    frame.loc[:4, "steps"] = np.nan  # five missing values
    d = eda.describe(frame["steps"])
    assert d["missing"] == f"masked (<={eda.MIN_TAIL} people)"
    assert "p1" not in d["quantiles"] and "p99" not in d["quantiles"]  # 3 people beyond each: not shown
    assert set(d["quantiles"]) >= {"p25", "p50", "p75"}


def test_families_wear_dependence_and_informative_missingness_are_found():
    frame = _frame()
    dossier = eda.blind_dossier(frame, target="dx", survival_time="followup", features=FEATURES, covariates=["cov"],
                                availability=["days_worn"])
    families = dossier["families"]
    assert any(set(g["members"]) == {"steps", "active"} for g in families)
    assert dossier["wear_dependence"]["worn_share"]["max_abs_rho"] > 0.8
    assert dossier["wear_dependence"]["steps"]["max_abs_rho"] < 0.1
    assert dossier["missingness"]["hr"]["smd_age"] > 0.5  # those missing hr are older
    digest = eda.digest(dossier)
    assert "steps ~ active" in digest and "worn_share" in digest and "hr" in digest and "flat" in digest


def test_the_dossier_never_relates_a_feature_or_covariate_to_the_outcome():
    """Permuting the outcome within the strata the dossier summarises it by leaves the dossier unchanged."""
    frame = _frame()
    groups = eda.strata(frame)
    rng = np.random.default_rng(1)
    permuted = frame.copy()
    for _, idx in permuted.groupby(groups).groups.items():
        order = rng.permutation(len(idx))
        permuted.loc[idx, ["dx", "followup", "lab"]] = frame.loc[idx, ["dx", "followup", "lab"]].to_numpy()[order]
    for target, time in (("dx", "followup"), ("lab", None)):
        kw = dict(target=target, survival_time=time, features=FEATURES, covariates=["cov"], availability=["days_worn"])
        assert eda.blind_dossier(frame, **kw) == eda.blind_dossier(permuted, **kw)


def test_the_outcome_profile_matches_a_direct_count():
    frame = _frame()
    profile = eda.outcome_profile(frame, "dx", "followup")
    years = frame["followup"].sum() / 365.25
    assert profile["overall"]["events"] == int(frame["dx"].sum())
    assert profile["overall"]["per_1000_person_years"] == pytest.approx(1000 * frame["dx"].sum() / years, rel=1e-3)
    assert set(profile["by_stratum"]) == set(eda.strata(frame).unique())


def test_the_signal_map_adjusts_attenuates_and_corrects():
    frame = _frame()
    frame["confounded"] = frame["cov"] + np.random.default_rng(2).normal(0, 1, len(frame))
    frame["lab"] = frame["lab"] + 0.5 * (frame["cov"] - 27)
    signal = eda.signal_map(frame, target="lab", features=["active", "confounded"], covariates=["cov"])
    rows = {r["feature"]: r for r in signal["features"]}
    assert rows["active"]["t2"] > rows["active"]["t0"] > 0.05 and rows["active"]["q_t2"] < 0.001  # adjusting removes noise
    assert rows["confounded"]["attenuation"] > 0.8  # the covariate explains nearly all of it
    cox = eda.signal_map(frame, target="dx", survival_time="followup", features=["active"])
    assert cox["features"][0]["t0"] < 1  # more activity, lower hazard (planted)


def test_frozen_transforms_put_both_shares_on_the_discovery_scale():
    frame = _frame()
    t = preprocess.fit(frame["steps"])
    assert t["log"] is False or t["lo"] > 0
    skewed = pd.Series(np.exp(np.random.default_rng(3).normal(0, 1.5, 4000)))
    tl = preprocess.fit(skewed)
    assert tl["log"] is True
    z = preprocess.apply(skewed, tl)
    assert abs(z.mean()) < 0.05 and z.std() == pytest.approx(1, abs=0.05)
    # The holdout keeps the discovery unit: a holdout with twice the spread is not rescaled to SD 1.
    wider = pd.Series(np.random.default_rng(4).normal(0, 2, 4000))
    narrow = preprocess.fit(pd.Series(np.random.default_rng(5).normal(0, 1, 4000)))
    assert preprocess.apply(wider, narrow, clip=False).std() == pytest.approx(2, abs=0.1)
    assert "discovery SD" in preprocess.describe(narrow)
