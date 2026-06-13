"""Scientific-validity guards for longitudinal / wearable data (a fast subset of
scripts/scientific_validation.py). These lock the statistical behaviours a rigorous reviewer of
physiological time-series would demand: no pseudo-replication, autocorrelation/effective-n awareness,
confounder adjustment, cluster-honest bootstrap CIs, and surfacing within-subject signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas_core.discovery import DiscoveryRequest, run_discovery
from codas_core.validation import _bootstrap_distribution


def _hard_validated(df, **kw) -> set[str]:
    rep = run_discovery(df, DiscoveryRequest(validation_resamples=kw.pop("rs", 150), **kw))
    return {c.feature for c in rep.candidates if c.verdict == "validated"}


def _ar1(n: int, phi: float, rng) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_cluster_bootstrap_is_wider_than_row_bootstrap_on_repeated_measures():
    """The repeated-measures CI must reflect the effective (participant) n, not the inflated row n."""
    rng = np.random.default_rng(0)
    K, T = 15, 40
    subj = np.repeat(np.arange(K), T)
    tx, ty = rng.normal(size=K), rng.normal(size=K)
    x = tx[subj] + rng.normal(size=K * T) * 0.2
    y = ty[subj] + rng.normal(size=K * T) * 0.2
    row = _bootstrap_distribution(x, y, 400, 0)
    cluster = _bootstrap_distribution(x, y, 400, 0, groups=subj)
    assert cluster.std() > row.std() * 1.5, "cluster bootstrap must give a substantially wider CI"


def test_confounder_adjustment_drops_a_purely_confounded_feature():
    rng = np.random.default_rng(3)
    n = 600
    z = rng.normal(size=n)
    df = pd.DataFrame({"z": z, "x": z + rng.normal(size=n) * 0.4,
                       "noise": rng.normal(size=n), "y": z + rng.normal(size=n) * 0.4})  # x ⊥ y | z
    assert "x" in _hard_validated(df, target_column="y", rs=200)                       # spurious without z
    assert "x" not in _hard_validated(df, target_column="y", confounder_columns=["z"], rs=200)  # dropped with z


def test_independent_autocorrelated_series_are_not_hard_validated():
    fp = 0
    for s in range(6):
        rng = np.random.default_rng(200 + s)
        n = 600
        df = pd.DataFrame({"t": np.arange(n), "x": _ar1(n, 0.95, rng), "y": _ar1(n, 0.95, rng)})
        fp += "x" in _hard_validated(df, target_column="y", time_column="t", rs=150)
    assert fp == 0, f"independent AR(1) pairs must not be hard-validated (got {fp}/6)"


def test_participant_id_reduces_pseudo_replication_false_validation():
    naive = grouped = 0
    for s in range(6):
        rng = np.random.default_rng(100 + s)
        K, T = 20, 50
        subj = np.repeat(np.arange(K), T)
        tx, ty = rng.normal(size=K), rng.normal(size=K)
        df = pd.DataFrame({"pid": subj, "x": tx[subj] + rng.normal(size=K * T) * 0.2,
                           "y": ty[subj] + rng.normal(size=K * T) * 0.2})
        naive += "x" in _hard_validated(df, target_column="y", rs=120)
        grouped += "x" in _hard_validated(df, target_column="y", participant_id_column="pid", rs=120)
    assert grouped < naive, f"declaring the participant id must reduce false validation (naive={naive}, grouped={grouped})"


def test_within_subject_signal_surfaces_under_opposite_between_correlation():
    rng = np.random.default_rng(0)
    K, T = 30, 40
    subj = np.repeat(np.arange(K), T)
    b = rng.normal(size=K)
    x_mean = -1.2 * b + rng.normal(size=K) * 0.3      # between: high-x subjects have LOW outcome
    within_x = rng.normal(size=K * T)
    df = pd.DataFrame({"pid": subj, "x": x_mean[subj] + within_x, "noise": rng.normal(size=K * T),
                       "y": b[subj] + 0.8 * within_x + rng.normal(size=K * T) * 0.3})   # within: +slope
    rep = run_discovery(df, DiscoveryRequest(target_column="y", participant_id_column="pid", validation_resamples=150))
    assert any("within-subject" in w.lower() or "within-person" in w.lower() for w in rep.warnings), \
        "the within-subject signal hidden by an opposite between-subject correlation must be surfaced"
