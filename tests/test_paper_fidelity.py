"""The validation battery must do what the paper (§2.5, §2.6) says. These tests exercise the paper's
requirements that the golden characterization fixtures do not reach (repeated measures, several
validated candidates, small samples), so a regression in any of them is caught.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.core.discovery import DiscoveryRequest, run_discovery
from codas.core.models import Candidate
from codas.core.statistics import safe_spearman
from codas.core.validation import (
    ValidationConfig,
    _construct_validity_test,
    _leave_one_out_test,
    _split_holdout,
    validate_candidate,
)


def _candidate(df: pd.DataFrame, feature: str, target: str) -> Candidate:
    rho, p, n = safe_spearman(df[feature], df[target])
    return Candidate(feature=feature, rho=float(rho), p_value=float(p), q_value=0.0,
                     n=int(n), direction="positive" if rho >= 0 else "negative", score=abs(float(rho)))


def test_leave_one_out_excludes_participants_not_rows():
    # Paper 2.6-4: influence is judged by dropping a whole participant, not a single row.
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 12)
    trait = rng.normal(size=8)[groups]
    x = trait + rng.normal(size=96) * 0.2
    y = trait + rng.normal(size=96) * 0.2
    rho, _, _ = safe_spearman(pd.Series(x), pd.Series(y))
    cand = SimpleNamespace(rho=float(rho))
    with_groups = _leave_one_out_test(x, y, cand, ValidationConfig(), groups=groups)[0]
    without = _leave_one_out_test(x, y, cand, ValidationConfig(), groups=None)[0]
    assert "participant" in with_groups.details  # 8 participants dropped, not 96 rows
    assert "row" in without.details               # cross-sectional fallback


def test_holdout_keeps_one_observation_per_participant():
    # Paper 2.6-1: for repeated measures the confirmation set retains one observation per participant.
    rng = np.random.default_rng(1)
    pid = np.repeat(np.arange(40), 5)
    df = pd.DataFrame({"f": rng.normal(size=200), "y": rng.normal(size=200), "pid": pid})
    held = _split_holdout(df, "f", "y", "pid", random_state=0)
    assert not held["pid"].duplicated().any()
    assert 0 < len(held) <= df["pid"].nunique()


def test_prior_validated_biomarker_enters_causal_robustness():
    # Paper 2.6-8: a candidate is residualized against previously validated biomarkers.
    rng = np.random.default_rng(2)
    n = 300
    a = rng.normal(size=n)
    b = a + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"a": a, "b": b, "y": a + rng.normal(size=n) * 0.5})
    # validate_candidate mutates the candidate in place, so use a fresh one per call.
    without_prior = validate_candidate(df, _candidate(df, "b", "y"), "y", None, [], [], {})
    with_prior = validate_candidate(df, _candidate(df, "b", "y"), "y", None, [], [], {}, prior_validated_columns=["a"])
    conf_off = next(t for t in without_prior.tests if t.name == "confounder_adjusted_robustness")
    conf_on = next(t for t in with_prior.tests if t.name == "confounder_adjusted_robustness")
    assert not conf_off.applicable            # nothing to control for
    assert conf_on.applicable and "a" in conf_on.details  # controls for the prior biomarker


def test_permutation_uses_requested_resamples_on_a_realistic_cohort():
    # Paper 2.6-2: 1,000 permutation resamples; the work budget must not silently reduce them here.
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": 0.3 * x + rng.normal(size=n)})
    report = run_discovery(df, DiscoveryRequest(target_column="y", validation_resamples=1000))
    cand = next(c for c in report.candidates if c.feature == "x")
    perm = next(t for t in cand.tests if t.name == "permutation_test")
    assert "resamples=1000" in perm.details


def test_construct_validity_gate_adapts_for_small_n():
    # Paper 2.6-7: adaptive threshold for small N (rather than disabling the gate below N=30).
    rng = np.random.default_rng(4)
    x = rng.normal(size=20)
    y = x.copy()  # near-tautological, rho ~ 1.0
    df = pd.DataFrame({"x": x, "y": y})
    cand = _candidate(df, "x", "y")
    result = _construct_validity_test(np.asarray(x), np.asarray(y), cand, ValidationConfig())[0]
    assert result.applicable and not result.passed  # flagged as implausibly strong despite N<=30
