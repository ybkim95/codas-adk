"""Regression guards for the reporting-integrity (A) and pseudo-replication-visibility (C) hardening.

A: the grounding audit must flag fabricated clinical statistics the engine never produces (hazard/odds
ratios, sensitivity, specificity, F1), and the numeric-correction pass must not rewrite numbers written
in a citation or explicit-subset context. C: when the caller does not declare the participant/time
roles, an undeclared repeated-measures or autocorrelated structure must be surfaced as a warning rather
than failing silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.agents.grounding import engine_numbers, ungrounded_claims
from codas.agents.numeric_audit import verify_and_correct
from codas.core.discovery import DiscoveryRequest, run_discovery


def _ar1(n, phi, rng):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_grounding_flags_fabricated_clinical_statistics():
    ev = engine_numbers({"ml_metric_value": 0.61}, [{"rho": 0.252}], [])
    ungrounded, total = ungrounded_claims(
        "AUROC=0.97, hazard ratio HR=4.2, sensitivity 95%, odds ratio OR=3.1, specificity 90%, F1 0.92.", ev)
    flagged = {k.lower() for k, _ in ungrounded}
    assert total >= 5
    assert {"auroc", "hazard ratio", "sensitivity", "odds ratio", "specificity", "f1"} <= flagged


def test_grounding_does_not_flag_engine_backed_numbers():
    ev = engine_numbers({"ml_metric_value": 0.61}, [{"rho": 0.252, "p_value": 0.001}], [])
    ungrounded, total = ungrounded_claims("AUC = 0.61 and Spearman rho = 0.252 (p<0.001).", ev)
    assert total >= 2 and ungrounded == []


def test_numeric_audit_leaves_citation_and_subset_numbers_untouched():
    fs = {"rows": 7497, "internal_battery_passing_variants": 3}
    cite, corr_c = verify_and_correct("We build on a prior study of N = 7,400 participants.", fs)
    assert cite.endswith("7,400 participants.") and corr_c == []
    subset, corr_s = verify_and_correct("We highlight the top 2 validated biomarkers with the largest effects.", fs)
    assert "top 2 validated" in subset and corr_s == []
    # a plain transcription slip about this study is still corrected
    fixed, corr = verify_and_correct("The cohort (N = 7,480) was analysed.", fs)
    assert "N = 7,497" in fixed and corr[0]["to"] == 7497


def test_undeclared_numeric_participant_id_is_warned():
    rng = np.random.default_rng(0)
    K, T = 30, 100
    pid = np.repeat(np.arange(K), T)
    trait, out = rng.normal(size=K), rng.normal(size=K)  # per-subject constant trait & outcome, independent
    df = pd.DataFrame({"pid": pid, "x": trait[pid] + rng.normal(size=K * T) * 0.1, "y": out[pid]})
    report = run_discovery(df, DiscoveryRequest(target_column="y", validation_resamples=100))  # pid NOT declared
    assert any("repeated measures" in w and "pid" in w for w in report.warnings)


def test_undeclared_temporal_autocorrelation_is_warned():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"s0": _ar1(600, 0.95, rng), "s1": _ar1(600, 0.95, rng), "y": _ar1(600, 0.95, rng)})
    report = run_discovery(df, DiscoveryRequest(target_column="y", validation_resamples=100))  # no time declared
    assert any("autocorrelated in row order" in w for w in report.warnings)


def test_ordinary_low_cardinality_feature_is_not_mistaken_for_an_id():
    # A genuine integer feature that predicts a continuous target (target varies within its groups)
    # must NOT trigger the undeclared-id warning.
    rng = np.random.default_rng(2)
    n = 900
    sev = rng.integers(0, 8, n)  # a symptom-severity feature 0..7, many repeats
    df = pd.DataFrame({"severity": sev, "noise": rng.normal(size=n), "y": 0.3 * sev + rng.normal(size=n)})
    report = run_discovery(df, DiscoveryRequest(target_column="y", validation_resamples=100))
    assert not any("repeated measures" in w and "severity" in w for w in report.warnings)
