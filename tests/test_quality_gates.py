"""Unit tests for the deterministic Section 2.6 quality gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas_core.models import Candidate
from codas_core.quality_gates import (
    DEDUP_R,
    OVERFIT_RATIO,
    VIF_MAX,
    evaluate_quality_gates,
)


def _gate(gates: dict, name: str) -> dict:
    return next(g for g in gates["gates"] if g["name"] == name)


def _candidate(feature: str, rho: float, verdict: str = "validated") -> Candidate:
    return Candidate(feature=feature, rho=rho, p_value=0.0, q_value=0.0, n=100,
                     direction="positive" if rho > 0 else "negative", score=abs(rho), verdict=verdict)


def test_multicollinearity_gate_fires_on_redundant_features():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    frame = pd.DataFrame({"a": a, "b": a + rng.normal(scale=1e-3, size=300), "y": rng.normal(size=300)})
    gates = evaluate_quality_gates(frame, ["a", "b"], {"metric_name": "r2", "metric_value": 0.1}, [])
    g = _gate(gates, "multicollinearity")
    assert g["triggered"] and g["metric"] > VIF_MAX


def test_performance_gate_fires_below_auc_threshold():
    gates = evaluate_quality_gates(pd.DataFrame({"y": [0, 1]}), [], {"metric_name": "auc", "metric_value": 0.52}, [])
    assert _gate(gates, "performance")["triggered"]
    gates = evaluate_quality_gates(pd.DataFrame({"y": [0, 1]}), [], {"metric_name": "auc", "metric_value": 0.71}, [])
    assert not _gate(gates, "performance")["triggered"]


def test_overfitting_gate_fires_on_large_train_cv_gap():
    ml = {"metric_name": "r2", "metric_value": 0.10, "train_metric_value": 0.90}
    assert _gate(evaluate_quality_gates(pd.DataFrame(), [], ml, []), "overfitting")["triggered"]
    ml_ok = {"metric_name": "r2", "metric_value": 0.40, "train_metric_value": 0.48}
    g = _gate(evaluate_quality_gates(pd.DataFrame(), [], ml_ok, []), "overfitting")
    assert not g["triggered"] and g["metric"] <= OVERFIT_RATIO


def test_ablation_gate_fires_when_not_above_chance():
    ml = {"metric_name": "auc", "metric_value": 0.53, "above_chance": False, "metric_vs_null_p": 0.4}
    assert _gate(evaluate_quality_gates(pd.DataFrame(), [], ml, []), "ablation")["triggered"]
    ml_ok = {"metric_name": "auc", "metric_value": 0.7, "above_chance": True, "metric_vs_null_p": 0.001}
    assert not _gate(evaluate_quality_gates(pd.DataFrame(), [], ml_ok, []), "ablation")["triggered"]


def test_family_dedup_gate_caps_correlated_variants_at_two():
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)
    frame = pd.DataFrame({
        "f1": base + rng.normal(scale=1e-2, size=300),
        "f2": base + rng.normal(scale=1e-2, size=300),
        "f3": base + rng.normal(scale=1e-2, size=300),  # third near-duplicate -> one over the cap
        "y": base,
    })
    cands = [_candidate("f1", 0.9), _candidate("f2", 0.88), _candidate("f3", 0.86)]
    g = _gate(evaluate_quality_gates(frame, ["f1", "f2", "f3"], {"metric_name": "r2", "metric_value": 0.8}, cands),
              "family_dedup")
    assert g["triggered"] and g["metric"] >= 1

    independent = [_candidate("f1", 0.9), _candidate("y", 0.5, verdict="conditional")]
    frame2 = pd.DataFrame({"f1": rng.normal(size=50), "y": rng.normal(size=50)})
    assert not _gate(evaluate_quality_gates(frame2, ["f1"], {"metric_name": "r2", "metric_value": 0.3}, independent),
                     "family_dedup")["triggered"]


def test_gate_summary_lists_only_triggered():
    ml = {"metric_name": "auc", "metric_value": 0.50, "above_chance": False}
    gates = evaluate_quality_gates(pd.DataFrame(), [], ml, [])
    assert set(gates["triggered"]) == {"performance", "ablation"}
    assert DEDUP_R == 0.90  # documented threshold, guards against accidental drift
