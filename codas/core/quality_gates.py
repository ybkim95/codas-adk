"""Deterministic quality gates for report assembly.

Before a report is assembled, CoDaS evaluates five deterministic gates that decide whether a class of
result may be surfaced. Each gate is a pure function of the engine's own outputs — the model-feature
matrix, the cross-validated metrics, and the surviving candidates — so a gate decision is auditable
and reproducible. The gates do not mutate results; they return a structured verdict that the report
layer records (and, in the full system, uses to suppress the corresponding table):

  (i)   multicollinearity — flags the multivariate (OLS) view when the maximum variance-inflation
        factor across the model features exceeds ``VIF_MAX`` (50).
  (ii)  performance       — flags the predictive-model results when the best cross-validated AUC is
        below 0.55 (classification) or R^2 is below 0 (regression).
  (iii) overfitting       — flags the results when the in-sample-to-cross-validated performance ratio
        exceeds ``OVERFIT_RATIO`` (5).
  (iv)  ablation          — flags the feature-importance view when the model does not beat its own
        permutation null (no above-chance signal).
  (v)   family deduplication — limits any group of mutually near-collinear features (|r| > ``DEDUP_R``)
        to two representatives, preventing a single signal from dominating by correlated variants.

Gate (v) is deliberately correlation-based, not name-based: CoDaS applies no name rules, so "feature
family" means a cluster of features that measure the same thing statistically, identified by their
inter-feature correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

VIF_MAX = 50.0
OVERFIT_RATIO = 5.0
DEDUP_R = 0.90


@dataclass
class GateVerdict:
    name: str
    triggered: bool
    detail: str
    metric: float | None = None

    def to_dict(self) -> dict[str, Any]:
        # Round the metric so the serialized gate summary is stable across BLAS/platforms (it is
        # embedded in the Fact Sheet, which is fingerprinted by the golden report test).
        metric = (round(float(self.metric), 4)
                  if self.metric is not None and np.isfinite(self.metric) else None)
        return {"name": self.name, "triggered": bool(self.triggered), "detail": self.detail, "metric": metric}


def _max_vif(frame: pd.DataFrame, features: list[str]) -> float:
    """Largest variance-inflation factor across ``features`` (VIF_j = 1 / (1 - R^2_j) from regressing
    feature j on the others). Needs >= 2 features and enough complete rows; returns nan otherwise."""
    usable = [c for c in features if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    if len(usable) < 2:
        return float("nan")
    x = frame[usable].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < len(usable) + 2:
        return float("nan")
    values = x.to_numpy(dtype=float)
    worst = 0.0
    for j in range(values.shape[1]):
        others = np.delete(values, j, axis=1)
        if np.nanstd(values[:, j]) < 1e-12:
            continue
        r2 = float(LinearRegression().fit(others, values[:, j]).score(others, values[:, j]))
        vif = 1.0 / max(1.0 - r2, 1e-9)
        worst = max(worst, vif)
    return worst


def _multicollinearity_gate(frame: pd.DataFrame, model_features: list[str]) -> GateVerdict:
    vif = _max_vif(frame, model_features)
    triggered = bool(np.isfinite(vif) and vif > VIF_MAX)
    detail = (f"max VIF={vif:.1f} across {len(model_features)} model features"
              if np.isfinite(vif) else "VIF not computable (too few features/rows)")
    return GateVerdict("multicollinearity", triggered, detail, metric=vif if np.isfinite(vif) else None)


def _performance_gate(ml_metrics: dict[str, Any]) -> GateVerdict:
    name = ml_metrics.get("metric_name")
    value = ml_metrics.get("metric_value")
    if name is None or value is None:
        return GateVerdict("performance", False, "no predictive metric computed")
    value = float(value)
    if name == "auc":
        triggered = value < 0.55
        return GateVerdict("performance", triggered, f"CV AUC={value:.3f} (threshold 0.55)", metric=value)
    triggered = value < 0.0
    return GateVerdict("performance", triggered, f"CV R^2={value:.3f} (threshold 0)", metric=value)


def _overfitting_gate(ml_metrics: dict[str, Any]) -> GateVerdict:
    train = ml_metrics.get("train_metric_value")
    cv = ml_metrics.get("metric_value")
    if train is None or cv is None:
        return GateVerdict("overfitting", False, "train-vs-CV ratio not available")
    train, cv = float(train), float(cv)
    # Ratio is only meaningful when both are positive skill; a non-positive CV with strong train fit is
    # itself overfitting.
    if cv <= 0.0:
        triggered = train > 0.10
        return GateVerdict("overfitting", triggered, f"train={train:.3f} but CV={cv:.3f} (no held-out skill)", metric=float("inf") if triggered else 0.0)
    ratio = train / cv
    triggered = ratio > OVERFIT_RATIO
    return GateVerdict("overfitting", triggered, f"train/CV={ratio:.2f} (threshold {OVERFIT_RATIO})", metric=ratio)


def _ablation_gate(ml_metrics: dict[str, Any]) -> GateVerdict:
    if ml_metrics.get("metric_value") is None:
        return GateVerdict("ablation", False, "no model fitted")
    above = ml_metrics.get("above_chance")
    if above is None:
        return GateVerdict("ablation", False, "permutation null not evaluated")
    triggered = not bool(above)
    p = ml_metrics.get("metric_vs_null_p")
    detail = "model does not beat its permutation null" if triggered else "model beats its permutation null"
    return GateVerdict("ablation", triggered, detail, metric=(float(p) if isinstance(p, (int, float)) else None))


def _family_dedup_gate(frame: pd.DataFrame, candidates: list) -> GateVerdict:
    """Cluster the surviving candidates by inter-feature correlation (|r| > DEDUP_R) and count how many
    would be dropped to keep at most two representatives per family (the strongest by |rho|)."""
    passing = [c for c in candidates if getattr(c, "verdict", "") in {"validated", "conditional"}
               and getattr(c, "feature", None) in frame.columns]
    if len(passing) < 3:
        return GateVerdict("family_dedup", False, f"{len(passing)} surviving candidate(s); no family exceeds two")
    order = sorted(passing, key=lambda c: abs(c.rho) if np.isfinite(c.rho) else 0.0, reverse=True)
    families: list[list] = []
    for cand in order:
        placed = False
        for fam in families:
            try:
                r = float(frame[[cand.feature, fam[0].feature]].dropna().corr().iloc[0, 1])
            except Exception:
                r = 0.0
            if np.isfinite(r) and abs(r) > DEDUP_R:
                fam.append(cand)
                placed = True
                break
        if not placed:
            families.append([cand])
    dropped = sum(max(0, len(fam) - 2) for fam in families)
    triggered = dropped > 0
    return GateVerdict("family_dedup", triggered,
                       f"{len(families)} feature famil(y/ies); {dropped} correlated variant(s) beyond two per family",
                       metric=float(dropped))


def evaluate_quality_gates(
    frame: pd.DataFrame,
    model_features: list[str],
    ml_metrics: dict[str, Any],
    candidates: list,
) -> dict[str, Any]:
    """Run the five gates and return a JSON-safe summary for the Fact Sheet.

    ``triggered=True`` means the gate would suppress the corresponding table in the full report; here it
    is recorded as an auditable decision that shows exactly why a result was or was not shown.
    """
    verdicts = [
        _multicollinearity_gate(frame, model_features),
        _performance_gate(ml_metrics),
        _overfitting_gate(ml_metrics),
        _ablation_gate(ml_metrics),
        _family_dedup_gate(frame, candidates),
    ]
    return {
        "gates": [v.to_dict() for v in verdicts],
        "triggered": sorted(v.name for v in verdicts if v.triggered),
    }
