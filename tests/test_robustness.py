"""Production-robustness regression guards (a fast, CI-friendly subset of scripts/robustness_audit.py).

These lock the invariants that matter most for an analysis engine that runs on arbitrary uploads:
it never crashes on degenerate input, it does not manufacture findings from noise, it recovers a
genuine signal, it catches target leakage, and it is deterministic. The full scored audit (scale,
service, the complete degenerate-input matrix, and the live agent dimensions) lives in scripts/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codas_core.data import InsufficientDataError
from codas_core.discovery import DiscoveryRequest, run_discovery


def _run(df: pd.DataFrame, target: str, **kw):
    return run_discovery(df, DiscoveryRequest(target_column=target, validation_resamples=kw.pop("rs", 120), **kw))


# --- never crash: any input yields a report OR the designated InsufficientDataError, never anything else ---

def _degenerate_frames() -> dict[str, tuple[pd.DataFrame, str]]:
    rng = np.random.default_rng(0)
    n = 60
    return {
        "empty": (pd.DataFrame({"y": []}), "y"),
        "single_row": (pd.DataFrame({"x": [1.0], "y": [2.0]}), "y"),
        "all_nan_target": (pd.DataFrame({"x": rng.normal(size=n), "y": [np.nan] * n}), "y"),
        "constant_target": (pd.DataFrame({"x": rng.normal(size=n), "y": [3.0] * n}), "y"),
        "constant_feature": (pd.DataFrame({"x": [5.0] * n, "y": rng.normal(size=n)}), "y"),
        "duplicate_columns": (pd.DataFrame(np.c_[rng.normal(size=n), rng.normal(size=n)],
                                           columns=["x", "x"]).assign(y=rng.normal(size=n)), "y"),
        "binary_text_target": (pd.DataFrame({"x": rng.normal(size=n), "y": rng.choice(["a", "b"], n)}), "y"),
        "multiclass_text_target": (pd.DataFrame({"x": rng.normal(size=n), "y": rng.choice(["a", "b", "c"], n)}), "y"),
        "inf_feature": (pd.DataFrame({"x": [np.inf, -np.inf] + list(rng.normal(size=n - 2)),
                                      "y": rng.normal(size=n)}), "y"),
        "target_absent": (pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)}), "missing"),
        "unicode_columns": (pd.DataFrame({"变量": rng.normal(size=n), "结果": rng.normal(size=n)}), "结果"),
        "wide_p_gt_n": (pd.DataFrame(rng.normal(size=(20, 60)),
                                     columns=[f"f{i}" for i in range(59)] + ["y"]), "y"),
    }


@pytest.mark.parametrize("name", list(_degenerate_frames().keys()))
def test_engine_never_crashes_on_degenerate_input(name):
    df, target = _degenerate_frames()[name]
    try:
        report = _run(df, target, top_k=5)
    except InsufficientDataError:
        return  # the designated graceful boundary
    mv = report.fact_sheet.get("ml_metric_value")
    assert mv is None or (isinstance(mv, float) and np.isfinite(mv)), "report headline metric must be finite"


# --- scientific integrity: no false positives, real recall, leakage caught ---

def test_no_validated_predictor_on_pure_noise():
    for seed in range(5):
        rng = np.random.default_rng(500 + seed)
        df = pd.DataFrame({f"f{i}": rng.normal(size=200) for i in range(6)} | {"y": rng.normal(size=200)})
        rep = _run(df, "y", top_k=6)
        assert sum(c.verdict == "validated" for c in rep.candidates) == 0, f"false positive on noise (seed {seed})"


def test_recovers_a_genuine_moderate_signal():
    rng = np.random.default_rng(2024)
    x1 = rng.normal(size=300)
    df = pd.DataFrame({"x1": x1, "n1": rng.normal(size=300), "y": 0.6 * x1 + rng.normal(size=300)})
    rep = _run(df, "y", top_k=6)
    assert any(c.feature == "x1" and c.verdict in {"validated", "conditional"} for c in rep.candidates)


def test_exact_target_copy_is_not_reported_as_clean_predictor():
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    df = pd.DataFrame({"copy_of_y": y.copy(), "x2": rng.normal(size=300), "y": y})
    rep = _run(df, "y", top_k=6)
    copy_c = next((c for c in rep.candidates if c.feature == "copy_of_y"), None)
    caught = copy_c is None or copy_c.verdict != "validated" or any(
        t.hard_gate and t.applicable and not t.passed for t in copy_c.tests)
    assert caught, "an exact copy of the target must be rejected as leakage, not validated"


def test_repeated_measures_are_not_pseudo_replicated():
    rng = np.random.default_rng(5)
    subj = np.repeat(np.arange(40), 25)
    trait_s = rng.normal(size=40)
    target_s = trait_s + rng.normal(size=40) * 0.1            # target constant within subject
    df = pd.DataFrame({"pid": subj, "feat": trait_s[subj] + rng.normal(size=len(subj)) * 0.3,
                       "y": target_s[subj]})
    rep = run_discovery(df, DiscoveryRequest(target_column="y", participant_id_column="pid", validation_resamples=120))
    records = list(rep.warnings) + list(rep.audit_log)
    assert any(k in r.lower() for r in records for k in ("aggregat", "effective", "pseudo", "cluster")), \
        "the engine must record that it corrected for repeated measures"


# --- determinism ---

def test_same_input_and_seed_is_reproducible():
    rng = np.random.default_rng(11)
    x1 = rng.normal(size=300)
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=300), "y": 0.7 * x1 + rng.normal(size=300)})

    def fingerprint():
        rep = _run(df.copy(), "y", top_k=6)
        return (rep.fact_sheet.get("ml_metric_value"),
                tuple((c.feature, round(float(c.rho), 8), c.verdict) for c in rep.candidates))

    assert fingerprint() == fingerprint()
