#!/usr/bin/env python3
"""Deterministic production-robustness audit for the CoDaS engine + service.

Prints a scored report card across five offline dimensions (R1 no-crash, R2 determinism,
R3 statistical correctness, R6 service layer, R8 scale). Needs no API key. The live agent
dimensions (R4 orchestration, R5 grounding integrity, R7 prompt injection) are in
``scripts/agent_robustness.py``.

    python scripts/robustness_audit.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from codas_core.data import InsufficientDataError
from codas_core.discovery import DiscoveryRequest, run_discovery


def _run(df: pd.DataFrame, target: str, **kw):
    return run_discovery(df, DiscoveryRequest(target_column=target, validation_resamples=kw.pop("rs", 150), **kw))


# ----------------------------------------------------------------------------------------------
# R1 — engine never crashes on degenerate input (must return a report OR InsufficientDataError)
# ----------------------------------------------------------------------------------------------

def _degenerate_cases() -> list[tuple[str, pd.DataFrame, str]]:
    rng = np.random.default_rng(0)
    n = 80
    base = {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "y": rng.normal(size=n)}
    cases: list[tuple[str, pd.DataFrame, str]] = [
        ("empty_dataframe", pd.DataFrame({"y": []}), "y"),
        ("single_row", pd.DataFrame({"x1": [1.0], "y": [2.0]}), "y"),
        ("two_rows", pd.DataFrame({"x1": [1.0, 2.0], "y": [2.0, 4.0]}), "y"),
        ("only_target_column", pd.DataFrame({"y": rng.normal(size=n)}), "y"),
        ("target_absent", pd.DataFrame(base), "not_a_column"),
        ("all_nan_target", pd.DataFrame({"x1": rng.normal(size=n), "y": [np.nan] * n}), "y"),
        ("all_nan_feature", pd.DataFrame({"x1": [np.nan] * n, "y": rng.normal(size=n)}), "y"),
        ("constant_target", pd.DataFrame({"x1": rng.normal(size=n), "y": [3.0] * n}), "y"),
        ("constant_feature", pd.DataFrame({"x1": [5.0] * n, "y": rng.normal(size=n)}), "y"),
        ("single_class_binary_target", pd.DataFrame({"x1": rng.normal(size=n), "y": [1] * n}), "y"),
        ("binary_text_target", pd.DataFrame({"x1": rng.normal(size=n), "y": rng.choice(["yes", "no"], n)}), "y"),
        ("multiclass_text_target", pd.DataFrame({"x1": rng.normal(size=n), "y": rng.choice(["a", "b", "c"], n)}), "y"),
        ("mostly_numeric_text_target", pd.DataFrame({"x1": rng.normal(size=n),
            "y": [str(v) for v in rng.normal(size=n - 3)] + ["x", "y", "z"]}), "y"),
        ("inf_in_feature", pd.DataFrame({"x1": [np.inf, -np.inf] + list(rng.normal(size=n - 2)),
            "y": rng.normal(size=n)}), "y"),
        ("huge_magnitudes", pd.DataFrame({"x1": rng.normal(size=n) * 1e300, "y": rng.normal(size=n) * 1e300}), "y"),
        ("tiny_magnitudes", pd.DataFrame({"x1": rng.normal(size=n) * 1e-300, "y": rng.normal(size=n) * 1e-300}), "y"),
        ("unicode_columns", pd.DataFrame({"变量": rng.normal(size=n), "café_score": rng.normal(size=n),
            "结果": rng.normal(size=n)}), "结果"),
        ("spaces_special_cols", pd.DataFrame({"my feat (1)": rng.normal(size=n), "f/2": rng.normal(size=n),
            "the target!": rng.normal(size=n)}), "the target!"),
        ("datetime_feature", pd.DataFrame({"t": pd.date_range("2020-01-01", periods=n, freq="D"),
            "x1": rng.normal(size=n), "y": rng.normal(size=n)}), "y"),
        ("boolean_feature", pd.DataFrame({"flag": rng.integers(0, 2, n).astype(bool), "y": rng.normal(size=n)}), "y"),
        ("high_cardinality_string", pd.DataFrame({"sid": [f"id_{i}" for i in range(n)], "y": rng.normal(size=n)}), "y"),
        ("duplicate_columns", pd.DataFrame(np.c_[rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)],
            columns=["x1", "x1", "y"]), "y"),
        ("duplicate_rows", pd.DataFrame({"x1": [1.0] * n, "x2": [2.0] * n, "y": [3.0] * n}), "y"),
        ("negatives_and_zeros", pd.DataFrame({"x1": rng.integers(-5, 1, n).astype(float), "y": rng.normal(size=n)}), "y"),
        ("mixed_type_column", pd.DataFrame({"x1": [1, 2.5, "three", None] * (n // 4), "y": rng.normal(size=n)}), "y"),
        ("sparse_target", pd.DataFrame({"x1": rng.normal(size=n),
            "y": [1.0, 2.0, 3.0] + [np.nan] * (n - 3)}), "y"),
        ("wide_p_gt_n", pd.DataFrame(rng.normal(size=(25, 200)), columns=[f"f{i}" for i in range(199)] + ["y"]), "y"),
        ("exact_target_copy_feature", pd.DataFrame({"copy": base["y"], "x2": base["x2"], "y": base["y"]}), "y"),
        ("nan_target_and_features", pd.DataFrame({"x1": [np.nan, 1.0] * (n // 2),
            "y": [2.0, np.nan] * (n // 2)}), "y"),
        ("all_identical_value", pd.DataFrame({"x1": [7.0] * n, "x2": [7.0] * n, "y": [7.0] * n}), "y"),
    ]
    return cases


def audit_r1_no_crash() -> tuple[int, int, list[str]]:
    fails = []
    cases = _degenerate_cases()
    for name, df, target in cases:
        try:
            report = _run(df, target, top_k=5)
            # An "ok" result must itself be JSON/finite-sane (no NaN headline sneaking through).
            fs = report.fact_sheet
            mv = fs.get("ml_metric_value")
            if mv is not None and isinstance(mv, float) and not np.isfinite(mv):
                fails.append(f"{name}: non-finite ml_metric_value in report")
        except InsufficientDataError:
            pass  # the designated, graceful boundary — a PASS
        except Exception as exc:  # noqa: BLE001
            fails.append(f"{name}: {type(exc).__name__}: {str(exc)[:80]}")
    return len(cases) - len(fails), len(cases), fails


# ----------------------------------------------------------------------------------------------
# R2 — determinism: same input + seed -> identical results
# ----------------------------------------------------------------------------------------------

def _signal_df(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.normal(size=n),
                         "y": 1.5 * x1 - 0.8 * x2 + rng.normal(size=n) * 0.5})


def _fingerprint(report) -> tuple:
    fs = report.fact_sheet
    cands = tuple((c.feature, round(float(c.rho), 8), round(float(c.q_value), 8), c.verdict)
                  for c in report.candidates)
    return (fs.get("ml_metric_name"), fs.get("ml_metric_value"),
            fs.get("internal_battery_passing_variants"), cands)


def audit_r2_determinism() -> tuple[int, int, list[str]]:
    fails = []
    datasets = [("signal_s0", _signal_df(0)), ("signal_s7", _signal_df(7)), ("signal_s99", _signal_df(99))]
    total = 0
    for name, df in datasets:
        total += 1
        a = _fingerprint(_run(df.copy(), "y", top_k=8))
        b = _fingerprint(_run(df.copy(), "y", top_k=8))
        if a != b:
            fails.append(f"{name}: two runs differ")
    return total - len(fails), total, fails


# ----------------------------------------------------------------------------------------------
# R3 — statistical correctness against ground truth
# ----------------------------------------------------------------------------------------------

def audit_r3_statistics() -> tuple[int, int, list[str]]:
    checks: list[tuple[str, bool, str]] = []

    # (a) pure noise -> no false positives across seeds
    fp = 0
    seeds = range(12)
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        df = pd.DataFrame({f"f{i}": rng.normal(size=200) for i in range(8)} | {"y": rng.normal(size=200)})
        rep = _run(df, "y", top_k=8)
        if sum(c.verdict == "validated" for c in rep.candidates) > 0:
            fp += 1
    checks.append((f"noise_false_positive_rate ({fp}/{len(seeds)})", fp <= 1,
                   f"{fp} of {len(seeds)} noise datasets produced a validated predictor (target ~0)"))

    # (b) planted MODERATE linear signal -> recovered as validated across seeds. The signal is
    # deliberately ~0.5 (real but not leakage-strength): a near-perfect rho>0.95 feature is correctly
    # rejected by the construct-validity gate as a likely target restatement, so a recall test must
    # use a genuine, non-circular effect size.
    recovered = 0
    for s in range(10):
        rng = np.random.default_rng(2000 + s)
        x1 = rng.normal(size=300)
        df = pd.DataFrame({"x1": x1, "n1": rng.normal(size=300), "n2": rng.normal(size=300),
                           "y": 0.6 * x1 + rng.normal(size=300)})  # rho ~ 0.5
        rep = _run(df, "y", top_k=8)
        if any(c.feature == "x1" and c.verdict in {"validated", "conditional"} for c in rep.candidates):
            recovered += 1
    checks.append((f"planted_signal_recall ({recovered}/10)", recovered >= 9,
                   f"x1 recovered in {recovered}/10 (target 10)"))

    # (c) exact target copy must be caught (NOT a clean validated predictor)
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    df = pd.DataFrame({"copy_of_y": y.copy(), "x2": rng.normal(size=300), "y": y})
    rep = _run(df, "y", top_k=8)
    copy_c = next((c for c in rep.candidates if c.feature == "copy_of_y"), None)
    caught = (copy_c is None) or (copy_c.verdict != "validated") or any(
        t.hard_gate and t.applicable and not t.passed for t in (copy_c.tests if copy_c else []))
    checks.append(("exact_leakage_caught", caught,
                   f"copy-of-target verdict={getattr(copy_c, 'verdict', 'absent')} (must not be clean-validated)"))

    # (d) pseudo-replication: repeated measures with a per-subject constant target -> effective-n correction
    rng = np.random.default_rng(5)
    subj = np.repeat(np.arange(40), 25)
    trait_s = rng.normal(size=40)                       # per-subject trait (one value per subject)
    target_s = trait_s + rng.normal(size=40) * 0.1      # target constant within subject
    trait, target = trait_s[subj], target_s[subj]       # expand to the 1000 rows
    df = pd.DataFrame({"pid": subj, "feat": trait + rng.normal(size=len(subj)) * 0.3,
                       "noise": rng.normal(size=len(subj)), "y": target})
    rep = run_discovery(df, DiscoveryRequest(target_column="y", participant_id_column="pid", validation_resamples=150))
    # The correction is recorded in the audit trail (aggregation) and/or the warnings (cluster/effective-n).
    records = list(rep.warnings) + list(rep.audit_log)
    corrected = any(("aggregat" in r.lower() or "effective" in r.lower() or "pseudo" in r.lower()
                     or "repeated" in r.lower() or "cluster" in r.lower()) for r in records)
    checks.append(("pseudo_replication_corrected", corrected,
                   "repeated-measures aggregation / effective-n correction is recorded (warnings or audit log)"))

    passed = sum(1 for _, ok, _ in checks if ok)
    details = [f"{name}: {'PASS' if ok else 'FAIL'} — {msg}" for name, ok, msg in checks]
    return passed, len(checks), details


# ----------------------------------------------------------------------------------------------
# R6 — service layer (auth, malformed input, path sandbox)
# ----------------------------------------------------------------------------------------------

def audit_r6_service() -> tuple[int, int, list[str]]:
    import os
    os.environ["CODAS_AGENT_API_KEYS"] = "audit-key"
    from fastapi.testclient import TestClient

    from codas_service.app import app
    client = TestClient(app)
    H = {"X-CoDaS-Agent-Key": "audit-key"}
    good_csv = "x1,y\n" + "\n".join(f"{i},{2*i}" for i in range(40))
    checks: list[tuple[str, bool]] = [
        ("healthz_open", client.get("/healthz").status_code == 200),
        ("auth_rejects_missing_key", client.get("/v1/health").status_code == 401),
        ("auth_rejects_wrong_key", client.get("/v1/health", headers={"X-CoDaS-Agent-Key": "x"}).status_code == 401),
        ("auth_accepts_key", client.get("/v1/health", headers=H).status_code == 200),
        ("discover_unknown_target_400", client.post("/v1/discover", headers=H,
            json={"csv": good_csv, "target_column": "nope"}).status_code == 400),
        ("malformed_base64_400", client.post("/v1/discover", headers=H,
            json={"csv_base64": "!!!notbase64!!!", "target_column": "y"}).status_code == 400),
        ("empty_payload_400", client.post("/v1/discover", headers=H, json={"target_column": "y"}).status_code == 400),
        ("valid_discover_200", client.post("/v1/discover", headers=H,
            json={"csv": good_csv, "target_column": "y", "validation_resamples": 100}).status_code == 200),
    ]
    # path sandbox: the ADK tool guardrail must block a read outside the allowed roots
    from codas_agents.callbacks import before_tool_guardrail

    class _T: name = "profile_dataset"
    blocked = before_tool_guardrail(_T(), {"csv_path": "/etc/passwd"}, None)
    checks.append(("path_sandbox_blocks_etc_passwd", isinstance(blocked, dict) and "error" in blocked))
    passed = sum(1 for _, ok in checks if ok)
    return passed, len(checks), [f"{name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks]


# ----------------------------------------------------------------------------------------------
# R8 — scale / performance (must finish within the interactive budget, no crash)
# ----------------------------------------------------------------------------------------------

def audit_r8_scale() -> tuple[int, int, list[str]]:
    import os
    os.environ.setdefault("CODAS_DISCOVERY_BUDGET_SECONDS", "120")
    checks = []
    rng = np.random.default_rng(0)

    # tall: 200k rows x 20 cols
    n = 200_000
    tall = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(19)})
    tall["y"] = 1.2 * tall["f0"] + rng.normal(size=n)
    t0 = time.monotonic()
    try:
        _run(tall, "y", top_k=10, rs=200)
        dt = time.monotonic() - t0
        checks.append((f"tall_200k_rows ({dt:.0f}s)", dt < 180))
    except Exception as exc:  # noqa: BLE001
        checks.append((f"tall_200k_rows CRASH {type(exc).__name__}", False))

    # wide: 500 cols x 300 rows
    wide = pd.DataFrame(rng.normal(size=(300, 500)), columns=[f"f{i}" for i in range(499)] + ["y"])
    wide["y"] = 1.5 * wide["f0"] + rng.normal(size=300) * 0.5
    t0 = time.monotonic()
    try:
        _run(wide, "y", top_k=10, rs=200)
        dt = time.monotonic() - t0
        checks.append((f"wide_500_cols ({dt:.0f}s)", dt < 180))
    except Exception as exc:  # noqa: BLE001
        checks.append((f"wide_500_cols CRASH {type(exc).__name__}", False))

    passed = sum(1 for _, ok in checks if ok)
    return passed, len(checks), [f"{name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks]


# ----------------------------------------------------------------------------------------------

def main() -> int:
    dims = [
        ("R1  engine no-crash on degenerate input", audit_r1_no_crash),
        ("R2  determinism / reproducibility", audit_r2_determinism),
        ("R3  statistical correctness (ground truth)", audit_r3_statistics),
        ("R6  service layer (auth / malformed / sandbox)", audit_r6_service),
        ("R8  scale / performance", audit_r8_scale),
    ]
    print("=" * 90)
    print("CoDaS-ADK PRODUCTION ROBUSTNESS AUDIT (offline dimensions)")
    print("=" * 90)
    grand_pass = grand_total = 0
    failures: list[str] = []
    for title, fn in dims:
        passed, total, details = fn()
        grand_pass += passed
        grand_total += total
        mark = "✅" if passed == total else "❌"
        print(f"\n{mark} {title}:  {passed}/{total}")
        for d in details:
            if "FAIL" in d or "CRASH" in d or (": " in d and not d.endswith("PASS")):
                print(f"      • {d}")
        for d in details:
            if d.startswith(tuple(["•"])):
                pass
        # always show fail lines explicitly
        for d in details:
            if "FAIL" in d or "CRASH" in d:
                failures.append(f"{title} -> {d}")
    print("\n" + "=" * 90)
    pct = 100.0 * grand_pass / grand_total if grand_total else 0.0
    print(f"OVERALL: {grand_pass}/{grand_total} checks passed ({pct:.1f}%)")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
    print("=" * 90)
    return 0 if grand_pass == grand_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
