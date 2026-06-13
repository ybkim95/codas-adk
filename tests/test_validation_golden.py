"""Golden / characterization test for the validation battery.

``validate_candidate`` is large; this pins its EXACT output (every candidate's verdict and every
per-dimension test result, across cross-sectional / confounded / repeated-measures inputs) to a
hash. Any change to the battery — intentional or an accidental regression during a refactor — flips
the hash and fails here, so the behaviour can be reviewed deliberately rather than drifting silently.

If you intend to change the battery, recompute the hash and update _EXPECTED_SHA in the same commit.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from codas_core.discovery import DiscoveryRequest, run_discovery

_EXPECTED_SHA = "2db56e0b2b9256ef710482b627e4f46c1a3316f8c1bfafc61a0b4fd4fcb015a4"
# Full-report fingerprint (fact sheet + warnings + audit log + candidates) — pins the WHOLE discovery
# pipeline so _screen / _assemble_report / build_analysis_frame can also be refactored no-change.
_EXPECTED_REPORT_SHA = "b04ca2aa0afa5a7ddaed2f6ccb57d56e0ca5c337f0ba6a04d54608e41143a6e8"


def _fingerprint(df: pd.DataFrame, **kw) -> list:
    rep = run_discovery(df, DiscoveryRequest(validation_resamples=kw.pop("rs", 200), **kw))
    out = []
    for c in rep.candidates:
        tests = sorted((t.name, bool(t.passed), bool(t.applicable), bool(getattr(t, "hard_gate", False)))
                       for t in c.tests)
        out.append((c.feature, c.verdict, round(float(c.rho), 6), tests))
    return out


def _report_fingerprint(df: pd.DataFrame, **kw) -> dict:
    rep = run_discovery(df, DiscoveryRequest(validation_resamples=kw.pop("rs", 200), **kw))
    fs = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sorted(rep.fact_sheet.items())}
    cands = [(c.feature, c.verdict, round(float(c.rho), 6)) for c in rep.candidates]
    return {"fact_sheet": fs, "warnings": list(rep.warnings), "audit_log": list(rep.audit_log), "candidates": cands}


def _build_fixtures():
    """The three fixed inputs; RNG draw order is byte-identical to the hash-generating script."""
    rng = np.random.default_rng(0)
    n = 300
    d1 = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "y": None})
    d1["y"] = 0.5 * d1.x1 + rng.normal(size=n)
    z = rng.normal(size=n)
    d2 = pd.DataFrame({"z": z, "x": z + rng.normal(size=n) * 0.4, "y": z + rng.normal(size=n) * 0.4})
    subj = np.repeat(np.arange(25), 20)
    d3 = pd.DataFrame({"pid": subj, "x": rng.normal(size=25)[subj] + rng.normal(size=500) * 0.3,
                       "y": rng.normal(size=25)[subj]})
    return d1, d2, d3


def test_validation_battery_output_is_pinned():
    warnings.filterwarnings("ignore")
    d1, d2, d3 = _build_fixtures()
    fp = {
        "cross": _fingerprint(d1, target_column="y"),
        "confounded": _fingerprint(d2, target_column="y", confounder_columns=["z"]),
        "repeated": _fingerprint(d3, target_column="y", participant_id_column="pid"),
    }
    sha = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()
    assert sha == _EXPECTED_SHA, f"validation battery output changed (sha={sha}); review the diff and update if intended"


def test_full_discovery_report_is_pinned():
    """Pins the whole pipeline output (fact sheet, warnings, audit log, candidates), so _screen /
    _assemble_report / build_analysis_frame / run_discovery can be refactored with no silent change."""
    warnings.filterwarnings("ignore")
    d1, d2, d3 = _build_fixtures()
    fp = {
        "cross": _report_fingerprint(d1, target_column="y"),
        "confounded": _report_fingerprint(d2, target_column="y", confounder_columns=["z"]),
        "repeated": _report_fingerprint(d3, target_column="y", participant_id_column="pid"),
    }
    sha = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()
    assert sha == _EXPECTED_REPORT_SHA, f"discovery report changed (sha={sha}); review the diff and update if intended"
