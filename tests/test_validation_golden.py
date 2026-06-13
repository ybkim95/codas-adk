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


def _fingerprint(df: pd.DataFrame, **kw) -> list:
    rep = run_discovery(df, DiscoveryRequest(validation_resamples=kw.pop("rs", 200), **kw))
    out = []
    for c in rep.candidates:
        tests = sorted((t.name, bool(t.passed), bool(t.applicable), bool(getattr(t, "hard_gate", False)))
                       for t in c.tests)
        out.append((c.feature, c.verdict, round(float(c.rho), 6), tests))
    return out


def test_validation_battery_output_is_pinned():
    warnings.filterwarnings("ignore")
    # NOTE: the RNG draw ORDER below must stay byte-identical to the script that generated the hash
    # (scripts capture in the commit message), or the synthetic inputs change and the hash shifts.
    rng = np.random.default_rng(0)
    n = 300
    d1 = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "y": None})
    d1["y"] = 0.5 * d1.x1 + rng.normal(size=n)
    z = rng.normal(size=n)
    d2 = pd.DataFrame({"z": z, "x": z + rng.normal(size=n) * 0.4, "y": z + rng.normal(size=n) * 0.4})
    subj = np.repeat(np.arange(25), 20)
    d3 = pd.DataFrame({"pid": subj, "x": rng.normal(size=25)[subj] + rng.normal(size=500) * 0.3,
                       "y": rng.normal(size=25)[subj]})
    fp = {
        "cross": _fingerprint(d1, target_column="y"),
        "confounded": _fingerprint(d2, target_column="y", confounder_columns=["z"]),
        "repeated": _fingerprint(d3, target_column="y", participant_id_column="pid"),
    }
    sha = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()
    assert sha == _EXPECTED_SHA, f"validation battery output changed (sha={sha}); review the diff and update if intended"
