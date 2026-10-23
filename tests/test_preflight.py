"""The deterministic preflight: before hours of agents, the holdout pipeline runs end to end on the
discovery share (never the sealed one) with stand-in hypotheses, and stops on what it cannot compute."""

import numpy as np
import pandas as pd

from codas.aou import runner


def _discovery(tmp_path, n=3000, broken=False):
    rng = np.random.default_rng(2)
    f = pd.DataFrame({"age": rng.normal(50, 15, n), "female": rng.integers(0, 2, n),
                      "act_steps_mean": rng.normal(8000, 2000, n), "act_days": rng.normal(150, 30, n)})
    f["hr_days"] = f["act_days"]                         # two sources counting the same days
    t, c = rng.exponential(1 / np.exp(0.3 * (f.age - 50) / 15)), rng.exponential(1.0, n)
    f["followup_days"] = np.ceil(np.minimum(t, c) * 365)
    f["y"] = (t <= c).astype(int)
    tn = rng.exponential(2.0, n)
    f["nc_ctl_days"] = -1.0 if broken else np.ceil(np.minimum(tn, c) * 365)
    f["nc_ctl"] = (tn <= c).astype(float)
    path = tmp_path / "explore_x.csv"
    f.to_csv(path, index=False)
    from codas.aou.cohort import holdout_path

    holdout_path(path).parent.mkdir(parents=True, exist_ok=True)
    holdout_path(path).write_text("never read\n")
    return path


def _run(path, tmp_path):
    return runner.preflight(path, tmp_path / "pre", stem="explore_x", target="y", survival_time="followup_days",
                            confounders=["age", "female", "act_days", "hr_days"], stand_ins=["act_steps_mean"],
                            negative_controls=["ctl"], positive_controls=["age"], availability=["act_days", "hr_days"])


def test_preflight_runs_the_holdout_pipeline_on_the_discovery_share(tmp_path, monkeypatch):
    seen = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: (seen.append(str(p)), real(p, *a, **k))[1])
    report = _run(_discovery(tmp_path), tmp_path)
    assert report["fatal"] == [] and any("hr_days" in f for f in report["flags"])  # duplicate noted, not fatal
    assert not any("holdout" in p for p in seen)                                     # the sealed share is untouched
    assert report["duplicates"] == [["act_days", "hr_days"]]


def test_preflight_stops_on_checks_that_cannot_be_computed(tmp_path):
    report = _run(_discovery(tmp_path, broken=True), tmp_path)
    assert any("negative control ctl" in f for f in report["fatal"])
