"""External replication from the replication package alone: features rebuilt on another cohort's own tables
(a daily mean, a phenotype definition, a mapped column), tested in the registered direction, triangulated with
laboratory values, masked."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from codas import external


def _cohort(seed=0, n=600, days=200):
    rng = np.random.default_rng(seed)
    level = rng.normal(8000, 2000, n)
    swing = rng.uniform(500, 3000, n)
    ids = [f"P{i}" for i in range(n)]
    rows = []
    start = pd.Timestamp("2021-03-01")
    for i in range(n):
        for d in range(days):
            if rng.random() < 0.85:
                rows.append((ids[i], start + pd.Timedelta(days=d), level[i] + swing[i] * rng.normal()))
    daily = pd.DataFrame(rows, columns=["pid", "date", "steps"])
    age = rng.normal(50, 10, n)
    people = pd.DataFrame({"pid": ids, "age": age, "sex_f": rng.integers(0, 2, n),
                           "hba1c": 6.0 - 0.0002 * (level - 8000) + 0.3 * rng.normal(size=n),
                           "crp": 1 + 0.0004 * swing + rng.normal(size=n), "bmi": rng.normal(28, 4, n)})
    return people, daily


def test_registered_hypotheses_replicate_on_another_cohort(tmp_path):
    people, daily = _cohort()
    folder = tmp_path / "package"
    folder.mkdir()
    sd_def = {"name": "steps_spread", "signal": "act_steps", "summary": "sd", "min_days": 14}
    prereg = {"hypotheses": [
        {"feature": "act_steps_mean", "direction": -1, "definition": {"components": ["act_steps_mean"]}},
        {"feature": "dsl_steps_spread_1a2b3c4d", "direction": 1, "definition": {"phenotype": sd_def}},
        {"feature": "bmi_cov", "direction": 1, "definition": {}},
        {"feature": "hrmin_resting_hr", "direction": 1, "definition": {}}]}
    (folder / "pre_registration.json").write_text(json.dumps(prereg))
    mapping = {"participant": "pid", "date": "date", "window_days": 180, "signals": {"act_steps": "steps"},
               "features": {"bmi_cov": "bmi"}, "outcome": {"column": "hba1c"},
               "covariates": {"age": "age", "female": "sex_f"}, "labs": ["crp"]}
    result = external.replicate(external.load_package(folder), people, daily, mapping)
    rows = {r["feature"]: r for r in result["hypotheses"]}
    assert rows["act_steps_mean"]["replicated"] and rows["act_steps_mean"]["rho"] < -0.3
    assert rows["act_steps_mean"]["built"] == "mean of act_steps over the first 180 days"
    assert rows["dsl_steps_spread_1a2b3c4d"]["built"] == "phenotype definition"
    assert rows["dsl_steps_spread_1a2b3c4d"]["labs"]["crp"]["rho"] > 0.2     # biology the first cohort lacked
    assert rows["bmi_cov"]["built"] == "column bmi" and not rows["bmi_cov"].get("replicated")
    assert rows["hrmin_resting_hr"]["status"] == "not testable here"
    assert result["tested"] == 3
    # Run from the command line: files written, counts of 1-20 masked.
    people.to_csv(tmp_path / "people.csv", index=False)
    daily.to_csv(tmp_path / "days.csv", index=False)
    (tmp_path / "map.json").write_text(json.dumps(mapping))
    few = people.copy()
    few.loc[few.index[15:], "hba1c"] = np.nan
    few.to_csv(tmp_path / "few.csv", index=False)
    assert external.main(["--package", str(folder), "--people", str(tmp_path / "few.csv"), "--days",
                          str(tmp_path / "days.csv"), "--map", str(tmp_path / "map.json"), "--out",
                          str(tmp_path / "out")]) == 0
    saved = json.loads((tmp_path / "out" / "external_replication.json").read_text())
    assert all(r.get("n") in (None, "<=20") for r in saved["hypotheses"])
    assert "External replication" in (tmp_path / "out" / "external_replication.md").read_text()
