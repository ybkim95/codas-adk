"""Pre-registration on discovery results and one-shot confirmation on a sealed holdout (dataset-agnostic)."""

import json

import numpy as np
import pandas as pd
import pytest

from codas.aou.confirm import confirm, confirmation_markdown, preregister


def _cand(feature, rho, components, verdict="validated"):
    return {"feature": feature, "rho": rho, "q_value": 0.001, "verdict": verdict, "components": components}


def _state(candidates, confounders, proposed=()):
    return {"target_column": "y", "confounder_columns": confounders, "excluded_columns": ["leak"],
            "latest_report": {"candidates": candidates}, "proposed_features": list(proposed)}


def test_continuous_outcome_replication_and_tamper_check(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    h = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "a": rng.uniform(1, 2, n),
                      "b": rng.uniform(1, 2, n), "c": rng.normal(size=n), "age": rng.normal(size=n)})
    h["y"] = 0.3 * h.f1 + 0.6 * (h.a / h.b) - 0.3 * h.c * h.f2 + 0.2 * h.age + rng.normal(size=n)
    h.to_csv(tmp_path / "x.holdout.csv", index=False)
    state = _state([_cand("f1", 0.2, ["f1"]), _cand("f2", 0.1, ["f2"]), _cand("a_over_b", 0.3, ["a", "b"]),
                    _cand("prod_cf2", -0.2, []), _cand("rejected_one", 0.5, ["f1"], verdict="rejected")],
                   ["age"], proposed=[{"name": "prod_cf2", "op": "product", "a": "c", "b": "f2"}])
    prereg = preregister(state, tmp_path, "x")
    assert [h["feature"] for h in prereg["hypotheses"]] == ["a_over_b", "f1", "prod_cf2", "f2"]  # validated only, by |rho|
    assert json.loads((tmp_path / "x_prereg.json").read_text())["sha256"] == prereg["sha256"]
    result = confirm(prereg, tmp_path / "x.holdout.csv", tmp_path)
    status = {r["feature"]: r["status"] for r in result["hypotheses"]}
    assert status == {"a_over_b": "replicated", "f1": "replicated", "prod_cf2": "replicated", "f2": "not replicated"}
    assert result["replicated"] == 3 and result["tested"] == 4 and result["confounders"] == ["age"]
    assert "3 of 4 pre-registered candidates replicated" in confirmation_markdown(result)
    tampered = {**prereg, "hypotheses": prereg["hypotheses"][:1]}
    with pytest.raises(ValueError, match="modified"):
        confirm(tampered, tmp_path / "x.holdout.csv", tmp_path)


def test_survival_outcome_uses_cox_and_never_adjusts_for_the_time_axis(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    h = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "age": rng.normal(size=n)})
    t = rng.exponential(1 / np.exp(0.4 * h.x + 0.3 * h.age))
    c = rng.exponential(1.0, n)
    h["followup_days"] = np.ceil(np.minimum(t, c) * 365)
    h["y"] = (t <= c).astype(int)
    h.to_csv(tmp_path / "s.holdout.csv", index=False)
    state = _state([_cand("x", 0.1, ["x"]), _cand("z", -0.05, ["z"])], ["age", "followup_days"])
    prereg = preregister(state, tmp_path, "s", survival_time="followup_days")
    result = confirm(prereg, tmp_path / "s.holdout.csv", tmp_path)
    assert result["confounders"] == ["age"]  # follow-up time is the time axis, not a covariate
    x, z = result["hypotheses"]
    assert x["test"] == "cox" and x["replicated"] and 1.3 < x["hr_per_sd"] < 1.7 and x["e_value"] > 1
    assert x["ci95"][0] < x["hr_per_sd"] < x["ci95"][1]
    assert not z["replicated"]
