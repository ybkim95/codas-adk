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


def _survival_holdout(rng, n=6000):
    """x raises the outcome hazard only; w raises every diagnosis, the outcome and the control alike
    (as more health-care contact would); age raises the outcome hazard."""
    h = pd.DataFrame({"x": rng.normal(size=n), "w": rng.normal(size=n), "age": rng.normal(size=n)})
    censor = rng.exponential(2.0, n)
    t = rng.exponential(1 / np.exp(0.5 * h.x + 0.5 * h.w + 0.4 * h.age))
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    t_nc = rng.exponential(1 / (0.5 * np.exp(0.6 * h.w)))
    h["nc_ctl_days"] = np.ceil(np.minimum(t_nc, censor) * 365)
    h["nc_ctl"] = (t_nc <= censor).astype(float)
    h.loc[h.index[:50], ["nc_ctl", "nc_ctl_days"]] = np.nan  # prevalent controls are left empty
    return h


def test_falsification_flags_a_candidate_that_also_predicts_a_negative_control(tmp_path):
    h = _survival_holdout(np.random.default_rng(5))
    h.to_csv(tmp_path / "f.holdout.csv", index=False)
    state = _state([_cand("x", 0.2, ["x"]), _cand("w", 0.2, ["w"])], ["age"])
    prereg = preregister(state, tmp_path, "f", survival_time="followup_days", negative_controls=["ctl", "absent"],
                         positive_controls=["age"], sensitivity_lag_days=90)
    result = confirm(prereg, tmp_path / "f.holdout.csv", tmp_path)
    x, w = result["hypotheses"]
    assert x["replicated"] and w["replicated"]  # both predict the outcome ...
    assert x["robust"] and not w["robust"]       # ... but only w also predicts a condition it cannot cause
    assert w["falsification_failed"] == ["negative control ctl"]
    assert {nc["control"]: nc["status"] for nc in x["negative_controls"]} == {
        "ctl": "no association", "absent": "not in the holdout table"}
    assert x["dose_response"]["monotone"] and x["dose_response"]["quantiles"] == 5
    assert x["lag_sensitivity"]["same_direction"] and x["lag_sensitivity"]["lag_days"] == 90
    assert result["positive_controls"][0]["control"] == "age" and result["positive_controls"][0]["detected"]
    assert result["robust"] == 1
    md = confirmation_markdown(result)
    assert "Negative-control outcomes" in md and "Positive controls" in md and "negative control ctl" in md


def test_dose_response_is_not_monotone_for_a_threshold_reversal(tmp_path):
    rng = np.random.default_rng(4)
    n = 4000
    h = pd.DataFrame({"x": rng.normal(size=n)})
    # Overall positive rank association, but the top fifth reverses: not a monotone dose-response.
    h["y"] = np.where(h.x > h.x.quantile(0.8), -1.0, h.x) + 0.3 * rng.normal(size=n)
    h.to_csv(tmp_path / "d.holdout.csv", index=False)
    prereg = preregister(_state([_cand("x", 0.3, ["x"])], []), tmp_path, "d")
    (x,) = confirm(prereg, tmp_path / "d.holdout.csv", tmp_path)["hypotheses"]
    assert x["same_direction"] and not x["dose_response"]["monotone"]
    assert x["falsification_failed"] == ["dose-response"] and not x["robust"]


def test_ledger_pools_every_hypothesis_tested_on_the_same_holdout(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    h = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(8)})
    h["y"] = 0.17 * h.f0 + rng.normal(size=n)
    h.to_csv(tmp_path / "l.holdout.csv", index=False)
    first = preregister(_state([_cand("f0", 0.2, ["f0"])], []), tmp_path, "l")
    alone = confirm(first, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")
    assert alone["ledger"] == {"key": "k", "hypotheses": 1, "registrations": 1}
    many = [_cand(f"f{i}", 0.1, [f"f{i}"]) for i in range(1, 8)] + [_cand("f0", 0.2, ["f0"])]
    second = preregister(_state(many, []), tmp_path, "l2")
    pooled = confirm(second, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")
    assert pooled["ledger"] == {"key": "k", "hypotheses": 9, "registrations": 2}
    f0 = next(r for r in pooled["hypotheses"] if r["feature"] == "f0")
    from codas.aou.confirm import _bh
    looks = [alone["hypotheses"][0]["p"]] + [r["p"] for r in pooled["hypotheses"]]
    at = 1 + [r["feature"] for r in pooled["hypotheses"]].index("f0")
    assert f0["q_ledger"] == pytest.approx(_bh(looks)[at])  # BH over all nine looks at the holdout
    assert f0["q_ledger"] > alone["hypotheses"][0]["q_ledger"]  # more looks, a stricter bar for f0
    again = confirm(second, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")  # the same registration
    assert again["ledger"]["hypotheses"] == 9  # is not counted twice
    other = confirm(first, tmp_path / "l.holdout.csv", tmp_path, ledger_key="other-holdout")
    assert other["ledger"]["hypotheses"] == 1
    from codas.aou.confirm import ledger_summary
    assert ledger_summary(tmp_path, "k")["hypotheses"] == 17  # the file keeps every look, repeats included


def _adherence_holdout(rng, n=6000):
    """``wear`` (days with data) drives both a wear-dependent feature and the outcome hazard, as health
    engagement would; ``x`` has a real effect of its own; ``bmi`` confounds x and is often missing."""
    wear = rng.normal(size=n)
    bmi = rng.normal(size=n)
    h = pd.DataFrame({"act_days": 150 + 20 * wear, "x": 0.6 * bmi + rng.normal(size=n),
                      "proxy": wear + 0.5 * rng.normal(size=n), "age": rng.normal(size=n)})
    t = rng.exponential(1 / np.exp(0.35 * h.x + 0.6 * wear + 0.5 * bmi + 0.2 * h.age))
    censor = rng.exponential(1.5, n)
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    # BMI recorded far more often for those diagnosed: complete-case is biased.
    h["bmi"] = np.where(rng.random(n) < np.where(h.y == 1, 0.9, 0.35), bmi, np.nan)
    return h


def test_adherence_adjustment_flags_a_wear_proxy_and_keeps_a_real_effect(tmp_path):
    h = _adherence_holdout(np.random.default_rng(11))
    h.to_csv(tmp_path / "w.holdout.csv", index=False)
    state = _state([_cand("x", 0.1, ["x"]), _cand("proxy", 0.12, ["proxy"])], ["age", "bmi"])
    prereg = preregister(state, tmp_path, "w", survival_time="followup_days", availability=["act_days"],
                         covariate_imputations=10)
    result = confirm(prereg, tmp_path / "w.holdout.csv", tmp_path)
    rows = {r["feature"]: r for r in result["hypotheses"]}
    assert rows["proxy"]["replicated"] and "adherence adjustment" in rows["proxy"]["falsification_failed"]
    assert not rows["proxy"]["robust"]
    assert rows["x"]["adherence_adjusted"]["holds"] and "adherence adjustment" in rows["x"]["falsification_passed"]
    assert [b["column"] for b in result["adherence_benchmark"]] == ["act_days"]
    md = confirmation_markdown(result)
    assert "wear adherence" in md and "imputed" in md


def test_missing_confounder_is_imputed_on_the_holdout(tmp_path):
    rng = np.random.default_rng(12)
    h = _adherence_holdout(rng)
    h.to_csv(tmp_path / "m.holdout.csv", index=False)
    state = _state([_cand("x", 0.1, ["x"])], ["age", "bmi", "act_days"])  # the correct outcome model
    mi = confirm(preregister(state, tmp_path, "m", survival_time="followup_days", availability=["act_days"],
                             covariate_imputations=10), tmp_path / "m.holdout.csv", tmp_path)["hypotheses"][0]
    cc = confirm(preregister(state, tmp_path, "m", survival_time="followup_days", availability=["act_days"],
                             covariate_imputations=0), tmp_path / "m.holdout.csv", tmp_path)["hypotheses"][0]
    target = 0.35 * h["x"].std(ddof=0)  # true log hazard ratio per SD of x
    assert mi["imputations"] == 10 and mi["n"] > cc["n"]
    assert abs(np.log(mi["hr_per_sd"]) - target) < abs(np.log(cc["hr_per_sd"]) - target)
    assert mi["complete_case"]["n"] == cc["n"]


def test_a_priori_hypotheses_form_their_own_family(tmp_path):
    rng = np.random.default_rng(21)
    n = 6000
    h = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "p1": rng.normal(size=n),
                      "p2": rng.normal(size=n), "age": rng.normal(size=n)})
    h["y"] = 0.25 * h.f1 + 0.04 * h.p1 + rng.normal(size=n)  # p1: a small real effect (p around 0.002)
    h.to_csv(tmp_path / "a.holdout.csv", index=False)
    state = _state([_cand("f1", 0.2, ["f1"]), _cand("f2", 0.1, ["f2"])], ["age"])
    state["a_priori_hypotheses"] = [
        {"feature": "p1", "direction": 1, "knownness": "emerging", "construct": "c1", "citation": "ref"},
        {"feature": "p2", "direction": -1, "knownness": "unexplored", "construct": "c2", "citation": ""},
        {"feature": "f1", "direction": 1, "knownness": "established", "construct": "c3", "citation": "ref"},
    ]
    prereg = preregister(state, tmp_path, "a")
    fam = {x["feature"]: x["family"] for x in prereg["hypotheses"]}
    assert fam == {"p1": "a_priori", "p2": "a_priori", "f1": "a_priori", "f2": "exploratory"}
    # Many earlier hypotheses on this holdout: the exploratory family pays for them, the a-priori one does not.
    ledger = tmp_path / "holdout_ledger.jsonl"
    ledger.write_text("".join(json.dumps({"key": "k", "prereg": f"old{i}", "stem": "old", "feature": f"x{i}",
                                          "p": 0.5, "same_direction": False}) + "\n" for i in range(200)))
    result = confirm(prereg, tmp_path / "a.holdout.csv", tmp_path, ledger_key="k")
    rows = {r["feature"]: r for r in result["hypotheses"]}
    assert rows["p1"]["family"] == "a_priori" and rows["p1"]["replicated"]       # BH over 3, not 205
    assert rows["p1"]["q_family"] < 0.05 < rows["p1"]["q_ledger"]
    assert not rows["p2"]["replicated"] and rows["f1"]["knownness"] == "established"
    md = confirmation_markdown(result)
    assert "A-priori hypotheses" in md and "Exploratory candidates" in md
