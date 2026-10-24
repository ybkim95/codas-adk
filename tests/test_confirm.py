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
    assert "3 of 4 novel hypotheses replicated" in confirmation_markdown(result)
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


def test_ledger_never_lets_a_rerun_or_a_known_outcome_count_again(tmp_path):
    """One cumulative ledger per holdout: a repeat of an (outcome, feature) test, or any later registration
    about an outcome already tested there, is reported but not confirmatory; novel tests are judged by online
    FDR in ledger order, with verdicts frozen when made."""
    from codas.aou import ledger

    rng = np.random.default_rng(5)
    n = 3000
    h = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(8)})
    h["y"] = 0.17 * h.f0 + rng.normal(size=n)
    h.to_csv(tmp_path / "l.holdout.csv", index=False)
    outcome_a, outcome_b = {"kind": "measurement", "concept_ids": [1]}, {"kind": "measurement", "concept_ids": [2]}
    first = preregister(_state([_cand("f0", 0.2, ["f0"])], []), tmp_path, "l", outcome=outcome_a)
    alone = confirm(first, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")
    (f0,) = alone["hypotheses"]
    assert f0["role"] == "novel" and f0["replicated"] and f0["test_level"] == pytest.approx(ledger.lord_level(1, []))
    # A second run on the same outcome (renamed, more candidates, f0 again): nothing in it is confirmatory.
    many = [_cand(f"f{i}", 0.1, [f"f{i}"]) for i in range(1, 8)] + [_cand("f0", 0.2, ["f0"])]
    second = preregister(_state(many, []), tmp_path, "l_renamed", outcome=outcome_a)
    rerun = confirm(second, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")
    roles = {r["feature"]: (r["role"], r.get("role_reason", "")) for r in rerun["hypotheses"]}
    assert all(role == "non_confirmatory" for role, _ in roles.values())
    assert "repeat" in roles["f0"][1] and "already tested" in roles["f1"][1]
    assert rerun["replicated"] == 0 and rerun["tested"] == 0 and rerun["non_confirmatory"] == 8
    # Confirming the same registration again adds nothing and changes no verdict.
    again = confirm(second, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")
    assert again["ledger"]["hypotheses"] == 9 and [r["status"] for r in again["hypotheses"]] == \
        [r["status"] for r in rerun["hypotheses"]]
    # A new outcome continues the online stream: its first test comes after f0's rejection.
    third = preregister(_state([_cand("f0", 0.2, ["f0"])], []), tmp_path, "m", outcome=outcome_b)
    fresh = confirm(third, tmp_path / "l.holdout.csv", tmp_path, ledger_key="k")
    assert fresh["hypotheses"][0]["role"] == "novel"
    assert fresh["hypotheses"][0]["test_level"] == pytest.approx(ledger.lord_level(2, [1]))
    summary = ledger.summary(tmp_path, "k")
    assert summary["novel_tested"] == 2 and summary["non_confirmatory"] == 8 and summary["hypotheses"] == 10
    assert confirm(first, tmp_path / "l.holdout.csv", tmp_path, ledger_key="other")["ledger"]["hypotheses"] == 1


def test_lord_levels_are_valid_and_reward_discoveries():
    from codas.aou import ledger

    assert sum(ledger.gamma(j) for j in range(1, 200000)) < 1  # the spending sequence never overspends
    levels = [ledger.lord_level(t, []) for t in range(1, 50)]
    assert all(b <= a for a, b in zip(levels, levels[1:])) and levels[0] < ledger.ALPHA
    assert ledger.lord_level(5, [1, 3]) > ledger.lord_level(5, [])  # earlier discoveries earn more wealth
    assert ledger.one_sided_p(0.04, True) == pytest.approx(0.02)
    assert ledger.one_sided_p(0.04, False) == pytest.approx(0.98)


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


def test_established_hypotheses_form_a_reference_family_and_old_entries_open_their_outcomes(tmp_path):
    rng = np.random.default_rng(21)
    n = 6000
    h = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "p1": rng.normal(size=n),
                      "p2": rng.normal(size=n), "age": rng.normal(size=n)})
    h["y"] = 0.25 * h.f1 + 0.08 * h.p1 + rng.normal(size=n)
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
    # Entries written before verdicts were frozen: their outcome ("old") counts as opened, nothing else.
    ledger = tmp_path / "holdout_ledger.jsonl"
    ledger.write_text("".join(json.dumps({"key": "k", "prereg": f"old{i}", "stem": "old", "feature": f"x{i}",
                                          "p": 0.5, "same_direction": False}) + "\n" for i in range(200)))
    result = confirm(prereg, tmp_path / "a.holdout.csv", tmp_path, ledger_key="k")
    rows = {r["feature"]: r for r in result["hypotheses"]}
    assert rows["f1"]["role"] == "reference" and rows["f1"]["reproduced"] and not rows["f1"]["replicated"]
    assert rows["p1"]["role"] == "novel" and rows["p1"]["replicated"]       # judged online, not against 200 old looks
    assert rows["p2"]["role"] == "novel" and not rows["p2"]["replicated"]
    assert result["reference"] == {"tested": 1, "reproduced": 1, "robust": result["reference"]["robust"]}
    assert result["ledger"]["legacy"] == 200 and result["ledger"]["legacy_outcomes"] == ["old"]
    md = confirmation_markdown(result)
    assert "Novel hypotheses" in md and "Reference associations" in md
    later = preregister({**state, "a_priori_hypotheses": []}, tmp_path, "old")  # an outcome tested before
    assert all(r["role"] == "non_confirmatory" for r in confirm(later, tmp_path / "a.holdout.csv", tmp_path,
                                                                 ledger_key="k")["hypotheses"])


def test_within_person_effect_is_reported_from_the_sealed_panel(tmp_path):
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(31)
    n = 1500
    h = pd.DataFrame({"x": rng.normal(size=n), "trait": rng.normal(size=n), "age": rng.normal(size=n)})
    h["y"] = 0.3 * h.x + 0.3 * h.trait + rng.normal(size=n)
    h.to_csv(tmp_path / "p.holdout.csv", index=False)
    people, visits = 400, 4
    pid = np.repeat(np.arange(people), visits)
    level = np.repeat(rng.normal(size=people), visits)
    panel = pd.DataFrame({"person_index": pid, "x": level + rng.normal(size=people * visits),
                          "trait": np.repeat(rng.normal(size=people), visits), "age": 50.0,
                          "days_from_start": np.tile([100, 200, 300, 400], people)})
    panel["y"] = np.repeat(rng.normal(size=people), visits) + 0.5 * (panel.x - level) + 0.5 * panel.trait \
        + rng.normal(size=len(panel))
    panel.to_csv(tmp_path / "p.panel.holdout.csv", index=False)
    prereg = preregister(_state([_cand("x", 0.2, ["x"]), _cand("trait", 0.2, ["trait"])], ["age"]), tmp_path, "p")
    result = confirm(prereg, tmp_path / "p.holdout.csv", tmp_path, panel_csv=tmp_path / "p.panel.holdout.csv")
    rows = {r["feature"]: r for r in result["hypotheses"]}
    assert rows["x"]["within_person"]["within"]["p"] < 1e-6
    trait = rows["trait"]["within_person"]
    assert "error" in trait or trait["within"]["ci95"][0] < 0 < trait["within"]["ci95"][1]
    assert "Within-person" in confirmation_markdown(result)


def test_one_failing_check_does_not_discard_the_confirmation(tmp_path, monkeypatch):
    from codas.aou import confirm as confirm_module

    h = _survival_holdout(np.random.default_rng(5))
    h.to_csv(tmp_path / "e.holdout.csv", index=False)
    state = _state([_cand("x", 0.2, ["x"]), _cand("w", 0.2, ["w"])], ["age"])
    prereg = preregister(state, tmp_path, "e", survival_time="followup_days", negative_controls=["ctl"],
                         positive_controls=["age"], sensitivity_lag_days=90)
    real_dose, real_effect = confirm_module._dose_response, confirm_module._effect

    def broken_dose(frame, x, *a, **k):
        raise ValueError("math domain error")

    def broken_effect(frame, x, *a, **k):
        if x.name == "w":
            raise ValueError("math domain error")
        return real_effect(frame, x, *a, **k)

    monkeypatch.setattr(confirm_module, "_dose_response", broken_dose)
    monkeypatch.setattr(confirm_module, "_effect", broken_effect)
    result = confirm(prereg, tmp_path / "e.holdout.csv", tmp_path)
    x, w = result["hypotheses"]
    assert x["replicated"] and "math domain error" in x["errors"]["dose_response"]
    assert x["dose_response"] is None                       # reported as not computed, not as a failure
    assert w["status"] == "test failed" and "math domain error" in w["errors"]["effect"] and not w["replicated"]
    assert result["tested"] == 2 and (tmp_path / "e_confirmation.json").exists()
    assert "test failed" in confirmation_markdown(result)
    monkeypatch.setattr(confirm_module, "_dose_response", real_dose)


def test_negative_controls_that_could_not_be_computed_never_pass(tmp_path):
    from codas.aou.confirm import audit

    h = _survival_holdout(np.random.default_rng(5))
    h["days_a"] = np.random.default_rng(6).normal(150, 40, len(h))
    h["days_b"] = h["days_a"]                                 # two identical wear-day columns
    h["nc_ctl_days"] = -1.0                                   # a control whose follow-up never starts
    h.to_csv(tmp_path / "n.holdout.csv", index=False)
    state = _state([_cand("x", 0.2, ["x"])], ["age", "days_a", "days_b"])
    prereg = preregister(state, tmp_path, "n", survival_time="followup_days", negative_controls=["ctl"],
                         positive_controls=["age"], sensitivity_lag_days=90)
    result = confirm(prereg, tmp_path / "n.holdout.csv", tmp_path)
    x = result["hypotheses"][0]
    assert np.isfinite(x["hr_per_sd"]) and x["replicated"]    # the duplicate is dropped, not fatal
    assert "days_b" in x["dropped_covariates"]
    nc = x["negative_controls"][0]
    assert nc["status"] == "could not be computed"
    assert not any("negative control" in t for t in x["falsification_passed"])
    flags = audit(result)
    assert any("days_b" in f for f in flags) and any("negative control" in f for f in flags)
    assert "could not be computed" in confirmation_markdown(result)


def test_robust_needs_at_least_one_check_that_ran(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    h = pd.DataFrame({"f1": rng.normal(size=n), "age": rng.normal(size=n)})
    h["y"] = 0.6 * h.f1 + rng.normal(size=n)
    h.to_csv(tmp_path / "r.holdout.csv", index=False)  # too few rows for quintiles: no check can run
    prereg = preregister(_state([_cand("f1", 0.4, ["f1"])], ["age"]), tmp_path, "r")
    row = confirm(prereg, tmp_path / "r.holdout.csv", tmp_path)["hypotheses"][0]
    assert row["replicated"] and not row["robust"] and row["falsification_passed"] == []


def _followup_confounded(rng, n=6000):
    """``x`` lowers the hazard but lengthens follow-up (those with high x joined earlier), so more of
    them are diagnosed in the window: the event flag rises with x while the hazard ratio is below 1."""
    h = pd.DataFrame({"x": rng.normal(size=n), "age": rng.normal(size=n)})
    t = rng.exponential(1 / (0.4 * np.exp(-0.3 * h.x + 0.3 * h.age)))
    censor = np.minimum(rng.uniform(0.2, 1.0, n) * np.exp(1.2 * h.x), 6)
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    return h


def test_survival_direction_comes_from_the_discovery_cox_model(tmp_path):
    discovery, holdout = _followup_confounded(np.random.default_rng(31)), _followup_confounded(np.random.default_rng(32))
    discovery.to_csv(tmp_path / "d.csv", index=False)
    holdout.to_csv(tmp_path / "d.holdout.csv", index=False)
    # The engine's binary screen sees the event flag rise with x.
    state = _state([_cand("x", 0.37, ["x"])], ["age"])
    prereg = preregister(state, tmp_path, "d", survival_time="followup_days", discovery_csv=tmp_path / "d.csv")
    (h,) = prereg["hypotheses"]
    assert h["direction"] == -1 and h["discovery_hr"] < 1 and h["discovery_rho"] == 0.37
    assert "disagreed" in h["direction_note"]
    result = confirm(prereg, tmp_path / "d.holdout.csv", tmp_path)
    (row,) = result["hypotheses"]
    assert row["replicated"] and row["hr_per_sd"] < 1 and row["discovery_hr"] == h["discovery_hr"]
    assert any("binary screen disagreed" in flag and "x" in flag for flag in result["audit"])
    assert "HR " in confirmation_markdown(result)


def test_a_priori_direction_is_kept_and_continuous_outcomes_are_unchanged(tmp_path):
    discovery = _followup_confounded(np.random.default_rng(33))
    discovery.to_csv(tmp_path / "a.csv", index=False)
    state = {**_state([], ["age"]), "a_priori_hypotheses": [{"feature": "x", "direction": 1, "knownness": "emerging"}]}
    prereg = preregister(state, tmp_path, "a", survival_time="followup_days", discovery_csv=tmp_path / "a.csv")
    (h,) = prereg["hypotheses"]
    assert h["direction"] == 1 and h["discovery_hr"] < 1  # the literature's prediction stands; the HR is recorded
    no_time = preregister(_state([_cand("x", 0.37, ["x"])], ["age"]), tmp_path, "c", discovery_csv=tmp_path / "a.csv")
    assert no_time["hypotheses"][0]["direction"] == 1 and "discovery_hr" not in no_time["hypotheses"][0]


def test_adherence_check_is_not_counted_when_wear_is_already_adjusted_for(tmp_path):
    h = _adherence_holdout(np.random.default_rng(13))
    h["hr_days"] = h["act_days"]  # a second days column from the same table: identical values
    h.to_csv(tmp_path / "v.holdout.csv", index=False)
    state = _state([_cand("x", 0.1, ["x"])], ["age", "bmi", "act_days"])
    prereg = preregister(state, tmp_path, "v", survival_time="followup_days", availability=["act_days", "hr_days"])
    result = confirm(prereg, tmp_path / "v.holdout.csv", tmp_path)
    (row,) = result["hypotheses"]
    assert "adherence_adjusted" not in row
    assert not any("adherence" in t for t in row["falsification_passed"] + row["falsification_failed"])
    assert result["availability"] == [] and "already in the adjustment set" in result["adherence_status"]
    assert "already in the adjustment set" in confirmation_markdown(result)


def test_duplicate_availability_columns_are_used_once(tmp_path):
    h = _adherence_holdout(np.random.default_rng(14))
    h["hr_days"] = h["act_days"]
    h.to_csv(tmp_path / "u.holdout.csv", index=False)
    state = _state([_cand("proxy", 0.12, ["proxy"])], ["age"])
    prereg = preregister(state, tmp_path, "u", survival_time="followup_days", availability=["act_days", "hr_days"])
    result = confirm(prereg, tmp_path / "u.holdout.csv", tmp_path)
    assert result["availability"] == ["act_days"] and [b["column"] for b in result["adherence_benchmark"]] == ["act_days"]
    (row,) = result["hypotheses"]
    assert np.isfinite(row["adherence_adjusted"]["p"]) and "adherence adjustment" in row["falsification_failed"]


def test_reconciliation_puts_engine_debate_and_holdout_side_by_side(tmp_path):
    from codas.aou.confirm import reconciliation_markdown

    h = _survival_holdout(np.random.default_rng(15))
    h.to_csv(tmp_path / "r.holdout.csv", index=False)
    state = {**_state([_cand("x", 0.1, ["x"]), _cand("w", 0.1, ["w"])], ["age"]),
             "a_priori_hypotheses": [{"feature": "age_x", "direction": 1}]}
    prereg = preregister(state, tmp_path, "r", survival_time="followup_days")
    result = confirm(prereg, tmp_path / "r.holdout.csv", tmp_path)
    next(r for r in result["hypotheses"] if r["feature"] == "w")["debate"] = "concede"
    md = reconciliation_markdown(result, {"x": "validated", "w": "validated"})
    assert md.startswith("## ") and "written before the holdout was opened" in md
    lines = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| ") and "---" not in line}
    assert "validated" in lines["x"] and "replicated" in lines["x"]
    assert "concede" in lines["w"]
    assert "not testable" in lines["age_x"] and "registered a priori" in lines["age_x"]


def _scaled_survival(rng, spread, n=8000):
    """Log hazard rises 0.3 per unit of x; ``spread`` is x's SD on this share."""
    h = pd.DataFrame({"x": rng.normal(0, spread, n), "age": rng.normal(size=n)})
    t = rng.exponential(1 / np.exp(0.3 * h.x + 0.3 * h.age))
    censor = rng.exponential(1.0, n)
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    return h


def test_hazard_ratios_per_sd_use_the_discovery_sd_on_the_holdout(tmp_path):
    """A holdout with twice the spread once doubled the log hazard ratio "per SD": each share was scaled by
    its own SD. The unit is now the discovery SD, frozen in the pre-registration."""
    discovery, holdout = _scaled_survival(np.random.default_rng(41), 1.0), _scaled_survival(np.random.default_rng(42), 2.0)
    discovery.to_csv(tmp_path / "f.csv", index=False)
    holdout.to_csv(tmp_path / "f.holdout.csv", index=False)
    prereg = preregister(_state([_cand("x", 0.2, ["x"])], ["age"]), tmp_path, "f", survival_time="followup_days",
                         discovery_csv=tmp_path / "f.csv")
    (h,) = prereg["hypotheses"]
    assert h["transform"]["scale"] == pytest.approx(1.0, abs=0.05) and "discovery SD" in h["transform_text"]
    (row,) = confirm(prereg, tmp_path / "f.holdout.csv", tmp_path)["hypotheses"]
    # Unclipped, the holdout effect is exp(0.3) per discovery SD, not exp(0.6) per its own SD. Clipped at the
    # discovery percentiles (a wider holdout loses more of its tails) it is smaller still, never exp(0.6).
    assert row["unclipped"]["hr_per_sd"] == pytest.approx(np.exp(0.3), abs=0.06)
    assert 1 < row["hr_per_sd"] < np.exp(0.45)
    assert row["discovery_hr"] == pytest.approx(np.exp(0.3), abs=0.06) and row["transform_text"] == h["transform_text"]
    assert "online false-discovery-rate" in prereg["correction"]
