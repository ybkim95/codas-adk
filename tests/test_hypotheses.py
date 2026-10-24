"""A hypothesis in the template strong biomarker papers use: every part is checked before registration
(codas.core.hypotheses), and every prediction it makes is scored on the sealed holdout (codas.aou.confirm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codas.aou import confirm as c
from codas.aou.confirm import confirm, confirmation_markdown, preregister
from codas.core import hypotheses as design
from codas.core import literature

COLUMNS = {"steps", "vigorous", "age", "bmi", "wear"}


def _check(change=None, *, measure="rho", testable=None, citable=(), blocked=(), confounders=()):
    item = {"feature": "steps", "direction": "negative", "construct": "activity volume", "incumbent": "vigorous",
            "shape": "monotone", "magnitude": {"measure": "rho", "low": -0.3, "high": -0.05, "source": "analogy"},
            "mechanism": ["steps raise insulin sensitivity", "insulin sensitivity lowers glucose"],
            "alternatives": ["wear"], "predictions": ["attenuation_below_half_after_incumbent"], **(change or {})}
    return design.validate(item, feature_ok=None, measure=measure, columns=COLUMNS, blocked=set(blocked),
                           confounders=set(confounders), citable=set(citable), cite_keys=literature.tokens,
                           testable=set(design.PREDICTIONS) if testable is None else set(testable))


def test_a_complete_hypothesis_is_put_in_normal_form():
    entry, problems = _check({"shape": "linear", "mechanism": "steps -> insulin sensitivity -> glucose",
                              "predictions": "attenuation_below_half_after_incumbent", "mediators": ["bmi"]})
    assert problems == []
    assert entry["direction"] == -1 and entry["shape"] == "monotone" and entry["incumbent"] == "vigorous"
    assert entry["mechanism_steps"] == ["steps", "insulin sensitivity", "glucose"]
    # Each alternative explanation brings the prediction that tests it.
    assert entry["alternatives"] == [{"explanation": "wear", "test": "adherence_robust"}]
    assert entry["predictions"] == ["attenuation_below_half_after_incumbent", "adherence_robust"]
    assert entry["magnitude"] == {"measure": "rho", "low": -0.3, "high": -0.05, "source": "analogy"}


@pytest.mark.parametrize("change, expected", [
    ({"construct": ""}, "name the construct"),
    ({"direction": "up"}, "direction must be positive or negative"),
    ({"shape": "wiggly"}, "shape must be one of"),
    ({"magnitude": {"measure": "rho", "low": 0.05, "high": 0.3, "source": "analogy"}}, "predicted side of 0"),
    ({"magnitude": {"measure": "rho", "low": -0.95, "high": -0.1, "source": "analogy"}}, "outside the plausible"),
    ({"magnitude": {"measure": "hr_per_sd", "low": 0.6, "high": 0.9, "source": "analogy"}}, "tested as rho"),
    ({"magnitude": {"measure": "rho", "low": "small", "high": -0.1, "source": "analogy"}}, "needs numbers"),
    ({"magnitude": {"measure": "rho", "low": -0.1, "high": -0.3, "source": "analogy"}}, "needs numbers"),
    ({"magnitude": {"measure": "rho", "low": -0.3, "high": -0.05, "source": "Smith 2020"}}, "source must be"),
    ({"magnitude": {"measure": "rho", "low": -0.3, "high": -0.05, "source": "[PMID:1]"}}, "source must be"),
    ({"mechanism": ["steps lower glucose"]}, "at least 2 steps"),
    ({"confounders": ["income"]}, "not columns of the table: income"),
    ({"mediators": ["bmi"], "confounders": ["bmi"]}, "cannot be both a mediator"),
    ({"alternatives": ["bad luck"]}, "alternative explanation 'bad_luck'"),
    ({"predictions": ["it replicates"]}, "unknown predictions"),
    ({"incumbent": "none"}, "needs an incumbent"),
    ({"incumbent": "steps"}, "must be another column"),
    ({"incumbent": "nowhere"}, "not a usable column"),
    ({"alternatives": [], "predictions": []}, "at least one falsification prediction"),
])
def test_each_missing_or_untestable_part_is_named(change, expected):
    entry, problems = _check(change)
    assert entry is None and any(expected in p for p in problems), problems


def test_the_measure_and_its_null_follow_the_outcome_type():
    entry, problems = _check({"magnitude": {"measure": "hr_per_sd", "low": 0.6, "high": 0.95,
                                            "source": "[PMID:30000001]"}},
                             measure="hr_per_sd", citable={"PMID:30000001"})
    assert problems == [] and entry["magnitude"]["source"] == "[PMID:30000001]"
    _, problems = _check({"magnitude": {"measure": "hr_per_sd", "low": 0.8, "high": 1.2, "source": "analogy"}},
                         measure="hr_per_sd")
    assert any("predicted side of 1" in p for p in problems)


def test_what_the_design_cannot_score_or_must_adjust_for_is_refused():
    _, problems = _check({"predictions": ["within_person_same_sign"]},
                         testable={"attenuation_below_half_after_incumbent", "adherence_robust"})
    assert "this design cannot score within_person_same_sign" in problems
    _, problems = _check({"mediators": ["bmi"]}, confounders={"bmi"})  # declared a confounder elsewhere
    assert any("cannot be both a mediator" in p for p in problems)
    _, problems = _check(blocked={"vigorous"})
    assert any("not a usable column" in p for p in problems)
    text = design.template_text("hr_per_sd")
    assert "hr_per_sd" in text and all(p in text for p in design.PREDICTIONS)
    assert all(a in text for a in design.ALTERNATIVES)


def test_attenuation_and_shrinkage_are_on_the_log_hazard_scale():
    assert c._attenuation({"hr_per_sd": 1.5}, {"hr_per_sd": 1.2}) == round(1 - np.log(1.2) / np.log(1.5), 3)
    assert c._attenuation({"rho": 0.2}, {"rho": 0.15}) == 0.25
    assert c._attenuation({"hr_per_sd": 1.5}, {"hr_per_sd": float("nan")}) is None
    assert c._shrinkage({"discovery_hr": 1.5, "hr_per_sd": 1.2}) == round(np.log(1.2) / np.log(1.5), 3)
    assert c._shrinkage({"discovery_hr": 1.0, "hr_per_sd": 1.2}) is None and c._shrinkage({}) is None


def test_shapes_are_read_from_the_effect_by_quantile():
    rising = {"estimates": [0, 0.1, 0.2, 0.3, 0.4], "monotone": True}
    assert c._shape_holds("monotone", rising, 1)[0] == "met"
    assert c._shape_holds("monotone", {**rising, "monotone": False}, 1)[0] == "failed"
    step = {"estimates": [0, 0.02, 0.05, 0.9, 1.0], "monotone": True}
    assert c._shape_holds("threshold", step, 1)[0] == "met"
    assert c._shape_holds("threshold", rising, 1)[0] == "failed"     # even steps: no threshold
    assert c._shape_holds("threshold", step, -1)[0] == "failed"      # the wrong way
    u = {"estimates": [0.5, 0.05, 0, 0.1, 0.6], "monotone": False}
    assert c._shape_holds("u_shaped", u, 1)[0] == "met"
    assert c._shape_holds("u_shaped", rising, 1)[0] == "failed"
    assert c._shape_holds("monotone", None, 1)[0] == "not testable"


def test_each_falsification_prediction_is_scored_from_its_check():
    row = {"negative_controls": [{"control": "a", "status": "no association"},
                                 {"control": "b", "status": "could not be computed"}],
           "lag_sensitivity": {"test": "cox", "lag_days": 730, "hr_per_sd": 1.3, "ci95": [1.1, 1.5], "p": 0.001,
                               "same_direction": True},
           "adherence_adjusted": {"p": 0.01, "holds": False},
           "incumbent_adjusted": {"incumbent": "inc", "attenuation": 0.2, "same_direction": True,
                                  "effect_without_excludes_null": True},
           "within_person": {"within": {"estimate": -0.1}, "same_direction": False}}
    got = {code: c._prediction(code, row)[0] for code in design.PREDICTIONS}
    assert got == {"negative_controls_null": "met", "survives_lag": "met", "adherence_robust": "failed",
                   "attenuation_below_half_after_incumbent": "met", "within_person_same_sign": "failed"}
    assert c._prediction("negative_controls_null",
                         {"negative_controls": [{"control": "a", "status": "associated"}]}) == ("failed", "associated: a")
    most = {"incumbent": "inc", "attenuation": 0.7, "same_direction": True, "effect_without_excludes_null": True}
    assert c._prediction("attenuation_below_half_after_incumbent",
                         {"incumbent_adjusted": most}) == ("failed", "inc removes 70% of the effect")
    # A share of an effect indistinguishable from none is noise, not evidence either way.
    none = {"incumbent_adjusted": {**most, "effect_without_excludes_null": False}}
    assert c._prediction("attenuation_below_half_after_incumbent", none)[0] == "not testable"
    assert all(c._prediction(code, {})[0] == "not testable" for code in design.PREDICTIONS)


def _incumbent_data(rng, n=6000, w_effect=0.3):
    """x adds to the incumbent ``inc``; ``dup`` is the incumbent measured again; ``z`` does nothing; ``w`` is an
    exploratory candidate whose effect is set per share."""
    h = pd.DataFrame({"x": rng.normal(size=n), "inc": rng.normal(size=n), "z": rng.normal(size=n),
                      "w": rng.normal(size=n), "age": rng.normal(size=n)})
    h["dup"] = h.inc + 0.3 * rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.4 * h.x + 0.5 * h.inc + w_effect * h.w + 0.2 * h.age))
    censor = rng.exponential(1.5, n)
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    return h


def _registered(feature, low, high):
    """An a-priori hypothesis as register_hypotheses stores it."""
    return {"feature": feature, "direction": 1, "knownness": "unexplored", "construct": "c", "citation": "",
            "incumbent": "inc", "shape": "monotone",
            "magnitude": {"measure": "hr_per_sd", "low": low, "high": high, "source": "analogy"},
            "mechanism_steps": ["one", "two"], "predictions": ["attenuation_below_half_after_incumbent"]}


def test_the_holdout_scores_every_registered_prediction(tmp_path):
    _incumbent_data(np.random.default_rng(41), w_effect=0.6).to_csv(tmp_path / "i.csv", index=False)
    _incumbent_data(np.random.default_rng(42)).to_csv(tmp_path / "i.holdout.csv", index=False)
    state = {"target_column": "y", "confounder_columns": ["age"], "excluded_columns": [],
             "latest_report": {"candidates": [{"feature": "w", "rho": 0.1, "q_value": 0.001, "verdict": "validated",
                                               "components": ["w"]}]},
             "a_priori_hypotheses": [_registered("x", 1.2, 1.8), _registered("dup", 1.2, 1.8),
                                     _registered("z", 1.3, 2.0)]}
    prereg = preregister(state, tmp_path, "i", survival_time="followup_days", discovery_csv=tmp_path / "i.csv")
    frozen = {h["feature"]: h for h in prereg["hypotheses"]}
    assert frozen["x"]["magnitude"]["low"] == 1.2 and frozen["x"]["incumbent"] == "inc"
    assert "magnitude" not in frozen["w"]          # an exploratory candidate predicts nothing of its own
    result = confirm(prereg, tmp_path / "i.holdout.csv", tmp_path)
    rows = {r["feature"]: r for r in result["hypotheses"]}

    def card(feature):
        return {i["prediction"]: i["result"] for i in rows[feature]["scorecard"]["items"]}

    assert card("x") == {"magnitude": "met", "shape: monotone": "met", "attenuation_below_half_after_incumbent": "met"}
    # The incumbent measured again: adding the incumbent removes most of its effect.
    assert card("dup")["attenuation_below_half_after_incumbent"] == "failed"
    assert rows["dup"]["incumbent_adjusted"]["attenuation"] > 0.5
    assert card("z")["magnitude"] == "failed"
    assert card("z")["attenuation_below_half_after_incumbent"] == "not testable"   # z has no effect to attenuate
    assert result["predictions"]["testable"] == 8 and "scorecard" not in rows["w"]
    # The discovery chose w where its effect was larger; the holdout shows how much of it was the search.
    assert 0.2 < rows["w"]["shrinkage"] < 0.8
    md = confirmation_markdown(result)
    assert "Registered predictions on the holdout" in md and "inc removes" in md and "Shrinkage" in md


def test_an_incumbent_already_adjusted_for_is_measured_against_the_effect_without_it(tmp_path):
    _incumbent_data(np.random.default_rng(43)).to_csv(tmp_path / "j.holdout.csv", index=False)
    state = {"target_column": "y", "confounder_columns": ["age", "inc"], "excluded_columns": [],
             "latest_report": {"candidates": []},
             "a_priori_hypotheses": [_registered("x", 1.2, 1.8), _registered("dup", 1.1, 2.0)]}
    result = confirm(preregister(state, tmp_path, "j", survival_time="followup_days"), tmp_path / "j.holdout.csv",
                     tmp_path)
    rows = {r["feature"]: r for r in result["hypotheses"]}
    x, dup = rows["x"]["incumbent_adjusted"], rows["dup"]["incumbent_adjusted"]
    assert x["already_adjusted"] and x["hr_per_sd"] == rows["x"]["hr_per_sd"]  # the main estimate includes it
    assert x["attenuation"] < 0.5 and dup["attenuation"] > 0.5   # not 0, as comparing the main estimate to itself
    results = {r["feature"]: {i["prediction"]: i["result"] for i in r["scorecard"]["items"]} for r in rows.values()}
    assert results["x"]["attenuation_below_half_after_incumbent"] == "met"
    assert results["dup"]["attenuation_below_half_after_incumbent"] == "failed"
