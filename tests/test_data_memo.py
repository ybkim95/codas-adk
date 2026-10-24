"""The data analyst's memo is checked claim by claim against the outcome-blind data summary, objective problems
in that summary refuse a hypothesis, an independent review is recorded, and the design's quality is measured."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.agents import tools
from codas.core import eda
from codas.core import hypotheses as design

DOSSIER = {
    "participants": 4000,
    "features": {"steps": {"n": 3900, "missing": 0.025, "mean": 8123.4, "sd": 3210.0, "skew": 2.31, "beyond_4sd": 0.001,
                           "quantiles": {"p1": 900.0, "p99": 19000.0}, "near_constant": False, "natural_unit": 4000.0},
                 "flat": {"n": 3900, "missing": 0.0, "mean": 1.0, "sd": 0.01, "skew": None, "near_constant": True},
                 "hr": {"n": 3000, "missing": 0.25, "mean": 64.2, "sd": 8.1, "skew": 0.4, "near_constant": False}},
    "families": [{"representative": "steps", "members": ["steps", "distance"], "min_abs_rho": 0.95, "max_abs_rho": 0.95}],
    "wear_dependence": {"steps": {"max_abs_rho": 0.62, "with": "days"}, "hr": {"max_abs_rho": 0.1, "with": "days"}},
    "missingness": {"hr": {"missing": 0.25, "smd_age": 0.31, "smd_female": 0.02}},
    "reliability": {"hr": {"reliability": 0.42}},
    "outcome": {"kind": "time to event", "overall": {"participants": 4000, "events": 312}},
}


def test_a_claim_is_verified_only_when_its_numbers_are_in_the_cited_entries():
    assert eda.check_claim(DOSSIER, "steps is right-skewed (skew 2.3)", ["features.steps.skew"]) == ("verified", "")
    assert eda.check_claim(DOSSIER, "a mean of 8,100 steps and 2.5% missing", ["features.steps"])[0] == "verified"
    assert eda.check_claim(DOSSIER, "312 events among 4000 people", ["outcome.overall"])[0] == "verified"
    assert eda.check_claim(DOSSIER, "it tracks wear at |rho| 0.62", ["wear_dependence.steps"])[0] == "verified"
    # A family named by any member; the count of a cited list counts.
    assert eda.check_claim(DOSSIER, "distance duplicates steps", ["families.distance"]) == ("cited", "")
    assert eda.check_claim(DOSSIER, "1 family", ["families"])[0] == "verified"
    status, detail = eda.check_claim(DOSSIER, "steps has skew 3.9", ["features.steps.skew"])
    assert status == "wrong value" and "3.9" in detail
    # A threshold written as a digit is not a value of the entry; written in words it is fine. (A rounding of
    # the value is: "skew 2" states 2.31 to its precision.)
    assert eda.check_claim(DOSSIER, "skew 2.31 is above 1.5", ["features.steps.skew"])[0] == "wrong value"
    assert eda.check_claim(DOSSIER, "skew 2.31 is above one and a half", ["features.steps.skew"])[0] == "verified"
    assert eda.check_claim(DOSSIER, "skew about 2", ["features.steps.skew"])[0] == "verified"
    assert eda.check_claim(DOSSIER, "hr10h and p99 have no number", ["features.hr"]) == ("cited", "")
    assert eda.check_claim(DOSSIER, "x", ["features.nowhere.skew"])[0] == "unknown field"
    assert eda.check_claim(DOSSIER, "x", [])[0] == "unknown field"


def test_the_memo_audit_counts_the_problems_it_noticed():
    found = {(i["problem"], i["subject"]) for i in eda.issues(DOSSIER)}
    assert found == {("near-constant", "flat"), ("strongly skewed", "steps"), ("one construct under several names", "steps"),
                     ("tracks wear", "steps"), ("missing not at random", "hr"), ("unreliable over the window", "hr")}
    memo = {"summary": "Four thousand people; the outcome has enough events.",
            "claims": [{"statement": "steps is skewed (skew 2.31)", "fields": ["features.steps.skew"],
                        "implication": "use a log scale"},
                       {"statement": "distance and steps are one construct", "fields": ["families.distance"]},
                       {"statement": "hr is reliable (0.9)", "fields": ["reliability.hr"]},   # wrong: it is 0.42
                       {"statement": "about bmi", "fields": ["covariates.bmi"]}]}            # not in the summary
    audit = eda.audit_memo(DOSSIER, memo)
    assert audit["checked"] == {"verified": 1, "cited": 1, "wrong value": 1, "unknown field": 1}
    assert audit["issues"] == 6 and audit["noticed"] == 2   # the skew and the family; a wrong claim notices nothing
    assert "unreliable over the window: hr" in audit["missed"]
    text = eda.memo_text(audit)
    assert "use a log scale" in text and "reliable (0.9)" not in text and "bmi" not in text


def _context(tmp_path, **state):
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    frame = pd.DataFrame({"y": rng.normal(size=n), "a": rng.normal(size=n), "twin": base, "inc": base + 0.01 * rng.normal(size=n),
                          "flat": np.r_[np.zeros(n - 5), np.ones(5)], "worn": rng.normal(size=n), "age": rng.normal(size=n)})
    frame.to_csv(tmp_path / "c.csv", index=False)
    return SimpleNamespace(state={"csv_path": str(tmp_path / "c.csv"), "target_column": "y", "context_columns": ["age"],
                                  "testable_predictions": list(design.PREDICTIONS), **state})


def _complete(feature, incumbent="none", predictions=("negative_controls_null",)):
    return {"feature": feature, "direction": "positive", "construct": "c", "incumbent": incumbent, "shape": "monotone",
            "magnitude": {"measure": "rho", "low": 0.05, "high": 0.3, "source": "analogy"},
            "mechanism": ["one", "two"], "predictions": list(predictions), "knownness": "emerging"}


def test_the_data_summary_refuses_what_cannot_be_tested_well(tmp_path):
    dossier = {"features": {"flat": {"near_constant": True}, "a": {}, "twin": {}, "worn": {}},
               "reliability": {"a": {"reliability": 0.3}},
               "wear_dependence": {"worn": {"max_abs_rho": 0.55, "with": "days"}}}
    ctx = _context(tmp_path, eda=dossier)
    out = tools.register_hypotheses(json.dumps([
        _complete("flat"), _complete("a"), _complete("twin", incumbent="inc"), _complete("worn"),
        _complete("y_free_feature_that_does_not_exist")]), ctx)
    reasons = {r["feature"]: " ".join(r["reasons"]) for r in out["rejected"]}
    assert "nearly constant" in reasons["flat"]
    assert "split-half reliability is 0.30" in reasons["a"]
    assert "incumbent measured again" in reasons["twin"]
    assert "tracks how long the device was worn" in reasons["worn"]
    attempts = {a["feature"]: a["parts"] for a in ctx.state["hypothesis_attempts"]}
    assert attempts["flat"] == ["data"] and attempts["y_free_feature_that_does_not_exist"] == ["feature"]
    # Named, the wear alternative brings its test, and the same hypothesis is registered.
    again = tools.register_hypotheses(json.dumps([{**_complete("worn"), "alternatives": ["wear"]}]), ctx)
    assert [h["feature"] for h in again["registered"]] == ["worn"]
    assert again["registered"][0]["predictions"] == ["negative_controls_null", "adherence_robust"]


def test_the_memo_and_the_review_tools_record_what_they_check(tmp_path):
    ctx = _context(tmp_path, eda=DOSSIER)
    out = tools.record_data_memo(json.dumps({"summary": "s", "claims": [
        {"statement": "skew 2.31", "fields": ["features.steps.skew"]},
        {"statement": "skew 7", "fields": ["features.steps.skew"]}]}), ctx)
    assert out["checked"] == {"verified": 1, "wrong value": 1} and len(out["to_correct"]) == 1
    assert ctx.state["data_memo"] == "s\n- skew 2.31 [features.steps.skew]"
    assert tools.record_data_memo("not json", ctx)["error"].startswith("memo_json is not valid JSON")
    assert ctx.state["data_memo_audit"]["revisions"] == 0
    tools.record_data_memo(json.dumps({"summary": "s2", "claims": []}), ctx)
    assert ctx.state["data_memo_audit"]["revisions"] == 1 and ctx.state["data_memo"] == "s2"

    assert "nothing to review" in tools.record_hypothesis_review("[]", ctx)["error"]
    ctx.state["a_priori_hypotheses"] = [{"feature": "a"}, {"feature": "worn"}]
    scores = {k: 4 for k in design.RUBRIC}
    bad = tools.record_hypothesis_review(json.dumps([{"feature": "a", **scores, "novelty": 6}]), ctx)
    assert "novelty must be a whole number from 1 to 5" in bad["error"]
    partial = tools.record_hypothesis_review(json.dumps([{"feature": "a", **scores}]), ctx)
    assert "review every registered hypothesis: worn" in partial["error"]
    ok = tools.record_hypothesis_review(json.dumps([{"feature": "a", **scores, "comment": "weak magnitude"},
                                                    {"feature": "worn", "scores": scores}]), ctx)
    assert ok == {"recorded": 2} and ctx.state["hypothesis_review"][0]["comment"] == "weak magnitude"


def test_the_quality_of_the_design_is_measured_not_asserted():
    attempts = [{"feature": "a", "accepted": False, "parts": ["magnitude", "mechanism"]},
                {"feature": "a", "accepted": True, "parts": []},
                {"feature": "b", "accepted": True, "parts": []},
                {"feature": "c", "accepted": False, "parts": ["data"]}]
    registered = [
        {"feature": "a", "construct": "Volume", "incumbent": "x", "shape": "monotone", "knownness": "emerging",
         "magnitude": {"source": "[PMID:1]"}, "citations": ["PMID:1"], "mechanism_steps": ["1", "2", "3"],
         "predictions": ["p", "q"]},
        {"feature": "b", "construct": "volume", "incumbent": "", "shape": "threshold", "knownness": "not assessed",
         "magnitude": {"source": "analogy"}, "citations": [], "mechanism_steps": ["1", "2"], "predictions": ["p"]}]
    review = [{"feature": "a", "scores": {k: 4 for k in design.RUBRIC}},
              {"feature": "b", "scores": {k: 2 for k in design.RUBRIC}}]
    memo = {"checked": {"verified": 3, "cited": 1, "wrong value": 1}, "issues": 6, "noticed": 4}
    confirmation = {"hypotheses": [
        {"feature": "a", "family": "a_priori", "same_direction": True, "replicated": True,
         "scorecard": {"items": [{"prediction": "magnitude", "result": "met"}, {"prediction": "shape: monotone", "result": "failed"},
                                 {"prediction": "survives_lag", "result": "not testable"}]}},
        {"feature": "b", "family": "a_priori", "same_direction": False, "replicated": False},
        {"feature": "w", "family": "exploratory", "same_direction": True, "replicated": True}]}
    q = design.quality(attempts, registered, review=review, memo=memo, confirmation=confirmation, proposed={"b"})
    d = q["design"]
    assert (d["submitted"], d["features_tried"], d["registered"], d["accepted_first_time"]) == (4, 3, 2, 1)
    assert d["refused_parts"] == {"magnitude": 1, "mechanism": 1, "data": 1}
    assert (d["with_incumbent"], d["magnitude_from_a_paper"], d["citing_retrieved_papers"]) == (0.5, 0.5, 0.5)
    assert (d["novel_by_evidence"], d["novelty_not_assessed"], d["constructs"]) == (0.5, 0.5, 1)
    assert (d["mean_mechanism_steps"], d["mean_predictions"], d["on_a_constructed_feature"]) == (2.5, 1.5, 0.5)
    assert q["data_understanding"]["confirmed_share"] == 0.8 and q["data_understanding"]["problems_noticed"] == 4
    assert q["review"]["novelty"] == 3.0 and q["review"]["hypotheses"] == 2
    h = q["holdout"]
    assert (h["a_priori_tested"], h["replicated"], h["predictions_met"], h["predictions_testable"]) == (2, 1, 1, 2)
    assert h["by_prediction"] == {"magnitude": {"met": 1, "testable": 1}, "shape": {"met": 0, "testable": 1}}
    text = design.quality_markdown(q)
    assert "## Hypothesis quality" in text and "2 hypotheses registered from 4 submissions" in text
    assert "1 of 2 a-priori hypotheses replicated" in text and "4 of 6 problems" in text
