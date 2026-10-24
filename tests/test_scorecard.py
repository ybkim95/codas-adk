"""The scorecard, the recipe coverage and the evidence ladder are read from a run's records, and say what was
not done as plainly as what was; the holdout checks each effect's consistency across strata without showing a
stratum's count."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from codas.aou import cohort, scorecard
from codas.aou import confirm as conf
from codas.aou.cohort import CohortSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"


def _row(**extra):
    """A time-to-event hypothesis that replicated, passed its falsification tests and adds predictive value."""
    row = {"feature": "x_mean", "family": "a_priori", "same_direction": True, "p": 1e-6, "p_one_sided": 5e-7,
           "test": "cox", "hr_per_sd": 1.3, "ci95": [1.2, 1.4], "e_value": 1.9, "imputations": 10,
           "replicated": True, "reproduced": False, "robust": True, "role": "novel",
           "effect_sizes": {"unit": "a doubling", "hr_per_unit": 1.25, "quarter_hr": [1, 1.1, 1.2, 1.5],
                            "cumulative_incidence": [0.02, 0.03, 0.03, 0.05], "risk_difference": 0.03,
                            "e_value_ci": 1.7, "prediction": {"metric": "C", "gain": 0.012, "gain_ci95": [0.004, 0.02],
                                                              "magnitude": "moderate"}},
           "dose_response": {"monotone": True}, "lag_sensitivity": {"p": 0.001, "same_direction": True},
           "negative_controls": [{"control": "a", "status": "no association"}, {"control": "b", "status": "associated"}],
           "adherence_adjusted": {"p": 0.001, "holds": True},
           "incumbent_adjusted": {"incumbent": "bmi_cov", "attenuation": 0.2},
           "heterogeneity": [{"by": "female", "strata": [], "q_p": 0.4, "i2": 0.1}]}
    return {**row, **extra}


CONFIRMATION = {"confounders": ["age", "female"], "negative_controls": ["a", "b"], "adherence_status": "tested: x",
                "replicated": 1, "tested": 2, "robust": 1, "predictions": {"met": 3, "testable": 4}}


def test_the_ladder_needs_each_rung_in_turn():
    assert scorecard.rung(_row()) == 3
    assert scorecard.rung(_row(effect_sizes={"prediction": {"gain": 0.01, "gain_ci95": [-0.001, 0.02]}})) == 2
    assert scorecard.rung(_row(robust=False)) == 1                     # a falsification test failed
    assert scorecard.rung(_row(replicated=False, robust=False)) == 1   # nominal, but not under the ledger's control
    assert scorecard.rung(_row(p_one_sided=0.2)) == 0
    assert scorecard.rung(_row(same_direction=False)) == 0
    tiny = {"prediction": {"gain": 0.002, "gain_ci95": [0.001, 0.003], "magnitude": "negligible"}}
    assert scorecard.rung(_row(effect_sizes=tiny)) == 2                # a real but negligible gain


def test_coverage_says_what_ran_what_did_not_and_what_does_not_apply():
    (record,) = scorecard.coverage({**CONFIRMATION, "hypotheses": [_row()]}, time_to_event=True,
                                   registered=[{"feature": "x_mean", "incumbent": "bmi_cov"}])
    cells = record["cells"]
    assert [key for key, _ in scorecard.STEPS] == list(cells)
    assert cells["adjusted"] == ["yes", "2 confounders, multiply imputed"]
    assert cells["negative_controls"][0] == "yes" and "associated: b" in cells["negative_controls"][1]
    assert cells["prediction"][0] == "yes" and "C gain +0.012" in cells["prediction"][1]
    assert cells["time_varying"][0] == "no"                            # never claimed when it did not run
    assert record["rung"] == 3 and "external cohort" in record["next"]
    # A value outcome: no lag, absolute risk or E-value apply; an exploratory candidate has no incumbent.
    value = _row(family="exploratory", test="partial_spearman", rho=0.2, hr_per_sd=None, effect_sizes={"per_unit": 0.1},
                 lag_sensitivity=None, incumbent_adjusted=None, adherence_adjusted=None,
                 heterogeneity=[{"by": "age", "strata": []}],
                 within_person={"error": "the feature cannot be built from the panel's period summaries"})
    (record,) = scorecard.coverage({**CONFIRMATION, "negative_controls": [], "hypotheses": [value],
                                    "adherence_status": "not counted: already in the adjustment set"},
                                   time_to_event=False)
    cells = record["cells"]
    assert {k: cells[k][0] for k in ("lag", "absolute_risk", "e_value", "incumbent", "negative_controls")} == {
        "lag": "n/a", "absolute_risk": "n/a", "e_value": "n/a", "incumbent": "n/a", "negative_controls": "n/a"}
    assert cells["adherence"] == ["yes", "already in the adjustment set"]
    assert cells["heterogeneity"] == ["no", "strata too small to estimate"] and cells["within_person"][0] == "no"
    # An a-priori hypothesis registered against an incumbent the holdout could not use says so.
    (record,) = scorecard.coverage({**CONFIRMATION, "hypotheses": [_row(incumbent_adjusted=None)]}, True,
                                   registered=[{"feature": "x_mean", "incumbent": "waist_cov"}])
    assert record["cells"]["incumbent"][0] == "no" and "waist_cov" in record["cells"]["incumbent"][1]
    (record,) = scorecard.coverage({**CONFIRMATION, "hypotheses": [_row(incumbent_adjusted=None)]}, True,
                                   registered=[{"feature": "x_mean", "incumbent": "none"}])
    assert record["cells"]["incumbent"][0] == "n/a"
    # Untested hypotheses are not scored; a non-confirmatory one is not called replicated or not.
    assert scorecard.coverage({**CONFIRMATION, "hypotheses": [{"feature": "y", "status": "not testable"}]}, True) == []
    (record,) = scorecard.coverage({**CONFIRMATION, "hypotheses": [_row(role="non_confirmatory", role_reason="repeat")]}, True)
    assert record["cells"]["replication"] == ["n/a", "repeat"]


def test_the_scorecard_reads_the_records_and_renders():
    state = {"confirmation": {**CONFIRMATION, "hypotheses": [_row(), _row(feature="z_sd", robust=False)]},
             "eda": {"features": {"x_mean": {}, "z_sd": {}}, "reliability": {"x_mean": {"reliability": 0.8},
                                                                             "z_sd": {"reliability": 0.3}}},
             "hypothesis_quality": {"design": {"registered": 2, "accepted_first_time": 1, "features_tried": 2,
                                               "novel_by_evidence": 0.5},
                                    "data_understanding": {"claims": 4, "verified": 2, "cited": 1,
                                                           "problems_noticed": 1, "problems_flagged": 2}},
             "citations": {"used": ["PMID:1", "PMID:2"], "verified": ["PMID:1"]}, "literature_status": "available"}
    card = scorecard.scorecard(state, time_to_event=True)
    values = {s["measure"]: s["value"] for s in card["summary"]}
    assert values["features measured reliably over the window (split-half >= 0.5)"] == "1 of 2 (50%)"
    assert values["memo claims the data summary confirms"] == "3 of 4 (75%)"
    assert values["citations among the papers retrieved in this run"] == "1 of 2 (50%)"
    assert values["hypotheses at each rung (0-3)"] == "0: 0, 1: 1, 2: 0, 3: 1"
    assert values["highest rung reached"].startswith("3 (adds predictive value beyond the established markers): x_mean")
    text = scorecard.markdown(card)
    assert "## What the analysis covered" in text and "| x_mean | 3 |" in text and "| z_sd | 1 |" in text
    assert scorecard.markdown({"summary": []}) == ""


def _sealed(rng, n=4000, modified=True):
    female = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    lin = (np.where(female == 1, np.log(1.6), 0.0) if modified else np.log(1.25)) * x
    t_event, t_censor = rng.exponential(1 / (0.06 * np.exp(lin))), rng.uniform(0.5, 3, n)
    return pd.DataFrame({"followup_days": np.ceil(np.minimum(t_event, t_censor) * 365),
                         "dx": (t_event <= t_censor).astype(int), "female": female, "age": rng.normal(50, 10, n),
                         "x": x, "_index_year": rng.choice([2019, 2020, 2021, 2022], n), "rare": 0})


def test_consistency_across_strata_finds_a_modifier_and_shows_no_count():
    frame = _sealed(np.random.default_rng(0))
    frame.loc[frame.index[:150], "rare"] = 1  # a stratum with few events: not estimated
    cov = frame[["age", "female"]]
    out = {h["by"]: h for h in conf._heterogeneity(frame, frame["x"], "dx", "followup_days", cov,
                                                    ["female", "age", "_index_year", "rare", "absent"], True)}
    assert set(out) == {"female", "age", "_index_year", "rare"}
    assert out["female"]["q_p"] < 0.01 and out["female"]["i2"] > 0.5
    women = next(s for s in out["female"]["strata"] if s["stratum"] == "female = 1")
    assert women["hr_per_sd"] > 1.4
    assert "too few to estimate" in [s.get("status") for s in out["rare"]["strata"]] and "q_p" not in out["rare"]
    for entry in out.values():  # effects only: a stratum's count with the total would give the other's
        assert all("n" not in s and "events" not in s for s in entry["strata"])
    calm = _sealed(np.random.default_rng(1), modified=False)  # the same effect in both strata
    (steady,) = conf._heterogeneity(calm, calm["x"], "dx", "followup_days", None, ["female"], True)
    assert steady["q_p"] > 0.01


def test_the_era_reaches_the_sealed_table_only_and_strata_are_registered():
    spec = CohortSpec.from_json(SPEC)
    assert cohort.ERA_COLUMN in cohort.holdout_columns(spec) and cohort.ERA_COLUMN not in cohort.output_columns(spec)
    sql, _ = cohort.build_cohort_sql(spec, "p.d")
    assert f"AS {cohort.ERA_COLUMN}" in sql
    person, _ = cohort.build_cohort_sql(replace(spec, design="incident"), "p.d", person_level=True)
    assert cohort.ERA_COLUMN not in person
    columns = cohort.heterogeneity_columns(spec)
    assert columns[:2] == ["female", "age"] and "wear_study" in columns and columns[-1] == cohort.ERA_COLUMN
    assert cohort.ERA_COLUMN not in cohort.heterogeneity_columns(replace(spec, holdout_fraction=0.0))
