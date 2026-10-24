"""The replication package: what an independent cohort needs to run the same test, and nothing about any
participant."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from codas.aou import replication
from codas.aou.cohort import CohortSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"


def _spec():
    spec = CohortSpec.from_json(SPEC)
    return replace(spec, name="explore_t2d", design="incident",
                   outcome=replace(spec.outcome, kind="condition", name="t2d", codes=["SNOMED:44054006"],
                                   concept_ids=[201826], min_occurrences=2, value_min=None, value_max=None))


PREREG = {"sha256": "abc123", "survival_time": "followup_days", "confounders": ["age", "female", "bmi_cov"],
          "covariate_imputations": 10, "sensitivity_lag_days": 365, "negative_controls": ["impacted_cerumen"],
          "availability": ["act_days"], "heterogeneity": ["female", "age", "_index_year"],
          "absolute_risk_horizon_days": 1095,
          "hypotheses": [{"feature": "act_steps_mean", "direction": -1, "family": "a_priori",
                          "magnitude": {"measure": "hr_per_sd", "low": 0.7, "high": 0.95},
                          "transform": {"scale": 2400.5, "log": False}, "effect_scale": {"unit": 1000}},
                         {"feature": "dsl_steps_spread_1a2b3c4d", "direction": 1, "family": "a_priori"}]}
RESULT = {"hypotheses": [{"feature": "act_steps_mean", "same_direction": True, "hr_per_sd": 0.82, "ci95": [0.75, 0.9],
                          "status": "reproduced", "robust": True, "events": 12}]}
STATE = {"defined_phenotypes": [{"column": "dsl_steps_spread_1a2b3c4d", "description": "the SD of steps_day over days",
                                 "definition": {"name": "steps_spread", "signal": "steps_day", "summary": "sd"}}],
         "scorecard": {"hypotheses": [{"feature": "act_steps_mean", "rung": 2}]}}


def test_the_package_holds_definitions_plan_and_protocol_but_no_participant_data(tmp_path):
    folder = replication.write_package(tmp_path, _spec(), "Type 2 diabetes", PREREG, RESULT, STATE,
                                       code_version="0123456789ab", cdr="wb-project-123.R2025Q4R6")
    names = {p.name for p in folder.iterdir()}
    assert {"extraction.sql", "phenotypes.sql", "cohort_spec.json", "features.json", "pre_registration.json",
            "holdout_results.json", "scorecard.json", "feature_definition_hashes.json", "protocol.md",
            "MANIFEST.json"} <= names
    sql = (folder / "extraction.sql").read_text()
    assert replication.PLACEHOLDER_CDR in sql and "wb-project-123" not in sql and "-- @outcome_concept_ids" in sql
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    assert manifest["cdr_release"] == "R2025Q4R6" and manifest["code_version"] == "0123456789ab"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_text().encode()).hexdigest() == digest
    features = json.loads((folder / "features.json").read_text())
    assert features["dsl_steps_spread_1a2b3c4d"]["phenotype"]["summary"] == "sd"
    assert "mean of daily steps" in features["act_steps_mean"]["definition"]
    protocol = (folder / "protocol.md").read_text()
    for text in ("SNOMED:44054006", "2 or more distinct dates", "## Success criteria", "Benjamini-Hochberg",
                 "the first 365 days of follow-up left out", "impacted_cerumen", "higher is protective",
                 "the SD of steps_day over days", "neither share", "hazard ratio per SD"):
        assert text in protocol, text
    # A count of 1-20 never appears unmasked; nothing in the package identifies a person.
    results = json.loads((folder / "holdout_results.json").read_text())
    assert results["hypotheses"][0]["events"] != 12
    every = "".join(p.read_text() for p in folder.iterdir())
    assert not re.search(r"\b\d{12}\b", every)
