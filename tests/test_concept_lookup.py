"""Concepts named by public code (ATC, SNOMED, LOINC) or by survey question text are resolved against the
CDR's own vocabulary before a run, never typed in as ids; what cannot be resolved unambiguously is
skipped with the reason. Survey answers become a 1/0 flag with skips and refusals left missing."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from codas.aou import cohort, feasibility as feas  # noqa: E402
from codas.aou.cohort import CohortSpec, CovariateSpec, OutcomeSpec, build_cohort_sql  # noqa: E402
from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query  # noqa: E402

from test_aou_sql import _window_spec  # noqa: E402


@pytest.fixture
def cdr(monkeypatch):
    con = duckdb.connect()
    build_synthetic_cdr(con)
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    return con


def test_catalog_codes_resolve_to_the_cdrs_concept_ids(cdr):
    known = feas.CatalogOutcome(name="t2d_code", label="T2D", kind="condition", concept_ids=[],
                                codes=["SNOMED:44054006"], expect_name="type 2 diabetes")
    unknown = feas.CatalogOutcome(name="nope", label="Nope", kind="condition", concept_ids=[],
                                  codes=["SNOMED:000000"], expect_name="x")
    resolved = feas.resolve_codes([known, unknown], SYNTHETIC_CDR, None)
    assert resolved[0].concept_ids == [201826] and resolved[1].concept_ids == []
    problems = feas.verify_catalog(resolved, SYNTHETIC_CDR, None)
    assert problems["t2d_code"] is None and "SNOMED:000000" in problems["nope"]


def test_a_drug_class_code_covers_its_ingredients(cdr):
    base = replace(_window_spec(), covariates=[
        CovariateSpec(name="glucose_lowering", kind="drug", concept_ids=[], codes=["ATC:A10"], lookback_days=None)])
    spec, failed = feas.verify_covariates(base, SYNTHETIC_CDR, None)
    assert failed == {} and spec.covariates[0].concept_ids == [21600712]
    df = duckdb_query(cdr)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    assert set(df["glucose_lowering"].unique()) <= {0, 1} and df["glucose_lowering"].sum() > 10  # metformin users


def test_survey_answers_become_a_flag_with_skips_missing(cdr):
    survey = CovariateSpec(name="smoking_ever", kind="survey", concept_ids=[], question_name="100 cigs lifetime",
                           positive_answers="yes", lookback_days=None, any_time=True)
    spec, failed = feas.verify_covariates(replace(_window_spec(), covariates=[survey]), SYNTHETIC_CDR, None)
    assert failed == {} and spec.covariates[0].concept_ids == [1585857]
    answers = feas.survey_answers(spec.covariates[0], SYNTHETIC_CDR, None)
    assert {"100 Cigs Lifetime: Yes", "100 Cigs Lifetime: No", "PMI: Skip"} <= set(answers["answer"])
    df = duckdb_query(cdr)(None, *build_cohort_sql(spec, SYNTHETIC_CDR))
    obs = cdr.execute("SELECT person_id, value_source_concept_id AS a, observation_date FROM observation").df()
    assert set(df["smoking_ever"].dropna().unique()) <= {0.0, 1.0}
    assert df["smoking_ever"].isna().any() and (df["smoking_ever"] == 1).any() and (df["smoking_ever"] == 0).any()
    assert len(obs) > 0


def test_ambiguous_or_unknown_survey_questions_are_skipped_with_the_reason(cdr):
    vague = CovariateSpec(name="vague", kind="survey", concept_ids=[], question_name="smoking", positive_answers="yes")
    absent = CovariateSpec(name="absent", kind="survey", concept_ids=[], question_name="no such question",
                           positive_answers="yes")
    _spec, failed = feas.verify_covariates(replace(_window_spec(), covariates=[vague, absent]), SYNTHETIC_CDR, None)
    assert "2 questions" in failed["vague"] and "Smoking Frequency" in failed["vague"]
    assert "no survey question" in failed["absent"]
