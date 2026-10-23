"""All of Us integration: the privacy guard, the backend check, and the cohort extractor (offline)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from codas.agents.callbacks import after_tool_logger, report_grounding_audit
from codas.agents.tools import preview_columns
from codas.aou import cohort
from codas.aou.cohort import CohortSpec, build_cohort_sql, missing_columns, output_columns
from codas.aou.privacy import (
    MASK,
    AouBackendError,
    check_llm_backend,
    mask_small_counts,
    mask_small_counts_in_text,
    sanitize_tool_response,
)
from codas.core.discovery import DiscoveryRequest, run_discovery

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_dataset.csv"
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"

def _window_spec() -> CohortSpec:
    """The example spec with the outcome-anchored design these tests characterise (no split, no lag)."""
    from dataclasses import replace

    spec = CohortSpec.from_json(SPEC)
    # Covariates named by code or survey text are resolved against a CDR first; these tests build SQL
    # straight from the spec, so they keep the ones given by concept id.
    return replace(spec, design="window", lag_days=0, horizon_days=None, valid_day=None,
                   holdout_fraction=0.0, min_wear_days=30, negative_controls=[],
                   covariates=[c for c in spec.covariates if c.concept_ids])

CDR = "fc-aou-cdr-prod.R2025Q4R6"


class _Tool:
    name = "profile_dataset"


@pytest.fixture
def aou(monkeypatch):
    monkeypatch.setenv("CODAS_AOU_MODE", "1")


# --- small-cell masking ---

def test_counts_1_to_20_are_masked_and_others_kept():
    out = mask_small_counts({"n": 12, "rows": 20, "n_cases": 1, "screen_effective_n": 7,
                             "count": 0, "n_participants": 21, "rho": 0.31})
    assert out == {"n": MASK, "rows": MASK, "n_cases": MASK, "screen_effective_n": MASK,
                   "count": 0, "n_participants": 21, "rho": 0.31}


def test_structural_counts_are_not_masked():
    out = mask_small_counts({"validated_count": 3, "n_features": 5, "warning_count": 2,
                             "screen_n_clusters": 4, "feature_count": 9})
    assert out == {"validated_count": 3, "n_features": 5, "warning_count": 2,
                   "screen_n_clusters": 4, "feature_count": 9}


def test_masking_recurses_into_candidates():
    out = mask_small_counts({"candidates": [{"feature": "steps", "n": 15}, {"feature": "rem", "n": 900}]})
    assert [c["n"] for c in out["candidates"]] == [MASK, 900]


def test_row_level_payloads_are_dropped():
    out = sanitize_tool_response({"columns": ["a"], "preview_rows": [{"a": 1}], "rows": 5})
    assert "preview_rows" not in out and out["rows"] == MASK


@pytest.mark.parametrize("text,expected", [
    ("n = 12 had data", f"n = {MASK} had data"),
    ("N=7,497 participants", "N=7,497 participants"),
    ("only 3 participants and 19 cases", f"only {MASK} participants and {MASK} cases"),
    ("rho = 0.12 across 25 people", "rho = 0.12 across 25 people"),
    ("q = 0.02, 2 features", "q = 0.02, 2 features"),
    ("in 2024 patients", "in 2024 patients"),
])
def test_prose_counts_are_masked(text, expected):
    assert mask_small_counts_in_text(text) == expected


# --- agent wiring ---

def test_after_tool_is_passthrough_outside_aou(monkeypatch):
    monkeypatch.delenv("CODAS_AOU_MODE", raising=False)
    ctx = SimpleNamespace(state={})
    assert after_tool_logger(_Tool(), {}, ctx, {"rows": 5, "preview_rows": [{}]}) is None


def test_after_tool_sanitizes_response_and_memory(aou):
    ctx = SimpleNamespace(state={"fact_sheet": {"rows": 14}, "rounds": [{"validated_count": 2}]})
    out = after_tool_logger(_Tool(), {}, ctx, {"rows": 14, "preview_rows": [{"x": 1}]})
    assert out == {"rows": MASK}
    assert ctx.state["fact_sheet"]["rows"] == MASK
    assert ctx.state["rounds"][0]["validated_count"] == 2  # convergence arithmetic still works


def test_preview_withholds_rows_in_aou(aou):
    out = preview_columns(str(SAMPLE))
    assert out["columns"] and "preview_rows" not in out


def test_report_counts_masked_in_aou(aou):
    ctx = SimpleNamespace(state={"report": "Among n = 9 participants with REM data...", "fact_sheet": {}})
    report_grounding_audit(ctx)
    assert "n = 9" not in ctx.state["report"] and MASK in ctx.state["report"]


# --- backend check ---

def test_backend_check_requires_vertex(aou, monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    with pytest.raises(AouBackendError):
        check_llm_backend()


def test_backend_check_accepts_workspace_vertex(aou, monkeypatch):
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "terra-vpc-sc-1234")
    check_llm_backend()


def test_backend_check_is_noop_outside_aou(monkeypatch):
    monkeypatch.delenv("CODAS_AOU_MODE", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    check_llm_backend()


# --- cohort extractor ---

def test_example_spec_loads():
    spec = _window_spec()
    assert spec.outcome.kind == "measurement" and len(spec.wearables) == 4
    assert [w.optional for w in spec.wearables] == [False, False, True, True]
    assert [p.kind for p in spec.phenotypes] == ["minute_activity", "sleep_stages", "minute_heart_rate"]
    assert spec.phenotypes[2].steps_from == spec.phenotypes[0].prefix
    full = CohortSpec.from_json(SPEC).covariates
    assert {"measurement", "drug", "survey"} <= {c.kind for c in full}
    assert all(c.concept_ids or c.codes or c.question_name for c in full)  # resolved against the CDR before a run


def test_measurement_sql_binds_values_and_omits_identifiers():
    sql, params = build_cohort_sql(_window_spec(), CDR)
    assert f"`{CDR}.measurement`" in sql and f"`{CDR}.sleep_daily_summary`" in sql
    assert "concept_ancestor" in sql and "ROW_NUMBER()" in sql
    names = {n: v for n, _, v in params}
    assert names["outcome_concept_ids"] == [3004410] and names["w1_filter"] is True
    select = sql.split("\nSELECT\n", 1)[1].split("\nFROM index_events")[0]
    assert "person_id" not in select and "index_date AS" not in select  # no identifiers or dates out


def test_condition_sql_labels_cases_and_controls():
    spec = CohortSpec.from_dict({
        "name": "t2d", "outcome": {"kind": "condition", "name": "t2d", "concept_ids": [201826]},
        "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act",
                       "columns": ["steps"]}],
    })
    sql, _ = build_cohort_sql(spec, CDR)
    assert "condition_occurrence" in sql and "IF(c.person_id IS NULL, 0, 1)" in sql


@pytest.mark.parametrize("bad", [
    {"table": "activity_summary; DROP TABLE x"},
    {"columns": ["steps`"]},
    {"prefix": "act-1"},
])
def test_spec_rejects_unsafe_identifiers(bad):
    wearable = {"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]} | bad
    with pytest.raises(ValueError):
        CohortSpec.from_dict({"name": "x", "outcome": {"kind": "measurement", "concept_ids": [1]},
                              "wearables": [wearable]})


def test_cdr_must_be_project_dataset(monkeypatch):
    monkeypatch.delenv("CODAS_AOU_CDR", raising=False)
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    with pytest.raises(ValueError):
        cohort.resolve_cdr("not a dataset")
    with pytest.raises(ValueError):
        cohort.resolve_cdr(None)
    monkeypatch.setenv("WORKSPACE_CDR", CDR)
    assert cohort.resolve_cdr(None) == CDR


def test_missing_columns_reports_schema_gaps():
    spec = _window_spec()
    present = [("activity_summary", c) for c in ["person_id", "date", *spec.wearables[0].columns]]
    missing = missing_columns(spec, present)
    assert "sleep_daily_summary.minute_asleep" in missing and not any(m.startswith("activity") for m in missing)


def _synthetic_cohort(spec: CohortSpec, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in output_columns(spec)})
    df["age"] = rng.integers(20, 80, n)
    df["female"] = rng.integers(0, 2, n)
    df["act_days"] = rng.integers(30, 180, n)
    df["hba1c"] = 5.6 - 0.4 * df["act_steps_mean"] + 0.01 * df["age"] + rng.normal(0, 0.5, n)
    return df


def test_extract_writes_table_under_codas_runs(monkeypatch, tmp_path):
    spec = _window_spec()
    table = _synthetic_cohort(spec)
    schema = pd.DataFrame(
        [(s.table, c) for s in spec.wearables
         for c in ["person_id", s.date_column, *s.columns, *([s.filter_column] if s.filter_column else [])]]
        + [("measurement", c) for c in ("person_id", "measurement_concept_id", "measurement_date",
                                        "value_as_number", "measurement_id")],
        columns=["table_name", "column_name"])
    calls = []

    def fake_query(client, sql, params):
        calls.append(sql)
        if "occasions AS" in sql:  # the repeated-measures panel
            return pd.DataFrame({c: [1.0, 2.0] for c in cohort.panel_columns(spec)})
        return schema if "INFORMATION_SCHEMA" in sql else table

    monkeypatch.setattr(cohort, "_query", fake_query)
    out_dir = ROOT / ".codas_runs" / "aou_test"
    path = cohort.extract_cohort(spec, CDR, out_dir, client=object())
    try:
        written = pd.read_csv(path)
        fitted, _notes = cohort.resolve_sources(spec, list(schema.itertuples(index=False, name=None)))
        assert not fitted.phenotypes                       # intraday tables absent from this schema: skipped
        assert list(written.columns) == output_columns(fitted) and len(calls) == 3  # schema, cohort, panel
        assert list(pd.read_csv(cohort.panel_path(path)).columns) == cohort.panel_columns(spec)
    finally:
        path.unlink()
        cohort.panel_path(path).unlink(missing_ok=True)
        out_dir.rmdir()
    with pytest.raises(ValueError):
        cohort.extract_cohort(spec, CDR, tmp_path, client=object())


def test_engine_runs_on_extractor_shaped_table():
    spec = _window_spec()
    report = run_discovery(_synthetic_cohort(spec), DiscoveryRequest(
        target_column="hba1c", confounder_columns=["age", "female"], validation_resamples=50))
    assert any(c.feature.startswith("act_steps_mean") for c in report.candidates)


def test_bigquery_jobs_record_and_print_the_bytes_billed(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    from codas.aou import runner

    class Job:
        total_bytes_billed = 2 * 10**12

        def result(self):
            return SimpleNamespace(to_dataframe=lambda: pd.DataFrame({"a": [1]}))

    monkeypatch.setattr(cohort, "_job_config", lambda params: None)
    start = len(cohort.QUERY_LOG)
    cohort._query(SimpleNamespace(query=lambda sql, job_config=None: Job()), "SELECT 1", [])
    assert cohort.QUERY_LOG[start]["bytes_billed"] == 2 * 10**12
    assert "2.00 TB billed" in capsys.readouterr().out
    out = tmp_path / "s"
    out.mkdir()
    path = runner._write_summary(out, "table", [], billed=cohort.QUERY_LOG[start:])
    assert "BigQuery in this run: 2.00 TB billed" in path.read_text()
