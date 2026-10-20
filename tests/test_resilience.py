"""Run resilience: a model call that keeps hitting a rate limit falls back to the next configured model,
and a free exploration survives a failing outcome and resumes without redoing finished ones.

Offline: fake models and the synthetic DuckDB CDR; no model is called.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import AsyncGenerator

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from google.genai.errors import ClientError

from codas.agents.agent import FallbackLlm
from codas.aou import runner
from codas.aou.cohort import CohortSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"

def _window_spec() -> CohortSpec:
    """The example spec with the outcome-anchored design these tests characterise (no split, no lag)."""
    from dataclasses import replace

    return replace(CohortSpec.from_json(SPEC), design="window", lag_days=0, horizon_days=None, valid_day=None,
                   holdout_fraction=0.0, min_wear_days=30, negative_controls=[])

CATALOG = ROOT / "scripts" / "aou" / "outcomes_catalog.json"
RUN_DIR = ROOT / ".codas_runs" / "resilience_test"


def _exhausted() -> ClientError:
    return ClientError(429, {"error": {"code": 429, "message": "Resource exhausted.", "status": "RESOURCE_EXHAUSTED"}})


SEEN: list[str] = []  # model names each fake received, in order


class _Fake(BaseLlm):
    fail_with: int = 0           # status code to raise, 0 = answer
    fail_after_first: bool = False

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        SEEN.append(llm_request.model)
        if self.fail_with and not self.fail_after_first:
            raise ClientError(self.fail_with, {"error": {"code": self.fail_with, "message": "x"}})
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"from {self.model}")]))
        if self.fail_after_first:
            raise _exhausted()


async def _collect(llm: BaseLlm) -> list[str]:
    request = LlmRequest(model=llm.model, contents=[types.Content(role="user", parts=[types.Part(text="hi")])])
    return [r.content.parts[0].text async for r in llm.generate_content_async(request)]


def test_rate_limited_model_falls_back_to_the_next_one():
    import asyncio

    SEEN.clear()
    llm = FallbackLlm(model="primary", models=[_Fake(model="primary", fail_with=429), _Fake(model="backup")])
    assert asyncio.run(_collect(llm)) == ["from backup"]
    assert SEEN == ["primary", "backup"]   # the backup received the request under its own model name


def test_errors_that_a_fallback_cannot_fix_are_raised():
    import asyncio

    bad_request = FallbackLlm(model="p", models=[_Fake(model="p", fail_with=400), _Fake(model="b")])
    with pytest.raises(ClientError):
        asyncio.run(_collect(bad_request))
    # Once part of an answer has been streamed, switching models would splice two answers together.
    midway = FallbackLlm(model="p", models=[_Fake(model="p", fail_after_first=True), _Fake(model="b")])
    with pytest.raises(ClientError):
        asyncio.run(_collect(midway))
    # When every model is exhausted, the last error surfaces.
    all_out = FallbackLlm(model="p", models=[_Fake(model="p", fail_with=429), _Fake(model="b", fail_with=429)])
    with pytest.raises(ClientError):
        asyncio.run(_collect(all_out))


def _fake_state() -> dict:
    return {"report": "report text", "fact_sheet": {"rows": 100},
            "rounds": [{"validated_count": 1, "ml_metric_name": "r2", "ml_metric_value": 0.1,
                        "passing_candidates": []}]}


@pytest.fixture
def synthetic_explore(monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    from codas.aou import cohort
    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    calls: list[str] = []
    failing: set[str] = set()

    def fake_run_agents(csv, goal, on_tool_call=None, context_columns=None, recorder=None, **_):
        name = Path(csv).stem
        calls.append(name)
        if name in failing:
            raise _exhausted()
        return _fake_state()

    monkeypatch.setattr(runner, "run_agents", fake_run_agents)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    def explore(**kw):
        return runner.explore(CATALOG, _window_spec(), SYNTHETIC_CDR, client=object(), top_k=1,
                              min_measurement_n=100, min_cases=30, out_dir=RUN_DIR, log=lambda *_: None, **kw)

    yield explore, calls, failing
    shutil.rmtree(RUN_DIR, ignore_errors=True)


def test_a_failing_outcome_does_not_stop_the_others(synthetic_explore):
    explore, calls, failing = synthetic_explore
    failing.add("explore_hdl")
    result = explore()
    assert calls == ["explore_hdl", "explore_depression"]
    failed, done = result["runs"]
    assert failed["cohort"] == "explore_hdl" and "ClientError" in failed["failed"]
    assert Path(done["report_path"]).exists()
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "explore_hdl" in summary and "failed" in summary and "explore_depression" in summary


def test_rerun_resumes_and_redoes_only_what_is_missing_or_stale(synthetic_explore, monkeypatch):
    explore, calls, failing = synthetic_explore
    failing.add("explore_hdl")
    explore()
    failing.clear()
    calls.clear()
    explore()                                   # same code: only the failed outcome runs again
    assert calls == ["explore_hdl"]
    calls.clear()
    explore()                                   # everything finished: nothing runs
    assert calls == []
    monkeypatch.setattr(runner, "_code_version", lambda: "a-later-commit")
    explore()                                   # a commit that leaves the analysis code alone: reused
    assert calls == []
    monkeypatch.setattr(runner, "_analysis_version", lambda root=None: "changed-analysis")
    explore()                                   # the analysis code changed: results are stale, both run again
    assert calls == ["explore_hdl", "explore_depression"]
    calls.clear()
    explore(rerun=True)                         # forced
    assert calls == ["explore_hdl", "explore_depression"]


def test_each_model_tier_backs_up_the_other():
    from codas.agents import agent

    reasoning, fast = agent.REASONING_MODEL, agent.FAST_MODEL
    if reasoning.model == fast.model:
        pytest.skip("a single model serves both tiers")
    assert [m.model for m in reasoning.models][:2] == [reasoning.model, fast.model]
    assert [m.model for m in fast.models][:2] == [fast.model, reasoning.model]


class _Silent(BaseLlm):
    """A model whose connection stalls: it never sends anything."""

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        SEEN.append(llm_request.model)
        await asyncio.sleep(3600)
        yield LlmResponse()


def test_a_stalled_model_call_times_out_retries_and_falls_back():
    import time

    from codas.agents.agent import ModelTimeout

    SEEN.clear()
    llm = FallbackLlm(model="stuck", models=[_Silent(model="stuck"), _Fake(model="backup")], timeout_s=0.2,
                      timeout_retries=1)
    started = time.monotonic()
    assert asyncio.run(_collect(llm)) == ["from backup"]
    assert SEEN == ["stuck", "stuck", "backup"] and time.monotonic() - started < 5
    SEEN.clear()
    alone = FallbackLlm(model="stuck", models=[_Silent(model="stuck")], timeout_s=0.2, timeout_retries=0)
    with pytest.raises(ModelTimeout):
        asyncio.run(_collect(alone))


def test_an_outcome_past_its_time_budget_is_stopped(monkeypatch):
    async def forever(*_args, **_kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(runner, "_run_agents_async", forever)
    monkeypatch.setattr(runner, "check_llm_backend", lambda: None)
    with pytest.raises(TimeoutError, match="did not finish within"):
        runner.run_agents("x.csv", "goal", timeout_min=0.002)


def test_trace_events_reach_disk_as_they_happen(tmp_path):
    from types import SimpleNamespace

    from codas.aou.trace import TraceRecorder

    path = tmp_path / "run_trace.jsonl"
    recorder = TraceRecorder(stream_path=path)
    part = SimpleNamespace(thought=None, function_call=None, function_response=None, text="n = 12 so far")
    recorder.record(SimpleNamespace(author="a", timestamp=1.0, content=SimpleNamespace(parts=[part])))
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and "<=20" in lines[0]  # written before the run ends, and masked


def test_identifiers_in_error_text_are_masked():
    from codas.aou.privacy import mask_identifiers

    text = mask_identifiers("int64 overflow: 200001110695 * 2654435761; 15 participants; p=0.000123456789")
    assert "200001110695" not in text and "2654435761" not in text and "<id> * <id>" in text
    assert "<=20 participants" in text and "0.000123456789" in text


def test_the_resume_key_follows_the_analysis_code_only(tmp_path):
    for rel in ("codas/core/engine.py", "codas/agents/prompts.py", "codas/aou/cohort.py", "codas/aou/job.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n")
    before = runner._analysis_version(tmp_path)
    (tmp_path / "codas/aou/job.py").write_text("x = 2\n")     # orchestration / display code
    assert runner._analysis_version(tmp_path) == before
    (tmp_path / "codas/agents/prompts.py").write_text("x = 2\n")  # analysis code
    assert runner._analysis_version(tmp_path) != before


def test_a_reused_run_without_its_pdf_gets_one_and_loses_the_stale_error(synthetic_explore):
    explore, calls, _failing = synthetic_explore
    explore()
    # An older run whose PDF failed: its summary still carries that error and no PDF.
    run_file = RUN_DIR / "explore_hdl_run.json"
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    files = saved["summary"]["files"]
    Path(files.pop("pdf")).unlink()
    files["pdf_error"] = "FPDFException: No page open, you need to call add_page() first"
    run_file.write_text(json.dumps(saved), encoding="utf-8")
    calls.clear()
    result = explore()
    assert calls == []                                          # reused, not rerun
    hdl = next(r for r in result["runs"] if r["cohort"] == "explore_hdl")
    assert Path(hdl["files"]["pdf"]).exists() and "pdf_error" not in hdl["files"]
    assert "pdf_error" not in json.loads(run_file.read_text(encoding="utf-8"))["summary"]["files"]
    assert "No page open" not in Path(result["summary_path"]).read_text(encoding="utf-8")


def test_wear_adherence_is_adjustment_only_and_covariates_are_imputed(synthetic_explore, monkeypatch):
    from codas.aou import cohort

    seen = {}

    def capture(csv, goal, on_tool_call=None, context_columns=None, recorder=None, state=None, **_):
        seen.update(goal=goal, context=list(context_columns or []), state=dict(state or {}))
        return _fake_state()

    monkeypatch.setattr(runner, "run_agents", capture)
    spec = _window_spec()
    days = cohort.availability_columns(spec)
    assert days == [f"{w.prefix}_days" for w in [*spec.wearables, *spec.phenotypes]]
    explore, _calls, _failing = synthetic_explore
    explore()
    assert set(days) <= set(seen["context"])                    # never screened as candidates
    assert all(d in seen["goal"] for d in days) and "wear" in seen["goal"]
    assert seen["state"]["covariate_imputations"] >= 5          # missing covariates are imputed, not dropped
    assert "imputed" in seen["goal"]
