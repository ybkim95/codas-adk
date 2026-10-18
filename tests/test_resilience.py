"""Run resilience: a model call that keeps hitting a rate limit falls back to the next configured model,
and a free exploration survives a failing outcome and resumes without redoing finished ones.

Offline: fake models and the synthetic DuckDB CDR; no model is called.
"""

from __future__ import annotations

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

    def fake_run_agents(csv, goal, on_tool_call=None, context_columns=None, recorder=None):
        name = Path(csv).stem
        calls.append(name)
        if name in failing:
            raise _exhausted()
        return _fake_state()

    monkeypatch.setattr(runner, "run_agents", fake_run_agents)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    def explore(**kw):
        return runner.explore(CATALOG, CohortSpec.from_json(SPEC), SYNTHETIC_CDR, client=object(), top_k=1,
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
    explore()                                   # the code changed: results are stale, both run again
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
