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
    lite = agent.LITE_MODEL
    if lite.model not in (reasoning.model, fast.model):
        assert [m.model for m in lite.models] == [lite.model, fast.model, reasoning.model]


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


def test_lessons_from_one_outcome_reach_the_next_without_changing_resume_keys(synthetic_explore, monkeypatch):
    goals: list[str] = []
    states: list[dict] = []

    def capture(csv, goal, on_tool_call=None, context_columns=None, recorder=None, state=None, **_):
        goals.append(goal)
        states.append(dict(state or {}))
        out = _fake_state()
        if len(goals) == 1:
            out["lessons"] = [{"lesson": "act_days measures how much the device was worn, not physiology",
                               "about": ["act_days"], "evidence": "critic"}]
        return out

    monkeypatch.setattr(runner, "run_agents", capture)
    explore, _calls, _failing = synthetic_explore
    explore()
    assert len(goals) == 2
    # Lessons come from other outcomes' discovery statistics: never in the goal the a-priori stage reads,
    # only in state, from which the tools reveal them once registration has closed.
    assert not any("act_days measures how much the device was worn" in g for g in goals)
    assert states[0]["prior_lessons"] == []
    assert "act_days measures how much the device was worn" in "; ".join(states[1]["prior_lessons"])
    memory = (RUN_DIR / "codas_memory.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(memory) == 1 and "act_days" in memory[0]
    goals.clear()
    explore()                      # same code and specs: both reused although lessons now exist
    assert goals == []


def test_record_lessons_validates_entries(tmp_path):
    import json
    from types import SimpleNamespace

    from codas.agents import tools

    ctx = SimpleNamespace(state={"dataset_columns": ["act_days", "y"]}, actions=SimpleNamespace(escalate=False))
    out = tools.record_lessons(json.dumps([
        {"lesson": "act_days measures wear", "about": ["act_days"], "evidence": "critic round 1"},
        {"lesson": "", "about": []},
        {"lesson": "about a column that does not exist", "about": ["nope"]}]), ctx)
    assert [x["lesson"] for x in out["recorded"]] == ["act_days measures wear"] and len(out["rejected"]) == 2
    assert ctx.state["lessons"][0]["about"] == ["act_days"]
    assert "error" in tools.record_lessons("{}", ctx)


def test_agent_work_survives_a_failure_after_the_agents(synthetic_explore, monkeypatch):
    """A crash after the agents (pre-registration, holdout test, saving) reruns only those steps."""
    explore, calls, failing = synthetic_explore
    real_save = runner.save_outputs
    crashes = {"explore_hdl": 1}

    def flaky_save(state, stem, *a, **k):
        if crashes.get(stem):
            crashes[stem] -= 1
            raise ValueError("math domain error")
        return real_save(state, stem, *a, **k)

    monkeypatch.setattr(runner, "save_outputs", flaky_save)
    first = explore()
    assert calls == ["explore_hdl", "explore_depression"] and "math domain error" in first["runs"][0]["failed"]
    assert (RUN_DIR / "explore_hdl_agents.json").exists()
    calls.clear()
    second = explore()
    assert calls == []                                     # the agents' saved work was reused
    assert not second["runs"][0].get("failed") and not (RUN_DIR / "explore_hdl_agents.json").exists()
    crashes["explore_hdl"] = 1
    explore(rerun=True)                                    # crash again, then change the analysis code
    calls.clear()
    monkeypatch.setattr(runner, "_analysis_version", lambda root=None: "changed-analysis")
    explore()
    assert "explore_hdl" in calls                          # a checkpoint from other analysis code is not reused


def test_a_failed_preflight_stops_the_outcome_before_the_agents(synthetic_explore, monkeypatch):
    explore, calls, failing = synthetic_explore
    monkeypatch.setattr(runner, "PREFLIGHT", True)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: {"flags": ["x"], "fatal": ["negative control a could "
                                                                                         "not be computed"],
                                                               "duplicates": []})
    result = explore()
    assert calls == [] and all("PreflightError" in r["failed"] for r in result["runs"])


async def _collect_request(llm: BaseLlm, request: LlmRequest) -> list[str]:
    return [r.content.parts[0].text async for r in llm.generate_content_async(request)]


def _llm_agents(node):
    from google.adk.agents import LlmAgent

    if isinstance(node, LlmAgent):
        yield node
    for child in getattr(node, "sub_agents", None) or []:
        yield from _llm_agents(child)


class _RejectsThinking(BaseLlm):
    """A model that refuses requests carrying a thinking level, as a model without thinking levels does."""

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        thinking = getattr(llm_request.config, "thinking_config", None)
        SEEN.append(f"{llm_request.model}:{'thinking' if thinking else 'plain'}")
        if thinking is not None:
            raise ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                              "message": "Thinking level is not supported for this model."}})
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"from {self.model}")]))


def test_a_model_that_rejects_the_thinking_level_is_asked_again_without_it():
    SEEN.clear()
    llm = FallbackLlm(model="older", models=[_RejectsThinking(model="older")])
    request = LlmRequest(model="older", contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
                         config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="HIGH")))
    out = asyncio.run(_collect_request(llm, request))
    assert out == ["from older"] and SEEN == ["older:thinking", "older:plain"]
    # Any other bad request is still raised, not retried.
    with pytest.raises(ClientError):
        asyncio.run(_collect(FallbackLlm(model="p", models=[_Fake(model="p", fail_with=400)])))


def test_agents_run_on_the_current_model_tiers_with_a_thinking_level():
    from codas.agents import agent
    from codas.core import model_tiers

    lite = {"gapcheck_agent", "artifact_agent", "lessons_agent"}
    fast = {"search_agent", "statistical_interpreter", "ml_interpreter"}
    superseded = set(model_tiers.SUPERSEDED)
    for a in _llm_agents(agent.root_agent):
        tier = "lite" if a.name in lite else "fast" if a.name in fast else "reasoning"
        assert a.model.model == model_tiers.model(tier), a.name
        assert not superseded & {m.model for m in a.model.models}, a.name
        level = a.generate_content_config.thinking_config.thinking_level
        expected = "HIGH" if a.name == "hypotheses_agent" else model_tiers.THINKING[tier]
        assert str(getattr(level, "value", level)) == expected, a.name


def test_outcomes_already_tested_on_this_holdout_are_not_selected_again(synthetic_explore):
    """A ranked selection skips outcomes whose holdout results are already known; an outcome named
    explicitly still runs (the ledger reports its registrations as non-confirmatory)."""
    from dataclasses import replace

    from codas.aou import ledger
    from codas.aou.synthetic import SYNTHETIC_CDR

    spec = replace(_window_spec(), holdout_fraction=0.5)
    ledger.append(RUN_DIR, runner.holdout_key(spec, SYNTHETIC_CDR), [{"prereg": "earlier", "stem": "explore_hdl",
                                                                      "feature": "act_steps_mean"}])

    def screen(**kw):
        return runner.explore(CATALOG, spec, SYNTHETIC_CDR, client=object(), top_k=1, min_measurement_n=100,
                              min_cases=30, out_dir=RUN_DIR, screen_only=True, log=lambda *_: None, **kw)

    ranked = screen()
    assert "hdl" not in ranked["chosen"] and "depression" in ranked["chosen"]
    assert "already tested on this holdout" in ranked["table"]
    assert screen(outcome_names=["hdl"])["chosen"] == ["hdl"]
    assert len(screen(max_outcomes=1)["chosen"]) == 1


def test_running_again_resumes_an_outcome_this_exploration_opened(tmp_path):
    """An outcome tested on the holdout is skipped by a ranked selection, unless its run with this code
    finished here: then running the exploration again reuses it instead of starting another outcome."""
    from dataclasses import replace

    from codas.aou import feasibility as feas
    from codas.aou import ledger

    base = replace(_window_spec(), holdout_fraction=0.5)
    outcome = next(o for o in feas.load_catalog(CATALOG) if o.name == "hdl")
    spec = feas.spec_for(outcome, base)
    key = "cdr|salt=1|permille=500"
    ledger.append(tmp_path, key, [{"prereg": "p", "stem": spec.name, "feature": "x"}])
    assert runner._opened_outcomes(tmp_path, [key], {"hdl": outcome}, base) == {"hdl"}
    report = tmp_path / f"{spec.name}_report.md"
    report.write_text("report", encoding="utf-8")
    run_key = runner._run_key(spec, runner.goal_for(spec, outcome.label), True)
    (tmp_path / f"{spec.name}_run.json").write_text(json.dumps({"key": run_key, "summary": {
        "cohort": spec.name, "report_path": str(report)}}), encoding="utf-8")
    assert runner._opened_outcomes(tmp_path, [key], {"hdl": outcome}, base) == set()
    # Finished with older code: the outcome counts as opened again, and the selection moves on.
    monkeypatch_key = runner._run_key(spec, runner.goal_for(spec, outcome.label), False)
    assert monkeypatch_key != run_key
    assert runner._opened_outcomes(tmp_path, [key], {"hdl": outcome}, base, test_holdout=False) == {"hdl"}


def test_an_instrumentation_run_leaves_the_sealed_holdout_unopened(synthetic_explore, monkeypatch):
    """With test_holdout off the agents run on the discovery share, but nothing is registered or tested, so
    the outcome stays fresh for a confirmatory run (whose key differs, so it is not served this run)."""
    from dataclasses import replace

    from codas.aou import ledger
    from codas.aou.synthetic import SYNTHETIC_CDR

    _, calls, _ = synthetic_explore
    monkeypatch.setattr(runner, "PREFLIGHT", False)
    tested = []
    monkeypatch.setattr(runner, "confirm", lambda *a, **k: tested.append(a) or {})
    spec = replace(_window_spec(), holdout_fraction=0.5)
    result = runner.explore(CATALOG, spec, SYNTHETIC_CDR, client=object(), top_k=1, min_measurement_n=100,
                            min_cases=30, out_dir=RUN_DIR, log=lambda *_: None, test_holdout=False, max_outcomes=1)
    run = result["runs"][0]
    assert calls == [run["cohort"]] and run["holdout_tested"] is False and tested == []
    assert not (RUN_DIR / ledger.LEDGER_NAME).exists()
    assert "left unopened" in Path(run["report_path"]).read_text(encoding="utf-8")
    assert "not tested in this run" in Path(result["summary_path"]).read_text(encoding="utf-8")
    goal = runner.goal_for(spec, "x")
    assert runner._run_key(spec, goal, test_holdout=False) != runner._run_key(spec, goal, test_holdout=True)
