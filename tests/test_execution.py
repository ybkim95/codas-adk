"""The execution report: every agent's model calls, tokens, tool calls and time, and how the agents fed
one another. Run with the scripted graph, so the counts can be checked exactly."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from codas.agents import tools
from codas.aou import runner
from codas.aou.execution import ExecutionPlugin, execution_markdown, execution_tables
from codas.aou.trace import TraceRecorder
from codas.core import gemini, model_tiers

from test_aou_e2e import (RECORDS, RUN_DIR, SCRIPTED_USAGE, _cohort_table, _window_spec,  # noqa: F401
                          aou_env)


def test_every_agent_that_ran_shows_its_model_calls_tokens_tools_and_time(aou_env, monkeypatch):
    monkeypatch.setenv("CODAS_MODEL_PRICES_JSON", json.dumps({"scripted": [1.0, 2.0]}))
    csv = RUN_DIR / "cohort.csv"
    _cohort_table().to_csv(csv, index=False)
    recorder = TraceRecorder()
    runner.run_agents(csv, runner.goal_for(_window_spec(), "HbA1c"), recorder=recorder)
    tables = recorder.execution_tables()
    rows = {a["agent"]: a for a in tables["agents"]}

    # Every model call the graph made is counted for the agent that made it, with its tokens.
    calls = Counter(name for name, _ in RECORDS)
    assert calls and {name: rows[name]["model_calls"] for name in calls} == dict(calls)
    usage = SCRIPTED_USAGE
    for name, n in calls.items():
        assert rows[name]["prompt"] == n * usage["prompt"] and rows[name]["thoughts"] == n * usage["thoughts"]
        assert rows[name]["models"] == {"scripted": n}
        expected = n * (usage["prompt"] * 1.0 + (usage["output"] + usage["thoughts"]) * 2.0) / 1e6
        assert rows[name]["cost_usd"] == pytest.approx(expected, abs=1e-4)
    totals = tables["totals"]
    assert totals["model_calls"] == sum(calls.values())
    assert totals["tokens"] == sum(a["total"] for a in tables["agents"]) == sum(calls.values()) * usage["total"]

    # Tool calls by agent agree with the trace, and every agent turn is timed, the deterministic ones too.
    traced = Counter((e["author"], e["name"]) for e in recorder.events if e["kind"] == "tool_call")
    counted = Counter({(a["agent"], tool): n for a in tables["agents"] for tool, n in a["tools"].items()})
    assert counted == traced
    assert rows["method_runner"]["turns"] >= 1 and rows["method_runner"]["model_calls"] == 0
    assert all(t["end_s"] >= t["start_s"] >= 0 for t in tables["timeline"])
    assert totals["wall_s"] >= max(a["active_s"] for a in tables["agents"]) - 0.1

    # How the agents fed one another: memory one agent wrote and a later agent's instructions read.
    flow = {(f["writer"], f["key"], f["reader"]) for f in tables["flow"]}
    assert ("hypotheses_agent", "hypotheses", "critic_agent") in flow
    assert ("method_runner", "analyses_digest", "statistical_interpreter") in flow
    # Rewriting a value unchanged is not a write: the masking of shared memory once made every agent that
    # called a tool look like the author of the round records.
    assert not any(w == "candidate_defender" and k in ("rounds", "fact_sheet") for w, k, _ in flow)
    # Who changed the design, and in which round.
    changes = [(c["agent"], c["tool"], c["round"]) for c in tables["design_changes"]]
    assert ("scout_agent", "set_target", 0) in changes and ("hypotheses_agent", "register_hypotheses", 0) in changes
    assert any(a == "critic_agent" and t == "exclude_features" and r >= 1 for a, t, r in changes)

    text = execution_markdown(tables)
    assert "| scout_agent |" in text and "Information flow" in text and "Design changes" in text


def test_the_execution_record_survives_a_resumed_run(aou_env):
    csv = RUN_DIR / "cohort.csv"
    _cohort_table().to_csv(csv, index=False)
    recorder = TraceRecorder()
    runner.run_agents(csv, runner.goal_for(_window_spec(), "HbA1c"), recorder=recorder)
    restored = TraceRecorder.restore(json.loads(json.dumps(recorder.snapshot(), default=str)))
    assert restored.execution_tables()["totals"] == recorder.execution_tables()["totals"]
    files = restored.write(RUN_DIR, "cohort")
    assert json.loads(open(files["execution_json"]).read())["totals"]["model_calls"] > 0
    assert open(files["execution_md"]).read().startswith("# Agent execution: cohort")


def test_a_tools_own_model_calls_count_for_the_agent_that_called_it(monkeypatch):
    """The literature tool calls Gemini itself, outside the agent's model; its tokens and model still
    count for the calling agent."""
    usage = {"model": "gemini-3.8-flash", "prompt": 100, "output": 50, "thoughts": 20, "total": 170,
             "search_queries": 2, "attempts": 1, "rate_limited": 0}
    monkeypatch.delenv("CODAS_AOU_MODE", raising=False)
    monkeypatch.setattr(gemini, "generate_grounded_reply", lambda query, context="": gemini.GroundedResponse(
        text="Evidence.", model="gemini-3.8-flash", configured=True, sources=[{"title": "a", "url": "u"}],
        queries=["q"], usage=usage))
    plugin = ExecutionPlugin()
    ctx = SimpleNamespace(function_call_id="call-1", agent_name="mechanism_agent", state={})
    tool = SimpleNamespace(name="search_literature")
    asyncio.run(plugin.before_tool_callback(tool=tool, tool_args={"query": "q"}, tool_context=ctx))
    result = tools.search_literature("q", ctx)
    assert "usage" not in json.dumps(result)  # the model never sees the bookkeeping
    asyncio.run(plugin.after_tool_callback(tool=tool, tool_args={"query": "q"}, tool_context=ctx, result=result))
    row = execution_tables(plugin.snapshot())["agents"][0]
    assert (row["agent"], row["tool_calls"], row["total"]) == ("mechanism_agent", 1, 170)
    assert row["models"] == {"gemini-3.8-flash": 1} and row["search_queries"] == 2
    assert row["cost_usd"] == pytest.approx((100 * 0.75 + 70 * 3.75) / 1e6, abs=1e-4)
    assert tools.MODEL_USAGE == {}  # taken by the plugin, so nothing accumulates over a long run


def test_costs_use_the_listed_price_of_the_served_model(monkeypatch):
    monkeypatch.delenv("CODAS_MODEL_PRICES_JSON", raising=False)
    assert model_tiers.price("gemini-3.5-flash-lite-001") == (0.30, 2.50)  # a dated version: its base model
    assert model_tiers.price("gemini-3.5-flash-001") == (1.5, 9.0)
    assert model_tiers.price("an-unlisted-model") is None
    monkeypatch.setenv("CODAS_MODEL_PRICES_JSON", "not json")
    assert model_tiers.price("gemini-4-argon") == (2.0, 10.0)
    records = {"calls": [{"kind": "model", "agent": "a", "start": 0, "end": 1, "model": "an-unlisted-model",
                            "prompt": 10, "output": 5, "total": 15}], "reads": {}, "started": 0}
    tables = execution_tables(records)
    assert tables["agents"][0]["cost_usd"] is None and tables["totals"]["cost_usd"] is None
    assert "unknown" in execution_markdown(tables)


def test_bookkeeping_never_stops_a_run():
    """ADK turns an exception inside a plugin into a failed run; a record that cannot be made is skipped."""
    plugin = ExecutionPlugin()
    broken = SimpleNamespace()  # no name, call id or invocation id: every record attempt fails inside
    calls = {"before_agent_callback": {"agent": broken, "callback_context": broken},
             "after_agent_callback": {"agent": broken, "callback_context": broken},
             "before_model_callback": {"callback_context": broken, "llm_request": None},
             "after_model_callback": {"callback_context": broken, "llm_response": broken},
             "on_model_error_callback": {"callback_context": broken, "llm_request": None, "error": ValueError()},
             "before_tool_callback": {"tool": broken, "tool_args": {}, "tool_context": broken},
             "after_tool_callback": {"tool": broken, "tool_args": {}, "tool_context": broken, "result": {}},
             "on_tool_error_callback": {"tool": broken, "tool_args": {}, "tool_context": broken, "error": ValueError()},
             "on_event_callback": {"invocation_context": None, "event": SimpleNamespace(actions=SimpleNamespace(
                 state_delta={"k": 1}))}}
    for name, kwargs in calls.items():
        assert asyncio.run(getattr(plugin, name)(**kwargs)) is None, name
    assert plugin.calls == []
