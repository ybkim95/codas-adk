"""Tests for the paper-faithful agent graph and its stateful loop tools (offline, no Gemini).

These exercise the parts the live LLM run depends on but that must be correct deterministically:
the six-phase structure, shared-memory tools (``set_target``/``run_discovery_round``), and the
GapChecker's convergence/escalate logic. A live end-to-end Gemini trace is exercised separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent

from codas_agents.agent import discovery_loop, reporting, root_agent
from codas_agents.tools import check_convergence, run_discovery_round, set_target


# --- a minimal stand-in for ADK's ToolContext (the tools use only .state and .actions.escalate) ---

class _Actions:
    def __init__(self) -> None:
        self.escalate = False


class _Ctx:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.actions = _Actions()


def _signal_csv(path: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(0)
    drv1, drv2 = rng.normal(size=n), rng.normal(size=n)
    frame = {f"noise{i}": rng.normal(size=n) for i in range(6)}
    frame["drv1"], frame["drv2"] = drv1, drv2
    frame["outcome"] = 0.6 * drv1 + 0.3 * drv2 + rng.normal(size=n) * 0.4
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def _null_csv(path: Path, n: int = 300) -> Path:
    rng = np.random.default_rng(1)
    frame = {f"g{i}": rng.normal(size=n) for i in range(8)}
    frame["outcome"] = rng.normal(size=n)
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


# --- structure: the graph IS the paper's six-phase orchestrator ---

def test_root_is_sequential_orchestrator():
    assert isinstance(root_agent, SequentialAgent)
    assert [a.name for a in root_agent.sub_agents] == ["data_understanding", "discovery_loop", "reporting"]


def test_discovery_is_a_loop_wrapping_a_parallel_dual_track():
    assert isinstance(discovery_loop, LoopAgent)
    assert discovery_loop.max_iterations >= 1
    names = [a.name for a in discovery_loop.sub_agents]
    assert names[0] == "search_agent" and names[-1] == "gapcheck_agent"
    parallels = [a for a in discovery_loop.sub_agents if isinstance(a, ParallelAgent)]
    assert len(parallels) == 1
    assert {a.name for a in parallels[0].sub_agents} == {"statistical_interpreter", "ml_interpreter"}


def test_reporting_phase_ends_with_the_report_agent():
    assert isinstance(reporting, SequentialAgent)
    assert reporting.sub_agents[-1].name == "report_agent"


# --- shared-memory tools ---

def test_set_target_records_design_in_memory():
    ctx = _Ctx({})
    set_target("outcome", ctx, participant_id_column="pid", confounder_columns_csv="a, b")
    assert ctx.state["target_column"] == "outcome"
    assert ctx.state["participant_id_column"] == "pid"
    assert ctx.state["confounder_columns"] == ["a", "b"]


def test_run_discovery_round_deepens_and_records_each_round(tmp_path):
    ctx = _Ctx({"csv_path": str(_signal_csv(tmp_path / "signal.csv"))})
    set_target("outcome", ctx)
    r0 = run_discovery_round(ctx)
    r1 = run_discovery_round(ctx)
    # the search deepens: later rounds widen the engineered-feature budget and the candidate count
    assert r0["round"] == 0 and r1["round"] == 1
    assert r1["ratio_feature_budget"] > r0["ratio_feature_budget"]
    assert r1["top_k"] > r0["top_k"]
    # every round is appended to shared memory, and the latest Fact Sheet is published
    assert len(ctx.state["rounds"]) == 2
    assert ctx.state["fact_sheet"]["target_column"] == "outcome"
    assert r0["validated_count"] >= 1  # the planted signal is recovered


def test_run_discovery_round_errors_without_path_or_target():
    assert "error" in run_discovery_round(_Ctx({}))
    # path present but no target set
    assert "error" in run_discovery_round(_Ctx({"csv_path": "/tmp/whatever.csv"}))


# --- GapChecker: convergence + escalate ---

def test_gapcheck_escalates_on_saturation(tmp_path):
    ctx = _Ctx({"csv_path": str(_signal_csv(tmp_path / "signal.csv"))})
    set_target("outcome", ctx)
    run_discovery_round(ctx)
    assert check_convergence(ctx)["converged"] is False  # one round in: keep going
    assert ctx.actions.escalate is False
    run_discovery_round(ctx)
    verdict = check_convergence(ctx)  # second round saturates (no new validated, tiny metric gain)
    assert verdict["converged"] is True
    assert ctx.actions.escalate is True


def test_gapcheck_escalates_on_a_clear_null_after_one_round(tmp_path):
    ctx = _Ctx({"csv_path": str(_null_csv(tmp_path / "null.csv"))})
    set_target("outcome", ctx)
    run_discovery_round(ctx)
    verdict = check_convergence(ctx)
    assert verdict["converged"] is True  # no signal: deeper search cannot help
    assert ctx.actions.escalate is True


def test_gapcheck_without_any_round_does_not_escalate():
    ctx = _Ctx({})
    verdict = check_convergence(ctx)
    assert verdict["converged"] is False and ctx.actions.escalate is False


# --- JSON-safety of tool results (real data has missing values; NaN/inf are not valid JSON) ---

def test_tools_return_strict_json_on_missing_data(tmp_path):
    import json

    from codas_agents.tools import _json_safe, preview_columns, profile_dataset

    path = tmp_path / "missing.csv"
    # NaN and inf are exactly what broke the ADK -> Gemini tool-response payload on real datasets.
    pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.inf, 2.0, np.nan], "t": [0, 1, 0]}).to_csv(path, index=False)
    for out in (preview_columns(str(path)), profile_dataset(str(path))):
        json.dumps(out, allow_nan=False)  # raises ValueError if any NaN/inf survived

    assert _json_safe(float("nan")) is None and _json_safe(float("inf")) is None
    assert _json_safe({"x": [float("inf"), np.float64("nan"), np.int64(7)]}) == {"x": [None, None, 7]}
