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

from codas.agents.agent import discovery_loop, reporting, root_agent
from codas.agents.tools import check_convergence, declare_confounders, run_discovery_round, set_target


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

    from codas.agents.tools import _json_safe, preview_columns, profile_dataset

    path = tmp_path / "missing.csv"
    # NaN and inf are exactly what broke the ADK -> Gemini tool-response payload on real datasets.
    pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.inf, 2.0, np.nan], "t": [0, 1, 0]}).to_csv(path, index=False)
    for out in (preview_columns(str(path)), profile_dataset(str(path))):
        json.dumps(out, allow_nan=False)  # raises ValueError if any NaN/inf survived

    assert _json_safe(float("nan")) is None and _json_safe(float("inf")) is None
    assert _json_safe({"x": [float("inf"), np.float64("nan"), np.int64(7)]}) == {"x": [None, None, 7]}


# --- production hardening: pluggable session store + runtime grounding guardrail ---

def test_session_factory_defaults_to_memory_and_falls_back_safely(monkeypatch):
    from google.adk.sessions import InMemorySessionService

    from codas.agents.runtime import new_session_service

    monkeypatch.delenv("CODAS_SESSION_BACKEND", raising=False)
    assert isinstance(new_session_service(), InMemorySessionService)
    # vertex requested but misconfigured (no project) must DEGRADE to in-memory, never crash
    monkeypatch.setenv("CODAS_SESSION_BACKEND", "vertex")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert isinstance(new_session_service(), InMemorySessionService)


def test_grounding_guardrail_flags_fabricated_figures():
    import logging

    from codas.agents.callbacks import LOGGER, report_grounding_audit

    class _Ctx:
        def __init__(self, state):
            self.state = state

    msgs: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            msgs.append(record.getMessage())

    handler = _Cap()
    prior_level = LOGGER.level
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)  # the guardrail logs at INFO; ensure the logger emits it
    try:
        fs = {"ml_metric_value": 0.4256, "rows": 220}
        cands = {"candidates": [{"rho": 0.52, "q_value": 0.001, "n": 220}]}
        report_grounding_audit(_Ctx({"fact_sheet": fs, "latest_report": cands,
                                     "report": "The model R2 of 0.43; the top predictor rho = 0.52."}))
        report_grounding_audit(_Ctx({"fact_sheet": fs, "latest_report": cands,
                                     "report": "We obtained an AUC of 0.98 and recommend deployment."}))
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(prior_level)
    assert any("grounding: 2/2" in m for m in msgs), f"a grounded report should verify all figures: {msgs}"
    assert any("unverified figure" in m for m in msgs), f"a fabricated AUC must be flagged: {msgs}"


# --- declaring a confounder must actually change what the engine validates ---------------------

def _confounded_csv(path: Path, n: int = 400) -> Path:
    """`driver` causes both `artefact` and the outcome; `genuine` has a real, independent effect."""
    rng = np.random.default_rng(7)
    driver = rng.normal(45, 12, size=n)
    frame = {
        "driver": driver,
        "artefact": 80 - 0.7 * driver + rng.normal(0, 6, size=n),
        "genuine": rng.normal(size=n),
        "noise": rng.normal(size=n),
    }
    frame["outcome"] = 50 - 0.45 * driver + 3.2 * frame["genuine"] + rng.normal(0, 5, size=n)
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def test_declare_confounders_merges_without_clearing_the_design(tmp_path):
    ctx = _Ctx({"csv_path": str(_confounded_csv(tmp_path / "c.csv"))})
    set_target("outcome", ctx, excluded_columns_csv="noise")
    out = declare_confounders("driver", ctx)
    assert out["confounder_columns"] == ["driver"]
    assert ctx.state["target_column"] == "outcome"      # the rest of the design survives
    assert ctx.state["excluded_columns"] == ["noise"]
    # merging, not replacing; unknown names and the target itself are reported, not recorded
    out = declare_confounders("genuine, not_a_column, outcome", ctx)
    assert out["confounder_columns"] == ["driver", "genuine"]
    assert set(out["ignored"]) == {"not_a_column", "outcome"}


def test_declare_confounders_requires_a_design_first(tmp_path):
    assert "error" in declare_confounders("driver", _Ctx({}))


def test_declaring_the_confounder_flips_the_verdict(tmp_path):
    """Without the declaration the confounded artefact validates; with it, it is rejected.

    This is the whole reason the tool exists: the domain judgement that `driver` is a common cause
    has to reach the engine, or the pipeline reports a feature that explains nothing.
    """
    csv = str(_confounded_csv(tmp_path / "c.csv"))

    def verdicts_for(ctx) -> dict[str, str]:
        run_discovery_round(ctx)
        return {c["feature"]: c["verdict"] for c in ctx.state["latest_report"]["candidates"]}

    blind = _Ctx({"csv_path": csv})
    set_target("outcome", blind)
    assert verdicts_for(blind).get("artefact") == "validated"

    aware = _Ctx({"csv_path": csv})
    set_target("outcome", aware)
    declare_confounders("driver", aware)
    aware_verdicts = verdicts_for(aware)
    assert aware_verdicts.get("artefact") == "rejected", aware_verdicts
    # What survives is built on `genuine`. Which variant carries it is up to collinearity demotion —
    # the bare column can be collapsed into an engineered ratio that ranks above it — so assert the
    # signal survives rather than pinning the label the engine happens to prefer.
    survivors = [f for f, v in aware_verdicts.items() if v == "validated"]
    assert survivors and all("genuine" in f for f in survivors), aware_verdicts
