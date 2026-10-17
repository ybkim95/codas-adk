"""End-to-end audit of the All of Us privacy guard with a scripted (offline) model.

The full CoDaS agent graph runs in AoU mode on a synthetic, extractor-shaped cohort. Every agent's
model is replaced by a scripted stand-in that calls the real deterministic tools, and every request
the graph sends to a model is recorded. The test then checks what a real Vertex model would have
received: no participant rows, no participant counts of 1-20, and a masked final report.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import pandas as pd
import pytest
from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from codas.agents import agent as agent_module
from codas.agents import tools
from codas.aou import runner
from codas.aou.cohort import CohortSpec, output_columns
from codas.aou.privacy import MASK, SMALL_CELL_MAX, _is_count_key

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"
CATALOG = ROOT / "scripts" / "aou" / "outcomes_catalog.json"
RUN_DIR = ROOT / ".codas_runs" / "aou_e2e_test"


def _call(name: str, **args: Any) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


# What each agent does: a sequence of tool calls, then a text answer.
SCRIPTS: dict[str, list[types.Part]] = {
    "scout_agent": [_call("profile_dataset"), _call("preview_columns"),
                    _call("set_target", target_column="hba1c", confounder_columns_csv="age,female")],
    "hypotheses_agent": [_call("search_literature", query="daily steps and HbA1c in n = 12 participants")],
    "search_agent": [_call("run_discovery_round")],
    "statistical_interpreter": [_call("propose_feature", operation="ratio", feature_a="act_steps_mean",
                                      feature_b="sleep_rare_mean")],
    "gapcheck_agent": [_call("check_convergence")],
    # The critic judges one input to be target leakage; the next round must run without it.
    "critic_agent": [_call("exclude_features", columns_csv="act_calories_out_mean,act_calories_out_sd",
                           reason="treated as derived from the outcome in this test")],
}
REPORT_TEXT = ("Mean daily steps were inversely associated with HbA1c. A subgroup of n = 7 participants "
               "and 15 cases lacked sleep data.")


# Every (agent name, request) the graph sends to a model. Module-level because pydantic copies
# list-valued fields, so a list passed to each model instance would not be shared.
RECORDS: list[tuple[str, LlmRequest]] = []


class ScriptedLlm(BaseLlm):
    """Replays SCRIPTS[agent_name], one step per model call within a turn, and records requests."""

    agent_name: str = ""

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        RECORDS.append((self.agent_name, llm_request))
        step = 0  # tool responses this agent has already received in the current turn
        for content in reversed(llm_request.contents or []):
            parts = content.parts or []
            if any(p.function_response for p in parts):
                step += 1
            elif any(p.function_call for p in parts):
                continue
            else:
                break
        script = SCRIPTS.get(self.agent_name, [])
        if step < len(script):
            part = script[step]
            if part.function_call.name == "set_target":  # target named in the goal ("The target column is x")
                goal = " ".join(p.text or "" for c in llm_request.contents or [] for p in c.parts or [])
                found = re.search(r"target column is (\w+)", goal)
                part = _call("set_target", target_column=found.group(1) if found else "hba1c",
                             confounder_columns_csv="age,female")
        else:
            text = REPORT_TEXT if self.agent_name == "report_agent" else f"{self.agent_name}: done."
            part = types.Part(text=text)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _llm_agents(node: Any):
    if isinstance(node, LlmAgent):
        yield node
    for child in getattr(node, "sub_agents", []) or []:
        yield from _llm_agents(child)


def _cohort_table(n: int = 400) -> pd.DataFrame:
    spec = CohortSpec.from_json(SPEC)
    rng = np.random.default_rng(11)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in output_columns(spec)})
    df["age"] = rng.integers(20, 80, n)
    df["female"] = rng.integers(0, 2, n)
    df["act_days"] = rng.integers(30, 180, n)
    df["act_steps_mean"] = rng.normal(7000, 2400, n)
    df["hba1c"] = 5.7 - 0.00012 * (df["act_steps_mean"] - 7000) + 0.004 * df["age"] + rng.normal(0, 0.35, n)
    rare = np.full(n, np.nan)
    rare[:15] = rng.normal(400, 30, 15)  # a column only 15 participants have: its counts must be masked
    df["sleep_rare_mean"] = rare
    return df


def _serialize(request: LlmRequest) -> str:
    chunks = [str(getattr(request.config, "system_instruction", "") or "")]
    for content in request.contents or []:
        for part in content.parts or []:
            if part.text:
                chunks.append(part.text)
            if part.function_call:
                chunks.append(json.dumps(part.function_call.args or {}, default=str))
            if part.function_response:
                chunks.append(json.dumps(part.function_response.response or {}, default=str))
    return "\n".join(chunks)


def _small_counts(obj: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (_is_count_key(str(key)) and isinstance(value, (int, float)) and not isinstance(value, bool)
                    and float(value).is_integer() and 1 <= value <= SMALL_CELL_MAX):
                found.append(f"{path}.{key}={value}")
            found += _small_counts(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found += _small_counts(item, f"{path}[{i}]")
    return found


@pytest.fixture
def aou_env(monkeypatch):
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CODAS_AOU_MODE", "1")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "aou-workspace-test")
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "0")
    monkeypatch.setattr(tools, "_ROUND_RESAMPLES", 60)

    RECORDS.clear()
    originals = {}
    for a in _llm_agents(agent_module.root_agent):
        originals[a.name] = a.model
        a.model = ScriptedLlm(model="scripted", agent_name=a.name)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield RECORDS
    finally:
        for a in _llm_agents(agent_module.root_agent):
            a.model = originals[a.name]
        shutil.rmtree(RUN_DIR, ignore_errors=True)


def test_full_graph_sends_only_masked_aggregates(aou_env):
    recorder = aou_env
    table = _cohort_table()
    csv = RUN_DIR / "cohort.csv"
    table.to_csv(csv, index=False)

    state = runner.run_agents(csv, runner.goal_for(CohortSpec.from_json(SPEC), "HbA1c"))
    summary = runner.save_outputs(state, "cohort", RUN_DIR)

    agents_called = {name for name, _ in recorder}
    assert {"scout_agent", "search_agent", "gapcheck_agent", "report_agent"} <= agents_called
    assert state.get("rounds"), "the discovery loop must have run"

    sent = [_serialize(req) for _, req in recorder]
    everything = "\n".join(sent)

    # 1. No participant rows: no preview payload, and no raw cell value at full precision.
    assert "preview_rows" not in everything
    raw = pd.read_csv(csv).head(5)
    for value in raw.select_dtypes("float").to_numpy().ravel():
        if np.isfinite(value):
            assert repr(float(value)) not in everything, f"raw cell {value!r} reached the model"

    # 2. No participant count of 1-20 in any tool response the model received.
    for _, req in recorder:
        for content in req.contents or []:
            for part in content.parts or []:
                if part.function_response:
                    leaks = _small_counts(part.function_response.response or {})
                    assert not leaks, f"small counts reached the model: {leaks}"
    # ...the exploratory association on the 15-participant column was masked, not dropped.
    assert f'"n": "{MASK}"' in everything

    # 3. The literature tool was off, and its query never left.
    assert "Literature search is turned off" in everything

    # 4. The critic's exclusion reached the engine: a later round ran without the column or anything
    #    built from it, and the round summary the report reads records it.
    last = state["rounds"][-1]
    assert len(state["rounds"]) >= 2 and "act_calories_out_mean" in last["excluded_features"]
    assert not any("act_calories_out" in c["feature"] for c in state["latest_report"]["candidates"])

    # 5. The saved report is masked.
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert "n = 7" not in report and "15 cases" not in report and MASK in report
    assert "n = 7" not in str(state.get("report"))


def test_explore_end_to_end_on_synthetic_cdr(aou_env, monkeypatch):
    """Catalog check -> feasibility -> extraction -> agents -> summary, on the synthetic DuckDB CDR."""
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    from codas.aou import cohort
    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    # top 1 of each outcome kind: one measurement and one condition run end to end
    result = runner.explore(CATALOG, CohortSpec.from_json(SPEC), SYNTHETIC_CDR, client=object(), top_k=1,
                            min_measurement_n=100, min_cases=30, out_dir=RUN_DIR, log=lambda *_: None)

    assert [r["cohort"] for r in result["runs"]] == ["explore_hdl", "explore_depression"]
    for run in result["runs"]:
        assert run["rounds"] >= 1 and Path(run["report_path"]).exists()
    hdl = result["runs"][0]
    # The planted signal is one latent activity factor behind steps, active minutes and calories (rank
    # correlation ~0.99 between them). The scripted critic excludes calories as leakage, and the engine
    # then also rejects features that are near-copies of an excluded column, so the signal surfaces
    # through whatever survives: an active-minutes or steps feature, or a ratio built on one.
    planted = ("act_steps_mean", "act_very_active_minutes_mean")
    assert any(p in c for c in hdl["top_candidates"] for p in planted), hdl["top_candidates"]
    # The base spec's covariate was extracted, and stayed out of the candidates.
    table = pd.read_csv(RUN_DIR / "explore_hdl.csv")
    assert table["bmi_cov"].notna().any() and not any("bmi_cov" in c for c in hdl["top_candidates"])
    assert "preview_rows" not in "\n".join(_serialize(req) for _, req in aou_env)
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "explore_hdl" in summary and "explore_depression" in summary and "adjusted for:" in summary


def test_run_agents_refuses_consumer_api_key(aou_env, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "consumer-key")
    with pytest.raises(Exception, match="Vertex"):
        runner.run_agents(RUN_DIR / "missing.csv", "goal")


def test_raw_values_would_be_caught_without_the_guard():
    """Sanity check on the audit itself: an unguarded preview carries full-precision cells."""
    table = _cohort_table()
    preview = json.dumps(table.head(3).to_dict(orient="records"))
    assert re.search(r"\d\.\d{8,}", preview)


def test_run_agents_inside_a_running_event_loop(aou_env):
    """Jupyter kernels already run an event loop; run_agents must work from inside one."""
    import asyncio

    csv = RUN_DIR / "cohort.csv"
    _cohort_table().to_csv(csv, index=False)

    async def notebook_cell():
        return runner.run_agents(csv, runner.goal_for(CohortSpec.from_json(SPEC), "HbA1c"))

    state = asyncio.run(notebook_cell())
    assert state.get("rounds")


def _confounded_table(n: int = 600) -> pd.DataFrame:
    """Body size drives calories, the covariate and the outcome; steps act on the outcome on their own."""
    spec = CohortSpec.from_json(SPEC)
    rng = np.random.default_rng(5)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in output_columns(spec)})
    body, steps = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df["age"] = rng.integers(20, 80, n)
    df["female"] = rng.integers(0, 2, n)
    df["act_steps_mean"] = 7000 + 2400 * steps
    df["act_calories_out_mean"] = 2200 + 300 * body + rng.normal(0, 40, n)
    df["bmi_cov"] = 28 + 4.5 * body + rng.normal(0, 0.6, n)
    df["hba1c"] = 5.7 + 0.30 * body - 0.25 * steps + rng.normal(0, 0.35, n)
    return df


def test_declared_covariate_confounder_is_adjusted_for_and_never_a_candidate(aou_env, monkeypatch):
    # The critic judges that calories track body size and declares the covariate; no leakage exclusion.
    monkeypatch.setitem(SCRIPTS, "critic_agent", [_call(
        "declare_confounders", confounder_columns_csv="bmi_cov",
        reason="body size drives both energy expenditure and the outcome")])
    csv = RUN_DIR / "confounded.csv"
    _confounded_table().to_csv(csv, index=False)
    spec = CohortSpec.from_json(SPEC)
    goal = runner.goal_for(spec, "HbA1c")
    assert "bmi_cov" in goal and "not candidate biomarkers" in goal

    state = runner.run_agents(csv, goal, context_columns=[c.name for c in spec.covariates])
    rounds = state["rounds"]
    first, last = rounds[0], rounds[-1]

    def passing(r):
        return {c["feature"] for c in r["passing_candidates"]}

    # The covariate is adjustment-only: never screened, alone or inside an engineered feature.
    for r in rounds:
        assert not any("bmi_cov" in f for f in passing(r)), passing(r)
    assert not any("bmi_cov" in c["feature"] for c in state["latest_report"]["candidates"])
    # Before adjustment the body-size proxy validates; once the covariate is declared, the loop runs
    # again on the adjusted design and the proxy no longer passes, while the independent signal does.
    assert "act_calories_out_mean" in passing(first)
    assert len(rounds) >= 2 and last["confounders"][-1] == "bmi_cov"
    assert last["confounder_log"][0]["reason"].startswith("body size")
    assert "act_calories_out_mean" not in passing(last)
    assert "act_steps_mean" in passing(last)
    assert 0 < last["adjustment_covariates"]["bmi_cov"]["non_missing_fraction"] <= 1
