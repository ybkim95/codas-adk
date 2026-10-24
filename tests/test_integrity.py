"""Guards for the scientific-integrity rules, run on every milestone.

* Outcome-permutation invariance: until the first discovery round closes registration, nothing the agents
  receive may depend on the outcome. Permuting the outcome within age x sex strata must leave every model
  request byte-identical, and with the barrier switched off the same test must detect the leak.
* Sealed invariance: the agents never open the sealed share or an all-participant cache, whatever path they
  pass.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codas.agents import agent as agent_module
from codas.agents import callbacks, tools
from codas.aou import runner
from codas.aou.cohort import holdout_path
from codas.core import eda

from test_aou_e2e import (RECORDS, RUN_DIR, SCRIPTS, ScriptedLlm, _call, _cohort_table, _llm_agents,  # noqa: F401
                          _serialize, _steps_hypothesis, _window_spec, aou_env)

# Every pre-registration tool, as a hypotheses agent might use them before registering.
_PRE_REGISTRATION = [
    _call("search_literature", query="daily steps and HbA1c"),
    _call("propose_feature", operation="ratio", feature_a="act_steps_mean", feature_b="act_days"),
    _call("declare_confounders", confounder_columns_csv="age", reason="ages both"),
    _call("exclude_features", columns_csv="act_calories_out_mean", reason="derived from the outcome here"),
    _call("register_hypotheses", hypotheses_json=json.dumps([_steps_hypothesis()])),
]


class _Stop(Exception):
    """Raised when the first discovery round would run: registration ends there."""


class _StopAtSearch(ScriptedLlm):
    async def generate_content_async(self, llm_request, stream=False):
        RECORDS.append((self.agent_name, llm_request))
        raise _Stop()
        yield  # pragma: no cover - makes this an async generator like the base class


def _pre_registration_requests(table: pd.DataFrame, name: str) -> list[str]:
    RECORDS.clear()
    csv = RUN_DIR / f"{name}.csv"
    table.to_csv(csv, index=False)
    with pytest.raises(_Stop):
        runner.run_agents(csv, runner.goal_for(_window_spec(), "HbA1c"))
    return [f"{agent}\n{_serialize(req)}".replace(name, "<dataset>") for agent, req in RECORDS]


def _permuted_outcome(table: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """The outcome shuffled within the strata the data dossier summarises it by (age band x sex)."""
    rng = np.random.default_rng(seed)
    out = table.copy()
    for _, idx in out.groupby(eda.strata(out)).groups.items():
        out.loc[idx, "hba1c"] = rng.permutation(out.loc[idx, "hba1c"].to_numpy())
    return out


def test_nothing_the_agents_see_before_registration_depends_on_the_outcome(aou_env, monkeypatch):
    monkeypatch.setitem(SCRIPTS, "hypotheses_agent", _PRE_REGISTRATION)
    search = next(a for a in _llm_agents(agent_module.root_agent) if a.name == "search_agent")
    # A plain assignment: aou_env puts every agent's real model back. monkeypatch would restore the scripted
    # stand-in after aou_env's teardown and leak it into later tests.
    search.model = _StopAtSearch(model="scripted", agent_name="search_agent")
    table = _cohort_table()
    permuted = _permuted_outcome(table)
    assert not np.allclose(table["hba1c"], permuted["hba1c"])
    original = _pre_registration_requests(table, "original")
    assert any("propose_feature" in r or "withheld" in r for r in original)  # the leaky tool was exercised
    assert any("Candidate features:" in r for r in original)  # the data dossier reached the designer
    assert original == _pre_registration_requests(permuted, "permuted")
    # The guard is only worth having if it can fail: without the barrier, the association leaks.
    monkeypatch.setattr(tools, "registration_open", lambda state: False)
    assert _pre_registration_requests(table, "original") != _pre_registration_requests(permuted, "permuted")


def test_agents_never_open_the_sealed_share(aou_env, monkeypatch):
    table = _cohort_table()
    csv = RUN_DIR / "cohort.csv"
    table.to_csv(csv, index=False)
    sealed = holdout_path(csv)
    sealed.parent.mkdir(parents=True, exist_ok=True)
    table.sample(frac=1.0, random_state=1).to_csv(sealed, index=False)
    opened: list[str] = []
    real_open, real_read = builtins.open, pd.read_csv
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    monkeypatch.setattr(pd, "read_csv", lambda f, *a, **k: (opened.append(str(f)), real_read(f, *a, **k))[1])
    # An agent that names the sealed file directly gets the discovery table, never the sealed one.
    monkeypatch.setitem(SCRIPTS, "scout_agent", [_call("profile_dataset", csv_path=str(sealed)),
                                                 _call("preview_columns", csv_path=str(sealed)),
                                                 _call("set_target", target_column="hba1c")])
    state = runner.run_agents(csv, runner.goal_for(_window_spec(), "HbA1c"))
    assert state.get("rounds")
    touched = [p for p in opened if "sealed" in Path(p).parts or p.endswith(".holdout.csv")]
    assert touched == [], touched


def test_the_guardrail_refuses_sealed_paths_and_aou_tools_ignore_agent_paths(monkeypatch, tmp_path):
    root = callbacks.ALLOWED_DATA_ROOTS[0]
    assert callbacks._is_allowed_path(str(root / "aou" / "explore_x.csv"))
    assert not callbacks._is_allowed_path(str(root / "aou" / "sealed" / "explore_x.holdout.csv"))
    assert not callbacks._is_allowed_path(str(root / "aou" / "sealed" / "phenotype_cache" / "p.csv"))
    assert not callbacks._is_allowed_path(str(root / "aou" / "explore_x.holdout.csv"))
    monkeypatch.setenv("CODAS_AOU_MODE", "1")
    assert tools._resolve_csv_path("/data/discovery.csv", "/data/sealed/x.holdout.csv") == Path("/data/discovery.csv")
    monkeypatch.delenv("CODAS_AOU_MODE")
    assert tools._resolve_csv_path("/data/discovery.csv", "/data/other.csv") == Path("/data/other.csv")


def test_propose_feature_withholds_the_outcome_association_until_registration_closes(tmp_path):
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=200), "b": rng.uniform(1, 2, 200)})
    frame["y"] = frame.a + rng.normal(size=200)
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    ctx = SimpleNamespace(state={"csv_path": str(path), "target_column": "y"})
    before = tools.propose_feature("ratio", "a", "b", ctx)
    assert before["exploratory_association"] is None and "withheld" in before["note"]
    ctx.state["rounds"] = [{"round": 0}]
    after = tools.propose_feature("ratio", "a", "b", ctx, name="a_over_b_again")
    assert after["exploratory_association"]["spearman_rho"] is not None
