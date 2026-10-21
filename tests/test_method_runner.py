"""The discovery loop's method pools as the agents use them: the deterministic runner applies the next
methods to the round's candidates, the gap finder's requests are validated, and the loop stops when the
pools are exhausted or the candidate set has stopped changing (the paper's stop rules)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.agents import tools


def _ctx(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    n = 1200
    f = pd.DataFrame({"age": rng.normal(50, 10, n), "female": rng.integers(0, 2, n), "x": rng.normal(size=n),
                      "z": rng.normal(size=n), "m": rng.normal(size=n), "leak": rng.normal(size=n)})
    f["y"] = 0.5 * f.x + 0.3 * f.m + rng.normal(size=n)
    csv = tmp_path / "d.csv"
    f.to_csv(csv, index=False)
    state = {"csv_path": str(csv), "target_column": "y", "confounder_columns": ["age", "female"],
             "excluded_columns": ["leak"], "context_columns": [], "rounds": [{"round": 0}],
             "latest_report": {"candidates": [{"feature": "x", "components": ["x"], "verdict": "validated", "rho": 0.4},
                                              {"feature": "z", "components": ["z"], "verdict": "conditional", "rho": 0.05}]}}
    return SimpleNamespace(state=state, actions=SimpleNamespace(escalate=False))


def test_batch_runs_requested_then_default_methods_on_the_candidates(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.state["next_methods"] = [{"method": "mediation", "mediator": "m"}]
    delta = tools.run_method_batch(ctx.state, per_pool=1)
    names = [r["tool"] for r in delta["analyses"]]
    assert names == ["method:mediation", "method:gradient_boosting"]
    assert delta["methods_explored"] == ["mediation", "gradient_boosting"] and delta["next_methods"] == []
    assert delta["rounds"][-1]["methods_run"] == ["mediation", "gradient_boosting"]
    mediation = delta["analyses"][0]["result"]
    assert mediation["features"] == ["x", "z"] and mediation["results"][0]["mediator"] == "m"
    assert "method:mediation" in delta["analyses_digest"]
    ctx.state.update(delta)
    second = tools.run_method_batch(ctx.state, per_pool=1)
    assert [r["tool"] for r in second["analyses"][2:]] == ["method:partial_correlation", "method:elastic_net"]


def test_gap_analysis_requests_are_validated(tmp_path):
    ctx = _ctx(tmp_path)
    out = tools.record_gap_analysis(json.dumps({
        "gaps": ["is the x effect confined to one sex?", "is part of it mediated by m?"],
        "next_methods": [{"method": "subgroup_stratified", "by": "female"}, {"method": "mediation", "mediator": "m"},
                         {"method": "mediation", "mediator": "leak"}, {"method": "astrology"}],
        "constraints": ["do not build features on z"]}), ctx)
    assert [m["method"] for m in out["accepted"]] == ["subgroup_stratified", "mediation"]
    assert len(out["rejected"]) == 2
    assert ctx.state["next_methods"] == out["accepted"]
    assert ctx.state["constraints"] == ["do not build features on z"]
    assert ctx.state["gap_analysis"][-1]["gaps"][0].startswith("is the x effect")
    tools.record_gap_analysis(json.dumps({"constraints": ["do not build features on z", "another"]}), ctx)
    assert ctx.state["constraints"] == ["do not build features on z", "another"]  # unioned, never replaced
    assert "error" in tools.record_gap_analysis("[not an object]", ctx)


def _round(i, features, validated=2, metric=0.1, methods_run=("a",)):
    return {"round": i, "validated_count": validated, "ml_metric_name": "r2", "ml_metric_value": metric,
            "passing_candidates": [{"feature": f} for f in features], "methods_run": list(methods_run)}


def test_loop_keeps_going_while_new_methods_run_and_stops_on_the_papers_rules(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.state["method_pool_active"] = True
    ctx.state["methods_explored"] = ["partial_correlation", "gradient_boosting"]
    ctx.state["rounds"] = [_round(0, ["a", "b"]), _round(1, ["a", "c"])]
    out = tools.check_convergence(ctx)
    assert not out["converged"] and not ctx.actions.escalate  # nothing gained, but a new method ran
    ctx.state["rounds"] = [_round(0, ["a", "b"]), _round(1, ["a", "b"]), _round(2, ["a", "b"])]
    out = tools.check_convergence(ctx)
    assert out["converged"] and "candidate set" in out["reason"] and ctx.actions.escalate
    ctx.actions.escalate = False
    from codas.core import methods

    ctx.state["methods_explored"] = [m for pool in methods.available(False).values() for m in pool]
    ctx.state["rounds"] = [_round(0, ["a"]), _round(1, ["b"], validated=5)]
    out = tools.check_convergence(ctx)
    assert out["converged"] and "exhausted" in out["reason"]
