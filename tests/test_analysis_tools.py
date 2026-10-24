"""The analysis tools as the agents call them: they read the shared design from memory, rebuild candidate
features, refuse columns the design blocks, record each analysis, and in AoU mode return masked
aggregates only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.agents import tools


def _ctx(tmp_path, survival=False, seed=0):
    rng = np.random.default_rng(seed)
    n = 1500
    f = pd.DataFrame({"age": rng.normal(50, 10, n), "female": rng.integers(0, 2, n), "a": rng.uniform(1, 2, n),
                      "b": rng.uniform(1, 2, n), "noise": rng.normal(size=n), "leak": rng.normal(size=n),
                      "cov": rng.normal(size=n)})
    f.loc[rng.random(n) < 0.4, "cov"] = np.nan
    if survival:
        t = rng.exponential(1 / (0.1 * np.exp(1.2 * f.a / f.b)))
        c = rng.uniform(0, 15, n)
        f["fu"], f["y"] = np.minimum(t, c), (t <= c).astype(int)
    else:
        f["y"] = np.where(f.female == 1, 1.0, 0.2) * (f.a / f.b) + rng.normal(scale=0.3, size=n)
    csv = tmp_path / "d.csv"
    f.to_csv(csv, index=False)
    state = {"csv_path": str(csv), "target_column": "y", "confounder_columns": ["age"], "excluded_columns": ["leak"],
             "context_columns": ["cov", "fu"] if survival else ["cov"], "covariate_imputations": 3,
             "survival_time_column": "fu" if survival else "",
             "latest_report": {"candidates": [{"feature": "a_over_b", "components": ["a", "b"], "verdict": "validated",
                                               "rho": 0.4}, {"feature": "noise", "components": ["noise"],
                                                             "verdict": "conditional", "rho": 0.05}]}}
    return SimpleNamespace(state=state, actions=SimpleNamespace(escalate=False))


def test_subgroup_tool_rebuilds_engine_features_and_records_the_analysis(tmp_path):
    ctx = _ctx(tmp_path)
    out = tools.run_subgroup_analysis("a_over_b", "female", ctx)
    assert "error" not in out and out["heterogeneity_p"] < 1e-6
    assert [s["stratum"] for s in out["strata"]] == ["female=0", "female=1"]
    assert ctx.state["analyses"][-1]["tool"] == "subgroup_analysis"


def test_tools_refuse_blocked_or_unknown_columns(tmp_path):
    ctx = _ctx(tmp_path)
    for name in ("leak", "y", "cov", "no_such_feature"):
        assert "error" in tools.run_dose_response(name, ctx), name
    assert "error" in tools.run_subgroup_analysis("a_over_b", "leak", ctx)


def test_multivariable_tool_defaults_to_the_declared_confounders_and_candidates(tmp_path):
    ctx = _ctx(tmp_path)
    out = tools.run_multivariable_model(ctx)
    assert [c["term"] for c in out["coefficients"]] == ["a_over_b", "noise", "age"]
    assert out["incremental"]["delta"] > 0.05 and out["incremental"]["base_terms"] == ["age"]
    out2 = tools.run_multivariable_model(ctx, features_csv="a_over_b", covariates_csv="age,cov")
    assert [c["term"] for c in out2["coefficients"]] == ["a_over_b", "age", "cov"] and out2["n"] == 1500


def test_survival_tools_use_the_follow_up_time(tmp_path):
    ctx = _ctx(tmp_path, survival=True)
    screen = tools.run_survival_screen(ctx)
    assert screen["results"][0]["feature"] == "a_over_b" and screen["results"][0]["hr_per_sd"] > 1.2
    curve = tools.run_dose_response("a_over_b", ctx)
    assert curve["model"] == "cox"
    nonlinear = tools.run_nonlinear_model(ctx, features_csv="a_over_b,noise")
    assert nonlinear["note"].startswith("time to event")


def test_tool_results_are_masked_in_aou_mode(tmp_path, monkeypatch):
    from codas.agents.callbacks import after_tool_logger

    monkeypatch.setenv("CODAS_AOU_MODE", "1")
    ctx = _ctx(tmp_path)
    ctx.state["confounder_columns"] = ["age"]
    out = tools.run_subgroup_analysis("noise", "age", ctx)
    tiny = {"strata": [{"stratum": "x", "n": 7, "rho": 0.1}]}
    masked = after_tool_logger(SimpleNamespace(name="run_subgroup_analysis"), {}, ctx, tiny)
    assert masked["strata"][0]["n"] != 7
    assert "error" not in out


def test_register_hypotheses_validates_and_stores_a_priori_hypotheses(tmp_path):
    import json

    ctx = _ctx(tmp_path)
    tools.propose_feature("ratio", "a", "b", ctx, name="a_per_b")
    out = tools.register_hypotheses(json.dumps([
        {"feature": "a_per_b", "direction": "positive", "construct": "relative load", "mechanism": "m",
         "knownness": "emerging", "citation": "doi:1"},
        {"feature": "noise", "direction": -1, "construct": "c", "knownness": "established"},
        {"feature": "leak", "direction": 1, "knownness": "unexplored"},          # excluded column
        {"feature": "nothing_here", "direction": 1, "knownness": "unexplored"},  # not a column or proposal
        {"feature": "a", "direction": "up", "knownness": "unexplored"},          # bad direction
    ]), ctx)
    assert [h["feature"] for h in out["registered"]] == ["a_per_b", "noise"]
    assert len(out["rejected"]) == 3
    stored = ctx.state["a_priori_hypotheses"]
    # No literature evidence in this run, so the knownness that counts is "not assessed"; the label is kept.
    assert [(h["feature"], h["direction"], h["knownness"], h["declared_knownness"]) for h in stored] == [
        ("a_per_b", 1, "not assessed", "emerging"), ("noise", -1, "not assessed", "established")]
    assert "error" in tools.register_hypotheses("not json", ctx)
    # Registration closes once a discovery round has run: the hypotheses must precede the statistics.
    ctx.state["rounds"] = [{"round": 0}]
    assert "error" in tools.register_hypotheses(json.dumps([{"feature": "a", "direction": 1,
                                                             "knownness": "emerging"}]), ctx)


def test_a_repeated_analysis_returns_the_earlier_result_without_recomputing(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    first = tools.run_subgroup_analysis("a_over_b", "female", ctx)
    calls = []
    real = tools._analysis_setup
    monkeypatch.setattr(tools, "_analysis_setup", lambda c: (calls.append(1), real(c))[1])
    again = tools.run_subgroup_analysis("a_over_b", "female", ctx)
    assert calls == [] and again["heterogeneity_p"] == first["heterogeneity_p"] and "Already run" in again["note"]
    assert len(ctx.state["analyses"]) == 1
    ctx.state["confounder_columns"] = ["age", "female"]           # a new adjustment set is a new analysis
    tools.run_subgroup_analysis("a_over_b", "age", ctx)
    tools.run_subgroup_analysis("a_over_b", "female", ctx)
    assert calls == [1, 1] and len(ctx.state["analyses"]) == 3


def test_every_analysis_run_is_counted_for_the_report_writers(tmp_path):
    from codas.agents import prompts

    ctx = _ctx(tmp_path)
    tools.run_subgroup_analysis("a_over_b", "female", ctx)
    tools.run_subgroup_analysis("a_over_b", "age", ctx)
    tools.run_dose_response("a_over_b", ctx)
    assert ctx.state["analyses_counts"] == "subgroup_analysis x2; dose_response x1"
    assert "{analyses_counts?}" in prompts.REPORT and "{analyses_counts?}" in prompts.REPORT_REVIEWER
