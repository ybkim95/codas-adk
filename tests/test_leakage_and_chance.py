"""Agent-declared leakage exclusion, convergence on a revised design, and the chance-claim audit.

All dataset-agnostic: the tests use the bundled synthetic sample and arbitrary column names.
"""

from pathlib import Path
from types import SimpleNamespace

from codas.agents.grounding import chance_claim_conflict
from codas.agents.tools import check_convergence, exclude_features, propose_feature, run_discovery_round, set_target

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_dataset.csv"


def _ctx():
    ctx = SimpleNamespace(state={"csv_path": str(SAMPLE)}, actions=SimpleNamespace(escalate=False))
    set_target("depression_score", ctx, participant_id_column="participant_id")
    return ctx


def test_exclusion_is_recorded_and_withdraws_dependent_proposals():
    ctx = _ctx()
    propose_feature("ratio", "steps_mean", "resting_hr_mean", ctx)
    propose_feature("product", "age", "sleep_duration_mean", ctx)
    out = exclude_features("steps_mean, depression_score, not_a_column", "computed from the outcome", ctx)
    assert out["newly_excluded"] == ["steps_mean"]
    assert set(out["ignored"]) == {"depression_score", "not_a_column"}   # target and unknown names
    assert out["withdrawn_proposals"] == ["steps_mean_ratio_resting_hr_mean"]
    assert [p["name"] for p in ctx.state["proposed_features"]] == ["age_product_sleep_duration_mean"]
    assert ctx.state["design_revision"] == 1
    assert ctx.state["exclusion_log"][0]["reason"] == "computed from the outcome"


def test_exclusion_requires_a_reason():
    assert "error" in exclude_features("steps_mean", "  ", _ctx())


def test_excluded_column_and_its_derivatives_leave_the_next_round(monkeypatch):
    from codas.agents import tools

    monkeypatch.setattr(tools, "_ROUND_RESAMPLES", 50)
    ctx = _ctx()
    run_discovery_round(ctx)
    assert any("steps_mean" in c["feature"] for c in ctx.state["latest_report"]["candidates"])
    exclude_features("steps_mean", "encodes the outcome", ctx)
    second = run_discovery_round(ctx)
    screened = [c["feature"] for c in ctx.state["latest_report"]["candidates"]]
    assert not any("steps_mean" in f for f in screened)
    assert second["excluded_features"] == ["steps_mean"] and second["design_revision"] == 1


def test_convergence_waits_for_a_round_on_the_revised_design():
    ctx = _ctx()
    same = {"validated_count": 2, "ml_metric_name": "r2", "ml_metric_value": 0.1, "ml_above_chance": True}
    ctx.state["rounds"] = [dict(same, round=0, design_revision=0), dict(same, round=1, design_revision=0)]
    assert check_convergence(ctx)["converged"] is True          # saturated on an unchanged design
    ctx = _ctx()
    ctx.state["rounds"] = [dict(same, round=0, design_revision=0), dict(same, round=1, design_revision=0)]
    ctx.state["design_revision"] = 1                             # something was excluded after round 1
    out = check_convergence(ctx)
    assert out["converged"] is False and not ctx.actions.escalate


def test_chance_claim_flagged_only_when_the_engine_disagrees():
    below = {"ml_above_chance": False, "ml_metric_vs_null_p": 0.2}
    claim = "The model exceeds the permutation null expectation, confirming a stable signal."
    assert "did not beat its permutation null" in chance_claim_conflict(claim, below)
    assert "0.2" in chance_claim_conflict(claim, below)
    assert chance_claim_conflict("The model does not exceed the permutation null.", below) is None
    assert chance_claim_conflict("Performance was above chance.", {"ml_above_chance": True}) is None
    assert chance_claim_conflict("Performance was above chance.", {}) is None   # no engine verdict
