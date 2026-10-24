"""The rehearsal gates the exploration: a critical failure stops it before any query, a pass is kept for this
code and these models, and the round limit a rehearsal asks for is honoured."""

from __future__ import annotations

import json
import sys
import types

import pytest

from codas.aou import job, rehearsal


def _result(critical=()):
    checks = [{"check": "the pipeline ran to the end", "passed": not critical, "critical": True},
              {"check": "the planted reactivity was found and replicated on the sealed half", "passed": False,
               "critical": False}]
    return {"key": rehearsal.key(), "checks": checks, "critical_failures": list(critical),
            "warnings": ["the planted reactivity was found and replicated on the sealed half"], "found": [], "defined": []}


def test_a_critical_failure_stops_before_any_query_and_a_pass_is_kept(tmp_path, monkeypatch):
    calls = []

    def fake(out_dir, **kwargs):
        calls.append(out_dir)
        folder = tmp_path / rehearsal.FOLDER
        folder.mkdir(exist_ok=True)
        result = _result(critical=["the pipeline ran to the end (RuntimeError: no model answered)"] if len(calls) == 1 else ())
        (folder / f"rehearsal_{result['key']}.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(rehearsal, "rehearse", fake)
    with pytest.raises(rehearsal.RehearsalError, match="no All of Us query was run"):
        rehearsal.run_once(tmp_path, log=lambda *_: None)
    assert not (tmp_path / rehearsal.FOLDER / f"rehearsal_{rehearsal.key()}.json").exists()  # tried again next time
    passed = rehearsal.run_once(tmp_path, log=lambda *_: None)
    assert not passed["critical_failures"] and len(calls) == 2
    rehearsal.run_once(tmp_path, log=lambda *_: None)                    # kept: not repeated
    assert len(calls) == 2
    latest = json.loads((tmp_path / rehearsal.FOLDER / rehearsal.LATEST).read_text())
    assert latest["warnings"] == passed["warnings"]
    from codas.aou import notebook_views as views

    table = views.rehearsal_table({"summary_path": str(tmp_path / "explore_summary.md")})
    assert list(table["result"]) == ["passed", "not met (a warning)"]


def test_the_job_rehearses_before_exploring_and_stops_on_a_failure(tmp_path, monkeypatch):
    from codas.aou import runner
    import codas.aou.cohort as cohort

    fake_bq = types.ModuleType("google.cloud.bigquery")
    fake_bq.Client = lambda **_: object()
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.bigquery = fake_bq
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    monkeypatch.setattr(cohort.CohortSpec, "from_json", classmethod(lambda cls, path: "spec"))
    monkeypatch.setenv("CODAS_REHEARSAL", "1")
    order = []

    def failing(out, **kwargs):
        order.append("rehearsal")
        raise rehearsal.RehearsalError("the rehearsal on synthetic data failed, so no All of Us query was run: x")

    monkeypatch.setattr(rehearsal, "run_once", failing)
    monkeypatch.setattr(runner, "explore", lambda *a, **k: order.append("explore"))
    out = tmp_path / "aou"
    stdout, stderr = sys.stdout, sys.stderr
    try:
        assert job.main([json.dumps({"catalog": "c", "spec": "s", "cdr": "p.d", "out_dir": str(out)})]) == 1
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    assert order == ["rehearsal"] and "no All of Us query was run" in job.status(out)["error"]


def test_a_run_can_limit_its_rounds():
    from codas.agents import tools

    context = types.SimpleNamespace(state={"rounds": [{"validated_count": 3, "ml_above_chance": True}],
                                           "round_limit": 1}, actions=types.SimpleNamespace(escalate=False))
    out = tools.check_convergence(context)
    assert out["converged"] and context.actions.escalate and "limit of 1" in out["reason"]
    context.state.pop("round_limit")
    context.actions.escalate = False
    assert not tools.check_convergence(context)["converged"]           # without a limit, one round is not enough
