"""The exploration as a background job: start, attach instead of a second start, follow, and record
how it ended, even when the process dies or the page that started it goes away."""

import json
import sys
import time

import pytest

from codas.aou import job


def _wait_until_stopped(out, timeout=20):
    deadline = time.time() + timeout
    while job.status(out)["state"] == "running" and time.time() < deadline:
        time.sleep(0.2)
    return job.status(out)


def test_start_attaches_to_a_running_job_and_notices_when_it_dies(tmp_path):
    out = tmp_path / "aou"
    fake = [sys.executable, "-c", "import time; print('hello from the job', flush=True); time.sleep(2)"]
    first = job.start({"top_k": 1}, out, command=fake)
    assert first["state"] == "running"
    again = job.start({"top_k": 1}, out, command=fake)
    assert again["pid"] == first["pid"]  # attached; no second process
    final = _wait_until_stopped(out)
    assert final["state"] == "stopped"  # it ended without recording an outcome: a crash, not a finish
    assert "hello from the job" in "\n".join(final["log_tail"])
    printed = []
    with pytest.raises(RuntimeError, match="stopped"):
        job.run_or_attach({"top_k": 1}, out, poll_s=0.1, log=printed.append, command=fake)
    # run_or_attach started a fresh job (the old one was not running) and followed it to its end.
    assert any("started in the background" in line for line in printed)


def test_the_job_records_its_result_or_its_masked_error(tmp_path, monkeypatch):
    import types

    import codas.aou.cohort as cohort
    from codas.aou import runner

    out = tmp_path / "aou"
    fake_bq = types.ModuleType("google.cloud.bigquery")
    fake_bq.Client = lambda **_: object()
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.bigquery = fake_bq
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    args = json.dumps({"catalog": "c.json", "spec": "s.json", "cdr": "p.d", "top_k": 1, "out_dir": str(out)})

    monkeypatch.setattr(cohort.CohortSpec, "from_json", classmethod(lambda cls, path: "spec"))
    seen = {}

    def fake_explore(catalog, base, cdr, **kwargs):
        seen.update(base=base, cdr=cdr, top_k=kwargs["top_k"])
        kwargs["on_agent"]("scout_agent")
        return {"summary_path": str(out / "explore_summary.md"), "runs": [], "chosen": [], "table": ""}

    stdout, stderr = sys.stdout, sys.stderr
    try:
        monkeypatch.setattr(runner, "explore", fake_explore)
        assert job.main([args]) == 0
        assert job.status(out)["state"] == "finished" and job.status(out)["result"]["runs"] == []
        assert seen == {"base": "spec", "cdr": "p.d", "top_k": 1}

        def failing(*_a, **_k):
            raise RuntimeError("BigQuery said: int64 overflow: 200001110695 * 2654435761")

        monkeypatch.setattr(runner, "explore", failing)
        assert job.main([args]) == 1
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    st = job.status(out)
    assert st["state"] == "failed" and "200001110695" not in st["error"] and "<id>" in st["error"]
