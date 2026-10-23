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


def test_heartbeat_and_environment_snapshot(tmp_path):
    beats = tmp_path / "beats.jsonl"
    hb = job._Heartbeat(beats, period=0.05).start()
    hb.now.update(outcome="Some outcome", agent="scout_agent")
    time.sleep(0.3)
    hb.stop()
    rows = [json.loads(line) for line in beats.read_text().splitlines()]
    assert len(rows) >= 2 and rows[-1]["outcome"] == "Some outcome" and rows[-1]["agent"] == "scout_agent"
    assert {"elapsed_min", "rss_gb", "memory_available_gb", "load_1min"} <= set(rows[-1])
    env = job.environment(tmp_path)
    assert env["cpus"] and len(env["analysis_code"]) == 12 and "packages" in env
    assert all(k.startswith(("GOOGLE_CLOUD", "CODAS_")) for k in env["settings"])  # never secrets


def test_the_notebook_stops_when_the_page_is_an_older_copy(tmp_path):
    """Cell 1's guard, run against a repository whose committed notebook carries another version."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cells = json.loads((root / "notebooks/aou_codas_quickstart.ipynb").read_text())["cells"]
    setup = next("".join(c["source"]) for c in cells if "def check_this_page_is_current" in "".join(c["source"]))
    stamped = __import__("re").search(r'NOTEBOOK_VERSION = "([0-9a-f]+)"', setup).group(1)
    assert stamped != "__NOTEBOOK_VERSION__"
    definitions = setup.split("\nupdate_checkout()\n")[0]
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "notebooks/aou_codas_quickstart.ipynb").write_text(json.dumps({"cells": [{"source": setup}]}))
    for cmd in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    ns: dict = {}
    exec(definitions.replace("assert (REPO", "# assert (REPO"), ns)
    ns["REPO"] = repo
    ns["check_this_page_is_current"]()          # the page matches the checkout: carries on
    ns["NOTEBOOK_VERSION"] = "0123456789ab"     # an older page
    with pytest.raises(SystemExit, match="older copy of the notebook"):
        ns["check_this_page_is_current"]()


def test_the_job_hides_the_known_harmless_library_notices():
    import re

    from codas.aou import job

    notice = "grpcio < 1.83.0 does not support Post-Quantum Cryptography (PQC). Support for non-PQC ..."
    assert any(re.match(p, notice) for p in job.QUIET_WARNINGS)
    assert not any(re.match(p, "ValueError: something real") for p in job.QUIET_WARNINGS)


def _collapsed(lines, **kwargs):
    import io

    sink = io.StringIO()
    stream = job._Masked(sink, **kwargs)
    for line in lines:
        stream.write(line + "\n")
    stream.close()
    return sink.getvalue().splitlines()


def test_a_flood_of_one_line_becomes_a_few_lines_and_a_count():
    flood = [f"12:{i // 600 % 60:02d}:{i // 10 % 60:02d}  WARNING  Warning: there are non-text parts in the response"
             for i in range(10_000)]
    # a warning and the source line under it, alternating, as Python prints them
    pair = ["/x/site-packages/lib.py:12: ConvergenceWarning: did not converge", "  warnings.warn("] * 3000
    written = _collapsed(["12:00:00  environment: ok", *flood, "12:59:59  === Next outcome ===", *pair,
                          "13:00:00  exploration finished"], summary_every=5000)
    assert len(written) < 30, written
    assert written[0].endswith("environment: ok") and "=== Next outcome ===" in "\n".join(written)
    assert written[-1].startswith("[repeated") or "exploration finished" in "\n".join(written)
    counted = {}
    for line in written:
        hit = job._REPEAT.match(line)
        if hit:
            counted[hit.group("text")] = counted.get(hit.group("text"), 0) + int(hit.group("n"))
    shown = sum(line.endswith("non-text parts in the response") for line in written if not job._REPEAT.match(line))
    assert shown + counted["WARNING  Warning: there are non-text parts in the response"] == 10_000
    assert counted["  warnings.warn("] + 3 == 3000


def test_progress_lines_and_rare_repeats_are_always_written():
    rounds = []
    for outcome in range(6):
        rounds.append(f"10:00:00  === Outcome {outcome} ===")
        for r in range(5):
            rounds += ["10:00:01    search_agent             -> run_discovery_round()"] + [f"10:00:02  other {outcome} {r} {k}"
                                                                                           for k in range(60)]
    written = _collapsed(rounds)
    assert sum("run_discovery_round" in line for line in written) == 30  # spread out: never collapsed
    assert written[1] == "10:00:01    search_agent             -> run_discovery_round()"  # indentation kept


def test_warnings_show_once_even_when_other_code_resets_the_registry():
    import io
    import subprocess

    code = r'''
import sys, warnings
from codas.aou import job
job.quiet_warnings()
def noisy():
    warnings.warn("did not converge", UserWarning)
for _ in range(50):
    noisy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("ignored here", UserWarning)
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stderr.count("did not converge") == 1, out.stderr
    without = subprocess.run([sys.executable, "-c", code.replace("job.quiet_warnings()", "")],
                             capture_output=True, text=True, check=True)
    assert without.stderr.count("did not converge") == 50  # the reason: each catch_warnings resets the registry


def test_quiet_follow_prints_milestones_and_updates_one_status_block(tmp_path):
    out = tmp_path / "aou"
    script = (
        "import sys, time\n"
        "from codas.aou import job\n"
        "job._say('environment: commit abc', milestone=True)\n"
        "for i in range(400):\n"
        "    job._say(f'  search_agent is working {i}')\n"
        "job._say('\\n=== Some outcome ===', milestone=True)\n"
        "job._say('  finished: 3 rounds', milestone=True)\n"
        "time.sleep(0.5)\n")
    command = [sys.executable, "-c", script]
    printed, blocks = [], []
    with pytest.raises(RuntimeError, match="stopped"):
        job.run_or_attach({"top_k": 1}, out, poll_s=0.1, log=printed.append, command=command, quiet=True,
                          live=blocks.append)
    text = "\n".join(printed)
    assert "environment: commit abc" in text and "=== Some outcome ===" in text and "finished: 3 rounds" in text
    assert "search_agent is working" not in text and len(printed) < 10
    assert blocks and "search_agent is working 399" in blocks[-1] and len(blocks[-1].splitlines()) <= 14


def test_log_digest_counts_the_most_repeated_lines(tmp_path):
    log = tmp_path / "explore.log"
    log.write_text("\n".join(["09:00:00  start"] + [f"09:{i:02d}:00  WARNING  same thing" for i in range(40)]
                             + ["[repeated 500 more times, 09:40:00-10:00:00] WARNING  same thing",
                                "10:00:01  other", "10:00:02  other"]) + "\n")
    rows = job.log_digest(log, top=2)
    assert rows[0] == {"line": "WARNING  same thing", "count": 540, "first": "09:00:00", "last": "10:00:00"}
    assert rows[1]["line"] == "other" and rows[1]["count"] == 2
    assert job.log_digest(tmp_path / "missing.log") == []


def test_a_new_job_keeps_the_earlier_logs(tmp_path):
    out = tmp_path / "aou"
    fake = [sys.executable, "-c", "print('first run', flush=True)"]
    job.start({"top_k": 1}, out, command=fake)
    _wait_until_stopped(out)
    job.start({"top_k": 1}, out, command=[sys.executable, "-c", "print('second run', flush=True)"])
    _wait_until_stopped(out)
    earlier = job.earlier_logs(out)
    assert len(earlier) == 1 and "first run" in earlier[0].read_text()


def test_a_job_that_floods_its_output_keeps_a_short_log(tmp_path):
    out = tmp_path / "aou"
    script = (
        "import logging, warnings\n"
        "from codas.aou import job\n"
        "job._capture_output()\n"
        "model_log = logging.getLogger('some.library')\n"
        "for i in range(2000):\n"
        "    model_log.warning(\"Warning: there are non-text parts in the response: ['function_call']\")\n"
        "    warnings.warn('did not converge', UserWarning)\n"
        "    with warnings.catch_warnings():\n"
        "        warnings.simplefilter('ignore')\n"
        "job._say('exploration finished', milestone=True)\n"
        "job._close_output()\n")
    job.start({"top_k": 1}, out, command=[sys.executable, "-c", script])
    st = _wait_until_stopped(out)
    lines = open(st["log_path"]).read().splitlines()
    assert len(lines) < 20, lines
    assert sum("did not converge" in line for line in lines) == 1
    top = job.log_digest(st["log_path"], top=1)[0]
    assert top["count"] == 2000 and "non-text parts" in top["line"]


def test_the_status_cell_shows_what_flooded_a_log(tmp_path):
    from codas.aou import notebook_views as views

    log = tmp_path / "explore.log"
    log.write_text("\n".join(["09:00:00  start", "09:00:01  rare"] + ["09:00:02  WARNING  flood"] * 50) + "\n")
    table = views.log_digest_table(log)
    assert list(table["times"]) == [50] and table.iloc[0]["log line (time stamp removed)"] == "WARNING  flood"
    assert views.log_digest_table(tmp_path / "none.log") is None


def test_stop_ends_a_running_job_and_records_it(tmp_path):
    out = tmp_path / "aou"
    job.start({"top_k": 1}, out, command=[sys.executable, "-c", "import time; time.sleep(60)"])
    assert job.status(out)["state"] == "running"
    st = job.stop(out)
    assert st["state"] == "stopped" and "stopped by user" in st.get("error", "")
    assert job.stop(out)["state"] == "stopped"  # stopping again is harmless
