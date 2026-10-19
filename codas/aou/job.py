"""Run the free exploration as a background job that outlives the notebook page.

A run over several outcomes takes hours. In a notebook cell it lives only as long as the browser
keeps the connection: after a reload or a sleeping laptop the cell stops showing output, and nobody
can tell whether the run is progressing, finished or dead. Here the exploration runs in its own
process, writes a timestamped log, and records its state (running, finished, failed) on disk:

* ``run_or_attach`` starts the job, or, when one is already running, attaches to it instead of
  starting a second, then follows its log until it ends. Re-running the cell after a reload is safe.
* ``status`` reports the state and the end of the log from any kernel.

The log holds what the notebook used to print (agent turns, tool names, outcome summaries), masked
for counts of 1-20 and for long digit runs (identifiers that a database error might quote). It is
written next to the mirrored reports (``codas_reports/aou/explore.log``) so JupyterLab can open it.
The job inherits the notebook's environment (Vertex project, location, models, CDR).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from codas.aou.cohort import DEFAULT_OUT_DIR, ROOT
from codas.aou.privacy import mask_identifiers

STATE_NAME = "explore_job.json"
RESULT_NAME = "explore_result.json"
LOG_NAME = "explore.log"
ENV_NAME = "explore_env.json"
HEARTBEAT_NAME = "explore_heartbeat.jsonl"
HEARTBEAT_S = float(os.getenv("CODAS_HEARTBEAT_S", "60"))
# Settings worth recording for debugging; nothing else from the environment is written.
_SETTINGS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "CODAS_AOU_CDR", "CODAS_GEMINI_REASONING_MODEL",
             "CODAS_GEMINI_FAST_MODEL", "CODAS_GEMINI_FALLBACK_MODELS", "CODAS_MODEL_TIMEOUT_S",
             "CODAS_MODEL_TIMEOUT_RETRIES", "CODAS_OUTCOME_TIMEOUT_MIN", "CODAS_MODEL_RETRY_ATTEMPTS")


def _meminfo() -> dict[str, float]:
    """Total and available memory of the machine, in GB (Linux)."""
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                out[key] = round(int(value.split()[0]) / 1024 / 1024, 2)
    except OSError:
        pass
    return out


def _rss_gb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024 / 1024, 3)
    except OSError:
        pass
    return None


def environment(out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    """What the run is running on: code versions, packages, machine size and the run's settings."""
    import platform
    import shutil
    from importlib import metadata

    from codas.aou import runner

    packages = {}
    for name in ("google-adk", "google-genai", "google-cloud-bigquery", "pandas", "numpy", "scikit-learn", "scipy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    mem = _meminfo()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return {"commit": runner._code_version()[:12], "analysis_code": runner._analysis_version()[:12],
            "python": platform.python_version(), "packages": packages, "cpus": os.cpu_count(),
            "memory_total_gb": mem.get("MemTotal"), "memory_available_gb": mem.get("MemAvailable"),
            "disk_free_gb": round(shutil.disk_usage(out).free / 1e9, 1),
            "settings": {k: os.environ[k] for k in _SETTINGS if os.environ.get(k)}}


class _Heartbeat:
    """Writes a status line every HEARTBEAT_S seconds: how long the run has gone, what it is doing, and
    the memory it uses. A log that just stops, with no error, means the process or machine was stopped."""

    def __init__(self, path: Path, period: float = HEARTBEAT_S):
        import threading

        self.path, self.period, self.started = path, period, time.time()
        self.now = {"outcome": None, "agent": None, "tool": None}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_Heartbeat":
        self.path.write_text("", encoding="utf-8")
        self._thread.start()
        return self

    def beat(self) -> dict[str, Any]:
        mem = _meminfo()
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_min": round((time.time() - self.started) / 60, 1),
                  **self.now, "rss_gb": _rss_gb(), "memory_available_gb": mem.get("MemAvailable"),
                  "load_1min": None if load is None else round(load, 2)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            r = self.beat()
            _say(f"heartbeat: {r['elapsed_min']} min, outcome {r['outcome']}, agent {r['agent']}, "
                 f"process memory {r['rss_gb']} GB, machine memory free {r['memory_available_gb']} GB, load {r['load_1min']}")

    def stop(self) -> None:
        self._stop.set()


def _paths(out_dir: str | Path) -> tuple[Path, Path, Path]:
    from codas.aou.runner import reports_dir

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_dir = reports_dir(out)
    log_dir.mkdir(parents=True, exist_ok=True)
    return out / STATE_NAME, log_dir / LOG_NAME, out / RESULT_NAME


def _alive(pid: int) -> bool:
    """True while ``pid`` runs; a finished child the kernel has not reaped yet (a zombie) counts as ended."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def status(out_dir: str | Path = DEFAULT_OUT_DIR, tail: int = 20) -> dict[str, Any]:
    """The job's state (none, running, finished, failed, or stopped if its process died), when it
    started, and the last ``tail`` lines of its log."""
    state_path, log_path, result_path = _paths(out_dir)
    state = _read(state_path) or {"state": "none"}
    if state.get("state") == "running" and not _alive(int(state.get("pid") or 0)):
        state["state"] = "stopped"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
    state["log_path"], state["log_tail"] = str(log_path), lines[-tail:] if tail else []
    beats = log_path.with_name(HEARTBEAT_NAME)
    if beats.exists():
        rows = [json.loads(line) for line in beats.read_text(encoding="utf-8").splitlines() if line.strip()]
        state["last_heartbeat"] = rows[-1] if rows else None
    if state["state"] == "finished":
        state["result"] = _read(result_path)
    return state


def start(args: dict[str, Any], out_dir: str | Path = DEFAULT_OUT_DIR, *,
          command: list[str] | None = None) -> dict[str, Any]:
    """Start the exploration in a new process (unless one is running) and return its state.
    ``command`` replaces the job process (for tests)."""
    current = status(out_dir)
    if current["state"] == "running":
        return current
    state_path, log_path, result_path = _paths(out_dir)
    if log_path.exists():
        log_path.replace(log_path.with_suffix(".previous.log"))
    result_path.unlink(missing_ok=True)
    payload = {**args, "out_dir": str(Path(out_dir).expanduser().resolve())}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command or [sys.executable, "-m", "codas.aou.job", json.dumps(payload)], cwd=str(ROOT),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=dict(os.environ))
    state = {"state": "running", "pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": args}
    _write(state_path, state)
    return status(out_dir)


def follow(out_dir: str | Path = DEFAULT_OUT_DIR, poll_s: float = 10.0, log: Any = print) -> dict[str, Any]:
    """Print the job's log as it grows until the job is no longer running; return its final status."""
    _, log_path, _ = _paths(out_dir)
    printed = 0
    while True:
        st = status(out_dir, tail=0)
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[printed:]:
                log(line)
            printed = len(lines)
        if st["state"] != "running":
            return status(out_dir)
        time.sleep(poll_s)


def run_or_attach(args: dict[str, Any], out_dir: str | Path = DEFAULT_OUT_DIR, poll_s: float = 10.0,
                  log: Any = print, *, command: list[str] | None = None) -> dict[str, Any]:
    """Start the exploration (or attach to the one already running), follow it to the end, and return
    the exploration's result. Raises if the job failed or its process died."""
    st = status(out_dir)
    if st["state"] == "running":
        log(f"An exploration has been running since {st.get('started')}; following it (no second run started).")
    else:
        st = start(args, out_dir, command=command)
        log(f"Exploration started in the background at {st.get('started')}. It keeps running if this page "
            f"loses its connection; run this cell again to follow it. Log: {st['log_path']}")
    final = follow(out_dir, poll_s=poll_s, log=log)
    if final["state"] == "finished":
        return final["result"]
    raise RuntimeError(f"The exploration {final['state']}: {final.get('error') or 'its process ended unexpectedly'}. "
                       "Run the cell again to resume; finished outcomes are reused.")


class _Masked:
    """A text stream that masks counts of 1-20 and identifiers before writing each line."""

    def __init__(self, stream: Any):
        self.stream, self.buffer = stream, ""

    def write(self, text: str) -> int:
        self.buffer += text
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            self.stream.write(mask_identifiers(line) + "\n")
        self.stream.flush()
        return len(text)

    def flush(self) -> None:
        if self.buffer:
            self.stream.write(mask_identifiers(self.buffer))
            self.buffer = ""
        self.stream.flush()


def _say(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


def main(argv: list[str]) -> int:
    """The job process: run the exploration and record how it ended."""
    sys.stdout, sys.stderr = _Masked(sys.stdout), _Masked(sys.stderr)
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    args = json.loads(argv[0])
    out = Path(args["out_dir"])
    state_path, log_path, result_path = _paths(out)
    ending: dict[str, Any] = {}
    heartbeat = _Heartbeat(log_path.with_name(HEARTBEAT_NAME))
    try:
        env = environment(out)
        _write(log_path.with_name(ENV_NAME), env)
        _say(f"environment: commit {env['commit']}, analysis code {env['analysis_code']}, python {env['python']}, "
             f"{env['cpus']} CPUs, memory {env['memory_available_gb']} of {env['memory_total_gb']} GB free, "
             f"disk {env['disk_free_gb']} GB free")
        _say(f"settings: {json.dumps(env['settings'])}")
        heartbeat.start()

        def on_log(message: str) -> None:
            text = str(message)
            if text.strip().startswith("=== ") and text.strip().endswith(" ==="):
                heartbeat.now.update(outcome=text.strip()[4:-4], agent=None, tool=None)
            _say(text.strip("\n") if text.strip() else text)

        def on_agent(agent: str) -> None:
            heartbeat.now.update(agent=agent, tool=None)
            _say(f"  {agent} is working")

        def on_tool(agent: str, tool: str) -> None:
            heartbeat.now.update(tool=tool)
            _say(f"    {agent:24s} -> {tool}()")

        from google.cloud import bigquery

        from codas.aou import runner
        from codas.aou.cohort import CohortSpec

        client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
        result = runner.explore(
            args["catalog"], CohortSpec.from_json(args["spec"]), args.get("cdr"), client=client,
            top_k=int(args.get("top_k", 3)), outcome_names=args.get("outcomes") or None, out_dir=out, log=on_log,
            on_tool_call=on_tool, on_agent=on_agent)
        _write(result_path, result)
        ending = {"state": "finished", "summary_path": result["summary_path"]}
        _say("exploration finished")
        return 0
    except BaseException as exc:  # noqa: BLE001 - record any ending, including an interrupt
        traceback.print_exc()
        ending = {"state": "failed", "error": mask_identifiers(f"{type(exc).__name__}: {str(exc)[:300]}")}
        return 1
    finally:
        heartbeat.stop()
        # Re-read the state: the launcher may have written it after this process started.
        _write(state_path, {**_read(state_path), **ending, "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
