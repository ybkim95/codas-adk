"""Run the free exploration as a background job that outlives the notebook page.

A run over several outcomes takes hours. In a notebook cell it lives only as long as the browser
keeps the connection: after a reload or a sleeping laptop the cell stops showing output, and nobody
can tell whether the run is progressing, finished or dead. Here the exploration runs in its own
process, writes a timestamped log, and records its state (running, finished, failed) on disk:

* ``run_or_attach`` starts the job, or, when one is already running, attaches to it instead of
  starting a second, then follows its log until it ends. Re-running the cell after a reload is safe.
* ``status`` reports the state and the end of the log from any kernel.
* ``log_digest`` lists the lines a log repeated most, to diagnose a run that flooded its log.

The log holds every agent turn and tool call, masked for counts of 1-20 and for long digit runs
(identifiers that a database error might quote). A line that floods the log (a library warning raised in
a loop, a retry message) is written a few times and then counted. The notebook shows only the
milestones (environment, feasibility, each outcome's start and end, failures) and one status block
updated in place; the full log is next to the mirrored reports (``codas_reports/aou/explore.log``) so
JupyterLab can open it, and earlier runs' logs are kept beside it. The job inherits the notebook's
environment (Vertex project, location, models, CDR).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from codas.aou.cohort import DEFAULT_OUT_DIR, ROOT
from codas.aou.privacy import mask_identifiers

STATE_NAME = "explore_job.json"
RESULT_NAME = "explore_result.json"
LOG_NAME = "explore.log"
MILESTONES_NAME = "explore_milestones.log"
MILESTONES_ENV = "CODAS_JOB_MILESTONES"
ENV_NAME = "explore_env.json"
HEARTBEAT_NAME = "explore_heartbeat.jsonl"
HEARTBEAT_S = float(os.getenv("CODAS_HEARTBEAT_S", "60"))
# Settings worth recording for debugging; nothing else from the environment is written.
_SETTINGS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "CODAS_AOU_CDR", "CODAS_GEMINI_REASONING_MODEL",
             "CODAS_GEMINI_FAST_MODEL", "CODAS_GEMINI_LITE_MODEL", "CODAS_GEMINI_FALLBACK_MODELS", "CODAS_MODEL_TIMEOUT_S",
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


def _tail(path: Path, n: int) -> list[str]:
    """The last ``n`` lines of a text file, read from its end."""
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = block = fh.tell()
        data = b""
        while data.count(b"\n") <= n and block > 0:
            step = min(65536, block)
            block -= step
            fh.seek(block)
            data = fh.read(step) + data
    return data.decode("utf-8", errors="replace").splitlines()[-n:] if size else []


def _read_new(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete lines added to ``path`` since byte ``offset``, and the offset after them."""
    if not path.exists():
        return [], offset
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    end = data.rfind(b"\n") + 1
    return data[:end].decode("utf-8", errors="replace").splitlines(), offset + end


def status(out_dir: str | Path = DEFAULT_OUT_DIR, tail: int = 20) -> dict[str, Any]:
    """The job's state (none, running, finished, failed, or stopped if its process died), when it
    started, and the last ``tail`` lines of its log."""
    state_path, log_path, result_path = _paths(out_dir)
    state = _read(state_path) or {"state": "none"}
    if state.get("state") == "running" and not _alive(int(state.get("pid") or 0)):
        state["state"] = "stopped"
    state["log_path"], state["log_tail"] = str(log_path), _tail(log_path, tail) if tail else []
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
    if log_path.exists():  # keep it: an earlier run's log is how its problems get diagnosed
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(log_path.stat().st_mtime))
        archived, k = log_path.with_name(f"explore.{stamp}.log"), 1
        while archived.exists():
            archived, k = log_path.with_name(f"explore.{stamp}-{k}.log"), k + 1
        log_path.replace(archived)
    milestones = log_path.with_name(MILESTONES_NAME)
    milestones.write_text("", encoding="utf-8")
    result_path.unlink(missing_ok=True)
    payload = {**args, "out_dir": str(Path(out_dir).expanduser().resolve())}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command or [sys.executable, "-m", "codas.aou.job", json.dumps(payload)], cwd=str(ROOT),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env={**os.environ, MILESTONES_ENV: str(milestones)})
    state = {"state": "running", "pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": args}
    _write(state_path, state)
    return status(out_dir)


def earlier_logs(out_dir: str | Path = DEFAULT_OUT_DIR) -> list[Path]:
    """Logs of earlier runs (kept when a new run starts), newest first."""
    _, log_path, _ = _paths(out_dir)
    return sorted(log_path.parent.glob("explore.*.log"), key=lambda p: p.stat().st_mtime, reverse=True)


def _live_text(st: dict[str, Any], lines: list[str]) -> str:
    beat = st.get("last_heartbeat") or {}
    head = f"Exploration {st['state']} (started {st.get('started')})"
    if beat:
        head += (f" | {beat.get('elapsed_min')} min | outcome: {beat.get('outcome') or '-'} | agent: "
                 f"{beat.get('agent') or '-'}" + (f" -> {beat['tool']}()" if beat.get("tool") else "")
                 + f" | job memory {beat.get('rss_gb')} GB")
    return "\n".join([head, f"Latest log lines (full log: {st['log_path']}):",
                      *(line if len(line) <= 200 else line[:197] + "..." for line in lines)])


def stop(out_dir: str | Path = DEFAULT_OUT_DIR, wait_s: float = 10.0) -> dict[str, Any]:
    """Stop the running exploration (its whole process group) and record that it was stopped by the user.
    Finished outcomes stay reusable; a later run resumes."""
    import signal

    state_path, _, _ = _paths(out_dir)
    st = status(out_dir, tail=0)
    if st["state"] != "running":
        return st
    pid = int(st.get("pid") or 0)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.time() + wait_s
        while _alive(pid) and time.time() < deadline:
            time.sleep(0.2)
        if not _alive(pid):
            break
    _write(state_path, {**_read(state_path), "state": "stopped", "error": "stopped by user",
                        "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
    return status(out_dir)


def follow(out_dir: str | Path = DEFAULT_OUT_DIR, poll_s: float = 10.0, log: Any = print, *, quiet: bool = False,
           live: Any = None) -> dict[str, Any]:
    """Follow the job until it is no longer running and return its final status.

    By default every log line is printed. With ``quiet`` only the milestones are printed (environment,
    feasibility, each outcome's start and end, failures) and ``live``, when given, is called with a short
    status (state, elapsed time, current outcome and agent, last 10 log lines) on every poll, for a
    display that is updated in place."""
    _, log_path, _ = _paths(out_dir)
    source = log_path.with_name(MILESTONES_NAME) if quiet else log_path
    offset = 0
    while True:
        st = status(out_dir, tail=10 if live else 0)
        lines, offset = _read_new(source, offset)
        for line in lines:
            log(line)
        if live:
            live(_live_text(st, st["log_tail"]))
        if st["state"] != "running":
            lines, offset = _read_new(source, offset)
            for line in lines:
                log(line)
            return status(out_dir)
        time.sleep(poll_s)


def _live_display() -> Any:
    """A callable that replaces one notebook output in place, or None outside a notebook."""
    try:
        from IPython import get_ipython
        from IPython.display import Pretty, display
    except ImportError:
        return None
    if get_ipython() is None:
        return None
    handle = display(Pretty("Exploration status: starting"), display_id=True)
    return None if handle is None else (lambda text: handle.update(Pretty(text)))


def run_or_attach(args: dict[str, Any], out_dir: str | Path = DEFAULT_OUT_DIR, poll_s: float = 10.0,
                  log: Any = print, *, command: list[str] | None = None, quiet: bool = False,
                  live: Any = None) -> dict[str, Any]:
    """Start the exploration (or attach to the one already running), follow it to the end, and return
    the exploration's result. Raises if the job failed or its process died. ``quiet`` prints only the
    milestones and keeps one status block updated in place (in a notebook; ``live`` replaces it)."""
    st = status(out_dir)
    if st["state"] == "running":
        log(f"An exploration has been running since {st.get('started')}; following it (no second run started).")
    else:
        st = start(args, out_dir, command=command)
        log(f"Exploration started in the background at {st.get('started')}. It keeps running if this page "
            f"loses its connection; run this cell again to follow it. Log: {st['log_path']}")
    if quiet and live is None:
        live = _live_display()
    final = follow(out_dir, poll_s=poll_s, log=log, quiet=quiet, live=live)
    if final["state"] == "finished":
        return final["result"]
    raise RuntimeError(f"The exploration {final['state']}: {final.get('error') or 'its process ended unexpectedly'}. "
                       "Run the cell again to resume; finished outcomes are reused.")


_STAMP = re.compile(r"^(\d{2}:\d{2}:\d{2})  ")  # as written by _say and the logging format
_REPEAT = re.compile(r"^\[repeated (?P<n>\d+) more times?(?:, (?P<first>[\d:]+)-(?P<last>[\d:]+))?\] (?P<text>.*)$")


def _split_stamp(line: str) -> tuple[str | None, str]:
    stamp = _STAMP.match(line)
    return (stamp.group(1), line[stamp.end():]) if stamp else (None, line)


def log_digest(path: str | Path, top: int = 10) -> list[dict[str, Any]]:
    """The ``top`` most repeated lines of a log (compared without their time stamps), with how often
    each appeared and when first and last, counting the repeats the log collapsed."""
    path = Path(path)
    if not path.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            hit = _REPEAT.match(line)
            if hit:
                text, n, first, last = hit["text"], int(hit["n"]), hit["first"], hit["last"]
            else:
                first, text = _split_stamp(line)
                n, last = 1, first
            if not text.strip():
                continue
            row = rows.setdefault(text, {"line": text, "count": 0, "first": None, "last": None})
            row["count"] += n
            row["first"] = row["first"] or first
            row["last"] = last or row["last"]
    return sorted(rows.values(), key=lambda r: -r["count"])[:top]


class _Floods:
    """Writes log lines to ``stream``, collapsing floods: a line (compared without its time stamp) that
    already appeared ``repeat_after`` times within the previous ``window`` lines is counted instead of
    written. The count is written as one ``[repeated N more times, first-last] <line>`` when the flood
    ends, every ``summary_every`` repeats, and at the end. Lines repeated further apart (an agent's turn
    each round) are always written. One instance serves stdout and stderr, which share the log."""

    def __init__(self, stream: Any, repeat_after: int = 3, window: int = 20, summary_every: int = 500):
        import threading
        from collections import Counter, deque

        self.stream, self.lock = stream, threading.RLock()  # the heartbeat writes from a thread
        self.repeat_after, self.window, self.summary_every = repeat_after, window, summary_every
        self.recent: Any = deque()
        self.in_window: Any = Counter()
        self.held: dict[str, list[Any]] = {}  # line -> [count, first stamp, last stamp, last line number]
        self.n = 0

    def _summary(self, text: str) -> None:
        count, first, last, _ = self.held.pop(text)
        span = f", {first}-{last}" if first and last else ""
        self.stream.write(f"[repeated {count} more time{'s' if count != 1 else ''}{span}] {text}\n")

    def line(self, line: str) -> None:
        stamp, text = _split_stamp(mask_identifiers(line))
        self.n += 1
        for old in [t for t, h in self.held.items() if self.n - h[3] > self.window]:
            self._summary(old)  # that flood has ended
        if text.strip() and self.in_window[text] >= self.repeat_after:
            held = self.held.setdefault(text, [0, stamp, stamp, self.n])
            held[0], held[2], held[3] = held[0] + 1, stamp or held[2], self.n
            if held[0] >= self.summary_every:
                self._summary(text)
        else:
            if text in self.held:
                self._summary(text)
            self.stream.write((f"{stamp}  {text}" if stamp else text) + "\n")
        self.recent.append(text)
        self.in_window[text] += 1
        if len(self.recent) > self.window:
            gone = self.recent.popleft()
            self.in_window[gone] -= 1
            if not self.in_window[gone]:
                del self.in_window[gone]

    def close(self) -> None:
        for text in list(self.held):
            self._summary(text)
        self.stream.flush()


class _Masked:
    """A text stream that masks counts of 1-20 and identifiers in each line and collapses floods
    (``_Floods``, shared with the process's other output stream when given)."""

    def __init__(self, stream: Any, floods: _Floods | None = None, **settings: int):
        self.floods, self.buffer = floods or _Floods(stream, **settings), ""

    def write(self, text: str) -> int:
        with self.floods.lock:
            self.buffer += text
            *lines, self.buffer = self.buffer.split("\n")
            for line in lines:
                self.floods.line(line)
            self.floods.stream.flush()
        return len(text)

    def flush(self) -> None:
        self.floods.stream.flush()

    def close(self) -> None:
        """Write the last partial line and every pending repeat count."""
        with self.floods.lock:
            if self.buffer:
                self.floods.line(self.buffer)
                self.buffer = ""
            self.floods.close()


# Known, harmless library notices (the notebook filters the same ones); real errors still show.
QUIET_WARNINGS = (".*automatic function calling.*", ".*EXPERIMENTAL.*", ".*Post-Quantum.*", ".*grpcio.*")


def quiet_warnings() -> None:
    """Show each distinct warning (category and message) once per job, and the known harmless notices
    never. Python normally shows a warning once per code location, but every
    ``warnings.catch_warnings()`` block (used by numerical code to silence expected notices) clears that
    record, and the "once" filter's too, so a warning raised in a loop would be printed on every pass.
    The record kept here is not cleared: ``catch_warnings`` restores this display function on exit."""
    for pattern in QUIET_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)
    shown: set[tuple[Any, str]] = set()
    show = warnings.showwarning

    def show_once(message: Any, category: Any, filename: str, lineno: int, file: Any = None,
                  line: str | None = None) -> None:
        key = (category, str(message))
        if key not in shown:
            shown.add(key)
            show(message, category, filename, lineno, file, line)

    warnings.showwarning = show_once


def _say(message: str, milestone: bool = False) -> None:
    """Write a time-stamped line to the log; a milestone also goes to the milestones file the notebook
    follows (set by ``start``)."""
    line = f"{time.strftime('%H:%M:%S')}  {message}"
    print(line, flush=True)
    path = os.getenv(MILESTONES_ENV)
    if milestone and path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(mask_identifiers(part) for part in line.split("\n")) + "\n")


def _capture_output() -> None:
    """Route the process's output through the masking, flood-collapsing log stream."""
    import logging

    out = _Masked(sys.stdout)
    sys.stdout, sys.stderr = out, _Masked(sys.stdout, floods=out.floods)  # one log, one record of repeats
    quiet_warnings()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    for handler in logging.getLogger().handlers:  # the same notices, when a library logs them instead
        handler.addFilter(lambda record: not any(re.match(p, record.getMessage()) for p in QUIET_WARNINGS))


def _close_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        (stream.close if isinstance(stream, _Masked) else stream.flush)()


def main(argv: list[str]) -> int:
    """The job process: run the exploration and record how it ended."""
    _capture_output()
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
             f"disk {env['disk_free_gb']} GB free", milestone=True)
        _say(f"settings: {json.dumps(env['settings'])}")
        heartbeat.start()

        def on_log(message: str) -> None:
            text = str(message)
            if text.strip().startswith("=== ") and text.strip().endswith(" ==="):
                heartbeat.now.update(outcome=text.strip()[4:-4], agent=None, tool=None)
            _say(text.strip("\n") if text.strip() else text, milestone=True)

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
        if os.getenv("CODAS_REHEARSAL", "1") != "0":  # the real models on synthetic data, before any query
            from codas.aou import rehearsal

            heartbeat.now.update(outcome="rehearsal (synthetic data)", agent=None, tool=None)
            rehearsal.run_once(out, log=on_log, on_tool_call=on_tool, on_agent=on_agent)
        result = runner.explore(
            args["catalog"], CohortSpec.from_json(args["spec"]), args.get("cdr"), client=client,
            top_k=int(args.get("top_k", 3)), outcome_names=args.get("outcomes") or None,
            max_outcomes=args.get("max_outcomes") or None, test_holdout=bool(args.get("test_holdout", True)),
            out_dir=out, log=on_log, on_tool_call=on_tool, on_agent=on_agent)
        _write(result_path, result)
        ending = {"state": "finished", "summary_path": result["summary_path"]}
        _say("exploration finished", milestone=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - record any ending, including an interrupt
        traceback.print_exc()
        ending = {"state": "failed", "error": mask_identifiers(f"{type(exc).__name__}: {str(exc)[:300]}")}
        _say(f"exploration failed: {ending['error']}", milestone=True)
        return 1
    finally:
        heartbeat.stop()
        # Re-read the state: the launcher may have written it after this process started.
        _write(state_path, {**_read(state_path), **ending, "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        _close_output()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
