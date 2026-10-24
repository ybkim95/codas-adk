"""Run CoDaS on All of Us cohorts: one table, or a free exploration over a catalog of outcomes.

Shared by ``scripts/aou/run_codas_aou.py`` and the Workbench notebook. ``run_agents`` turns on
``CODAS_AOU_MODE`` and re-checks the model backend before every run, so the privacy guard is on
whichever entry point is used. Everything written goes under ``.codas_runs/aou/`` and is masked for
counts of 1-20.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

# The paper's loop runs up to five rounds (its stop rules usually end it sooner); set before the agent
# graph is imported, which reads it. Main keeps its own default.
os.environ.setdefault("CODAS_MAX_DISCOVERY_ROUNDS", "5")
# Five rounds, the per-candidate debate and the report review take 1-2 h per outcome on a cohort of this size.
os.environ.setdefault("CODAS_OUTCOME_TIMEOUT_MIN", "180")

from codas.aou import cohort  # noqa: E402
from codas.aou import feasibility as feas
from codas.aou import ledger as ledger_book
from codas.aou.cohort import (DEFAULT_OUT_DIR, FOLLOWUP_COLUMN, CohortSpec, extract_cohort, holdout_path,
                              is_survival, resolve_cdr)
from codas.aou.phenotypes import describe as describe_phenotypes
from codas.aou.confirm import (confirm, confirmation_markdown, ledger_summary, preregister,
                              reconciliation_markdown)
from codas.aou.privacy import check_llm_backend, mask_identifiers, mask_small_counts, mask_small_counts_in_text
from codas.aou.table1 import cohort_table
from codas.core import model_tiers
from codas.aou.execution import totals_text
from codas.aou.trace import TraceRecorder

GOAL_TEMPLATE = (
    "Discover and validate wearable biomarkers of {label} in this All of Us cohort. The target column is "
    "{name} ({kind}: {target_note}). age and female are confounders. The features summarise Fitbit data over "
    "the {window}-day window before each participant's index date. {index_note}{covariates} Iterate and write a "
    "short report."
)
# How the extractor fixes the index date (codas/aou/cohort.py), so the analysis reads the design correctly.
INDEX_NOTES = {
    "measurement": "The index date is the participant's first qualifying measurement that has wearable data "
                   "in the window before it.",
    "condition": "The index date is the first recorded diagnosis for cases and the last wearable day for "
                 "everyone else, so a case's window ends before the first recorded diagnosis (which can "
                 "postdate onset); cases and non-cases get index dates by different rules.",
}
INCIDENT_NOTE = (
    "This is a prospective design: everyone's index date is {window} days after their first valid wear day, "
    "the features summarise those baseline days, and the outcome is observed from {lag} days after the index "
    "date{outcome_rule}. Cases and non-cases share this index rule."
)
HOLDOUT_NOTE = (" A sealed share of participants is held out: your candidates are pre-registered and tested once "
                "on it after you finish, so report what this discovery share supports.")
COVARIATE_NOTE = (
    " These columns are clinical covariates recorded up to the index date, available for confounder "
    "adjustment and not candidate biomarkers: {columns}. Declare as confounders the ones you judge to "
    "be common causes of a candidate feature and the outcome."
)


PHENOTYPE_NOTE = (
    " Intraday phenotypes, computed from minute-level data over the same window (each a published "
    "definition; see codas/aou/phenotypes.py): {columns}."
)
ADHERENCE_NOTE = (
    " These columns count the days each wearable recorded data in the window, i.e. how much a participant "
    "wore and synced the device: {columns}. They measure wear adherence, not physiology, so they are not "
    "candidate biomarkers; adherence tracks health engagement and access to care, which also drive "
    "diagnosis and lab testing. Declare them as confounders when a candidate could reflect how much a "
    "participant wears the device rather than what the device measures."
)
IMPUTATION_NOTE = (
    " Missing values of a declared confounder are multiply imputed ({m} draws, pooled), so adjusting for a "
    "covariate keeps every participant instead of dropping those without it."
)
# Missing confounder values: imputed this many times in the adjusted tests (0 = drop incomplete rows).
COVARIATE_IMPUTATIONS = int(os.getenv("CODAS_COVARIATE_IMPUTATIONS", "5"))
CONFIRM_IMPUTATIONS = int(os.getenv("CODAS_CONFIRM_IMPUTATIONS", "10"))  # the one-shot holdout test affords more


PANEL_NOTE = (" A repeated-measures panel is also available for a within-person analysis: every later measurement "
              "of the outcome paired with the wearable summary of the {period} days before it (run_mixed_model). A "
              "within-person effect compares a person with themselves, so stable differences between people "
              "(adherence, body size, socioeconomic position) cannot produce it.")
MEMORY_NAME = "codas_memory.jsonl"


def goal_for(spec: CohortSpec, label: str | None = None) -> str:
    kind = spec.outcome.kind
    note = "first qualifying lab value" if kind == "measurement" else "1 = diagnosed, 0 = not"
    index_note = INDEX_NOTES[kind]
    if spec.design == "incident":
        if is_survival(spec):
            note = "1 = new diagnosis during follow-up, 0 = none"
            rule = (f"; anyone diagnosed before then is excluded, and {FOLLOWUP_COLUMN} is each participant's "
                    "follow-up time (longer follow-up means more opportunity for a diagnosis)")
        else:
            note = "first value measured after the baseline and lag"
            rule = "; the target is the first value measured from then on"
        index_note = INCIDENT_NOTE.format(window=spec.window_days, lag=spec.lag_days, outcome_rule=rule)
    if spec.holdout_fraction > 0:
        index_note += HOLDOUT_NOTE
    covariates = ""
    if spec.covariates:
        columns = ", ".join(f"{c.name} ({_describe(c)})" for c in spec.covariates)
        covariates = COVARIATE_NOTE.format(columns=columns)
    if spec.phenotypes:
        covariates += PHENOTYPE_NOTE.format(columns="; ".join(
            f"{column} ({text})" for source in spec.phenotypes for column, text in describe_phenotypes(source)))
    if cohort.availability_columns(spec):
        covariates += ADHERENCE_NOTE.format(columns=", ".join(cohort.availability_columns(spec)))
    if COVARIATE_IMPUTATIONS > 0:
        covariates += IMPUTATION_NOTE.format(m=COVARIATE_IMPUTATIONS)
    if cohort.has_panel(spec):
        covariates += PANEL_NOTE.format(period=spec.panel["period_days"])
    return GOAL_TEMPLATE.format(label=label or spec.outcome.name, name=spec.outcome.name, kind=kind,
                                target_note=note, window=spec.window_days, index_note=index_note,
                                covariates=covariates)


def _describe(covariate: Any) -> str:
    span = f"in the {covariate.lookback_days} days" if covariate.lookback_days is not None else "at any time"
    if covariate.kind == "measurement":
        return f"most recent value {span} up to the index date"
    return f"1 if a {covariate.kind} record starts {span} before the index date"


# Wall-clock budget for one outcome's agent run; a run past it is stopped and reported as failed.
OUTCOME_TIMEOUT_MIN = float(os.getenv("CODAS_OUTCOME_TIMEOUT_MIN", "120"))


async def _run_agents_async(csv: Path, goal: str, on_tool_call: Any = None,
                            context_columns: list[str] | None = None,
                            recorder: TraceRecorder | None = None, on_agent: Any = None,
                            state: dict[str, Any] | None = None) -> dict[str, Any]:
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from codas.agents.agent import root_agent  # AoU mode checks the model backend on import
    from codas.aou.execution import ExecutionPlugin

    plugin = ExecutionPlugin()  # every agent turn, model call and tool call, for the execution report
    if recorder is not None:
        recorder.bind(root_agent)
        recorder.attach(plugin)

    sessions = InMemorySessionService()
    sid = uuid.uuid4().hex[:12]
    await sessions.create_session(app_name="codas_aou", user_id="aou", session_id=sid,
                                  state={**(state or {}), "csv_path": str(csv),
                                         "context_columns": list(context_columns or [])})
    runner = Runner(app=App(name="codas_aou", root_agent=root_agent, plugins=[plugin]), session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=goal)])
    speaking = None
    async for event in runner.run_async(user_id="aou", session_id=sid, new_message=message):
        if recorder is not None:
            recorder.record(event)
        if on_agent and event.author != speaking:
            speaking = event.author
            on_agent(speaking)
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call and on_tool_call:
                on_tool_call(event.author, call.name)
    session = await sessions.get_session(app_name="codas_aou", user_id="aou", session_id=sid)
    return dict(session.state)


def run_agents(csv: str | Path, goal: str, on_tool_call: Any = None,
               context_columns: list[str] | None = None, recorder: TraceRecorder | None = None,
               on_agent: Any = None, timeout_min: float | None = None,
               state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the CoDaS agent graph on ``csv`` with the AoU guard on, and return the final session state.

    ``context_columns`` are adjustment-only columns (the spec's covariates): never screened as
    candidates, available to be declared as confounders. ``state`` seeds further session settings (e.g.
    ``covariate_imputations``). ``recorder`` collects the full agent trace;
    ``on_agent`` is called with each agent's name when it starts its turn. A run longer than
    ``timeout_min`` (default OUTCOME_TIMEOUT_MIN) raises TimeoutError."""
    os.environ["CODAS_AOU_MODE"] = "1"
    check_llm_backend()  # the graph may have been imported before AoU mode was switched on
    minutes = OUTCOME_TIMEOUT_MIN if timeout_min is None else timeout_min
    coro = _run_agents_async(Path(csv).expanduser().resolve(), goal, on_tool_call, context_columns, recorder,
                             on_agent, state)
    try:
        return _run_sync(asyncio.wait_for(coro, timeout=minutes * 60))
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"the agents did not finish within {minutes:g} minutes") from exc


def _run_sync(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, including from a Jupyter cell.

    A notebook kernel already runs an event loop, where ``asyncio.run`` refuses to start; there the
    coroutine runs on its own loop in a worker thread, and this call blocks until it finishes.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def reports_dir(out: Path) -> Path:
    """Where a run's aggregate outputs are mirrored for viewing: ``codas_reports/<run folder>`` in the
    checkout (or CODAS_REPORTS_DIR). Jupyter does not serve files inside hidden directories such as
    ``.codas_runs``, so links point here. Participant-level tables are never copied."""
    base = Path(os.getenv("CODAS_REPORTS_DIR") or cohort.ROOT / "codas_reports")
    return base.expanduser().resolve() / out.name


def _publish(paths: dict[str, str], out: Path) -> dict[str, str]:
    target = reports_dir(out)
    target.mkdir(parents=True, exist_ok=True)
    published = {}
    for kind, path in paths.items():
        src = Path(path)
        if src.suffix.lower() == ".csv" or not src.exists():
            continue
        published[kind] = str(shutil.copy2(src, target / src.name))
    return published


# Session-state keys each phase publishes (the agents' output_key values plus the engine's records).
STATE_KEYS = ("target_column", "confounder_columns", "excluded_columns", "context_columns", "exclusion_log",
              "confounder_log", "hypotheses", "statistical_interpretation", "ml_interpretation", "critique",
              "defense", "mechanisms", "novelty", "strategy", "artifacts", "report", "rounds", "fact_sheet",
              "confirmation", "cohort_table", "analyses", "a_priori_hypotheses", "gap_analysis", "methods_explored",
              "constraints", "candidate_debates", "lessons", "report_review", "preflight")


def save_outputs(state: dict[str, Any], stem: str, out_dir: str | Path = DEFAULT_OUT_DIR,
                 recorder: TraceRecorder | None = None) -> dict[str, Any]:
    """Write the masked report, round summary, every phase's output and (with ``recorder``) the full
    agent trace; return a masked one-line summary of the run."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = mask_small_counts_in_text(str(state.get("report") or ""))
    rounds = state.get("rounds") or []
    fact_sheet = state.get("fact_sheet") or {}
    (out / f"{stem}_report.md").write_text(report, encoding="utf-8")
    (out / f"{stem}_rounds.json").write_text(
        json.dumps(mask_small_counts({"rounds": rounds, "fact_sheet": fact_sheet}), indent=2, default=str),
        encoding="utf-8")
    phases = {k: (mask_small_counts_in_text(v) if isinstance(v, str) else mask_small_counts(v))
              for k, v in state.items() if k in STATE_KEYS}
    (out / f"{stem}_state.json").write_text(json.dumps(phases, indent=2, default=str), encoding="utf-8")
    files = {"state": str(out / f"{stem}_state.json")}
    coverage_missing, execution = None, None
    if recorder is not None:
        files.update(recorder.write(out, stem))
        coverage_missing = recorder.coverage()["missing"]
        execution = recorder.execution_tables()
    files.update(_compile_pdf(out, stem, phases, recorder.events if recorder else [],
                              recorder.coverage_markdown() if recorder else "", fact_sheet.get("rows"), execution))
    files["report_md"] = str(out / f"{stem}_report.md")
    files.update(_publish({k: v for k, v in files.items() if not k.endswith("_error")}, out))
    last = rounds[-1] if rounds else {}
    return mask_small_counts({
        "cohort": stem,
        "rows": fact_sheet.get("rows"),
        "rounds": len(rounds),
        "validated_count": last.get("validated_count"),
        "ml_metric": f"{last.get('ml_metric_name')}={last.get('ml_metric_value')}" if last else None,
        "confounders": list(state.get("confounder_columns") or []),
        "covariate_coverage": {name: info.get("non_missing_fraction")
                               for name, info in (last.get("adjustment_covariates") or {}).items()},
        "excluded": list(state.get("excluded_columns") or []),
        "top_candidates": [f"{c.get('feature')} (rho={c.get('rho')}, {c.get('verdict')})"
                           for c in (last.get("passing_candidates") or [])[:5]],
        "report_path": str(out / f"{stem}_report.md"),
        "files": files,
        "coverage_missing": coverage_missing,
        "replication": (f"{state['confirmation']['replicated']}/{state['confirmation']['tested']}"
                        if state.get("confirmation") else None),
        "robust": state["confirmation"].get("robust") if state.get("confirmation") else None,
        "execution": execution["totals"] if execution and execution.get("agents") else None,
    })


def _compile_pdf(out: Path, stem: str, phases: dict[str, Any], events: list[dict[str, Any]], coverage_md: str,
                 rows: Any, execution: dict[str, Any] | None = None) -> dict[str, str]:
    """Build ``<stem>_report.pdf``; a failure, or a section that could not be rendered, leaves its full
    traceback in ``<stem>_pdf_error.txt``. Returns the file entries for the run summary."""
    import traceback

    error_file = out / f"{stem}_pdf_error.txt"
    try:
        from codas.aou.pdf_report import build_pdf

        served = ((execution or {}).get("totals") or {}).get("models") or {}
        meta = {"rows": rows, "code_version": _code_version()[:12],
                "models": (", ".join(f"{m} x{n} calls" for m, n in served.items()) if served else
                           ", ".join(dict.fromkeys(model_tiers.model(t) for t in model_tiers.TIERS)))}
        files = {"pdf": str(build_pdf(out, stem, phases, events, coverage_md, meta, execution))}
        if error_file.exists():
            files["pdf_warnings"] = str(error_file)
        return files
    except Exception as exc:  # noqa: BLE001 - the PDF is a convenience; the run's files are already written
        error_file.write_text(mask_identifiers(traceback.format_exc()), encoding="utf-8")
        return {"pdf_error": f"{type(exc).__name__}: {str(exc)[:200]}", "pdf_warnings": str(error_file)}


def rebuild_pdf(out_dir: str | Path, stem: str) -> dict[str, str]:
    """Compile an outcome's PDF again from what its run saved (``<stem>_state.json`` and the trace), without
    running the agents. Returns the file entries (``pdf``, or ``pdf_error`` and ``pdf_warnings``), mirrored."""
    out = Path(out_dir).expanduser().resolve()
    state_file = out / f"{stem}_state.json"
    if not state_file.exists():
        state_file = reports_dir(out) / f"{stem}_state.json"
    phases = json.loads(state_file.read_text(encoding="utf-8"))
    trace = next((p for p in (out / f"{stem}_trace.jsonl", reports_dir(out) / f"{stem}_trace.jsonl") if p.exists()), None)
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()] if trace else []
    coverage = "Rebuilt from the saved outputs; the trace file shows which agents ran in each phase and round."
    execution_file = next((p for p in (out / f"{stem}_execution.json", reports_dir(out) / f"{stem}_execution.json")
                           if p.exists()), None)
    execution = json.loads(execution_file.read_text(encoding="utf-8")) if execution_file else None
    files = _compile_pdf(out, stem, phases, events, coverage, (phases.get("fact_sheet") or {}).get("rows"), execution)
    files.update(_publish({k: v for k, v in files.items() if not k.endswith("_error")}, out))
    return files


def _refresh_pdf(out: Path, summary: dict[str, Any], key: str) -> dict[str, Any]:
    """A reused run whose PDF is missing (an older run's PDF step failed) gets it compiled from its saved
    outputs; the old error goes, and the run's saved summary is updated so later summaries agree."""
    files = dict(summary.get("files") or {})
    if files.get("pdf") and Path(files["pdf"]).exists():
        return summary
    for stale in ("pdf", "pdf_error", "pdf_warnings"):
        files.pop(stale, None)
    try:
        files.update(rebuild_pdf(out, summary["cohort"]))
    except (OSError, ValueError) as exc:  # the saved state is gone or unreadable: say so, keep the rest
        files["pdf_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    summary = {**summary, "files": files}
    (out / f"{summary['cohort']}_run.json").write_text(json.dumps({"key": key, "summary": summary}),
                                                       encoding="utf-8")
    return summary


def _with_coverage(confounders: list[str] | None, coverage: dict[str, Any] | None) -> str:
    """Confounder names, each adjustment covariate with the share of participants who have a value
    (adjusting for it drops the rest)."""
    parts = []
    for name in confounders or []:
        share = (coverage or {}).get(name)
        if isinstance(share, (int, float)):
            parts.append(f"{name} ({share:.0%} have a value)")
        elif share is not None:
            parts.append(f"{name} ({share})")
        else:
            parts.append(name)
    return ", ".join(parts) or "none"


def _code_version() -> str:
    """The checkout's commit, so results produced by older code are not reused."""
    try:
        head = subprocess.run(["git", "-C", str(cohort.ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
        return head.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - no git: fall back to a constant
        return "unknown"


# The code that decides what an outcome's results are. Changes elsewhere (the notebook, the background
# job, report rendering, views) leave finished outcomes reusable; a change here reruns them.
ANALYSIS_CODE = ("codas/core", "codas/agents", "codas/aou/cohort.py", "codas/aou/phenotypes.py",
                 "codas/aou/feasibility.py", "codas/aou/confirm.py", "codas/aou/ledger.py",
                 "codas/aou/privacy.py", "codas/aou/table1.py")


def _analysis_version(root: Path | None = None) -> str:
    """A hash of the analysis code's source files."""
    base = Path(root or cohort.ROOT)
    digest = hashlib.sha256()
    for entry in ANALYSIS_CODE:
        path = base / entry
        for f in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            if f.exists():
                digest.update(f.relative_to(base).as_posix().encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


def _run_key(spec: CohortSpec, goal: str, test_holdout: bool = True) -> str:
    payload = json.dumps({"spec": asdict(spec), "goal": goal, "code": _analysis_version(),
                          "holdout_test": test_holdout}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _plain(value: Any) -> Any:
    """JSON fallback for numpy scalars and arrays in the agents' state."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _agents_checkpoint(out: Path, name: str) -> Path:
    return out / f"{name}_agents.json"


def _save_agents(out: Path, name: str, key: str, state: dict[str, Any], recorder: TraceRecorder) -> None:
    """Keep the agents' finished work (aggregate state and trace) until the run is saved, so a failure in a
    later step (pre-registration, the holdout test, writing the outputs) does not cost the agent run."""
    _agents_checkpoint(out, name).write_text(json.dumps({"key": key, "state": state, "trace": recorder.snapshot()},
                                                        default=_plain), encoding="utf-8")


def _saved_agents(out: Path, name: str, key: str) -> tuple[dict[str, Any], TraceRecorder] | None:
    try:
        saved = json.loads(_agents_checkpoint(out, name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if saved.get("key") != key:
        return None
    return saved["state"], TraceRecorder.restore(saved.get("trace") or {})


def _finished_run(out: Path, name: str, key: str) -> dict[str, Any] | None:
    """The saved summary of a run of ``name`` with this exact ``key`` whose report is still on disk."""
    try:
        saved = json.loads((out / f"{name}_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = saved.get("summary") or {}
    if saved.get("key") != key or not Path(str(summary.get("report_path", ""))).exists():
        return None
    return summary


_MAX_RECALLED = int(os.getenv("CODAS_MAX_RECALLED_LESSONS", "12"))


def _remember(out: Path, key: str, stem: str) -> None:
    """Append the lessons a finished run recorded (from its saved, masked state) to the cohort memory."""
    try:
        lessons = json.loads((out / f"{stem}_state.json").read_text(encoding="utf-8")).get("lessons") or []
    except (OSError, ValueError):
        return
    if lessons:
        with (out / MEMORY_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "stem": stem, "lessons": lessons}) + "\n")


def _recalled_lessons(out: Path, key: str, stem: str) -> list[str]:
    """Lessons other outcomes recorded on the same participants, newest last, without repeats. The goal
    carries them; the resume key does not, so remembering more never invalidates a finished outcome."""
    path = out / MEMORY_NAME
    if not path.exists():
        return []
    seen: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("key") != key or entry.get("stem") == stem:
            continue
        for lesson in entry.get("lessons") or []:
            text = str(lesson.get("lesson") if isinstance(lesson, dict) else lesson).strip()
            if text and text not in seen:
                seen.append(text)
    return [mask_small_counts_in_text(t) for t in seen[-_MAX_RECALLED:]]


def _opened_outcomes(out: Path, keys: list[str], outcomes: dict[str, Any], base: CohortSpec,
                     test_holdout: bool = True) -> set[str]:
    """Catalog outcomes already tested on this holdout, by run name or by outcome definition. An outcome whose
    run with the current code already finished here, or whose agents finished before a later step failed, is
    not counted: running the exploration again resumes it rather than moving on to another outcome."""
    if not keys:
        return set()
    stems, outcome_keys = ledger_book.opened(out, keys[0])
    opened = set()
    for name, outcome in outcomes.items():
        spec = feas.spec_for(outcome, base)
        if spec.name not in stems and ledger_book.outcome_key(outcome_identity(spec), spec.name) not in outcome_keys:
            continue
        key = _run_key(spec, goal_for(spec, outcome.label), test_holdout)
        if _finished_run(out, spec.name, key) is None and _saved_agents(out, spec.name, key) is None:
            opened.add(name)
    return opened


def holdout_key(spec: CohortSpec, cdr: str | None) -> str:
    """Which participants a spec's holdout holds: the split depends only on the person id, the salt and
    the share, so every spec with the same three seals the same people."""
    return f"{resolve_cdr(cdr)}|salt={spec.split_salt}|permille={int(round(spec.holdout_fraction * 1000))}"


def _ledger_lines(out: Path, keys: list[str]) -> list[str]:
    lines = []
    for key in dict.fromkeys(keys):
        led = ledger_summary(out, key)
        if not led["hypotheses"]:
            continue
        lines += ["## Sealed holdout ledger", "",
                  f"{led['hypotheses']} hypotheses from {led['registrations']} registrations have been tested on "
                  f"this holdout. Novel tests under online false-discovery-rate control: {led['novel_tested']}, of "
                  f"which {len(led['novel_replicated'])} replicated"
                  + (": " + "; ".join(led["novel_replicated"]) if led["novel_replicated"] else "")
                  + f". Reference (established) associations: {len(led['reference_reproduced'])} of "
                  f"{led['reference_tested']} reproduced. Not confirmatory: {led['non_confirmatory']} (repeats or "
                  f"outcomes already tested here). The next novel test will be judged at level "
                  f"{led['next_level']:.3g}.", ""]
        if led["legacy"]:
            lines += [f"{led['legacy']} earlier hypotheses were tested before verdicts were frozen; their outcomes "
                      f"({', '.join(led['legacy_outcomes'])}) count as already tested, so new registrations about "
                      "them are not confirmatory.", ""]
    return lines


def _write_summary(out: Path, table: str, runs: list[dict[str, Any]], ledger_keys: list[str] | None = None,
                   billed: list[dict[str, Any]] | None = None) -> Path:
    lines = ["# CoDaS free exploration on All of Us", "",
             "Counts of 1-20 are masked. Review before anything leaves the Workbench.", ""]
    if billed:
        lines += [f"BigQuery in this run: {cohort.billed_text(sum(b['bytes_billed'] for b in billed))} over "
                  f"{len(billed)} jobs.", ""]
    ran = [r["execution"] for r in runs if r.get("execution")]
    if ran:
        costs = [t.get("cost_usd") for t in ran]
        cost = ("" if any(c is None for c in costs) else
                f"; model cost about ${sum(costs):,.2f} at list prices")
        lines += [f"Agents in this run: {sum(t['model_calls'] for t in ran):,} model calls, "
                  f"{sum(t['tool_calls'] for t in ran):,} tool calls, {sum(t['tokens'] for t in ran):,} tokens "
                  f"({sum(t.get('thoughts', 0) for t in ran):,} thinking), "
                  f"{sum(t['wall_s'] for t in ran) / 60:.0f} agent minutes{cost}.", ""]
    lines += ["## Feasibility screen", "", table, "", "## Runs", ""]
    for r in runs:
        if r.get("failed"):
            lines += [f"### {r['cohort']}", f"- failed: {r['failed']}",
                      "- run the exploration again to retry; finished outcomes are reused", ""]
            continue
        coverage = r.get("covariate_coverage") or {}
        confounders, excluded = r.get("confounders") or [], r.get("excluded") or []
        unused = [c for c in coverage if c not in confounders and c not in excluded]
        lines += [f"### {r['cohort']}", f"- participants: {r['rows']}, rounds: {r['rounds']}, "
                  f"validated: {r['validated_count']}, {r['ml_metric']}"
                  + (f", replicated on the sealed holdout: {r['replication']}" if r.get("replication") else "")
                  + (f" ({r['robust']} also passed the falsification tests)" if r.get("robust") is not None else ""),
                  f"- adjusted for: {_with_coverage(confounders, coverage)}; "
                  f"excluded as leakage: {', '.join(excluded) or 'none'}"]
        if unused:
            lines.append(f"- covariates not adjusted for: {_with_coverage(unused, coverage)}")
        if r.get("coverage_missing"):
            lines.append(f"- steps without output: {'; '.join(r['coverage_missing'])}")
        if r.get("execution"):
            lines.append(f"- agents: {totals_text(r['execution'])}")
        if r.get("holdout_tested") is False:
            lines.append("- sealed holdout: not tested in this run; the outcome stays fresh for a confirmatory run")
        lines += [*[f"- {c}" for c in r["top_candidates"]], f"- report: {r['report_path']}"]
        lines += [f"- {kind}: {path}" for kind, path in (r.get("files") or {}).items()] + [""]
    lines += _ledger_lines(out, ledger_keys or [])
    path = out / "explore_summary.md"
    path.write_text(mask_small_counts_in_text("\n".join(lines)), encoding="utf-8")
    _publish({"summary": str(path)}, out)
    return path


PREFLIGHT = os.getenv("CODAS_PREFLIGHT", "1").strip().lower() not in ("0", "false", "no", "off")
PREFLIGHT_IMPUTATIONS = 2  # enough to exercise the imputation path in minutes


class PreflightError(RuntimeError):
    """The deterministic pipeline could not compute something on this cohort's discovery share."""


def _duplicate_columns(frame: Any) -> list[list[str]]:
    """Groups of columns with identical values (missing values included), in column order."""
    import pandas as pd

    groups: dict[int, list[str]] = {}
    for column in frame.columns:
        groups.setdefault(int(pd.util.hash_pandas_object(frame[column], index=False).sum()), []).append(column)
    out = []
    for names in groups.values():
        while len(names) > 1:
            same = [n for n in names[1:] if frame[n].equals(frame[names[0]])]
            if same:
                out.append([names[0], *same])
            names = [n for n in names[1:] if n not in same]
    return out


def preflight(csv: str | Path, out_dir: str | Path, *, stem: str, target: str, survival_time: str | None,
              confounders: list[str], stand_ins: list[str], negative_controls: list[str] | tuple[str, ...] = (),
              positive_controls: list[str] | tuple[str, ...] = (), availability: list[str] | tuple[str, ...] = (),
              panel_csv: str | Path | None = None) -> dict[str, Any]:
    """Run the holdout pipeline end to end on the DISCOVERY share with stand-in hypotheses (one wearable
    column per source), adjusting for every covariate the agents could choose, before any agent runs. It
    never touches the sealed share and writes no ledger entry. ``fatal`` lists what could not be computed:
    code or data the real holdout test would also fail on, found in minutes instead of after hours."""
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv)
    candidates = [{"feature": c, "rho": 0.1, "q_value": 0.001, "verdict": "validated", "components": [c]}
                  for c in stand_ins if c in frame.columns]
    state = {"target_column": target, "confounder_columns": [c for c in confounders if c in frame.columns],
             "excluded_columns": [], "latest_report": {"candidates": candidates}, "proposed_features": []}
    prereg = preregister(state, out, stem, survival_time=survival_time, negative_controls=list(negative_controls),
                         positive_controls=list(positive_controls), availability=list(availability),
                         covariate_imputations=PREFLIGHT_IMPUTATIONS, discovery_csv=csv)
    result = confirm(prereg, Path(csv), out, ledger_key=None,
                     panel_csv=Path(panel_csv) if panel_csv and Path(panel_csv).exists() else None)
    # The stand-ins carry a placeholder screen correlation, so a direction disagreement means nothing here.
    flags = [f for f in result.get("audit") or [] if "binary screen disagreed" not in f]
    if not candidates:
        flags.append("no stand-in hypothesis could be built from this cohort's columns")
    target_values = pd.to_numeric(frame[target], errors="coerce") if target in frame.columns else pd.Series(dtype=float)
    if survival_time and target_values.sum() < 10:  # say why nothing could be estimated
        flags.append(f"the discovery share has {int(target_values.sum())} events among {len(frame)} participants: "
                     "too few to estimate anything (could not be computed)")
    fatal = [f for f in flags if "could not be computed" in f or "raised an error" in f or "no stand-in" in f]
    numeric = frame.drop(columns=[c for c in (target, survival_time) if c and c in frame.columns])
    return {"flags": flags, "fatal": fatal, "duplicates": _duplicate_columns(numeric)}


def outcome_identity(spec: CohortSpec) -> dict[str, Any]:
    """What the outcome is, independent of what the run is called: the holdout ledger's outcome key."""
    o = spec.outcome
    return {"kind": o.kind, "concept_ids": sorted(int(c) for c in o.concept_ids), "codes": sorted(o.codes),
            "value_min": o.value_min, "value_max": o.value_max}


def _preflight_args(spec: CohortSpec, csv: Path) -> dict[str, Any]:
    """What the holdout test of this spec would use: every covariate the agents could adjust for and one
    stand-in hypothesis per wearable source and phenotype source."""
    demographics = ["age", "female"] if spec.include_demographics else []
    stand_ins = ([f"{s.prefix}_{s.columns[0]}_mean" for s in spec.wearables if s.columns]
                 + [p.output_columns()[1] for p in spec.phenotypes if len(p.output_columns()) > 1])
    return {"stem": spec.name, "target": spec.outcome.name, "survival_time": FOLLOWUP_COLUMN if is_survival(spec) else None,
            "confounders": demographics + [c.name for c in spec.covariates] + cohort.availability_columns(spec),
            "stand_ins": stand_ins, "negative_controls": [c.name for c in spec.negative_controls],
            "positive_controls": list(spec.positive_controls), "availability": cohort.availability_columns(spec),
            "panel_csv": cohort.panel_path(csv)}


def run_spec(spec: CohortSpec, cdr: str | None = None, *, label: str | None = None, client: Any = None,
             out_dir: str | Path = DEFAULT_OUT_DIR, check_schema: bool = True, on_tool_call: Any = None,
             on_agent: Any = None, lessons: list[str] | None = None, test_holdout: bool = True) -> dict[str, Any]:
    """Extract one cohort, run the agents on it, save the masked outputs and the agent trace. ``lessons``
    are earlier outcomes' lessons about this cohort. They come from other outcomes' discovery statistics on
    the same people, so the agents see them only once registration has closed (state ``cohort_lessons``),
    never in the goal. With ``test_holdout`` off the agents run on the discovery share as usual but nothing
    is registered or tested: the sealed holdout stays unopened and the outcome stays fresh for a
    confirmatory run."""
    if check_schema:  # fit optional sources to this CDR first, so the goal and context match the table
        if client is None:
            from google.cloud import bigquery

            client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
        present = cohort._query(client, *cohort.schema_check_sql(spec, resolve_cdr(cdr)))
        spec, _skipped = cohort.resolve_sources(spec, list(zip(present["table_name"], present["column_name"])))
    csv = extract_cohort(spec, cdr, out_dir, client=client, check_schema=check_schema)
    out = Path(out_dir).expanduser().resolve()
    key = _run_key(spec, goal_for(spec, label), test_holdout)  # as explore keys finished runs (lessons excluded)
    saved = _saved_agents(out, spec.name, key)
    if saved is not None:  # the agents finished earlier and a later step failed: redo only the later steps
        state, recorder = saved
    else:
        checked = None
        goal = goal_for(spec, label)
        if PREFLIGHT:  # minutes on the discovery share, before hours of agents
            checked = preflight(csv, out / "preflight", **_preflight_args(spec, csv))
            if checked["fatal"]:
                raise PreflightError("; ".join(checked["fatal"])[:400])
            if checked["duplicates"]:
                goal += ("\nThese columns hold identical values in this cohort; adjust for at most one of each "
                         "group: " + "; ".join(" = ".join(g) for g in checked["duplicates"]) + ".")
        recorder = TraceRecorder(stream_path=out / f"{spec.name}_trace.jsonl")
        context = ([c.name for c in spec.covariates] + cohort.availability_columns(spec)
                   + ([FOLLOWUP_COLUMN] if is_survival(spec) else []))
        state = run_agents(csv, goal, on_tool_call, context, recorder=recorder,
                           on_agent=on_agent,
                           state={"covariate_imputations": COVARIATE_IMPUTATIONS,
                                  "survival_time_column": FOLLOWUP_COLUMN if is_survival(spec) else "",
                                  "prior_lessons": list(lessons or []),
                                  "panel_csv_path": (str(cohort.panel_path(csv)) if cohort.panel_path(csv).exists()
                                                     else "")})
        if checked is not None:
            state["preflight"] = checked
        _save_agents(out, spec.name, key, state, recorder)
    state["cohort_table"] = cohort_table(spec, csv)  # deterministic, masked; written after the agents finish
    sealed = holdout_path(csv)
    tested = test_holdout and spec.holdout_fraction > 0 and sealed.exists()
    if spec.holdout_fraction > 0 and not test_holdout:
        state["report"] = ("_The sealed holdout was left unopened in this run: nothing was registered or tested on "
                           f"it, so nothing here is confirmed._\n\n{state.get('report') or ''}")
    if tested:
        prereg = preregister(state, out, spec.name, survival_time=FOLLOWUP_COLUMN if is_survival(spec) else None,
                             negative_controls=[c.name for c in spec.negative_controls],
                             positive_controls=list(spec.positive_controls),
                             availability=cohort.availability_columns(spec),
                             covariate_imputations=CONFIRM_IMPUTATIONS, discovery_csv=csv,
                             outcome=outcome_identity(spec))
        # The pre-registration is on disk before the holdout is read.
        sealed_panel = holdout_path(cohort.panel_path(csv))
        result = confirm(prereg, sealed, out, ledger_key=holdout_key(spec, cdr),
                         panel_csv=sealed_panel if sealed_panel.exists() else None)
        # The per-candidate debate next to the holdout result, so a candidate the debate conceded but the
        # holdout replicated (or the reverse) is visible rather than buried in two sections.
        verdicts = {d["feature"]: d.get("verdict") for d in state.get("candidate_debates") or [] if d.get("feature")}
        if verdicts:
            for row in result["hypotheses"]:
                if row["feature"] in verdicts:
                    row["debate"] = verdicts[row["feature"]]
            (out / f"{spec.name}_confirmation.json").write_text(json.dumps(result, indent=2, default=str),
                                                                encoding="utf-8")
        state["confirmation"] = result
        engine = {c.get("feature"): c.get("verdict") for c in (state.get("latest_report") or {}).get("candidates") or []}
        state["report"] = (f"{reconciliation_markdown(result, engine)}\n{state.get('report') or ''}\n\n"
                           f"{confirmation_markdown(result)}")
    summary = {**save_outputs(state, spec.name, out_dir, recorder), "holdout_tested": tested}
    _agents_checkpoint(out, spec.name).unlink(missing_ok=True)
    return summary


def explore(
    catalog_path: str | Path,
    base: CohortSpec,
    cdr: str | None = None,
    *,
    client: Any = None,
    top_k: int = 3,
    per_kind: bool = True,
    min_measurement_n: int = feas.MIN_MEASUREMENT_N,
    min_cases: int = feas.MIN_CASES,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    screen_only: bool = False,
    outcome_names: list[str] | None = None,
    max_outcomes: int | None = None,
    test_holdout: bool = True,
    on_tool_call: Any = None,
    log: Any = print,
    rerun: bool = False,
    on_agent: Any = None,
) -> dict[str, Any]:
    """Verify the catalog, screen feasibility, then run CoDaS on the top-K outcomes.

    With ``per_kind`` (the default) the top K is taken for each outcome kind in the catalog, so one
    run covers measurements and conditions alike. ``outcome_names`` names catalog entries to run instead of the ranked top K; they still have to pass the
    concept check and have at least one qualifying participant. A ranked selection skips outcomes already
    tested on this holdout (their holdout results are known, so a new test of them would not be
    confirmatory); a named one still runs, and the ledger reports its registrations as non-confirmatory.
    ``max_outcomes`` caps how many run in all. With ``test_holdout`` off nothing is registered or tested on the
    sealed holdout (an instrumentation run that keeps the outcomes fresh).

    An outcome whose run fails (a model that stays out of capacity, a query error) is recorded as
    failed and the exploration moves on; ``explore_summary.md`` is rewritten after every outcome. Calling
    again with the same arguments resumes: an outcome already finished with the same spec, goal and code
    version is reused rather than run again, so only failed or missing outcomes run. ``rerun`` forces
    every outcome to run.

    Returns the masked screen table, the chosen outcomes and one summary per run (with ``failed`` set
    for a failed one), and writes ``explore_summary.md`` next to the per-outcome reports.
    """
    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
    cdr = resolve_cdr(cdr)
    first_job = len(cohort.QUERY_LOG)  # this run's BigQuery jobs, for the bytes-billed line
    base, failed_covariates = feas.verify_covariates(base, cdr, client, log)
    for name, reason in failed_covariates.items():
        log(f"covariate {name} skipped: {reason}")
    if base.covariates:
        log(f"covariates: {', '.join(c.name for c in base.covariates)} (adjustment only, not candidates)")
    present = cohort._query(client, *cohort.schema_check_sql(base, cdr))
    pairs = list(zip(present["table_name"], present["column_name"]))
    base, skipped = cohort.resolve_sources(base, pairs)
    for note in skipped:
        log(note)
    if base.phenotypes:
        log("intraday phenotypes: " + ", ".join(f"{p.prefix} from {p.table}" for p in base.phenotypes))
    missing = cohort.missing_columns(base, pairs)
    if missing:
        raise ValueError("The CDR lacks columns the wearable spec needs: " + ", ".join(missing)
                         + ". Edit the spec's wearables to match the CDR schema.")
    outcomes = feas.resolve_codes(feas.load_catalog(catalog_path), cdr, client, log)
    problems = feas.verify_catalog(outcomes, cdr, client)
    verified = [o for o in outcomes if problems.get(o.name) is None]
    log(f"catalog: {len(verified)}/{len(outcomes)} outcomes verified against the concept table")

    rows = feas.feasibility(verified, base, cdr, client)
    out = Path(out_dir).expanduser().resolve()
    keys = [holdout_key(base, cdr)] if base.holdout_fraction > 0 else []
    by_name = {o.name: o for o in verified}
    opened = _opened_outcomes(out, keys, by_name, base, test_holdout)
    if outcome_names:
        unknown = sorted(set(outcome_names) - {o.name for o in outcomes})
        if unknown:
            raise ValueError(f"Not in the catalog: {', '.join(unknown)}.")
        chosen = [r for r in rows if r["name"] in set(outcome_names) and r["n_participants"] > 0]
        skipped = sorted(set(outcome_names) - {r["name"] for r in chosen})
        if skipped:
            log(f"skipped (failed the concept check or no qualifying participants): {', '.join(skipped)}")
    else:
        chosen = feas.rank_outcomes([r for r in rows if r["name"] not in opened], top_k=top_k,
                                    min_measurement_n=min_measurement_n, min_cases=min_cases, per_kind=per_kind)
        if opened:
            log(f"already tested on this holdout, not selected again: {', '.join(sorted(opened))}")
    if max_outcomes:
        chosen = chosen[:max_outcomes]
    table = feas.feasibility_table(rows, problems, chosen, opened)
    log(table)

    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    memory_key = keys[0] if keys else cdr  # lessons are shared across outcomes on the same participants
    summary_path = _write_summary(out, table, runs, keys, cohort.QUERY_LOG[first_job:])
    if not screen_only:
        for row in chosen:
            o = by_name[row["name"]]
            spec = feas.spec_for(o, base)
            key = _run_key(spec, goal_for(spec, o.label), test_holdout)
            done = None if rerun else _finished_run(out, spec.name, key)
            if done:
                log(f"\n=== {o.label} === finished earlier with this code; reusing its report")
                runs.append(_refresh_pdf(out, done, key))
            else:
                log(f"\n=== {o.label} ===")
                try:
                    result = run_spec(spec, cdr, label=o.label, client=client, out_dir=out, check_schema=False,
                                      on_tool_call=on_tool_call, on_agent=on_agent,
                                      lessons=_recalled_lessons(out, memory_key, spec.name),
                                      test_holdout=test_holdout)
                except Exception as exc:  # noqa: BLE001 - one outcome's failure must not discard the others
                    reason = mask_identifiers(f"{type(exc).__name__}: {str(exc)[:300]}")
                    log(f"  failed: {reason}\n  Run again to retry; finished outcomes are reused.")
                    runs.append({"cohort": spec.name, "failed": reason})
                else:
                    (out / f"{spec.name}_run.json").write_text(json.dumps({"key": key, "summary": result}),
                                                               encoding="utf-8")
                    _remember(out, memory_key, spec.name)
                    runs.append(result)
                    log(f"  finished: {result.get('rounds')} rounds, {result.get('validated_count')} validated in "
                        f"the last round" + (f", replicated on the sealed holdout: {result['replication']}"
                                             if result.get("replication") else ""))
            summary_path = _write_summary(out, table, runs, keys, cohort.QUERY_LOG[first_job:])
    billed = cohort.QUERY_LOG[first_job:]
    if billed:
        log(f"BigQuery in this run: {cohort.billed_text(sum(b['bytes_billed'] for b in billed))}")

    return {"table": table, "chosen": [r["name"] for r in chosen], "runs": runs, "summary_path": str(summary_path),
            "rows": mask_small_counts(rows)}
