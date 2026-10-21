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

from codas.aou import cohort  # noqa: E402
from codas.aou import feasibility as feas
from codas.aou.cohort import (DEFAULT_OUT_DIR, FOLLOWUP_COLUMN, CohortSpec, extract_cohort, holdout_path,
                              is_survival, resolve_cdr)
from codas.aou.phenotypes import describe as describe_phenotypes
from codas.aou.confirm import confirm, confirmation_markdown, ledger_summary, preregister
from codas.aou.privacy import check_llm_backend, mask_identifiers, mask_small_counts, mask_small_counts_in_text
from codas.aou.table1 import cohort_table
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
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from codas.agents.agent import root_agent  # AoU mode checks the model backend on import

    if recorder is not None:
        recorder.bind(root_agent)

    sessions = InMemorySessionService()
    sid = uuid.uuid4().hex[:12]
    await sessions.create_session(app_name="codas_aou", user_id="aou", session_id=sid,
                                  state={**(state or {}), "csv_path": str(csv),
                                         "context_columns": list(context_columns or [])})
    runner = Runner(app_name="codas_aou", agent=root_agent, session_service=sessions)
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
              "confirmation", "cohort_table", "analyses", "a_priori_hypotheses")


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
    coverage_missing = None
    if recorder is not None:
        files.update(recorder.write(out, stem))
        coverage_missing = recorder.coverage()["missing"]
    files.update(_compile_pdf(out, stem, phases, recorder.events if recorder else [],
                              recorder.coverage_markdown() if recorder else "", fact_sheet.get("rows")))
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
    })


def _compile_pdf(out: Path, stem: str, phases: dict[str, Any], events: list[dict[str, Any]], coverage_md: str,
                 rows: Any) -> dict[str, str]:
    """Build ``<stem>_report.pdf``; a failure, or a section that could not be rendered, leaves its full
    traceback in ``<stem>_pdf_error.txt``. Returns the file entries for the run summary."""
    import traceback

    error_file = out / f"{stem}_pdf_error.txt"
    try:
        from codas.aou.pdf_report import build_pdf

        meta = {"rows": rows, "code_version": _code_version()[:12],
                "models": ", ".join(dict.fromkeys(filter(None, (os.getenv("CODAS_GEMINI_REASONING_MODEL"),
                                                                 os.getenv("CODAS_GEMINI_FAST_MODEL")))))}
        files = {"pdf": str(build_pdf(out, stem, phases, events, coverage_md, meta))}
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
    files = _compile_pdf(out, stem, phases, events, coverage, (phases.get("fact_sheet") or {}).get("rows"))
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
                 "codas/aou/feasibility.py", "codas/aou/confirm.py", "codas/aou/privacy.py",
                 "codas/aou/table1.py")


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


def _run_key(spec: CohortSpec, goal: str) -> str:
    payload = json.dumps({"spec": asdict(spec), "goal": goal, "code": _analysis_version()}, sort_keys=True,
                         default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


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
        # A candidate registered by several runs is listed once, with how many registrations replicated it.
        hits = led["replicated"]
        led["replicated"] = [h if hits.count(h) == 1 else f"{h} (x{hits.count(h)})" for h in dict.fromkeys(hits)]
        lines += ["## Sealed holdout ledger", "",
                  f"{led['hypotheses']} hypotheses from {led['registrations']} pre-registrations have been tested on "
                  "this holdout. Corrected together (BH, q <= 0.05, same direction as discovery), "
                  f"{len(led['replicated'])} distinct candidates replicate"
                  + (": " + "; ".join(led["replicated"]) if led["replicated"] else "")
                  + ".", ""]
    return lines


def _write_summary(out: Path, table: str, runs: list[dict[str, Any]], ledger_keys: list[str] | None = None) -> Path:
    lines = ["# CoDaS free exploration on All of Us", "",
             "Counts of 1-20 are masked. Review before anything leaves the Workbench.", "",
             "## Feasibility screen", "", table, "", "## Runs", ""]
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
        lines += [*[f"- {c}" for c in r["top_candidates"]], f"- report: {r['report_path']}"]
        lines += [f"- {kind}: {path}" for kind, path in (r.get("files") or {}).items()] + [""]
    lines += _ledger_lines(out, ledger_keys or [])
    path = out / "explore_summary.md"
    path.write_text(mask_small_counts_in_text("\n".join(lines)), encoding="utf-8")
    _publish({"summary": str(path)}, out)
    return path


def run_spec(spec: CohortSpec, cdr: str | None = None, *, label: str | None = None, client: Any = None,
             out_dir: str | Path = DEFAULT_OUT_DIR, check_schema: bool = True, on_tool_call: Any = None,
             on_agent: Any = None) -> dict[str, Any]:
    """Extract one cohort, run the agents on it, save the masked outputs and the agent trace."""
    if check_schema:  # fit optional sources to this CDR first, so the goal and context match the table
        if client is None:
            from google.cloud import bigquery

            client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
        present = cohort._query(client, *cohort.schema_check_sql(spec, resolve_cdr(cdr)))
        spec, _skipped = cohort.resolve_sources(spec, list(zip(present["table_name"], present["column_name"])))
    csv = extract_cohort(spec, cdr, out_dir, client=client, check_schema=check_schema)
    recorder = TraceRecorder(stream_path=Path(out_dir).expanduser().resolve() / f"{spec.name}_trace.jsonl")
    context = ([c.name for c in spec.covariates] + cohort.availability_columns(spec)
               + ([FOLLOWUP_COLUMN] if is_survival(spec) else []))
    state = run_agents(csv, goal_for(spec, label), on_tool_call, context, recorder=recorder, on_agent=on_agent,
                       state={"covariate_imputations": COVARIATE_IMPUTATIONS,
                              "survival_time_column": FOLLOWUP_COLUMN if is_survival(spec) else ""})
    state["cohort_table"] = cohort_table(spec, csv)  # deterministic, masked; written after the agents finish
    sealed = holdout_path(csv)
    if spec.holdout_fraction > 0 and sealed.exists():
        out = Path(out_dir).expanduser().resolve()
        prereg = preregister(state, out, spec.name, survival_time=FOLLOWUP_COLUMN if is_survival(spec) else None,
                             negative_controls=[c.name for c in spec.negative_controls],
                             positive_controls=list(spec.positive_controls),
                             availability=cohort.availability_columns(spec),
                             covariate_imputations=CONFIRM_IMPUTATIONS)
        # The pre-registration is on disk before the holdout is read.
        result = confirm(prereg, sealed, out, ledger_key=holdout_key(spec, cdr))
        state["confirmation"] = result
        state["report"] = f"{state.get('report') or ''}\n\n{confirmation_markdown(result)}"
    return save_outputs(state, spec.name, out_dir, recorder)


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
    on_tool_call: Any = None,
    log: Any = print,
    rerun: bool = False,
    on_agent: Any = None,
) -> dict[str, Any]:
    """Verify the catalog, screen feasibility, then run CoDaS on the top-K outcomes.

    With ``per_kind`` (the default) the top K is taken for each outcome kind in the catalog, so one
    run covers measurements and conditions alike. ``outcome_names`` names catalog entries to run instead of the ranked top K; they still have to pass the
    concept check and have at least one qualifying participant.

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
    base, failed_covariates = feas.verify_covariates(base, cdr, client)
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
    outcomes = feas.load_catalog(catalog_path)
    problems = feas.verify_catalog(outcomes, cdr, client)
    verified = [o for o in outcomes if problems.get(o.name) is None]
    log(f"catalog: {len(verified)}/{len(outcomes)} outcomes verified against the concept table")

    rows = feas.feasibility(verified, base, cdr, client)
    if outcome_names:
        unknown = sorted(set(outcome_names) - {o.name for o in outcomes})
        if unknown:
            raise ValueError(f"Not in the catalog: {', '.join(unknown)}.")
        chosen = [r for r in rows if r["name"] in set(outcome_names) and r["n_participants"] > 0]
        skipped = sorted(set(outcome_names) - {r["name"] for r in chosen})
        if skipped:
            log(f"skipped (failed the concept check or no qualifying participants): {', '.join(skipped)}")
    else:
        chosen = feas.rank_outcomes(rows, top_k=top_k, min_measurement_n=min_measurement_n, min_cases=min_cases,
                                    per_kind=per_kind)
    table = feas.feasibility_table(rows, problems, chosen)
    log(table)

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    keys = [holdout_key(base, cdr)] if base.holdout_fraction > 0 else []
    summary_path = _write_summary(out, table, runs, keys)
    if not screen_only:
        by_name = {o.name: o for o in verified}
        for row in chosen:
            o = by_name[row["name"]]
            spec = feas.spec_for(o, base)
            key = _run_key(spec, goal_for(spec, o.label))
            done = None if rerun else _finished_run(out, spec.name, key)
            if done:
                log(f"\n=== {o.label} === finished earlier with this code; reusing its report")
                runs.append(_refresh_pdf(out, done, key))
            else:
                log(f"\n=== {o.label} ===")
                try:
                    result = run_spec(spec, cdr, label=o.label, client=client, out_dir=out, check_schema=False,
                                      on_tool_call=on_tool_call, on_agent=on_agent)
                except Exception as exc:  # noqa: BLE001 - one outcome's failure must not discard the others
                    reason = mask_identifiers(f"{type(exc).__name__}: {str(exc)[:300]}")
                    log(f"  failed: {reason}\n  Run again to retry; finished outcomes are reused.")
                    runs.append({"cohort": spec.name, "failed": reason})
                else:
                    (out / f"{spec.name}_run.json").write_text(json.dumps({"key": key, "summary": result}),
                                                               encoding="utf-8")
                    runs.append(result)
            summary_path = _write_summary(out, table, runs, keys)

    return {"table": table, "chosen": [r["name"] for r in chosen], "runs": runs, "summary_path": str(summary_path),
            "rows": mask_small_counts(rows)}
