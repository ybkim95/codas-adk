"""Run CoDaS on All of Us cohorts: one table, or a free exploration over a catalog of outcomes.

Shared by ``scripts/aou/run_codas_aou.py`` and the Workbench notebook. ``run_agents`` turns on
``CODAS_AOU_MODE`` and re-checks the model backend before every run, so the privacy guard is on
whichever entry point is used. Everything written goes under ``.codas_runs/aou/`` and is masked for
counts of 1-20.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from codas.aou import cohort
from codas.aou import feasibility as feas
from codas.aou.cohort import DEFAULT_OUT_DIR, CohortSpec, extract_cohort, resolve_cdr
from codas.aou.privacy import check_llm_backend, mask_small_counts, mask_small_counts_in_text

GOAL_TEMPLATE = (
    "Discover and validate wearable biomarkers of {label} in this All of Us cohort. The target column is "
    "{name} ({kind}: {target_note}). age and female are confounders. The features summarise Fitbit data over "
    "the {window}-day window before each participant's index date.{covariates} Iterate and write a short report."
)
COVARIATE_NOTE = (
    " These columns are clinical covariates recorded up to the index date, available for confounder "
    "adjustment and not candidate biomarkers: {columns}. Declare as confounders the ones you judge to "
    "be common causes of a candidate feature and the outcome."
)


def goal_for(spec: CohortSpec, label: str | None = None) -> str:
    kind = spec.outcome.kind
    note = "first qualifying lab value" if kind == "measurement" else "1 = diagnosed, 0 = not"
    covariates = ""
    if spec.covariates:
        columns = ", ".join(f"{c.name} ({_describe(c)})" for c in spec.covariates)
        covariates = COVARIATE_NOTE.format(columns=columns)
    return GOAL_TEMPLATE.format(label=label or spec.outcome.name, name=spec.outcome.name, kind=kind,
                                target_note=note, window=spec.window_days, covariates=covariates)


def _describe(covariate: Any) -> str:
    span = f"the {covariate.lookback_days} days" if covariate.lookback_days is not None else "any time"
    if covariate.kind == "measurement":
        return f"most recent value in {span} up to the index date"
    return f"1 if a {covariate.kind} record starts in {span} before the index date"


async def _run_agents_async(csv: Path, goal: str, on_tool_call: Any = None,
                            context_columns: list[str] | None = None) -> dict[str, Any]:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from codas.agents.agent import root_agent  # AoU mode checks the model backend on import

    sessions = InMemorySessionService()
    sid = uuid.uuid4().hex[:12]
    await sessions.create_session(app_name="codas_aou", user_id="aou", session_id=sid,
                                  state={"csv_path": str(csv), "context_columns": list(context_columns or [])})
    runner = Runner(app_name="codas_aou", agent=root_agent, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=goal)])
    async for event in runner.run_async(user_id="aou", session_id=sid, new_message=message):
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call and on_tool_call:
                on_tool_call(event.author, call.name)
    session = await sessions.get_session(app_name="codas_aou", user_id="aou", session_id=sid)
    return dict(session.state)


def run_agents(csv: str | Path, goal: str, on_tool_call: Any = None,
               context_columns: list[str] | None = None) -> dict[str, Any]:
    """Run the CoDaS agent graph on ``csv`` with the AoU guard on, and return the final session state.

    ``context_columns`` are adjustment-only columns (the spec's covariates): never screened as
    candidates, available to be declared as confounders."""
    os.environ["CODAS_AOU_MODE"] = "1"
    check_llm_backend()  # the graph may have been imported before AoU mode was switched on
    return _run_sync(_run_agents_async(Path(csv).expanduser().resolve(), goal, on_tool_call, context_columns))


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


def save_outputs(state: dict[str, Any], stem: str, out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    """Write the masked report and round summary; return a masked one-line summary of the run."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = mask_small_counts_in_text(str(state.get("report") or ""))
    rounds = state.get("rounds") or []
    fact_sheet = state.get("fact_sheet") or {}
    (out / f"{stem}_report.md").write_text(report, encoding="utf-8")
    (out / f"{stem}_rounds.json").write_text(
        json.dumps(mask_small_counts({"rounds": rounds, "fact_sheet": fact_sheet}), indent=2, default=str),
        encoding="utf-8")
    last = rounds[-1] if rounds else {}
    return mask_small_counts({
        "cohort": stem,
        "rows": fact_sheet.get("rows"),
        "rounds": len(rounds),
        "validated_count": last.get("validated_count"),
        "ml_metric": f"{last.get('ml_metric_name')}={last.get('ml_metric_value')}" if last else None,
        "confounders": list(state.get("confounder_columns") or []),
        "excluded": list(state.get("excluded_columns") or []),
        "top_candidates": [f"{c.get('feature')} (rho={c.get('rho')}, {c.get('verdict')})"
                           for c in (last.get("passing_candidates") or [])[:5]],
        "report_path": str(out / f"{stem}_report.md"),
    })


def run_spec(spec: CohortSpec, cdr: str | None = None, *, label: str | None = None, client: Any = None,
             out_dir: str | Path = DEFAULT_OUT_DIR, check_schema: bool = True, on_tool_call: Any = None) -> dict[str, Any]:
    """Extract one cohort, run the agents on it, save the masked outputs."""
    csv = extract_cohort(spec, cdr, out_dir, client=client, check_schema=check_schema)
    state = run_agents(csv, goal_for(spec, label), on_tool_call, [c.name for c in spec.covariates])
    return save_outputs(state, spec.name, out_dir)


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
) -> dict[str, Any]:
    """Verify the catalog, screen feasibility, then run CoDaS on the top-K outcomes.

    With ``per_kind`` (the default) the top K is taken for each outcome kind in the catalog, so one
    run covers measurements and conditions alike. ``outcome_names`` names catalog entries to run instead of the ranked top K; they still have to pass the
    concept check and have at least one qualifying participant.

    Returns the masked screen table, the chosen outcomes and one summary per run, and writes
    ``explore_summary.md`` next to the per-outcome reports.
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
    missing = cohort.missing_columns(base, list(zip(present["table_name"], present["column_name"])))
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

    runs = []
    if not screen_only:
        by_name = {o.name: o for o in verified}
        for row in chosen:
            o = by_name[row["name"]]
            log(f"\n=== {o.label} ===")
            runs.append(run_spec(feas.spec_for(o, base), cdr, label=o.label, client=client, out_dir=out_dir,
                                 check_schema=False, on_tool_call=on_tool_call))

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# CoDaS free exploration on All of Us", "",
             "Counts of 1-20 are masked. Review before anything leaves the Workbench.", "",
             "## Feasibility screen", "", table, "", "## Runs", ""]
    for r in runs:
        lines += [f"### {r['cohort']}", f"- participants: {r['rows']}, rounds: {r['rounds']}, "
                  f"validated: {r['validated_count']}, {r['ml_metric']}",
                  f"- adjusted for: {', '.join(r.get('confounders') or []) or 'none'}; "
                  f"excluded as leakage: {', '.join(r.get('excluded') or []) or 'none'}",
                  *[f"- {c}" for c in r["top_candidates"]], f"- report: {r['report_path']}", ""]
    summary_path = out / "explore_summary.md"
    summary_path.write_text(mask_small_counts_in_text("\n".join(lines)), encoding="utf-8")
    return {"table": table, "chosen": [r["name"] for r in chosen], "runs": runs, "summary_path": str(summary_path)}
