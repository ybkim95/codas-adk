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
    "the {window}-day window before each participant's index date. Iterate and write a short report."
)


def goal_for(spec: CohortSpec, label: str | None = None) -> str:
    kind = spec.outcome.kind
    note = "first qualifying lab value" if kind == "measurement" else "1 = diagnosed, 0 = not"
    return GOAL_TEMPLATE.format(label=label or spec.outcome.name, name=spec.outcome.name, kind=kind,
                                target_note=note, window=spec.window_days)


async def _run_agents_async(csv: Path, goal: str, on_tool_call: Any = None) -> dict[str, Any]:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from codas.agents.agent import root_agent  # AoU mode checks the model backend on import

    sessions = InMemorySessionService()
    sid = uuid.uuid4().hex[:12]
    await sessions.create_session(app_name="codas_aou", user_id="aou", session_id=sid,
                                  state={"csv_path": str(csv)})
    runner = Runner(app_name="codas_aou", agent=root_agent, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=goal)])
    async for event in runner.run_async(user_id="aou", session_id=sid, new_message=message):
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            call = getattr(part, "function_call", None)
            if call and on_tool_call:
                on_tool_call(event.author, call.name)
    session = await sessions.get_session(app_name="codas_aou", user_id="aou", session_id=sid)
    return dict(session.state)


def run_agents(csv: str | Path, goal: str, on_tool_call: Any = None) -> dict[str, Any]:
    """Run the CoDaS agent graph on ``csv`` with the AoU guard on, and return the final session state."""
    os.environ["CODAS_AOU_MODE"] = "1"
    check_llm_backend()  # the graph may have been imported before AoU mode was switched on
    return asyncio.run(_run_agents_async(Path(csv).expanduser().resolve(), goal, on_tool_call))


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
        "top_candidates": [f"{c.get('feature')} (rho={c.get('rho')}, {c.get('verdict')})"
                           for c in (last.get("passing_candidates") or [])[:5]],
        "report_path": str(out / f"{stem}_report.md"),
    })


def run_spec(spec: CohortSpec, cdr: str | None = None, *, label: str | None = None, client: Any = None,
             out_dir: str | Path = DEFAULT_OUT_DIR, check_schema: bool = True, on_tool_call: Any = None) -> dict[str, Any]:
    """Extract one cohort, run the agents on it, save the masked outputs."""
    csv = extract_cohort(spec, cdr, out_dir, client=client, check_schema=check_schema)
    state = run_agents(csv, goal_for(spec, label), on_tool_call)
    return save_outputs(state, spec.name, out_dir)


def explore(
    catalog_path: str | Path,
    base: CohortSpec,
    cdr: str | None = None,
    *,
    client: Any = None,
    top_k: int = 3,
    min_measurement_n: int = feas.MIN_MEASUREMENT_N,
    min_cases: int = feas.MIN_CASES,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    screen_only: bool = False,
    on_tool_call: Any = None,
    log: Any = print,
) -> dict[str, Any]:
    """Verify the catalog, screen feasibility, then run CoDaS on the top-K outcomes.

    Returns the masked screen table, the chosen outcomes and one summary per run, and writes
    ``explore_summary.md`` next to the per-outcome reports.
    """
    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT") or None)
    cdr = resolve_cdr(cdr)
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
    chosen = feas.rank_outcomes(rows, top_k=top_k, min_measurement_n=min_measurement_n, min_cases=min_cases)
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
                  *[f"- {c}" for c in r["top_candidates"]], f"- report: {r['report_path']}", ""]
    summary_path = out / "explore_summary.md"
    summary_path.write_text(mask_small_counts_in_text("\n".join(lines)), encoding="utf-8")
    return {"table": table, "chosen": [r["name"] for r in chosen], "runs": runs, "summary_path": str(summary_path)}
