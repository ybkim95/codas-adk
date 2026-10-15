"""Run CoDaS on All of Us data inside the Researcher Workbench.

    # 1. build the participant-level table from the CDR (stays on the workspace disk)
    python scripts/aou/run_codas_aou.py extract --spec scripts/aou/hba1c_fitbit.example.json
    python scripts/aou/run_codas_aou.py extract --spec ... --dry-run      # print the SQL only

    # 2. run the agents on it (Vertex AI in the workspace project)
    python scripts/aou/run_codas_aou.py run --csv .codas_runs/aou/hba1c_fitbit.csv \
        --goal "Discover wearable biomarkers of HbA1c; the target is hba1c, adjust for age and female"

    # or both
    python scripts/aou/run_codas_aou.py all --spec ... --goal "..."

This script turns on CODAS_AOU_MODE, so the agent graph refuses any backend other than Vertex AI,
the model never sees a participant row, and counts of 1-20 are masked in what the model reads and in
the saved report. The report and round summary are written to .codas_runs/aou/ (git-ignored). Check
them against the Dissemination Policy yourself before anything leaves the Workbench.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CODAS_AOU_MODE"] = "1"

from codas.aou.cohort import DEFAULT_OUT_DIR, CohortSpec, build_cohort_sql, extract_cohort, resolve_cdr  # noqa: E402
from codas.aou.privacy import MASK, SMALL_CELL_MAX, mask_small_counts, mask_small_counts_in_text  # noqa: E402

DEFAULT_GOAL = (
    "Discover and validate wearable biomarkers of the outcome in this All of Us cohort. The first column "
    "is the outcome; age and female are confounders. Iterate and write a short report."
)


def _masked_count(n: int) -> str:
    return MASK if 1 <= n <= SMALL_CELL_MAX else f"{n:,}"


def cmd_extract(args: argparse.Namespace) -> Path | None:
    spec = CohortSpec.from_json(args.spec)
    if args.dry_run:
        sql, params = build_cohort_sql(spec, resolve_cdr(args.cdr))
        print(sql)
        print("\n-- parameters:\n" + "\n".join(f"--   @{n} {t} = {v!r}" for n, t, v in params))
        return None
    path = extract_cohort(spec, args.cdr, args.out_dir, check_schema=not args.skip_schema_check)
    rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print(f"wrote {path} ({_masked_count(rows)} participants). It is participant-level data: keep it in "
          "the Workbench.")
    return path


async def _run_agents(csv: Path, goal: str) -> dict:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from codas.agents.agent import root_agent  # imported here: AoU mode checks the backend on import

    sessions = InMemorySessionService()
    sid = uuid.uuid4().hex[:12]
    await sessions.create_session(app_name="codas_aou", user_id="aou", session_id=sid,
                                  state={"csv_path": str(csv)})
    runner = Runner(app_name="codas_aou", agent=root_agent, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=goal)])
    async for event in runner.run_async(user_id="aou", session_id=sid, new_message=message):
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            if getattr(part, "function_call", None):
                print(f"  {event.author:24s} -> {part.function_call.name}()")
    session = await sessions.get_session(app_name="codas_aou", user_id="aou", session_id=sid)
    return dict(session.state)


def cmd_run(args: argparse.Namespace, csv: Path | None = None) -> None:
    csv = (csv or Path(args.csv)).expanduser().resolve()
    state = asyncio.run(_run_agents(csv, args.goal))
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = csv.stem
    report = mask_small_counts_in_text(str(state.get("report") or ""))
    (out / f"{stem}_report.md").write_text(report, encoding="utf-8")
    summary = mask_small_counts({"rounds": state.get("rounds", []), "fact_sheet": state.get("fact_sheet", {})})
    (out / f"{stem}_rounds.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("-" * 96)
    print(report or "(no report produced)")
    print(f"\nsaved {out / (stem + '_report.md')} and {out / (stem + '_rounds.json')}. Review them against "
          "the Dissemination Policy before sharing anything outside the Workbench.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
        if name in ("extract", "all"):
            p.add_argument("--spec", required=True)
            p.add_argument("--cdr", default=None, help="project.dataset (default: CODAS_AOU_CDR or WORKSPACE_CDR)")
            p.add_argument("--skip-schema-check", action="store_true")
        if name == "extract":
            p.add_argument("--dry-run", action="store_true", help="print the SQL and parameters, run nothing")
        if name == "run":
            p.add_argument("--csv", required=True)
        if name in ("run", "all"):
            p.add_argument("--goal", default=DEFAULT_GOAL)
    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        args.dry_run = False
        cmd_run(args, csv=cmd_extract(args))


if __name__ == "__main__":
    main()
