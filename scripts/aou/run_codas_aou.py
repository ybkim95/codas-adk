"""Run CoDaS on All of Us data inside the Researcher Workbench.

    # free exploration: verify the outcome catalog, screen feasibility, run CoDaS on the top 3 of each kind
    python scripts/aou/run_codas_aou.py explore
    python scripts/aou/run_codas_aou.py explore --screen-only          # counts only, no agent runs

    # one cohort from a spec
    python scripts/aou/run_codas_aou.py extract --spec scripts/aou/hba1c_fitbit.example.json [--dry-run]
    python scripts/aou/run_codas_aou.py run --csv .codas_runs/aou/hba1c_fitbit.csv --goal "..."
    python scripts/aou/run_codas_aou.py all --spec scripts/aou/hba1c_fitbit.example.json

The agents run with CODAS_AOU_MODE on: the graph refuses any backend other than Vertex AI, the model
never sees a participant row, and counts of 1-20 are masked in what the model reads and in every
file written to .codas_runs/aou/ (git-ignored). Review those files against the Dissemination Policy
before anything leaves the Workbench.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CODAS_AOU_MODE"] = "1"

from codas.aou import runner  # noqa: E402
from codas.aou.cohort import DEFAULT_OUT_DIR, CohortSpec, build_cohort_sql, extract_cohort, resolve_cdr  # noqa: E402
from codas.aou.privacy import MASK, SMALL_CELL_MAX  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "hba1c_fitbit.example.json"
DEFAULT_CATALOG = HERE / "outcomes_catalog.json"


def _progress(author: str, tool: str) -> None:
    print(f"  {author:24s} -> {tool}()")


def cmd_extract(args: argparse.Namespace) -> Path | None:
    spec = CohortSpec.from_json(args.spec)
    if args.dry_run:
        sql, params = build_cohort_sql(spec, resolve_cdr(args.cdr))
        print(sql)
        print("\n-- parameters:\n" + "\n".join(f"--   @{n} {t} = {v!r}" for n, t, v in params))
        return None
    path = extract_cohort(spec, args.cdr, args.out_dir, check_schema=not args.skip_schema_check)
    rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
    shown = MASK if 1 <= rows <= SMALL_CELL_MAX else f"{rows:,}"
    print(f"wrote {path} ({shown} participants). It is participant-level data: keep it in the Workbench.")
    return path


def cmd_run(args: argparse.Namespace, csv: Path | None = None, spec: CohortSpec | None = None) -> None:
    csv = (csv or Path(args.csv)).expanduser().resolve()
    goal = args.goal or (runner.goal_for(spec) if spec else runner.GOAL_TEMPLATE.format(
        label="the outcome", name="the first column", kind="outcome", target_note="see the cohort spec",
        window="pre-index", index_note="", covariates=""))
    context = [c.name for c in spec.covariates] if spec else [
        c.strip() for c in (getattr(args, "covariates", "") or "").split(",") if c.strip()]
    state = runner.run_agents(csv, goal, _progress, context)
    summary = runner.save_outputs(state, csv.stem, args.out_dir)
    print("-" * 96)
    print(Path(summary["report_path"]).read_text(encoding="utf-8") or "(no report produced)")
    print(f"\nsaved under {Path(args.out_dir).resolve()}. Review before sharing anything outside the Workbench.")


def cmd_explore(args: argparse.Namespace) -> None:
    result = runner.explore(
        args.catalog, CohortSpec.from_json(args.spec), args.cdr, top_k=args.top_k, per_kind=not args.overall,
        min_measurement_n=args.min_n, min_cases=args.min_cases, out_dir=args.out_dir,
        screen_only=args.screen_only, on_tool_call=_progress,
        outcome_names=[o.strip() for o in args.outcomes.split(",") if o.strip()] or None)
    print(f"\nsummary: {result['summary_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "run", "all", "explore"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
        if name != "run":
            p.add_argument("--spec", default=str(DEFAULT_SPEC),
                           help="cohort spec (for explore: supplies the wearables, window and demographics)")
            p.add_argument("--cdr", default=None, help="project.dataset (default: CODAS_AOU_CDR or WORKSPACE_CDR)")
        if name in ("extract", "all"):
            p.add_argument("--skip-schema-check", action="store_true")
        if name == "extract":
            p.add_argument("--dry-run", action="store_true", help="print the SQL and parameters, run nothing")
        if name == "run":
            p.add_argument("--csv", required=True)
            p.add_argument("--covariates", default="",
                           help="comma-separated adjustment-only columns (never screened as candidates)")
        if name in ("run", "all"):
            p.add_argument("--goal", default=None)
        if name == "explore":
            p.add_argument("--catalog", default=str(DEFAULT_CATALOG))
            p.add_argument("--top-k", type=int, default=3, help="outcomes to run for each outcome kind")
            p.add_argument("--overall", action="store_true",
                           help="take the top K across all kinds instead of for each kind")
            p.add_argument("--min-n", type=int, default=runner.feas.MIN_MEASUREMENT_N,
                           help="minimum participants for a measurement outcome")
            p.add_argument("--min-cases", type=int, default=runner.feas.MIN_CASES,
                           help="minimum cases (and controls) for a condition outcome")
            p.add_argument("--screen-only", action="store_true", help="verify and count, run no agents")
            p.add_argument("--outcomes", default="", help="comma-separated catalog names to run instead of the top K")
    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "all":
        args.dry_run = False
        spec = CohortSpec.from_json(args.spec)
        cmd_run(args, csv=cmd_extract(args), spec=spec)
    else:
        cmd_explore(args)


if __name__ == "__main__":
    main()
