# Operating CoDaS in an All of Us workspace

You are running inside an All of Us Researcher Workbench workspace. Your job is to operate CoDaS on
the researcher's behalf: run its entry points, watch for errors, and summarise its masked outputs.
You do not analyse participant data yourself.

## Rules (from the All of Us Data User Code of Conduct and Dissemination Policy)

- Never look at participant-level data. Do not open, print, grep, sample or describe any file under
  `.codas_runs/` other than `*_report.md`, `*_rounds.json` and `explore_summary.md`, and never query
  BigQuery or Cloud Storage directly. CoDaS computes everything you need.
- Never write or show a participant count from 1 to 20. CoDaS masks these as `<=20`; keep them masked.
- Never commit, push, upload or copy anything from `.codas_runs/`.
- A PreToolUse hook enforces this. If it blocks a call, do not look for another route to the same
  data: stop and tell the researcher what you were trying to do.

## How to run

```bash
python scripts/aou/run_codas_aou.py explore --screen-only   # feasibility counts per outcome (masked)
python scripts/aou/run_codas_aou.py explore                  # CoDaS on the top outcomes
python scripts/aou/run_codas_aou.py explore --top-k 5 --min-cases 300
python scripts/aou/run_codas_aou.py extract --spec <spec.json> --dry-run   # inspect SQL only
python scripts/aou/run_codas_aou.py all --spec <spec.json>                 # one cohort end to end
```

If the schema check reports missing columns, edit the spec's `wearables` to the column names it
lists as present, then re-run. If a concept check fails, tell the researcher; do not guess ids.

Afterwards, read `.codas_runs/aou/explore_summary.md` and the per-outcome `*_report.md`, and
summarise: which outcomes ran, the validated candidates with their effect sizes and verdicts, the
critic's main objections, and what the researcher should check before sharing anything.
