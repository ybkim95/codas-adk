# Running CoDaS on All of Us

This branch runs CoDaS inside the All of Us Researcher Workbench 2.0 (Verily Workbench, Google
Cloud), with the agents served by Vertex AI in the workspace's own project. Nothing here changes the
public CoDaS behaviour unless `CODAS_AOU_MODE=1` is set.

## Where things run

```
Researcher Workbench (workspace GCP project)
├─ BigQuery CDR (OMOP) ──► codas.aou.cohort ──► .codas_runs/aou/<cohort>.csv   participant-level, stays on disk
├─ deterministic engine (codas.core) reads the CSV, computes every statistic
├─ privacy guard (codas.aou.privacy): no rows to the model, counts 1-20 masked
└─ Vertex AI (Gemini, or Claude via Model Garden) ◄── aggregates only
```

The Data User Code of Conduct forbids giving workspace access to an AI tool that could result in the
export of participant-level data. In AoU mode:

* the agent graph refuses to start unless `GOOGLE_GENAI_USE_VERTEXAI=TRUE` and `GOOGLE_CLOUD_PROJECT`
  are set, and refuses if `GOOGLE_API_KEY`, `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` is present;
* `preview_columns` returns column names and types only, never rows;
* every tool response, and the `fact_sheet` / `rounds` memory the prompts interpolate, has
  participant counts of 1-20 replaced by `<=20` (Data and Statistics Dissemination Policy);
* the final report is masked the same way;
* literature grounding sends only LLM-written literature queries (masked) to Google Search;
  `CODAS_AOU_LITERATURE_SEARCH=0` turns it off.

The guard only covers what CoDaS produces. You remain responsible for what you copy out of the
Workbench, and for confirming that the program accepts this configuration (see below).

## Setup in a workspace

```bash
git clone -b aou https://github.com/ybkim95/codas-adk && cd codas-adk
pip install -e ".[aou]"

export CODAS_AOU_MODE=1                      # set by run_codas_aou.py anyway
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
export GOOGLE_CLOUD_LOCATION=global   # preview Gemini models are served from "global"; us-central1 returns 404 for them
export CODAS_AOU_CDR=<cdr-project>.<cdr-dataset>   # the workspace's CDR; or WORKSPACE_CDR if exposed

# models: Gemini by default; any claude-* name is served as Claude on Vertex AI
export CODAS_GEMINI_REASONING_MODEL=gemini-3.1-pro-preview
export CODAS_GEMINI_FAST_MODEL=gemini-3-flash-preview
# export CODAS_GEMINI_REASONING_MODEL=<claude model id enabled in Model Garden>
```

Check that Vertex AI is enabled before the first run:

```bash
gcloud services list --enabled | grep aiplatform
```

## Run

```bash
# print the extraction SQL without running it
python scripts/aou/run_codas_aou.py extract --spec scripts/aou/hba1c_fitbit.example.json --dry-run

# build the table (checks every column against INFORMATION_SCHEMA first)
python scripts/aou/run_codas_aou.py extract --spec scripts/aou/hba1c_fitbit.example.json

# run the agents
python scripts/aou/run_codas_aou.py run --csv .codas_runs/aou/hba1c_fitbit.csv \
  --goal "Discover wearable biomarkers of HbA1c. The target is hba1c; age and female are confounders."
```

The report and round summary land in `.codas_runs/aou/`, which is git-ignored. Do not commit or push
anything from that directory: pushing to GitHub is an export.

## Cohort specs

A spec names an outcome by OMOP concept id and the wearable tables to summarise; see
`scripts/aou/hba1c_fitbit.example.json`.

* `outcome.kind = "measurement"`: index date is the first qualifying measurement with wearable data in
  the `window_days` before it; the target is `value_as_number`, with optional `value_min`/`value_max`.
* `outcome.kind = "condition"`: cases are indexed at first diagnosis, everyone else at their last
  wearable day; the target is 1/0. The two groups get index dates by different rules, so treat
  results as hypothesis-generating.
* each wearable column becomes `<prefix>_<column>_mean` and `_sd` over the window, plus
  `<prefix>_days`; `filter_column`/`filter_value` selects rows (e.g. `is_main_sleep = true`).
* `covariates` (optional) adds clinical context columns for confounder adjustment. Each names a
  `kind` (`measurement`, `condition` or `drug`), OMOP `concept_ids` (descendants included by
  default) and a `lookback_days` (default 365; `null` for no limit; the example spec uses `null` for
  BMI, since a 365-day lookback left most participants without a value). A measurement covariate is the
  most recent value in the lookback up to and including the index date (`value_min`/`value_max`
  apply); a condition or drug covariate is 1 if a record starts in the lookback before the index
  date. `expect_name` is checked against the concept table before a run, like the catalog. A
  covariate that shares a concept id with the outcome is refused in a single spec and left out
  automatically in an exploration.
* the output has no person_id or dates.

The Fitbit column names in the example follow earlier CDR documentation. The extractor stops with a
list of missing columns if your CDR differs; check with `bq show --schema <cdr>.<table>`.

## Free exploration

The fastest path is the notebook `notebooks/aou_codas_quickstart.ipynb`: open it in a workspace
JupyterLab app and choose Run > Run All Cells. Nothing needs editing. The first cell fast-forwards the
checkout to `origin/aou` (resetting the notebook's saved outputs, which would otherwise block the
pull) and says so if the notebook itself changed and needs reloading; the second finds the CDR
(`wb resource list`) and a Vertex location and Gemini models that answer from the workspace project.
The same flow from a terminal:

```bash
python scripts/aou/run_codas_aou.py explore --screen-only   # check + count, no agents
python scripts/aou/run_codas_aou.py explore                  # CoDaS on the top 3 outcomes of each kind
```

1. **Catalog check.** `scripts/aou/outcomes_catalog.json` lists about 15 Registered Tier outcomes
   (labs and conditions). Each outcome's concept ids must exist in the CDR concept table, sit in the
   expected domain, and have a name containing `expect_name`; failures are skipped and reported.
2. **Feasibility screen.** For each outcome, the participants who meet the extractor's wearable
   rule (and, for a condition, cases and controls), counted with the extractor's own SQL.
3. **Selection.** A fixed rule: keep measurements with at least `--min-n` participants (default
   1,000) and conditions with at least `--min-cases` cases and controls (default 200), order by that
   sample, and take `--top-k` (default 3) for each outcome kind, so one run covers lab measurements
   and diagnoses. `--overall` takes the top K across kinds instead.
4. **Runs.** Each selected outcome gets a spec built from the base spec's wearables and covariates,
   is extracted, and goes through the full agent graph. `explore_summary.md` collects the masked
   screen table and, for each run, the validated candidates, the confounders adjusted for and the
   features excluded as leakage.

### Outputs of a run

Each outcome writes, under `.codas_runs/aou/` (and, for everything except participant-level tables,
mirrored to `codas_reports/aou/`, because Jupyter does not serve files inside hidden directories):

* `<cohort>_report.pdf`: the compiled report: cover (participants, code commit, models), the final
  report, Phase A design (target, confounders, exclusions with reasons) and hypotheses, the round
  table and every round's critic/defender debate, mechanisms and novelty, strategy and artifacts,
  and an appendix with the candidates' effect sizes, the phase coverage and the fact sheet. Built
  with `fpdf2` and `markdown` (pure Python); a Unicode font is used when installed.
* `<cohort>_trace.jsonl` / `<cohort>_trace.md`: the full agent trace, every agent's text, tool call
  and tool response in order, labelled with phase (a child of the root agent) and discovery round.
  The responses are the ones the model received, masked for counts of 1-20.
* `<cohort>_state.json`: every phase's output (hypotheses, interpretations, critique, defense,
  mechanisms, novelty, strategy, artifacts, report, rounds, fact sheet, exclusion and confounder logs).
* Phase coverage (in the trace, the PDF and `explore_summary.md`): which agents produced output in
  each phase and round, with any silent step listed, so a reader can check that every step ran.

`codas_reports/` and `.codas_runs/` are git-ignored. Both hold only masked aggregates and LLM text,
but review them against the Dissemination Policy before anything leaves the Workbench.

### When the model service is busy

Vertex AI answers `429 RESOURCE_EXHAUSTED` when shared capacity is short (preview models on the
`global` endpoint are the most exposed). Three layers keep an exploration going:

* each model call retries with exponential backoff and jitter for about three minutes
  (`CODAS_MODEL_RETRY_ATTEMPTS`, `CODAS_MODEL_RETRY_INITIAL_DELAY`, `CODAS_MODEL_RETRY_MAX_DELAY`);
* if it is still refused (or the model is not served), the call goes to the next model: the two tiers
  back each other up, and `CODAS_GEMINI_FALLBACK_MODELS` can add more;
* an outcome that still fails is recorded as failed and the others continue. Running the exploration
  again resumes: outcomes already finished with the same spec, goal and code commit are reused
  (`<cohort>_run.json`), so only the failed or missing ones run; `rerun=True` forces all of them.

### Choosing outcomes yourself

The ranking favours whichever outcomes have the largest effective sample. To run specific catalog
entries instead, set `OUTCOMES = ["<name>", ...]` in the notebook's second cell, or pass
`--outcomes <name>,<name>` to `run_codas_aou.py explore`.

## Engine and agent behaviour added on this branch (dataset-agnostic)

* **Agent-declared leakage.** The Researcher and Critic agents can call `exclude_features` when they
  judge that a column encodes the outcome. Later rounds drop the column and everything engineered
  from it, the engine's construct-independence gate rejects features that are near-copies of it, and
  the loop runs at least one more round on the revised design. Exclusions and their reasons appear in
  the round summaries the report is written from.
* **Reciprocal and near-duplicate features.** Near-duplicate validated features are now detected with
  rank (Spearman) correlation, like the rest of the screen, so `a/b` and `b/a` count once.
* **Robust held-out model.** Features are clipped to training-fold quantiles before the held-out model
  is fit, so a single extreme row in a test fold can no longer drive cross-validated R^2 far below zero.
  On the bundled synthetic example this moves the held-out metric in the third decimal only; the
  full-report golden hash was re-pinned for it.
* **Adjustment-only covariates.** `DiscoveryRequest.context_columns` (empty by default) names columns
  that never enter the candidate screen but can be declared as confounders. The AoU runner fills it
  with the spec's covariates and lists them in the goal; the Researcher and Critic decide, with a
  recorded reason, whether to adjust for each (`declare_confounders`). Each round summary shows the
  share of rows that have each covariate, since adjusting drops the rows that lack it. Declaring a
  confounder after a round has run revises the design, so the loop runs at least once more with it.
* **Blocked columns stay blocked.** The target, the participant/time columns and every excluded,
  confounder and context column are kept out of engineered products and out of agent-proposed
  features (the proposal is refused, and the engine skips it with a warning if one arrives anyway),
  so a column excluded as leakage cannot return inside an interaction term.
* **Permutation null with enough draws.** The held-out model's permutation null always has at least
  39 draws, so p = 0.025 is attainable; before, large cohorts used 4 draws and could never report p
  below 0.2. The draw count is in the fact sheet (`ml_null_draws`). This moved only the null fields of
  the full-report golden fingerprint, which was re-pinned.
* **Design stated in the goal.** The goal tells the agents how the index date is fixed for the
  outcome kind, so a condition's pre-diagnosis window is not read as post-diagnosis behaviour.
* **Chance-claim audit.** If the report says the model beats chance while the engine's permutation
  test says it does not, a correction note is appended to the report.

## Testing outside the Workbench

`codas/aou/synthetic.py` builds a synthetic All of Us–shaped CDR in DuckDB (no real people; planted
associations), and `duckdb_query` runs the extractor's BigQuery SQL against it via sqlglot.

* `tests/test_aou_sql.py` checks the SQL against a hand-worked micro-CDR (window bounds, the
  index-day exclusion, the nap filter, re-indexing to a later measurement, the minimum-wear rule)
  and runs the catalog check, the feasibility counts and the screen on the synthetic CDR.
* `tests/test_aou_e2e.py` runs the whole agent graph in AoU mode with a scripted model that calls
  the real tools, records every request sent to the model, and asserts that none carries a
  participant row or a count of 1-20; it also runs `explore` end to end on the synthetic CDR.

## Claude Code inside the workspace (opt-in)

Workbench cloud apps ship with Claude Code. `workbench/claude/` holds a configuration that runs it
on Vertex AI in the workspace project and lets it operate CoDaS (run the notebook or
`run_codas_aou.py`, read the masked outputs, summarise them) without reading participant data:

* a PreToolUse hook (`aou_guard.py`) blocks reading `.codas_runs/` except masked outputs, direct
  BigQuery or Cloud Storage access, interpreters other than the CoDaS entry points, recursive grep
  and `find -exec`, writing code that reads the tables, and edits to the guard or privacy module;
* deny rules repeat the main file blocks, and a CLAUDE.md tells Claude to stop and ask when blocked.

The hook is an allow-list, not proof of isolation. All of Us has not published guidance on AI
coding agents in workspaces, so the installer refuses to run until you confirm you have the
program's written agreement:

```bash
scripts/aou/install_claude_guard.sh --confirmed   # copies into .claude/ (git-ignored)
```

## Open questions to confirm with All of Us

Ask the User Support Hub or the Resource Access Board before a production run:

1. Are Vertex AI generative models (Gemini; Claude via Model Garden) enabled in Workbench 2.0 All of Us
   workspaces, and is sending aggregate results to them acceptable under the Code of Conduct?
2. Is Google Search grounding acceptable when the query contains no participant data?
3. What is the CDR dataset identifier in Workbench 2.0 (the legacy Workbench exposed `WORKSPACE_CDR`)?
4. May the pre-installed Claude Code run in a workspace on Vertex AI, configured so it cannot read
   participant-level files and only operates CoDaS?
