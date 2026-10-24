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
* literature search sends only query text: the catalog's construct and outcome names before the agents
  start, and the agents' short queries (masked, long digit runs removed) to Europe PMC and Google Search;
  `CODAS_AOU_LITERATURE_SEARCH=0` turns both off.

The guard only covers what CoDaS produces. You remain responsible for what you copy out of the
Workbench, and for confirming that the program accepts this configuration (see below).

## Setup in a workspace

```bash
git clone -b aou https://github.com/ybkim95/codas-adk && cd codas-adk
pip install -e ".[aou]"

export CODAS_AOU_MODE=1                      # set by run_codas_aou.py anyway
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
export GOOGLE_CLOUD_LOCATION=global   # the current Gemini models are served from "global"
export CODAS_AOU_CDR=<cdr-project>.<cdr-dataset>   # the workspace's CDR; or WORKSPACE_CDR if exposed

# models: optional; each tier defaults to its newest model (codas/core/model_tiers.py)
# export CODAS_GEMINI_REASONING_MODEL=gemini-4-argon
# export CODAS_GEMINI_FAST_MODEL=gemini-3.8-flash
# export CODAS_GEMINI_LITE_MODEL=gemini-3.5-flash-lite
# export CODAS_GEMINI_REASONING_MODEL=<claude model id enabled in Model Garden>
```

### Models

The agents run on three tiers, all from one table (`codas/core/model_tiers.py`) that the agent graph,
the literature search, the notebook's model probe and the execution report share:

| tier | model (newest first) | agents | thinking level |
|---|---|---|---|
| reasoning | `gemini-4-argon`, then `gemini-3.8-flash` | scout, hypotheses (HIGH), critic, defender, gap finder, per-candidate debate, mechanism, novelty, strategy, report, reviewer, reviser | MEDIUM |
| fast | `gemini-3.8-flash`, then `gemini-3.5-flash` | search step, statistical and ML interpreters | LOW |
| lite | `gemini-3.5-flash-lite`, then `gemini-3.8-flash` | convergence check, artifacts, lessons | LOW |

The notebook probes each tier's candidates newest first and uses the first that answers from the
workspace project, so a newer model is picked up by editing that one table. A tier whose model is out of
capacity or not served falls back to the other tiers' models. A model that refuses a request's thinking
level is asked again without one. Superseded names (`gemini-3.1-pro-preview`, `gemini-3-flash-preview`,
the 2.x models) run on their successor. `CODAS_GEMINI_MODEL` forces one model everywhere. List prices
(USD per 1M input and output tokens, Oct 2026): `gemini-4-argon` 2 and 10 (4 and 20 after 2026-11-30),
`gemini-3.8-flash` 0.75 and 3.75 (1.50 and 7.50 after 2026-12-31), `gemini-3.5-flash-lite` 0.30 and 2.50;
thinking tokens bill as output. `CODAS_MODEL_PRICES_JSON` overrides them for the cost estimates.

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

## Prospective design and sealed replication (default)

The example spec uses `design: "incident"`, the design of the landmark All of Us wearable papers
(e.g. Master et al., Nat Med 2022; Zheng et al., Nat Med 2024):

* **One index rule for everyone.** The index date is `window_days` after the first valid wear day
  (`valid_day`, e.g. 100-45,000 steps); features summarise that baseline window.
* **Prevalent cases and early events excluded.** Anyone with the outcome before `index + lag_days` is
  dropped (a guard against reverse causation).
* **Outcome.** A condition becomes time to a new diagnosis: the target is 1/0 and `followup_days` runs
  from `index + lag_days` to the diagnosis or the last EHR visit (`visit_occurrence`). A measurement
  becomes the first value at or after `index + lag_days` (within `horizon_days` if set). In this design
  a baseline value of the outcome's own measure is allowed as a covariate.
* **Sealed holdout.** `holdout_fraction` of participants (a deterministic, overflow-free hash of the person id and `split_salt`, `cohort.holdout_bucket`; the
  flag is computed apart from the selected columns and dropped before writing) goes to
  `sealed/<cohort>.holdout.csv`. No agent or tool can open it: the tool guardrail refuses anything under
  `sealed/`, and in All of Us mode the tools ignore any path an agent passes.
* **Pre-registration and one-shot confirmation** (`codas/aou/confirm.py`). After the agents finish,
  the final round's validated candidates are written to `<cohort>_prereg.json` with their definitions,
  discovery direction, confounders and a SHA-256 hash, before the holdout is opened. Each is then
  tested once on the holdout: Cox hazard ratio per SD with 95% CI and E-value for a time-to-event
  outcome (`codas/core/survival.py`, Efron ties, matches lifelines), partial Spearman otherwise. "Per SD"
  is the discovery share's SD: each feature's transform (clipped at the discovery 0.5th-99.5th
  percentiles, logged when strongly skewed, scaled by the discovery SD; `codas/core/preprocess.py`) is
  frozen in the pre-registration and applied unchanged to the holdout, and the unclipped estimate is
  reported beside it. The results (`<cohort>_confirmation.json`) are appended to the report as a
  deterministic section; the agents never see them.
* **Falsification on the holdout.** Every candidate is also tested, with the same confounders, for:
  * *negative-control outcomes* (`negative_controls`): conditions the exposures should not cause.
    The example uses impacted cerumen, regular astigmatism and verruca vulgaris (OMOP 374375, 380706,
    140641, from the OHDSI negative-control lists). Each control is extracted into the sealed table
    only, as `nc_<name>` (new diagnosis after `index + lag_days`; empty for anyone diagnosed earlier)
    and `nc_<name>_days`. An association (Cox p below 0.05 divided by the number of controls tested)
    flags confounding or differences in health-care contact. A control is checked against the
    concept table like a covariate, and one that is the outcome itself is dropped.
  * *dose-response*: the effect across quintiles of the feature must change monotonically in the
    pre-registered direction;
  * *lag sensitivity* (time to event): dropping the first 365 days of follow-up must keep the
    direction.
  A candidate that replicates and passes every test that could run is reported as *robust*.
  *Positive controls* (`positive_controls`, columns with a known association with the outcome, e.g.
  age and BMI) are tested the same way, to show the holdout can detect a real effect.
* **Holdout ledger** (`codas/aou/ledger.py`). Every hypothesis tested on a holdout is appended to
  `.codas_runs/aou/holdout_ledger.jsonl`, keyed by which participants it holds (CDR, `split_salt`,
  `holdout_fraction`), and identified by its outcome's definition and its feature's definition:
  * a *repeat* (the same outcome and feature, for example after a code change) is reported but not tested
    again, so re-running cannot buy another chance at a false replication;
  * an outcome already tested is *opened*: its holdout results are known, so later registrations about it
    are reported as non-confirmatory, and the selection does not choose it again;
  * established associations form a *reference* family, each tested one-sided at 0.05 like a positive
    control and never counted as a discovery;
  * every other hypothesis is *novel*: tested one-sided in its registered direction, at a level set by
    online false-discovery-rate control (LORD++, Javanmard and Montanari 2018) over the ledger's order, so
    each verdict is final when it is made and never changes as the ledger grows.

  `explore_summary.md` ends with the ledger's totals and the level the next novel test will face.

## Data understanding before hypotheses

Before the agents register anything, `set_target` computes a data dossier (`codas/core/eda.py`). It never
relates a feature to the outcome, so it is safe to read before registration. The hypotheses agent reads
its digest and can look up any column with the `data_detail` tool. It holds:
* each candidate feature: coverage, mean and SD, quantiles, skew and excess kurtosis, shares of zeros and of
  extreme values (beyond 4 SD; beyond 3 IQR from the quartiles), whether it is near-constant, and a
  natural unit (its interquartile range, rounded);
* feature families: features that correlate at |rho| >= 0.7, with a representative, so one construct is
  not registered under several names;
* wear dependence: each feature's strongest correlation with a days-worn column;
* missingness: how the people missing a feature differ in age, sex and wear (standardised mean
  differences);
* the adjustment covariates: univariate summaries only, because a covariate can stand in for the outcome;
* the outcome on its own: its distribution, or events, person-years, incidence per 1,000 person-years
  and censoring, overall and by age band and sex.

A quantile is shown only when at least 20 people lie beyond it, and any count or share resting on 1-20
people is withheld. A test permutes the outcome within age band x sex and requires every request the
agents send before registration to stay byte-identical.

Once the first discovery round closes registration, the signal map follows. For every candidate it gives:
* its association with the outcome adjusted for the scout's base confounders (T0, e.g. age and sex), then
  also for every adjustment covariate (T2);
* the share of the T0 effect the covariates explain (attenuation);
* whether missing the feature is itself associated with the outcome;
* q-values across features.

The interpreters, critic, defender and gap finder read it. The dossier and the signal map are in the
notebook, the PDF ("Data understanding") and `<cohort>_state.json`.

Set `design: "window"` and `holdout_fraction: 0` for the earlier outcome-anchored design.

## Quality rules: which days, nights, cases and participants count

The example spec applies the rules the published All of Us wearable studies share (e.g. Master et al.,
Nat Med 2022; Zheng et al., Nat Med 2024). Each is a spec setting, so another dataset sets its own:
* a valid wear day has 100-45,000 steps (`valid_day`);
* minute heart rate counts only on days with at least 600 heart-rate minutes, 10 h of data
  (`min_minutes_per_day` on a `minute_heart_rate` source); a day worn for an hour would otherwise give a
  resting or night value from that hour;
* a main-sleep night counts only with 3-16 h asleep (`value_bounds` on the daily sleep source,
  `night_minutes` on the sleep-stage source);
* an intraday phenotype needs at least 14 contributing days (`min_days`); with fewer it is missing, and
  `<prefix>_days` still says how many there were;
* heart rate during sleep comes from each person's main-sleep periods (`sleep_from`: `sleep_hr`, and
  `sleep_dip` against the 10:00-19:59 mean); `night_hr` stays the clock-hour 00:00-05:59 mean. With a
  minute-steps source as well (`steps_from`), `sedentary_hr` is the mean in awake minutes without steps
  (outside every main-sleep period), a daytime resting level that sleep does not dominate;
* a diagnosis makes a case only on two distinct dates (`min_occurrences`: the catalog's default for every
  condition, and the example's for the negative controls). People with one date are left out rather than
  counted as controls, and any diagnosis before follow-up still excludes the person;
* everyone needs an EHR visit in the 365 days before the index date (`ehr_lookback_days`), so a missing
  diagnosis means no diagnosis rather than no records;
* `ehr_visits` (distinct visit dates in the year before the index date; `kind: "visits"`) and `wear_study`
  (listed in the All of Us WEAR study table, i.e. given a device rather than bringing one; `kind:
  "in_table"`) are adjustment covariates for health-care contact and device source.

The extraction also writes a quality summary from the discovery share only (`<cohort>.qc.json`):
* for a lab outcome, the units its values were recorded in: shares in whole percent, a unit with 1-20 people
  shown only as that, quartiles only with 40 or more people. When two units each hold at least 5% of the
  values, the dossier tells the agents the outcome is not on one scale, and `unit_concept_ids` on the
  outcome (or on a catalog entry) keeps one unit;
* for every daily feature, its split-half reliability: the Spearman correlation across people between its
  value over the first and over the second half of the window, and the Spearman-Brown reliability of the
  whole-window value, 2 rho / (1 + rho), computed with at least 40 people. A feature below 0.5 is named in
  the dossier as unreliable: its two halves barely agree, so it describes those weeks more than the person.

The rules in words (`cohort.quality_rules`) are in the agents' goal, the PDF ("Data understanding") and the
notebook. The sources whose values these rules change got new prefixes (`mainsleep`, `sleepseg`, `hr10h`,
replacing `sleep`, `sleepstage`, `hrmin`). `scripts/aou/feature_definitions.json` records what every prefix
means (a hash of its settings and SQL) and the retired ones; a test fails when a definition changes under an
existing prefix, so a name in the cumulative ledger always means one thing. The rules read the same tables
as before; the phenotype cache is rebuilt once because its SQL changed (about 13 TB, roughly $75 at
on-demand prices).

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
  `kind` (`measurement`, `condition`, `drug`, `survey`; or `visits`, the distinct visit dates in the lookback
  before the index date, and `in_table`, 1 if the person is listed in `table`, which name no concept), OMOP
  `concept_ids` (descendants included by
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
   and diagnoses. `--overall` takes the top K across kinds instead. Outcomes already tested on this
   holdout are not selected again (the screen table says "already tested on this holdout"): their
   holdout results are known, so a new test of them could not be confirmatory; an outcome this exploration
   tested itself and finished with the current code is reused, not skipped. `MAX_OUTCOMES` in the
   notebook caps how many run in all, and `TEST_ON_HOLDOUT = False` runs the agents without registering or
   testing anything on the sealed half (a run that measures the agents and keeps the outcome fresh).
4. **Runs.** Each selected outcome gets a spec built from the base spec's wearables and covariates,
   is extracted, and goes through the full agent graph. `explore_summary.md` collects the masked
   screen table and, for each run, the validated candidates, the confounders adjusted for and the
   features excluded as leakage.

### Outputs of a run

Each outcome writes, under `.codas_runs/aou/` (and, for everything except participant-level tables,
mirrored to `codas_reports/aou/`, because Jupyter does not serve files inside hidden directories):

* `<cohort>_report.pdf`: the compiled report: cover (participants, code commit, models, holdout
  replication), cohort characteristics of the discovery and sealed shares (masked), the final
  report, Phase A design (target, confounders, exclusions with reasons) and hypotheses, the round
  table and every round's critic/defender debate, mechanisms and novelty, strategy and artifacts,
  and an appendix with forest plots of the holdout effects (candidates and positive controls), the
  candidates' discovery effect sizes, the phase coverage and the fact sheet. Built
  with `fpdf2` and `markdown` (pure Python); a Unicode font is used when installed. Each section is
  rendered on its own: one that fails is replaced by a note and its full traceback is written to
  `<cohort>_pdf_error.txt`. `runner.rebuild_pdf(out_dir, cohort)` compiles a PDF again from the saved
  state and trace without rerunning the agents. A reused outcome whose PDF is missing gets one this
  way when the exploration runs again (its saved summary drops the old error), and the notebook's
  outputs cell does the same for any outcome still without a PDF.
* `<cohort>_trace.jsonl` / `<cohort>_trace.md`: the full agent trace, every agent's text, tool call
  and tool response in order, labelled with phase (a child of the root agent) and discovery round.
  The responses are the ones the model received, masked for counts of 1-20.
* `<cohort>_state.json`: every phase's output (hypotheses, interpretations, critique, defense,
  mechanisms, novelty, strategy, artifacts, report, rounds, fact sheet, exclusion and confounder logs).
* Phase coverage (in the trace, the PDF and `explore_summary.md`): which agents produced output in
  each phase and round, with any silent step listed, so a reader can check that every step ran.
* `<cohort>_execution.md` / `<cohort>_execution.json`: how each agent ran, recorded by an ADK plugin
  on the runner (`codas/aou/execution.py`). It covers:
  - per agent: turns, active minutes, model calls and failures, tool calls by tool and failures, tokens
    (prompt, output, thinking, cached), the models that actually served it (a fallback shows here) and
    an estimated cost at list prices. The literature tool's own Gemini calls count for the agent that
    called it.
  - phases: how long each sequence, loop and parallel step ran.
  - design changes: who set the target, declared confounders, excluded columns, proposed features,
    registered hypotheses or recorded a gap analysis, and in which round.
  - information flow: which agent wrote a memory key that a later agent's instructions read.
  - a timeline of agent turns, parallel turns included.

  The PDF has the same tables and the timeline. The notebook shows them per outcome, with tokens and
  cost in the runs table. `explore_summary.md` gives the totals for each outcome and for the run.

`codas_reports/` and `.codas_runs/` are git-ignored. Both hold only masked aggregates and LLM text,
but review them against the Dissemination Policy before anything leaves the Workbench.

### When the model service is busy

Vertex AI answers `429 RESOURCE_EXHAUSTED` when shared capacity is short (preview models on the
`global` endpoint are the most exposed). Three layers keep an exploration going:

* each model call retries with exponential backoff and jitter for about three minutes
  (`CODAS_MODEL_RETRY_ATTEMPTS`, `CODAS_MODEL_RETRY_INITIAL_DELAY`, `CODAS_MODEL_RETRY_MAX_DELAY`);
* if it is still refused (or the model is not served), the call goes to the next model: the three tiers
  back each other up, and `CODAS_GEMINI_FALLBACK_MODELS` can add more;
* an outcome that still fails is recorded as failed and the others continue. Running the exploration
  again resumes: outcomes already finished with the same spec, goal and code commit are reused
  (`<cohort>_run.json`), so only the failed or missing ones run; `rerun=True` forces all of them.

Two time limits stop a run from hanging: a model request that sends nothing for
`CODAS_MODEL_TIMEOUT_S` (default 600) is abandoned, retried `CODAS_MODEL_TIMEOUT_RETRIES` times
(default 1) and then sent to the next model; an outcome whose agents run longer than
`CODAS_OUTCOME_TIMEOUT_MIN` (default 180 in the AoU runner: five rounds, the per-candidate debate and the
report review take 1-2 hours on a cohort of about 11,000) is stopped and recorded as failed.

### Background job and progress log

The notebook runs the exploration as a background process (`codas/aou/job.py`), so it survives a
page reload, a sleeping laptop or a dropped connection. Running the cell again re-attaches to the
running job rather than starting a second one; `job.status()` reports running, finished, failed or
stopped (the process died) from any kernel. The job writes a timestamped log of agent turns and tool
calls to `codas_reports/aou/explore.log`, masked for counts of 1-20 and for long digit runs (a
database error can quote an identifier). Each outcome's trace (`<cohort>_trace.jsonl`) is appended as
events happen, so a run that dies midway still shows where it stopped.

The notebook cell prints only the milestones (environment, feasibility, each outcome's start and end,
failures, from `explore_milestones.log`) and one status block updated in place (elapsed time, current
outcome and agent, the latest log lines), so the printed notebook stays a few dozen pages however long
the run. The log itself is kept short too:
- Each distinct Python warning is shown once per job. Numerical code silences expected notices with
  `warnings.catch_warnings()`, and every such block clears Python's per-location record (and the
  "once" filter's), so without this a warning raised in a loop prints on every pass.
- A line that comes back three times within 20 lines (a library warning or retry message in a loop)
  is counted rather than written, and the count appears as `[repeated N more times, first-last] <line>`.
  Lines repeated further apart, such as an agent's turn each round, are always written.

A new run keeps the previous log as `explore.<date-time>.log`. The Status cell lists the most repeated
lines of the current and the three most recent earlier logs (`job.log_digest`), which shows what
flooded a run.

For debugging, the job also writes `codas_reports/aou/explore_env.json` (commit, analysis-code
version, package versions, CPUs, memory, free disk, the run's settings) and a heartbeat every minute
to `explore_heartbeat.jsonl` (elapsed time, current outcome and agent, process and machine memory,
load). A log whose heartbeat simply stops, with no error, means the process or machine was stopped
(for example by the app's autostop idle time); memory falling towards zero before that means it ran
out of memory. The Status cell charts it.

Finished outcomes are reused when the exploration runs again with the same spec, goal and analysis
code (a hash of `codas/core`, `codas/agents` and the AoU cohort, feasibility, confirmation, ledger,
privacy and Table 1 modules); changes to the notebook, the job, the views, the execution report or the
PDF do not force reruns.

The notebook is generated by `scripts/aou/make_notebook.py`; a test fails when the committed notebook
differs from what the script writes, so edit the script and regenerate rather than editing the notebook.

The agents' finished work is kept (`<cohort>_agents.json`: aggregate state and trace, keyed like the run)
until the outputs are saved. If a later step fails (pre-registration, the holdout test, writing the
outputs), the next run redoes only those steps. On the holdout, every check of every hypothesis
(complete case, adherence adjustment, dose-response, lag, each negative control, within-person) runs on
its own: a numerical failure is recorded on that hypothesis and listed under the replication table
("could not be computed"), never counted as passed, and never discards the other checks. A hypothesis
whose main estimate fails counts as tested and not replicated ("test failed").

The notebook stamps its own version into the first cell and stops when the page shows an older copy
than the checkout holds (the browser does not refresh a notebook that changed on disk). Each step
shows tables and figures (`codas/aou/notebook_views.py`): the environment and model probe, the
schema and concept checks, a feasibility chart, the runs table and time per phase, per-outcome
cohort characteristics, candidates and holdout results, a forest plot across outcomes, and the job's
memory over time. Charts scale to the output column and tables (`views.fit`, and `views.wrap_tables`
for the reports' Markdown tables) wrap long identifiers at underscores, so nothing is cut off on screen
or in a printed copy of the notebook.

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
  share of rows that have each covariate and how missing values are handled. Declaring a
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

## Beyond the correlation screen (phenotypes, adherence, imputation, analyses, a-priori hypotheses)

* **Intraday digital phenotypes** (`codas/aou/phenotypes.py`, spec key `phenotypes`). From a
  minute-activity table (`steps_intraday`): relative amplitude (M10/L5), interdaily stability,
  intradaily variability, cosinor amplitude and acrophase, L5 onset, peak-1 and peak-30 cadence, and
  activity fragmentation. From a sleep-stage table (`sleep_level`): sleep midpoint mean and SD, onset
  SD, social jetlag, wake bouts, stage transitions per hour and REM latency. Each is a published
  definition, computed in BigQuery over the baseline window on the primary wearable's valid days, and
  checked against an independent numpy implementation in the tests. Column names are given per role
  as a list of acceptable names, resolved against the CDR's schema; an optional source (all phenotype
  sources, and the heart-rate zone minutes in the example spec) that is not found is skipped with a
  note. The checks cell prints every wearable table's columns so the roles can be confirmed.
* **Wear adherence is adjustment-only.** Every `<prefix>_days` column (days with data per source) is a
  context column: never screened, explained to the agents as adherence, and available as a
  confounder. On the holdout each hypothesis is re-estimated with them added (a negative-control
  exposure); `robust` also requires the effect to hold there, and adherence's own association with
  the outcome is reported as the benchmark.
* **Missing covariates are imputed.** `codas/core/imputation.py`: chained equations with posterior
  draws, the exposure and outcome as auxiliaries (plus the Nelson-Aalen hazard for time to event),
  Rubin's rules. In discovery `DiscoveryRequest.covariate_imputations` (default 0, complete-case) is
  set to 5 by the AoU runner (`CODAS_COVARIATE_IMPUTATIONS`); the holdout uses 10
  (`CODAS_CONFIRM_IMPUTATIONS`) and keeps the complete-case estimate alongside.
* **Analyses the agents choose** (`codas/core/analyses.py`, tools in `codas/agents/tools.py`): a Cox
  survival screen for incident outcomes; subgroup effects with Cochran's Q and I^2; a multivariable
  model (OLS, logistic or Cox) with each feature's independent effect and the cross-validated
  incremental value over the confounders (delta R^2, AUC or C-index, bootstrap CI); gradient boosting
  against the linear model with permutation importance and partial-dependence shapes; restricted
  cubic spline dose-response. The statistical and ML interpreters, the critic and the defender call
  them; each call is recorded (`analyses` in the state file, a table in the notebook and the PDF).
  Round summaries now give each candidate's q, confounder-adjusted rho, failed checks, and the
  engine's warnings.
* **Hypotheses registered before the data** (`register_hypotheses`). After reading the literature
  evidence and before the first round, the Researcher registers falsifiable hypotheses (feature, predicted
  direction, construct, mechanism, a citation by key). They are tested on the holdout whatever discovery
  finds, as the `a_priori` family: one the counted literature classes as established is a reference
  (tested at alpha on its own), the rest join discovery's `exploratory` candidates in the novel family,
  under online false-discovery-rate control across the cumulative ledger. Registration closes when the
  first round runs.
* **Within-person mixed models** (spec key `panel`, measurement outcomes): every later outcome value is
  paired with the wearable summary of the `period_days` before it (`<name>.panel.csv`, same holdout
  split, a pseudonymous person index). `run_mixed_model` fits a random-intercept model with the
  exposure split into within-person and between-person parts (`codas/core/mixed.py`); a feature that
  only marks a stable trait shows a within-person effect near zero. The holdout reports each
  pre-registered feature's within-person effect. Intraday phenotypes are not in the panel.

## Checks that keep a result honest, and the wider data

**Collinear covariates.**
- `cox_ph` leaves out covariate columns that are constant or a linear combination of earlier ones
  (pivot by rank; the exposure column is never displaced) and reports them (`dropped_covariates`).
- Example: two wearable sources that count the same days give identical columns. v8 adjusted for both,
  which made every negative-control Cox model singular and `nan`.

**`nan` never passes.**
- A negative control or lag check whose estimate is not finite is "could not be computed" and is not
  counted as passed.
- `robust` needs at least one falsification test that actually ran.
- `confirm.audit` lists what could not be computed or should not be trusted as is: estimates, checks
  that ran for no hypothesis, covariates dropped as collinear, positive controls not computed or not
  detected. It prints above the replication table.

**Preflight.**
- Before any agent runs, `runner.preflight` runs the whole holdout pipeline on the discovery share (never
  the sealed one, no ledger entry), with one stand-in hypothesis per wearable and phenotype source and
  every covariate the agents could adjust for.
- Anything it cannot compute stops that outcome in minutes with the reason.
- Columns with identical values are named in the agents' goal.
- `CODAS_PREFLIGHT=0` turns it off.

**Repeated analyses.** An analysis tool called again with the same arguments and confounders returns
the recorded result instead of recomputing it.

**Permutation nulls.** The ML method pool uses 19 draws for every model, so the smallest p is 0.05.

**Concepts by public code.**
- Outcomes, covariates and negative controls can name concepts by code instead of id, e.g.
  `"codes": ["SNOMED:84114007"]`, `["LOINC:8480-6"]` or `["ATC:C10AA"]`.
- `feasibility.resolve_codes` looks the codes up in the CDR's own vocabulary. The existing check then
  confirms domain and name.
- Several codes are alternatives for one quantity (eGFR is reported under MDRD and CKD-EPI formulas):
  the codes found are used and the ones missing are named in the log. An item with none found is
  skipped with the reason.
- A drug class (ATC) covers its ingredients through `concept_ancestor`.

**Survey covariates** (`"kind": "survey"`):
- `question_name` is text that must match exactly one question used in `observation` (otherwise the
  covariate is skipped, listing what matched).
- `positive_answers` is a regular expression over the answer's name, coded 1. Skips and refusals are
  missing; any other answer is 0.
- `any_time` takes the answer nearest the index date, for lifetime traits.
- The checks cell shows each question's answers, how each is coded, and how many people gave it
  (1–20 masked).

**Minute heart rate** (`"kind": "minute_heart_rate"`, an optional phenotype source):
- `table` may list several acceptable names; the first one the CDR has is used.
- Phenotypes:
  - resting heart-rate proxy: mean of the day's 30 lowest minutes
  - night and day heart rate, and the nocturnal dip
  - night minute-to-minute variability: a proxy, not HRV
  - 24-h relative amplitude
  - with `steps_from`, heart-rate rise per step/min
- These queries read the minute table, which is large. The inventory cell shows its size; BigQuery bills
  by bytes read.

**BigQuery cost.**
- Under the incident design the intraday phenotypes depend on the person only, so they are computed once
  per CDR and phenotype definition, kept in `.codas_runs/aou/phenotype_cache/` (participant-level, never
  read by the agents) and merged into each outcome's table by person. The person id is dropped before
  the table is written.
- In the run after v8, every outcome re-read about 13 TB of minute tables; now one run reads them once.
  `CODAS_PHENOTYPE_CACHE=0` computes them inline again.
- Every BigQuery job's bytes billed are recorded. Jobs over 10 GB are printed to the job log, and the run
  total, with a cost estimate at `CODAS_BQ_USD_PER_TIB` (default the $6.25/TiB on-demand list price),
  goes to the log and `explore_summary.md`. The workspace bill is the authority.

**Time-to-event direction.**
- Discovery screens an incident outcome on its event flag, which ignores how long each person was
  followed. A feature that lengthens follow-up can raise the flag while lowering the hazard.
- So each exploratory candidate's pre-registered direction comes from a Cox model on the discovery
  share, adjusted for the final confounders, as the holdout tests it (`discovery_hr`). A disagreement
  with the screen's sign is noted and listed in the audit. A-priori hypotheses keep the direction they
  predicted.
- In the run after v8, a hyperlipidemia candidate had a binary screen rho of +0.20 but a holdout HR of
  0.85 (q 5e-5). It was labelled "not replicated" for the sign alone. Its holdout test is now spent;
  only new data can confirm it.

**Wear adherence already adjusted for.** Days-with-data columns that are already confounders, or
duplicate one, are left out of the adherence check. When none is left the check is reported as not
counted, never as passed.

**Report and holdout side by side.**
- Each report opens with one table per pre-registered hypothesis: the discovery engine's verdict, the
  debate's, and the sealed holdout's (with q and robust).
- The agents' narrative is written before the holdout is opened, so it cannot know the holdout result.
- The report writer and reviewer also get every analysis tool run with its count. The one-line digest
  shows only the latest analyses, and a report once said a mixed model was not run when it had run four
  times.

**Literature search.** A 429 from the search tool's quota is retried with exponential backoff and jitter
(`CODAS_GROUNDING_RETRIES`, default 3). Every query goes to the search again: Google's grounding terms
forbid caching or reusing its results, so the per-run cache of Phase 33 was removed. Each search's
tokens, served model, Google Search queries, tries and rate-limited tries are recorded for the agent that
asked.

**Literature sources** (checks cell, `codas/core/literature.py`). One tiny query to each source shows what
answers from the workspace (status, latency, bytes received). Only a fixed word is sent, never data, and
every literature request is counted and capped (`CODAS_LITERATURE_MAX_REQUESTS`), because All of Us
monitors outbound traffic. The sources:
- Europe PMC, Crossref, OpenAlex, NCBI E-utilities (PubMed) and Semantic Scholar: open bibliographic APIs
  that return PMIDs and DOIs and allow caching, so citations can point to them.
- Nature and the other Springer Nature journals: found by DOI through those indexes; the probe checks
  the Nature Portfolio prefix 10.1038 through Crossref. Springer Nature's own Metadata API is tried only
  when `SPRINGER_NATURE_API_KEY` is set as a workspace secret.
- Google Search grounding: web context for the agents' reasoning; its links are not citations.
- Google Scholar: not used, because it has no API and its terms forbid automated queries.

**Literature evidence and citations** (`codas/core/literature.py`, `scripts/aou/literature_terms.json`).
Before the agents start, the runner maps each wearable feature to the construct it measures (a versioned
catalog: an intraday phenotype by its source kind and name, a daily column by a name pattern) and counts in
Europe PMC, for each construct and the outcome, the papers naming both in the title or abstract, the reviews
and meta-analyses among them, and the earliest year; the most cited papers come with their keys. The class
is counted, not judged: established (at least one review or meta-analysis, or ten papers;
`CODAS_NOVELTY_ESTABLISHED_REVIEWS`, `CODAS_NOVELTY_ESTABLISHED_PAPERS`), emerging (fewer), unexplored (none),
not assessed (the service did not answer; after the first such answer the rest are not asked). The synonyms
are fixed in the catalog, so no phrasing chosen during a run can make a construct look newer than it is.
* The Researcher, the Mechanism and the Novelty agents read this evidence and can call `search_papers`
  (Europe PMC, with abstracts) for more. A paper is cited only by its key, `[PMID:...]` or `[DOI:...]`, and
  only if it was retrieved in this run; web context from Google Search grounding is not a citation.
* `register_hypotheses` refuses a hypothesis that cites no retrieved paper when papers were retrieved, or
  that cites a key nobody retrieved. Its knownness is the evidence class of the feature's construct (`not
  assessed` when the construct was not counted); the agent's own label is kept as `declared_knownness`. An
  established hypothesis is a reference (tested at alpha on its own, never a discovery); any other joins
  the novel family under online false-discovery-rate control.
* After the report, every key in the registered hypotheses, mechanisms, novelty and report is checked
  against what the run retrieved. The report gets a numbered reference list of the retrieved papers it
  cites, and a note naming any other key as not a citation (`citations` in `<cohort>_state.json`).
* Responses are cached in `.codas_runs/aou/literature_cache.jsonl` (these APIs allow it), so a repeated
  query is not sent again; each request is counted and capped as above.

**The research basis** (`docs/research/`). The papers CoDaS's design rests on: 51 wearable-biomarker studies
and 77 biomarker methods papers and exemplars, compiled from abstracts, with a table of each recipe step and
where CoDaS implements it. The checks cell looks every DOI in them up in Crossref from the workspace and
writes the result to `codas_reports/aou/references.json` (registered or not, the registered title, journal
and year, and how much of that title the entry contains).

**Data inventory** (checks cell):
- Every table with its columns, the participants each person-level table covers, and how many of them
  have Fitbit data (masked).
- Tables shaped like minute heart rate, found by their columns.
- Table sizes.

## Fidelity to the paper's CoDaS (v4) on this branch

| paper component | here |
|---|---|
| deterministic runners + LLM interpreters | engine (`codas/core`), `MethodRunner`, analysis tools |
| statistical method pool chosen per round | partial correlation, subgroup stratified, Huber robust regression, propensity stratification, mutual information, mediation, survival screen (`codas/core/methods.py`); the paper's "mixed-effects proxy" is replaced by a real within-person mixed model |
| ML method pool, nested CV | gradient boosting (sklearn histogram GBM), elastic net, random forest, lasso, ridge, Huber, logistic, each with nested CV, a bootstrap CI and a permutation null; XGBoost / LightGBM are not dependencies |
| gap finder, constraints unioned | `gap_finder` + `record_gap_analysis` |
| stop rules (pools exhausted, Jaccard > 0.9 twice, no new method/candidate) | `check_convergence` when the pools are active; up to 5 rounds on AoU |
| per-candidate critic -> defender debate | `candidate_debate` with recorded verdicts; the defender has analysis tools (the paper's had none) |
| event pruning between phases | `include_contents="none"` + memory placeholders |
| novelty / mechanism with search | as before, now also reading the registered hypotheses and debates |
| report review and revision | `report_reviewer` -> `report_reviser`; one document, not per-section writers |
| LaTeX report | fpdf2 PDF with phase outputs, method and gap tables, debates, lessons, forest / subgroup / dose-response / importance figures; no LaTeX in the Workbench |
| 80/20 row split | sealed 50% participant split, pre-registration hash, cumulative ledger, falsification battery (stricter) |
| cost table | per-agent execution report (calls, tokens, tools, time, estimated cost, models served), with the agents' information flow and design changes |
| not ported | causal-model (DAG) generator, reinvestigation planner as a separate agent (the gap finder covers re-running methods on candidates), literature updater inside the loop, section-wise parallel writers |

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
