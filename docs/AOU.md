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
export GOOGLE_CLOUD_LOCATION=us-central1
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
* the output has no person_id or dates.

The Fitbit column names in the example follow earlier CDR documentation. The extractor stops with a
list of missing columns if your CDR differs; check with `bq show --schema <cdr>.<table>`.

## Open questions to confirm with All of Us

Ask the User Support Hub or the Resource Access Board before a production run:

1. Are Vertex AI generative models (Gemini; Claude via Model Garden) enabled in Workbench 2.0 All of Us
   workspaces, and is sending aggregate results to them acceptable under the Code of Conduct?
2. Is Google Search grounding acceptable when the query contains no participant data?
3. What is the CDR dataset identifier in Workbench 2.0 (the legacy Workbench exposed `WORKSPACE_CDR`)?
