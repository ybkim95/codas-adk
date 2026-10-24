"""Write the All of Us quickstart notebook (notebooks/aou_codas_quickstart.ipynb).

The notebook is generated, never edited by hand: change this file, then run
``python scripts/aou/make_notebook.py`` (an output path may be given). A test checks that the committed
notebook is exactly what this script writes.
"""
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# CoDaS on All of Us: free biomarker exploration

Run this inside an All of Us Researcher Workbench 2.0 workspace (JupyterLab app): **Run > Run All Cells**.
Nothing needs editing; the first cell brings this checkout up to date with the `aou` branch.

1. **Setup**: update and install CoDaS (`aou` branch) and point it at Vertex AI in this workspace's project.
2. **Checks**: the CDR has the Fitbit columns the spec needs; every catalog outcome's, covariate's and
   negative control's concept ids exist, sit in the right domain and have the expected names.
3. **Feasibility screen**: participants (and cases) per outcome; counts of 1-20 are masked.
4. **Exploration**: CoDaS runs on the top outcomes of each kind (lab measurements and diagnoses) and
   writes masked reports to `.codas_runs/aou/`. The spec's covariates (e.g. BMI) are extracted for
   confounder adjustment only; the agents decide whether to adjust for them.

The default design is prospective: everyone is indexed 180 days after their first valid wear day,
the features summarise that baseline, and outcomes are counted after a 90-day lag (earlier diagnoses
exclude the person; diagnoses are analysed as time to event). Half of the participants are sealed
away: the agents work on the other half, their validated candidates are pre-registered (hashed), and
each is tested once on the sealed half (Cox hazard ratio with E-value for diagnoses, partial
correlation for lab values). Each candidate is also checked against negative-control diagnoses it
should not predict, for a dose-response across quintiles and, for diagnoses, after dropping the first
year of follow-up; known associations (age, BMI) are tested as positive controls. Every hypothesis
ever tested on the sealed half is corrected together, so re-running cannot manufacture a
replication. That replication table is the result to trust.

What the model sees: column names, aggregate statistics, and counts with 1-20 masked. Never a row.
The extracted tables stay on this workspace disk (git-ignored). Review every output against the
Data and Statistics Dissemination Policy before anything leaves the Workbench, and do not commit
`.codas_runs/`."""),
md("## 1. Setup"),
code("""import json, os, re, subprocess, sys, warnings
from pathlib import Path

NOTEBOOK_VERSION = "__NOTEBOOK_VERSION__"  # identifies this copy of the notebook; set by the generator
# Known, harmless library notices; real errors still show.
for pattern in (".*automatic function calling.*", ".*EXPERIMENTAL.*", ".*Post-Quantum.*", ".*grpcio.*"):
    warnings.filterwarnings("ignore", message=pattern)

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
assert (REPO / "codas" / "aou").exists(), f"Run this notebook from the codas-adk checkout (got {REPO})."

def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)

def update_checkout():
    \"\"\"Fast-forward this checkout to origin/aou. Jupyter saves outputs into this notebook file, which
    would block a pull, so the notebook is reset first (outputs only; there is nothing to edit in it).\"\"\"
    if git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "aou":
        print("not on the aou branch; skipping the update")
        return
    if git("fetch", "-q", "origin", "aou").returncode:
        print("could not reach GitHub; running the code already here")
        return
    old, new = (git("rev-parse", ref).stdout.strip() for ref in ("HEAD", "origin/aou"))
    if old == new or git("merge-base", "--is-ancestor", new, old).returncode == 0:
        print(f"CoDaS is up to date ({old[:7]})")
        return
    git("checkout", "--", "notebooks")
    pulled = git("pull", "-q", "--ff-only", "origin", "aou")
    if pulled.returncode:
        raise SystemExit("git pull failed:\\n" + pulled.stderr)
    print(f"CoDaS updated {old[:7]} -> {new[:7]}")
    if any(m == "codas" or m.startswith("codas.") for m in sys.modules):
        raise SystemExit("Kernel > Restart Kernel and Run All Cells, so the updated code is loaded.")

def check_this_page_is_current():
    \"\"\"Stop if this browser page shows an older copy of the notebook than the checkout holds (the page
    is not refreshed when the file changes on disk, and running an old copy runs old cells).\"\"\"
    shown = git("show", "HEAD:notebooks/aou_codas_quickstart.ipynb")
    if shown.returncode:
        return
    cells = json.loads(shown.stdout).get("cells", [])
    found = re.search(r'NOTEBOOK_VERSION = "([0-9a-f]+)"', "".join("".join(c.get("source", "")) for c in cells))
    if found and found.group(1) != NOTEBOOK_VERSION:
        raise SystemExit("This page shows an older copy of the notebook. File > Reload Notebook from Disk "
                         "(choose Revert/Discard if asked; do not save this page), then Run > Run All Cells.")

update_checkout()
check_this_page_is_current()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[aou]"], check=True)
sys.path.insert(0, str(REPO))"""),
code("""# --- optional settings: the defaults need no editing (everything is detected automatically) ----
CDR = os.getenv("CODAS_AOU_CDR") or os.getenv("WORKSPACE_CDR") or ""   # "<cdr-project>.<cdr-dataset>"
REASONING_MODEL = ""   # leave empty to auto-pick the newest that answers; or a Gemini id, or a Claude id
FAST_MODEL = ""        #   enabled in Model Garden
LITE_MODEL = ""
TOP_K = 1              # outcomes to run for EACH kind (lab values, diagnoses), best first by the scan
MAX_OUTCOMES = 2       # at most this many outcomes in all (this stage: one lab value and one diagnosis)
TEST_ON_HOLDOUT = True   # each selected outcome's registered hypotheses are tested once on the sealed half
OUTCOMES = []          # optional: catalog names to run instead, e.g. ["<name>", ...]; [] = ranked
# ---------------------------------------------------------------------------------------------
import json, re

def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

project = os.getenv("GOOGLE_CLOUD_PROJECT") or sh("gcloud", "config", "get-value", "project")
assert project, "No Google Cloud project found; set GOOGLE_CLOUD_PROJECT to this workspace's project."

def find_cdr():
    \"\"\"The Registered/Controlled Tier CDR among the workspace's BigQuery resources (wb resource list).\"\"\"
    try:
        resources = json.loads(sh("wb", "resource", "list", "--format=json") or "[]")
    except json.JSONDecodeError:
        return ""
    def walk(obj):
        if isinstance(obj, dict):
            ds = obj.get("datasetId") or obj.get("dataset_id")
            proj = obj.get("projectId") or obj.get("project_id")
            if isinstance(ds, str) and isinstance(proj, str) and re.fullmatch(r"[RC]\\d{4}Q\\dR\\d+", ds):
                yield f"{proj}.{ds}"
            for value in obj.values():
                yield from walk(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from walk(value)
    found = sorted(set(walk(resources)))
    return found[0] if found else ""

CDR = CDR or find_cdr()
assert CDR, ("Could not find the CDR. Open the workspace's Resources tab, click the R2025Q4R6 dataset, and set "
             "CDR = '<Project ID>.<Dataset ID>' at the top of this cell.")

# Pick a Vertex location and models that answer from this project. Every attempt is logged, so a
# failure says why (404 = model not offered there, 403 = permission or perimeter).
import warnings
from google import genai
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
for handler in logging.getLogger().handlers:  # the same notice, when the library logs it
    handler.addFilter(lambda record: "automatic function calling" not in record.getMessage())

# The candidates of each model tier, newest first, come from one table in the code (codas/core/model_tiers.py);
# the first that answers from this project is used, so a newer model is picked up without editing anything.
import time
from codas.core import model_tiers

LOCATIONS = ["global", "us-central1", "us-east4", "us-west1"]
attempts, answered = [], {}
clients = {}   # keep each client referenced: a client that is garbage-collected closes its connection

def answers(location, model):
    if (location, model) not in answered:
        client = clients.setdefault(location, genai.Client(vertexai=True, project=project, location=location))
        started = time.monotonic()
        try:
            client.models.generate_content(model=model, contents="Say OK")
            result = "OK"
        except Exception as exc:
            result = f"{type(exc).__name__}: {str(exc)[:110]}"
        attempts.append((location, model, result, round(time.monotonic() - started, 1)))
        answered[(location, model)] = result == "OK"
    return answered[(location, model)]

def first_working(location, preferred, tier):
    if preferred.lower().startswith("claude"):
        return preferred   # Claude on Vertex is not probed through the Gemini client
    candidates = dict.fromkeys([preferred, *model_tiers.CANDIDATES[tier]])
    return next((m for m in candidates if m and answers(location, m)), None)

chosen = None
for location in LOCATIONS:
    fast = first_working(location, FAST_MODEL, "fast")
    if not fast:
        continue
    reasoning = first_working(location, REASONING_MODEL, "reasoning") or fast
    lite = first_working(location, LITE_MODEL, "lite") or fast
    chosen = (location, reasoning, fast, lite)
    break
if not chosen:
    for row in attempts:
        print(" | ".join(map(str, row)))
    raise SystemExit("No Gemini model answered from this project. Copy the table above (it holds no participant "
                     "data) and share it to decide the fix.")
location, REASONING_MODEL, FAST_MODEL, LITE_MODEL = chosen

os.environ.update({
    "CODAS_AOU_MODE": "1",
    "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
    "GOOGLE_CLOUD_PROJECT": project,
    "GOOGLE_CLOUD_LOCATION": location,
    "CODAS_AOU_CDR": CDR,
    "CODAS_GEMINI_REASONING_MODEL": REASONING_MODEL,
    "CODAS_GEMINI_FAST_MODEL": FAST_MODEL,
    "CODAS_GEMINI_LITE_MODEL": LITE_MODEL,
    "CODAS_GEMINI_MODELS": ",".join(dict.fromkeys([FAST_MODEL, REASONING_MODEL])),  # literature search
})
for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(key, None)   # AoU mode refuses consumer API keys

from codas.aou.privacy import check_llm_backend
check_llm_backend()
print("model backend: Vertex AI in the workspace project (OK)")

from IPython.display import Markdown, display
import matplotlib.pyplot as plt  # enables inline figures
from codas.aou import job, notebook_views as views

def priced(model):
    price = model_tiers.price(model)
    return f"{model} (${price[0]:g} / ${price[1]:g} per 1M input / output tokens)" if price else f"{model} (no listed price)"

display(Markdown("**Environment** (where this run executes; useful when something goes wrong)"))
display(views.fit(views.environment_table(job.environment(), **{
    "project": project, "CDR": CDR, "Vertex location": location, "reasoning model": priced(REASONING_MODEL),
    "fast model": priced(FAST_MODEL), "lite model": priced(LITE_MODEL)})))
display(Markdown("**Model probe** (each tier's candidates, newest first; the first that answered is used)"))
display(views.fit(views.pd.DataFrame(attempts, columns=["location", "model", "result", "seconds"])))"""),
md("## 2. Checks: CDR schema and outcome catalog"),
code("""from google.cloud import bigquery
from IPython.display import Markdown, display

from codas.aou import cohort, feasibility as feas, runner
from codas.aou.cohort import CohortSpec

client = bigquery.Client(project=project)
BASE = CohortSpec.from_json(REPO / "scripts/aou/hba1c_fitbit.example.json")   # wearables, window, demographics
CATALOG = REPO / "scripts/aou/outcomes_catalog.json"

present = cohort._query(client, *cohort.schema_check_sql(BASE, CDR))
pairs = list(zip(present["table_name"], present["column_name"]))
display(Markdown("**Wearable columns the spec reads** (optional sources and phenotypes are skipped if absent)"))
display(views.fit(views.schema_table(BASE, present)))
BASE, skipped = cohort.resolve_sources(BASE, pairs)  # fit optional sources to this CDR (the run does the same)
for note in skipped:
    print(note)
missing = cohort.missing_columns(BASE, pairs)
if missing:
    print("Missing in this CDR:", ", ".join(missing))
    print("Columns that do exist in those tables:")
    for table, cols in present.groupby("table_name")["column_name"]:
        print(f"  {table}: {', '.join(sorted(cols))}")
    raise SystemExit("Edit scripts/aou/hba1c_fitbit.example.json to match, then re-run this cell.")

SPEC_AS_WRITTEN = BASE
BASE, failed_covariates = feas.verify_covariates(BASE, CDR, client)
for name, reason in failed_covariates.items():
    print(f"{name if name.startswith('negative control') else 'covariate ' + name} skipped: {reason}")

# Which wearable tables this CDR has (names only), including minute-level ones for richer phenotypes.
WEARABLE_TABLES = ["activity_summary", "sleep_daily_summary", "sleep_level", "steps_intraday",
                   "heart_rate_summary", "heart_rate_minute_level", "heart_rate_intra_day"]
tables = client.query(f"SELECT table_name FROM `{CDR}.INFORMATION_SCHEMA.TABLES`").result().to_dataframe()
have = set(tables["table_name"])
print("wearable tables in this CDR:", ", ".join(f"{t} {'yes' if t in have else 'no'}" for t in WEARABLE_TABLES))
# Their column names and types (schema only, no data), so intraday phenotype roles can be checked.
present_tables = [t for t in WEARABLE_TABLES if t in have]
if present_tables:
    schema = client.query(f"SELECT table_name, column_name, data_type FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS` "
                          f"WHERE table_name IN UNNEST({present_tables!r})").result().to_dataframe()
    display(Markdown("**Columns of the wearable tables in this CDR**"))
    display(views.fit(schema.groupby("table_name")[["column_name", "data_type"]].apply(
        lambda g: ", ".join(f"{c} ({t})" for c, t in zip(g.column_name, g.data_type))).rename("columns").reset_index()))

outcomes = feas.resolve_codes(feas.load_catalog(CATALOG), CDR, client, print)  # public codes -> this CDR's ids
problems = feas.verify_catalog(outcomes, CDR, client)
shown_covariates, _ = feas.resolve_survey_questions(feas.resolve_codes(SPEC_AS_WRITTEN.covariates, CDR, client),
                                                    CDR, client)
shown_covariates += [c for c in SPEC_AS_WRITTEN.covariates if c.name not in {s.name for s in shown_covariates}]
items = ([("outcome", o) for o in outcomes] + [("covariate", c) for c in shown_covariates]
         + [("negative control", c) for c in feas.resolve_codes(SPEC_AS_WRITTEN.negative_controls, CDR, client)])
concept_rows = cohort._query(client, *feas.concept_check_sql([i for _, i in items], CDR)).to_dict("records")
all_problems = {**problems, **{k.removeprefix("negative control "): v for k, v in failed_covariates.items()}}
display(Markdown("**Concepts** (each id as this CDR names it; any failure is skipped, never analysed)"))
display(views.fit(views.concept_table(items, concept_rows, all_problems)))
for covariate in [c for c in BASE.covariates if c.kind == "survey"]:
    display(Markdown(f"**Survey covariate `{covariate.name}`**: answers to the question matched by "
                     f"'{covariate.question_name}' and how each is coded (people, 1-20 masked)"))
    display(views.fit(feas.survey_answers(covariate, CDR, client)))"""),
md("""### Data inventory

Every table in this CDR with its number of columns, how many participants each person-level table covers,
and how many of them also have Fitbit data (counts of 1-20 masked; no row is read). Tables that look like
minute-level heart rate are found by their columns, not their name. Table sizes come from BigQuery's
table metadata."""),
code("""from codas.aou import inventory

inv = inventory.inventory(CDR, client)
display(views.fit(inv["tables"]))
print("minute heart-rate tables found by shape:", inv["minute_heart_rate"] or "none")
try:
    sizes = client.query(f"SELECT table_id, row_count, ROUND(size_bytes / 1e9, 1) AS size_gb FROM `{CDR}.__TABLES__` "
                         "ORDER BY size_bytes DESC").result().to_dataframe()
    display(Markdown("**Largest tables** (rows and GB; BigQuery bills queries by bytes read)"))
    display(views.fit(sizes.head(15)))
except Exception as exc:  # metadata may be unavailable; the inventory above does not depend on it
    print("table sizes unavailable:", type(exc).__name__)"""),
md("""### Literature sources

Which bibliographic sources answer from this workspace: one tiny query each, sending only a fixed word,
never data. The agents can cite only what these return. Google Search grounding gives web context for
their reasoning; its terms forbid caching or harvesting the results, so its links are not citations.
Google Scholar has no API and its terms forbid automated queries, so it is not used; Nature papers are
found by DOI through the open indexes."""),
code("""from codas.core import gemini, literature

display(views.fit(views.pd.DataFrame(literature.probe(grounding=gemini.generate_grounded_reply))))
print("literature traffic so far:", literature.EGRESS)"""),
md("""### The research basis, checked

`docs/research/` holds the papers CoDaS's design rests on (wearable-biomarker studies, biomarker methods),
compiled from abstracts. This looks every DOI in them up in Crossref from here (answers are cached, so a
second Run All asks nothing) and shows which are registered and how well each entry matches its record."""),
code("""import json, os

papers = literature.references_in("\\n".join(p.read_text() for p in sorted((REPO / "docs" / "research").glob("*.md"))))
checked = literature.verify_references(papers, cache=literature.Cache(REPO / ".codas_runs" / "aou" / "literature_cache.jsonl"))
reports = Path(os.getenv("CODAS_REPORTS_DIR") or REPO / "codas_reports") / "aou"
reports.mkdir(parents=True, exist_ok=True)
(reports / "references.json").write_text(json.dumps(checked, indent=1))
display(views.fit(views.references_summary(checked)))
doubtful = views.references_doubtful(checked)
if doubtful is not None:
    display(Markdown("**Not found, or the entry matches little of the registered title**"))
    display(views.fit(doubtful))"""),
md("""## 3. Choosing the outcomes: a phenome-wide scan

The background job below starts by screening every verified catalog outcome (about 100 diagnoses and 30 lab
values) against every wearable feature and phenotype, on the discovery half only: one BigQuery pass for the
participants' features and one for all their diagnoses and lab values, then a test of every pair adjusted
for age and sex, again after all covariates and wear time, and in two fixed halves. An outcome is selected
when it has a hit (significant across the whole matrix, not negligible in size, the same sign in both halves
and after adjustment), best first by hits the literature has not established. Nothing is tested on the
sealed half here. Its table and map are shown below once the job ends (counts of 1-20 masked)."""),
md("""## 4. Exploration

Runs CoDaS (extraction, profiling, the discovery loop, critic/defender debate, reporting) on each
selected outcome, typically 1-2 hours per outcome. The run happens in a **background job**, so it
keeps going if this page reloads, the laptop sleeps or the connection drops. This cell prints only the
milestones (environment, feasibility, each outcome's start and end, failures) and one status block,
updated in place: elapsed time, current outcome and agent, and the latest log lines. Every agent turn
and tool call is in the full log, `codas_reports/aou/explore.log` (a line that floods it, such as a
library warning raised in a loop, is written a few times and then counted). If the page lost its
connection, run the cell again (or Run All): it re-attaches to the running job instead of starting a
second one.

Before anything touches All of Us data the job runs a **rehearsal**: the real models run the whole pipeline
once, for one round, on synthetic data with a planted signal (about half an hour, once per code and model
version). If the pipeline cannot finish, register hypotheses, test the sealed half or write the report there,
the job stops before any query and says why; weaker results (no new phenotype defined, the planted signal
missed, the literature unreachable) are shown as warnings.

This stage (`TOP_K = 1`, `MAX_OUTCOMES = 2`, `TEST_ON_HOLDOUT = True`) runs the lab value and the diagnosis the
phenome scan ranks first, and tests each one's registered hypotheses once on the sealed half. The first run
on a CDR builds two tables that later runs reuse: the intraday phenotypes and the day-by-day baseline, each a
single BigQuery pass over the minute-level tables (roughly 13 TB each, under the cost caps; every job is dry-run
first). Outcomes already tested on the sealed half are not selected again; set `TEST_ON_HOLDOUT = False` to run
the agents without opening the sealed half.

Safeguards: a model request that sends nothing for 10 minutes is abandoned, retried, then sent to another
model; an outcome still running after 3 hours is stopped and marked failed, and the others
continue. Run All again to retry failed outcomes: finished ones are reused, not rerun, and an outcome
whose agents finished but whose holdout test or saving failed redoes only those steps."""),
code("""result = job.run_or_attach({"catalog": str(CATALOG), "spec": str(REPO / "scripts/aou/hba1c_fitbit.example.json"),
                            "cdr": CDR, "top_k": TOP_K, "outcomes": OUTCOMES or None,
                            "max_outcomes": MAX_OUTCOMES or None, "test_holdout": TEST_ON_HOLDOUT}, quiet=True)
rehearsal = views.rehearsal_table(result)
if rehearsal is not None:
    display(Markdown("### Rehearsal: the real models on synthetic data, before any All of Us query"))
    display(views.fit(rehearsal))
if result.get("scan"):
    display(Markdown(f"### Phenome scan: {result['scan']['hits']} hits among {result['scan']['pairs']:,} pairs"))
    display(views.phenome_figure(result))
    hits = views.phenome_hits_table(result)
    if hits is not None:
        display(views.fit(hits))
display(views.feasibility_figure(result["rows"], result["chosen"]))
display(Markdown("### Runs"))
display(views.fit(views.runs_table(result)))
literature = views.literature_table(result)
if literature is not None:
    display(Markdown("**Literature searches** (how many returned sources; the agents must not cite without them)"))
    display(views.fit(literature))
display(views.timing_figure(result))
display(Markdown(views.wrap_tables(Path(result["summary_path"]).read_text())))"""),
md("""## 5. Results per outcome

For each outcome: who is in the cohort (discovery and sealed shares); the quality rules applied at
extraction (wear-time days, sleep length, the case rule, EHR activity) and, for a lab outcome, the units
its values were recorded in; what the data look like (the dossier the agents read before registering anything: each candidate's distribution, feature families,
wear dependence and who is missing what; it never relates a feature to the outcome); how much has been
published on each measured construct with the outcome (counted in Europe PMC before the agents started; this
count, not the agents' opinion, decides whether a hypothesis is established); the hypotheses the
agents registered from the literature before any statistic existed; the signal map they saw once
registration closed (each candidate's adjusted association and how much the covariates explain); every method and analysis run, by round
(the statistical and ML method pools chosen by the gap finder, and the agents' own analyses: survival
screen, subgroups, multivariable model with incremental value, non-linear model, dose-response,
within-person mixed model); the per-candidate debate verdicts; the lessons recorded for later
outcomes; the candidates that survived the discovery loop; how each pre-registered hypothesis
did on the sealed holdout (novel hypotheses judged under online false-discovery-rate control across
everything tested on these participants; established ones as a reference family), with the
falsification checks including wear adherence; how each agent ran (model calls, tokens, tool calls,
time and estimated cost, the models that actually served it, who changed the design in which round,
and which agent's output each agent read); then the agents' report."""),
code("""for run in result["runs"]:
    stem = run["cohort"]
    display(Markdown(f"---\\n### {stem}"))
    if run.get("failed"):
        display(Markdown(f"Failed: {run['failed']}. Run All again to retry it."))
        continue
    for title, table in (("Cohort characteristics", views.cohort_table(result, stem)),
                         ("Quality rules applied at extraction, and the outcome's units", views.quality_table(result, stem)),
                         ("Data understanding: candidate features (read before any hypothesis)",
                          views.eda_table(result, stem)),
                         ("Feature families and features that track wear", views.families_table(result, stem)),
                         ("The data analyst's memo: each claim checked against the data summary",
                          views.data_memo_table(result, stem)),
                         ("Literature evidence, counted before the agents started (the class comes from counts)",
                          views.literature_evidence_table(result, stem)),
                         ("Phenotypes the agents defined on the day-by-day baseline",
                          views.defined_phenotypes_table(result, stem)),
                         ("Hypotheses registered before any statistic, each as designed",
                          views.hypotheses_table(result, stem)),
                         ("Submissions refused at registration, and why", views.refusals_table(result, stem)),
                         ("Independent review of the registered hypotheses (1-5; secondary evidence)",
                          views.hypothesis_review_table(result, stem)),
                         ("Hypothesis quality, measured", views.hypothesis_quality_table(result, stem)),
                         ("Signal map: adjusted associations once registration closed", views.signal_table(result, stem)),
                         ("Methods and analyses run, by round (method pools and agents' tools)",
                          views.analyses_table(result, stem)),
                         ("Per-candidate debate", views.debates_table(result, stem)),
                         ("Lessons recorded for later outcomes", views.lessons_table(result, stem)),
                         ("Sealed holdout: pre-registered hypotheses", views.confirmation_table(result, stem)),
                         ("Sealed holdout: effect sizes in units a reader can weigh", views.effect_sizes_table(result, stem)),
                         ("Sealed holdout: each registered prediction scored", views.scorecard_table(result, stem)),
                         ("Sealed holdout: each effect across strata of sex, age, era and sub-study",
                          views.heterogeneity_table(result, stem)),
                         ("What the analysis covered, and how far the evidence reaches",
                          views.outcome_scorecard_table(result, stem)),
                         ("Recipe coverage and the evidence ladder, per hypothesis",
                          views.recipe_coverage_table(result, stem)),
                         ("Agent execution: calls, tokens, tools, time and estimated cost per agent",
                          views.execution_table(result, stem)),
                         ("Design changes: who changed what the later agents work on, by round",
                          views.design_changes_table(result, stem)),
                         ("Information flow: which agent's output each agent read", views.flow_table(result, stem))):
        if table is not None:
            display(Markdown(f"**{title}**"))
            display(views.fit(table))
    for figure in (views.candidates_figure(result, stem), views.absolute_risk_figure(result, stem),
                   views.execution_timeline_figure(result, stem)):
        if figure is not None:
            display(figure)
    display(Markdown("**Report**\\n\\n" + views.wrap_tables(Path(run["report_path"]).read_text())))"""),
md("""## 6. Across outcomes: the sealed holdout

Every pre-registered candidate's effect on the sealed holdout with its 95% CI (filled = replicated after
correcting across every hypothesis tested on these participants)."""),
code("""figure = views.forest_figure(result)
display(figure if figure is not None else Markdown("No candidate was pre-registered."))"""),
md("""## Outputs of each run

For every outcome: the compiled PDF report (final report, every phase's output, the per-round
critic/defender debate, phase coverage and the candidates' effect sizes), the full agent trace
(`_trace.md` to read, `_trace.jsonl` with every tool call and response), and every phase's saved
output (`_state.json`). Click a link to open it in JupyterLab."""),
code("""from IPython.display import FileLink

labels = {"pdf": "PDF report", "trace_md": "agent trace (readable)", "trace_jsonl": "agent trace (JSONL)",
          "execution_md": "agent execution (readable)", "execution_json": "agent execution (JSON)",
          "state": "phase outputs"}
out_dir = Path(result["summary_path"]).parent
for run in result["runs"]:
    display(Markdown(f"**{run['cohort']}**" + (f" — failed: {run['failed']}" if run.get("failed") else "")))
    files = run.get("files") or {}
    if not run.get("failed") and not (files.get("pdf") and Path(files["pdf"]).exists()):
        # compile it now from the saved outputs (no agent rerun); an older attempt's error no longer applies
        for stale in [k for k in files if k.startswith("pdf")]:
            files.pop(stale)
        files.update(runner.rebuild_pdf(out_dir, run["cohort"]))
    for kind, label in labels.items():
        if kind in files:
            display(FileLink(os.path.relpath(files[kind]), result_html_prefix=f"{label}: "))
    if files.get("pdf_error"):
        print("PDF not built:", files["pdf_error"])
    if files.get("pdf_warnings") and Path(files["pdf_warnings"]).exists():
        print("PDF problems (full traceback for debugging):\\n" + Path(files["pdf_warnings"]).read_text()[-3000:])
    if run.get("coverage_missing"):
        print("steps without output:", "; ".join(run["coverage_missing"]))"""),
md("""## Status (run any time, from any kernel)

Shows whether the background exploration is running, finished, failed or stopped, the end of its log,
and the lines that repeated most in this run's log and in earlier runs' logs (kept beside it): what
flooded a run's output."""),
code("""from pathlib import Path

from IPython.display import Markdown, display

from codas.aou import job, notebook_views as views

st = job.status(tail=30)
print("state:", st["state"], "| started:", st.get("started"), "| ended:", st.get("ended", "-"))
if st.get("error"):
    print("error:", st["error"])
if st.get("last_heartbeat"):
    print("last heartbeat:", st["last_heartbeat"])
if st["state"] == "stopped":
    print("The job's process ended without recording an outcome: the kernel or machine was stopped, or it ran "
          "out of memory (see the chart). Run the exploration cell again to resume.")
figure = views.heartbeat_figure(st["log_path"])
if figure is not None:
    display(figure)
print("\\n".join(st["log_tail"]))
for path in [Path(st["log_path"]), *job.earlier_logs()[:3]]:
    table = views.log_digest_table(path)
    if table is not None:
        display(Markdown(f"**Most repeated lines in `{path.name}`**"))
        display(views.fit(table))"""),
md("""## Before anything leaves the Workbench

- Outputs are in `.codas_runs/aou/`: per-outcome `*_report.pdf`, `*_report.md`, `*_trace.md`,
  `*_trace.jsonl`, `*_state.json`, `*_rounds.json`, and `explore_summary.md`.
- Counts of 1-20 are masked by CoDaS, but check every table and figure yourself against the
  Dissemination Policy (including counts that could be derived by subtraction).
- The `*.csv` files are participant-level. Never download, copy, commit or paste them anywhere."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
import hashlib
import json

_version = hashlib.sha256(json.dumps([c["source"] for c in nb.cells]).encode()).hexdigest()[:12]
nb.cells[next(i for i, c in enumerate(nb.cells) if "__NOTEBOOK_VERSION__" in c["source"])]["source"] = \
    next(c for c in nb.cells if "__NOTEBOOK_VERSION__" in c["source"])["source"].replace("__NOTEBOOK_VERSION__", _version)
nbf.validate(nb)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "notebooks" / "aou_codas_quickstart.ipynb"
nbf.write(nb, str(out))
print("ok", len(cells), "cells ->", out)
