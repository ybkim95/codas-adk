# CoDaS: An AI co-data-scientist

CoDaS is an AI co-data-scientist that prioritizes candidate biomarkers from wearable and clinical data. Given a participant-level table and a clinical outcome, a team of Gemini agents profiles the data, grounds hypotheses in the literature, runs an iterative discovery loop, argues each candidate for and against, and drafts a report for human review. A deterministic Python engine supplies the statistical outputs used by the public workflow. This separation reduces the risk that a language model invents a numerical result, but it does not by itself verify labels, study design, clinical interpretation, or numbers assembled outside that workflow.

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --only-binary=:all: -r requirements-lock.txt
pip install .
```

Use a fresh virtualenv. A base environment with a broken numpy or pandas is the usual reason imports fail.

## Quick start

```bash
# try it on the bundled sample
python examples/run_agent.py

# run the agents on your own table (one row per participant, one column the outcome)
export GOOGLE_API_KEY=...
python examples/run_agent.py your_table.csv "Discover candidate biomarkers for <outcome> in this cohort"
```

The agents read the schema, choose the target and roles, iterate the discovery loop, and print a grounded report.

`examples/sample_dataset.csv` is the bundled sample, and the only file here that stands in for a participant table. It is synthetic and is shaped to exercise the public pipeline's expected input roles; it is not evidence that the governed cohort schemas, distributions or preprocessing are reproduced. The rows are not participants. The clinical cohorts are not redistributed here. The repository does include non-participant-level, per-feature and per-candidate artifacts from selected runs under `paper_artifacts/`; that directory documents their scope and omissions. `examples/DATA_MANIFEST.md` lists every tracked CSV, JSON and related data artifact, with the command used to reproduce that inventory.

For the deterministic engine on its own, name the target and skip the API key.

```python
from codas.core.data import read_csv_dataset
from codas.core.discovery import run_discovery, DiscoveryRequest

df = read_csv_dataset("examples/sample_dataset.csv")
report = run_discovery(df, DiscoveryRequest(target_column="depression_score"))
for c in report.candidates:
    print(c.verdict, c.feature, round(c.rho, 3), c.q_value)
```

The engine reads nothing from column names, so the same code runs on any table and any disease.

## Agent workflow

The orchestrator runs six phases over one shared memory and one deterministic tool set. Its core is a discovery loop that deepens the search until the configured iteration bound is reached or a GapChecker returns a stop decision. This is an orchestration rule, not a validated scientific criterion for concluding that a dataset contains no reportable signal.

```
   a table  +  a research goal in plain language
        |
        v
   ORCHESTRATOR      shared memory      deterministic tools
   |
   |-- Phase A       profile the data, pick target and roles, frame literature-grounded hypotheses
   |
   |-- Phase B/C     DISCOVERY LOOP, deepening each round
   |                   1. search       run a deeper discovery round
   |                   2. interpret    statistical and ML tracks read the round in parallel
   |                   3. validate     critic and defender argue each candidate
   |                   4. gapcheck     converged? stop. otherwise iterate
   |
   |-- Phase D/E/F   mechanism, novelty, strategy, then a grounded report
        |
        v
   an auditable report, with optional human feedback for another pass
```

The graph is built with [google-adk](https://google.github.io/adk-docs/). Each phase is a `SequentialAgent`, the loop is a `LoopAgent`, and the two interpreters run in a `ParallelAgent`, all sharing one `session.state`. To customize it, edit `codas/agents/agent.py`, where the two Gemini tiers and the loop depth are set (or override them with the `CODAS_*` environment variables in `.env.example`).

Numbers emitted by the public deterministic path come from the engine, and `tests/test_validation_golden.py` pins a bundled synthetic example by hash so a refactor cannot quietly change that example's verdict. This test does not establish the provenance of every number in a manuscript assembled from multiple runs and analysis environments. The engine (`codas.core`) is plain numpy, pandas, scipy, and scikit-learn with no LLM and no network. It screens univariate and engineered features with Spearman correlation under Benjamini-Hochberg FDR control, then puts each candidate through a validation battery covering replication, stability, robustness, and discriminative power. Leakage guards drop the target and its declared proxies before screening and demote features that duplicate a stronger one.

## Scope

CoDaS prioritizes candidate biomarkers as hypothesis-generating signals for expert review, not as validated clinical tools. It expects a participant-level table with a declared target and, for repeated measures, a declared participant or time column, which it uses to correct for clustering and temporal dependence. It warns when those roles look undeclared and flags a feature that separates the outcome implausibly strongly as possible leakage, but it does not infer roles from the data on its own.

## Checking selected reported effects

When supplied with matching participant-level analysis tables, the script below recomputes selected Spearman effect sizes and compares them with reference values in its configuration. The governed tables, their complete preprocessing lineage and several manuscript analysis layers are not redistributed here; consequently, a public clone by itself cannot reproduce or independently verify the manuscript effects. Edit the config to point at analysis tables you are authorized to use.

```bash
python scripts/reproduce_paper_biomarkers.py --config scripts/paper_cohorts.example.json
```

The reference values configured for comparison are:

| Finding | Spearman rho |
|---|---|
| DWB, main sleep-duration variability vs PHQ-8 | +0.252 |
| WEAR-ME, C-reactive protein vs HOMA-IR | +0.393 |
| WEAR-ME, HDL cholesterol vs HOMA-IR | -0.412 |

The count of validated candidates depends on the declared exclusions and thresholds, so the script reports it as computed rather than as a fixed number.
This harness does not cover GLOBEM, the record-linked retrospective analysis, the nested-model ablation, the human-review studies or end-to-end manuscript assembly.

## Citation

```bibtex
@article{kim2026codas,
  title={An AI Co-Data-Scientist for Prioritizing Candidate Biomarkers from Wearable Sensor Data},
  author={Kim, Yubin and others},
  journal={arXiv preprint arXiv:2604.14615},
  note={The preprint carries the earlier title, CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors},
  year={2026}
}
```

## Paper scripts

`scripts/paper/injected_error_stress_test.py` is the injected-error experiment of the Nature
Medicine revision (Supplementary Note 2). It plants known error classes into the governed
WEAR-ME table, which is not redistributed (set `CODAS_WEARME_CSV`), runs this repository's
validation battery on each planted feature over 20 seeds and writes the per-row gate fields.
Run on this commit it reproduces the released Source Data file `source_data_stress_test.csv`
row for row. The module evaluates fourteen checks and counts eleven toward the pass rate, so
the `pass_rate` column differs from the archived file, whose engine copy counted fourteen.

## License

See [LICENSE](LICENSE). The current evaluation notice does not grant permission to use, copy,
modify or distribute the software; the project owners must select an applicable license before
claiming a reusable open-source release.
