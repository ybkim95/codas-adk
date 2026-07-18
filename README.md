# CoDaS: An AI co-data-scientist

CoDaS is an AI co-data-scientist that prioritizes candidate biomarkers from wearable and clinical data. Given a participant-level table and a clinical outcome, a team of Gemini agents profiles the data, grounds hypotheses in the literature, runs an iterative discovery loop, argues each candidate for and against, and drafts a report you can audit. A deterministic Python engine computes every statistic, so the agents decide what to test and when to stop while never inventing a number.

## Install

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

For the deterministic engine on its own, name the target and skip the API key.

```python
from codas.core.data import read_csv_dataset
from codas.core.discovery import run_discovery, DiscoveryRequest

df = read_csv_dataset("your_table.csv")
report = run_discovery(df, DiscoveryRequest(target_column="outcome"))
for c in report.candidates:
    print(c.verdict, c.feature, round(c.rho, 3), c.q_value)
```

The engine reads nothing from column names, so the same code runs on any table and any disease.

## Agent workflow

The orchestrator runs six phases over one shared memory and one deterministic tool set. Its core is a discovery loop that deepens the search each round until a GapChecker judges another round will not help.

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

Every reportable number comes from the engine, and `tests/test_validation_golden.py` pins the whole pipeline output by hash so a refactor cannot quietly change a verdict. The engine (`codas.core`) is plain numpy, pandas, scipy, and scikit-learn with no LLM and no network. It screens univariate and engineered features with Spearman correlation under Benjamini-Hochberg FDR control, then puts each candidate through a validation battery covering replication, stability, robustness, and discriminative power. Leakage guards drop the target and its declared proxies before screening and demote features that duplicate a stronger one.

## Reproducing the reported effects

The reported effect sizes are recomputed by the engine from the participant-level analysis tables. The clinical cohorts are governed data and are not redistributed here, so edit the config to point at tables you already hold.

```bash
python scripts/reproduce_paper_biomarkers.py --config scripts/paper_cohorts.example.json
```

On the analysis tables the engine returns the reported effect sizes to three decimals.

| Finding | Spearman rho |
|---|---|
| DWB, main sleep-duration variability vs PHQ-8 | +0.252 |
| WEAR-ME, C-reactive protein vs HOMA-IR | +0.393 |
| WEAR-ME, HDL cholesterol vs HOMA-IR | -0.380 |

The count of validated candidates depends on the declared exclusions and thresholds, so the script reports it as computed rather than as a fixed number.

## Citation

```bibtex
@article{kim2026codas,
  title={CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors},
  author={Kim, Yubin and others},
  journal={arXiv preprint arXiv:2604.14615},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
