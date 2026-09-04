# Data manifest

Every file in this repository that carries data or points at data, with its provenance. The
list is exhaustive rather than representative, because an aggregate assurance about participant
data is worth nothing if it is wrong for one file.

Reproduce it with:

```bash
git ls-files | grep -Ei '\.(csv|tsv|parquet|json|xlsx|pkl|npy)$'
```

## What ships

| Path | Shape | What it is | Participant data |
|---|---|---|---|
| `examples/sample_dataset.csv` | 420 rows, 12 columns | Synthetic sample used by `examples/run_agent.py`, `examples/quickstart.py`, and the engine, service and golden-output tests | **None** |
| `scripts/ci_smoke_cohorts.json` | config | Points the reproduction harness at the synthetic sample so CI proves the script runs end to end | **None** |
| `scripts/paper_cohorts.example.json` | config | Example cohort config with placeholder paths and the published reference effect sizes | **None** |

`examples/sample_dataset.csv` is synthetic. Its columns carry the same semantics as the
production inputs, which is what allows the pipeline to run on it end to end and what makes
`tests/test_validation_golden.py` a real check rather than a smoke test. Its rows are not
people, and it is not a subset, sample or transformation of any study cohort.

The two JSON files are configuration. `paper_cohorts.example.json` contains paths you fill in
yourself and the reference effect sizes already published in the paper. Neither file carries
participant records.

## What never ships

Excluded by `.gitignore`, so they exist only in a working checkout.

| Path | What lands there |
|---|---|
| `.codas_runs/` | Run outputs, pipeline state, and anything uploaded through the agent service |
| `.cache/` | Datasets fetched at runtime by the benchmark and validation scripts |

If you have run the pipeline locally both directories will hold files. They are yours, they are
not part of this repository, and keeping them out is enforced by `.gitignore` rather than left
to convention.

## Cohort data

The clinical cohorts behind the reported effects are governed data and are not redistributed
here. Their access terms differ and each is stated in the paper's Data Availability statement.
`scripts/reproduce_paper_biomarkers.py` takes cohort tables as arguments and ships none of its
own, so an approved data holder can recompute the reported effects without anything further
from us.
