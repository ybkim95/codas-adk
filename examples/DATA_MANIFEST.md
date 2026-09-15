# Data manifest

Every tracked CSV, JSON and related data artifact in this repository, with the provenance
available in the supplied archive. The inventory is exhaustive for the filename classes in the
command below; it is not a claim that the repository contains complete provenance for every
manuscript result.

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
| `paper_artifacts/dwb/validated_candidates.json` | 33 candidates | Candidates with verdicts, reasons and per-test results, Digital Wellbeing run | **None**, per-candidate |
| `paper_artifacts/dwb/biomarker_proofs.json` | 33 candidates | Per-candidate evidence behind each verdict | **None**, per-candidate |
| `paper_artifacts/dwb/feature_registry.json` | 194 features | Feature generation history | **None**, per-feature |
| `paper_artifacts/dwb/full_stat_results_spearman.json` | 145 features, n = 5,999 | Derived effect sizes on the discovery split | **None**, per-feature |
| `paper_artifacts/dwb/full_stat_results_spearman_full_cohort.json` | 145 features, n = 7,497 | The same statistics on the full analytic sample, which is what the paper reports | **None**, per-feature |
| `paper_artifacts/dwb/numeric_verification_log.json` | 12 corrections | Numeric verification pass record | **None** |
| `paper_artifacts/dwb/debate_records.json` | 20 candidates | Critic assessment and Defender statement for each candidate that entered adversarial review, as the run logged them | **None**, per-candidate |
| `paper_artifacts/wearme/validated_candidates.json` | 23 candidates | As above, WEAR-ME run | **None**, per-candidate |
| `paper_artifacts/wearme/biomarker_proofs.json` | 23 candidates | As above, WEAR-ME run | **None**, per-candidate |
| `paper_artifacts/wearme/feature_registry.json` | 64 features | As above, WEAR-ME run | **None**, per-feature |
| `paper_artifacts/wearme/full_stat_results_spearman.json` | per-feature | As above, WEAR-ME run | **None**, per-feature |
| `paper_artifacts/wearme/numeric_verification_log.json` | 10 corrections | As above, WEAR-ME run | **None** |
| `paper_artifacts/wearme/debate_records.json` | 18 candidates | As above, WEAR-ME run | **None**, per-candidate |

`paper_artifacts/` holds selected archived run outputs for two cohorts whose participant-level
data cannot be redistributed by this repository. A reader can compare values in those files with
selected reported values, but the exports do not establish complete end-to-end lineage. Every
file is per-feature or per-candidate. `paper_artifacts/README.md` states the scope and omissions.

`examples/sample_dataset.csv` is synthetic. Its columns exercise the roles expected by the
public pipeline and support an end-to-end golden test of that synthetic example. This does not
show equivalence to a governed cohort's schema, value distribution or preprocessing. Its rows
are not people, and it is not a subset, sample or transformation of any study cohort.

The two JSON files are configuration. `paper_cohorts.example.json` contains paths users must fill
in and reference effect sizes stated in the manuscript. Neither file carries participant records.

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
own. An authorized data holder can use it to compare selected effects only if they also possess
matching analysis tables, column definitions, derived features and preprocessing. This repository
alone does not reconstruct those inputs.
