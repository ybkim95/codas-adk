# Paper artifacts

Selected archived run outputs for two cohorts whose participant-level data cannot be
redistributed. These files support row-level checks of the values they contain, but they are
not a complete manifest of every candidate or number in the accompanying manuscript. In
particular, this directory contains no GLOBEM run and should not be used to reconstruct the
manuscript's aggregate candidate count without a separate reconciliation manifest.

Nothing here is participant-level. Every file is per-feature or per-candidate. There are no
participant identifiers and no arrays of participant length.

The verdict field carries the run's own vocabulary, `VALIDATED`, `CONDITIONAL` and
`REJECTED`, and the file is named for it. The paper retired that wording after review and
calls the same label *internally screened*. These records are left as the run wrote them
rather than edited to match later prose, so `VALIDATED` here is not a claim of validation.

## Runs

| Directory | Run | Cohort |
|---|---|---|
| `dwb/` | `dwb_hourly_20260304_160315` | Digital Wellbeing, depression |
| `wearme/` | `wearme_20260304_160314` | WEAR-ME, insulin resistance |

The two are from the same analysis session.

## Files

| File | What it holds |
|---|---|
| `validated_candidates.json` | Candidate records present in the archived export, with verdicts, recorded reasons and per-test results. Rejected records are included. `discovery_round` is non-null for all 23 WEAR-ME records and for 12 of the 33 Digital Wellbeing records, so round-level reconstruction is complete for the former export and partial for the latter; neither fact proves that the exports cover every manuscript row. |
| `biomarker_proofs.json` | The per-candidate evidence fields present in the archived export. |
| `feature_registry.json` | Archived feature records with category, source columns and formula fields. The export does not establish a complete, cross-cohort feature-generation manifest. |
| `full_stat_results_spearman.json` | Per-feature statistics, effect sizes, p-values and confidence intervals computed on the archived Digital Wellbeing discovery split, `n = 5,999`. |
| `full_stat_results_spearman_full_cohort.json` | The same statistics computed on the full analytic sample, `n = 7,497`. It contains the Digital Wellbeing full-sample effect reported in the paper; it is not a cross-cohort results manifest. |
| `numeric_verification_log.json` | The numeric verification pass. Records each correction the pass made to the drafted report, with the before and after value. |

## What is not here, and why

**Participant-level tables.** The Digital Wellbeing cohort is not redistributable and WEAR-ME
is available only under an approved access process. See the paper's Data Availability
statement for the terms of each.

**The exploratory data analysis report.** It carries row indices into the participant table.
Those are positions rather than identifiers, but publishing outlier membership across many
features is not something we are willing to do for a cohort we cannot redistribute.

**Cross-validation fold assignments and the Critic and Defender exchanges.** The response to
referees said these would be released. They are not in the archived run outputs, so we cannot
release them, and the response has been corrected rather than left to promise what does not
exist.

**The nested-model ablation code.** Not part of this release. See the paper's Code Availability
statement.

## Reading `numeric_verification_log.json`

This file is worth reading directly, because it shows the verification pass doing its job and
also shows its limit. Each entry is a value the pass changed in the drafted report. For DWB it
records twelve corrections, several of the form `Sample size corrected: N=5511 -> N=7497`.

The pass compares numbers in the draft against the engine's own outputs. It does not check that
a label attached to a number describes what the number measures. The paper's Methods states
that boundary and gives the case where it mattered.

## Two Spearman files for Digital Wellbeing, and why

The pipeline screens candidates on a discovery split and reports effect sizes on the full
analytic sample. Those are different numbers for the same feature, so both files ship rather
than one. Main sleep duration variability is `rho = 0.2491` at `n = 5,999` in
`full_stat_results_spearman.json` and `rho = 0.2521` at `n = 7,497` in
`full_stat_results_spearman_full_cohort.json`. The paper reports 0.252, which is the second.

The full-cohort file was supplied as an output from an earlier run in the same family. It is
shipped because it carries the sample size and Digital Wellbeing effect used in the paper; the
repository does not include a run manifest that independently establishes its lineage. A reader
checking that effect against the split-sample file alone would therefore see a different value
from a different archived sample. WEAR-ME has no second file here. Its per-feature `n` varies
with measurement availability, so `derived_cardio_fitness` is `n = 865` within a cohort of
1,078.
