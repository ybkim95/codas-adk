# Paper artifacts

The run outputs behind the results reported in the accompanying paper, for the two cohorts
whose participant-level data cannot be redistributed. These let a reader check the reported
numbers against the records the pipeline produced, rather than take them on trust.

Nothing here is participant-level. Every file is per-feature or per-candidate. There are no
participant identifiers and no arrays of participant length.

## Runs

| Directory | Run | Cohort |
|---|---|---|
| `dwb/` | `dwb_hourly_20260304_160315` | Digital Wellbeing, depression |
| `wearme/` | `wearme_20260304_160314` | WEAR-ME, insulin resistance |

The two are from the same analysis session.

## Files

| File | What it holds |
|---|---|
| `validated_candidates.json` | Every candidate the discovery loop produced, with its verdict, the reason recorded for that verdict, the per-test results from the validation battery, and the `discovery_round` in which it first appeared. Rejected candidates are included with their rejection reason, which is the point of shipping it. The round is present for all 23 WEAR-ME candidates and for 12 of the 33 Digital Wellbeing candidates, so a per-round reconstruction is complete for one cohort and partial for the other. |
| `biomarker_proofs.json` | The per-candidate evidence assembled for each verdict. |
| `feature_registry.json` | Every feature considered, with its category, its source columns and the formula used to construct it. It records how each feature was built. The discovery round is not here; it is in `validated_candidates.json`. |
| `full_stat_results_spearman.json` | The derived per-candidate statistics, effect sizes with p-values and confidence intervals. |
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
