# The research basis for CoDaS on All of Us

This folder holds what the design of CoDaS on All of Us (AoU) rests on: how published studies found and
validated new biomarkers, and where CoDaS implements each step.

* `wearable_papers.md`: 51 wearable-biomarker papers (AoU, UK Biobank accelerometry, consumer devices, and the
  papers that define wearable phenotypes), with how each derived its marker, its design and its validation.
* `biomarker_methods.md`: 77 citations, in three groups:
  - 32 discovery and validation exemplars: proteomics, metabolomics, imaging and ECG AI, PheWAS;
  - 8 AI-driven discovery systems;
  - 37 methods papers and guidelines.

  It closes with a synthesis: the recipe, the evaluation ladder, metrics beyond p-values, how novelty is
  shown, how hypotheses are pre-specified, and standard preprocessing.

**How far to trust them.**
- Both were compiled from abstracts, indexes and search summaries. No full text could be opened from the
  machine that wrote them.
- A detail no source confirmed is marked UNVERIFIED (or [UNV]). A DOI is given only where a search result
  showed one.
- The notebook's checks cell looks every DOI in these files up in Crossref from inside the Workbench and
  writes what it found to `codas_reports/aou/references.json`: whether the DOI is registered, its title,
  journal and year, and how much of that title the entry itself contains (a low share means the DOI may
  belong to another paper). Until a run has done that, treat every entry as a pointer to check, not a
  citation.

## The recipe, and where CoDaS implements it

The recipe the papers share, step by step (synthesis in `biomarker_methods.md` section 4 and
`wearable_papers.md` part 2). For each step: the studies that model it, what CoDaS does, and where.

| # | step | what CoDaS does | where | state |
|---|---|---|---|---|
| 1 | Freeze the framework, not the hits: time zero, prevalent cases out, a lag, covariate tiers, the multiplicity plan, the splits (Master 2022, Zheng 2024, Gadd 2024) | Prospective incident design: index after the baseline window, lag before counting diagnoses, prevalent cases excluded. A sealed half of the participants. Every hypothesis registered (hashed) before the holdout is read. One ledger per holdout, under online false-discovery-rate control (LORD++) | `codas/aou/cohort.py`, `confirm.py`, `ledger.py` | done |
| 2 | Quality control and exploration: valid days, main sleep, missingness, collinearity, reliability | A data dossier the agents read before registering anything, which never relates a feature to the outcome: distributions, families, wear dependence, who is missing what, split-half reliability. A signal map they read after (adjusted associations and attenuation). The papers' valid-day, wear-time and main-sleep rules, a two-date case rule, EHR activity before the index date, and the units of a lab outcome | `codas/core/eda.py`, `codas/aou/cohort.py`, `codas/aou/phenotypes.py` | done |
| 3 | A hypothesis-free scan of features × outcomes, corrected across the grid, published in full | Every feature against about 140 catalog outcomes on the discovery share (Cox score tests adjusted for age and sex, again after the covariates, and in two halves; partial Spearman for lab values), Benjamini-Hochberg across the grid; outcomes are chosen by consistent hits the literature has not established, those already tested on the holdout left out; the full grid published | `codas/aou/phenome.py` | done |
| 4 | Remove artefacts: lag, negative controls, E-value, wear adjustment, dose-response, proportional hazards | Run on the holdout: lag sensitivity (the effect must survive it, not only keep its sign), negative-control diagnoses, E-values, adherence adjustment, quantile dose-response, positive controls; a planted prodrome (reverse causation) fails the lag check. Not yet: splines, a specification curve, proportional-hazards tests | `codas/aou/confirm.py` | done (splines and a specification curve not yet) |
| 5 | Replicate: holdout, then internal-external (site or period), then an external cohort; expect shrinkage | The sealed holdout, tested once; each effect's consistency across sex, age, enrolment era and sub-study membership (Q, I^2); a replication package with a methods protocol for another cohort or a later release's new participants | `confirm.py`, `codas/aou/replication.py` | done (the external test itself needs that cohort) |
| 6 | Effect sizes: HR per SD and per natural unit, quartiles, absolute risk | Per SD of the discovery share, frozen in the pre-registration; per natural unit; quarter HRs with a trend test; cumulative incidence by quarter with the risk difference; a magnitude class in every table | `codas/core/preprocess.py`, `codas/core/effects.py` | done |
| 7 | Incremental value over the established risk factors, head-to-head with the incumbent marker (ΔC, calibration, NRI, decision curves) | Models frozen on discovery; ΔC (or ΔR^2) with a bootstrap CI; head-to-head with the incumbent each hypothesis names. Not yet: calibration, NRI and decision curves (they need disease-specific thresholds) | `codas/core/effects.py`, `confirm.py` | done (calibration, NRI, decision curves not yet) |
| 8 | Triangulation and specificity: MR, lags, negative controls, disease-specific vs generic "sickness" | Negative controls on the holdout; the phenome scan's grid shows across how many outcomes a feature is hit (specificity, read by eye, not yet a test). MR needs genetic data (Controlled Tier), which is outside this workspace | `confirm.py`, `codas/aou/phenome.py` | partial |
| 9 | Report the rung reached on the Hlatky/Pepe ladder and what is missing | A per-outcome scorecard; for each hypothesis the recipe steps it went through and the rung it reached (1 associated on the holdout, 2 replicated and robust, 3 adds predictive value), with the next step | `codas/aou/scorecard.py` | done |
| 10 | A new construct from raw signals with a physiological story (VILPA, SRI, DHRPS, CosinorAge, deviation from a personal baseline) | Minute-level phenotypes: interdaily stability, intradaily variability, cosinor, resting HR, night dip, HR per step (DHRPS); a day-by-day baseline and a small phenotype language, so the agents define new constructs (variability, trend, weekend contrast, reactivity to the day before); a planted reactivity is found end to end | `codas/aou/phenotypes.py`, `codas/aou/daystore.py`, `codas/core/phenotype_dsl.py` | done |
| 11 | Hypotheses that are mechanism-first, comparative, quantitative and falsifiable: "X predicts D beyond Y, with this shape, robust to a lag, not associated with this negative control" | A hypothesis template: the incumbent it must beat, direction, shape, magnitude, mechanism with citations, and falsification predictions; a deterministic validator, one review, and each prediction scored on the holdout | `codas/core/hypotheses.py`, `confirm.py` | done |
| 12 | Novelty from prior evidence, citations with identifiers | Before the agents start, Europe PMC counts the papers and reviews on every measured construct with the outcome; the class (established, emerging, unexplored) comes from those counts. Hypotheses cite only papers retrieved in the run, by PMID or DOI; the report gets a reference list and a note on any other key | `codas/core/literature.py`, `scripts/aou/literature_terms.json`, `codas/agents/tools.py` | done |

**Defaults the AoU papers share, and CoDaS's.**

| | the papers | CoDaS now |
|---|---|---|
| baseline data | at least 6 months | a 180-day baseline window |
| valid day | at least 10 h worn (600 HR minutes) and 100–45,000 steps | 100–45,000 steps; heart-rate phenotypes only from days with 600 heart-rate minutes |
| sleep | main sleep only, nights of 3–16 h | main sleep only, 3–16 h asleep, in the daily summaries and the stage phenotypes |
| outcomes | counted at least 6 months after monitoring starts | a 90-day lag after the index date, with a one-year lag sensitivity analysis on the holdout |
| small cells | suppressed below 20 | counts of 1–20 masked everywhere |

## What is not possible here

- External replication: UK Biobank or another cohort needs its own access.
- Mendelian randomisation: needs Controlled Tier genetic data.
- A device-model check: AoU does not record the device model.

CoDaS writes a replication package (definitions, code version, frozen models) so an external team can repeat a
finding, and reports which rung of the ladder each finding reached.
