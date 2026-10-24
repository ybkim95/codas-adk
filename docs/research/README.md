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
| 3 | A hypothesis-free scan of features × outcomes, corrected across the grid, published in full | A phenome-wide Cox screen on the discovery share; outcomes already tested on the holdout are left out; the full grid published | `codas/aou/phenome.py` | M7 |
| 4 | Remove artefacts: lag, negative controls, E-value, wear adjustment, dose-response, proportional hazards | Run on the holdout: lag sensitivity, negative-control diagnoses, E-values, adherence adjustment, quantile dose-response, positive controls. Next: splines and a specification curve | `codas/aou/confirm.py` | done; more in M5 |
| 5 | Replicate: holdout, then internal-external (site or period), then an external cohort; expect shrinkage | The sealed holdout, tested once. Next: heterogeneity by enrollment era and by own device vs WEAR, and a replication package for a later AoU release or another cohort | `confirm.py`; M9 | holdout done |
| 6 | Effect sizes: HR per SD and per natural unit, quartiles, absolute risk | Per SD of the discovery share, frozen in the pre-registration. Next: natural units, quartile HRs, cumulative incidence by quartile and a magnitude class in every table | `codas/core/preprocess.py`; M5 | M2-M5 |
| 7 | Incremental value over the established risk factors, head-to-head with the incumbent marker (ΔC, calibration, NRI, decision curves) | Models frozen on discovery; ΔC with a bootstrap CI; head-to-head with the incumbent each hypothesis names | M5 | M5 |
| 8 | Triangulation and specificity: MR, lags, negative controls, disease-specific vs generic "sickness" | Negative controls now; specificity across the phenome scan next. MR needs genetic data (Controlled Tier), which is outside this workspace | M7 | partial |
| 9 | Report the rung reached on the Hlatky/Pepe ladder and what is missing | A per-outcome scorecard with the rung and the missing evidence | M8 | M8 |
| 10 | A new construct from raw signals with a physiological story (VILPA, SRI, DHRPS, CosinorAge, deviation from a personal baseline) | Minute-level phenotypes now: interdaily stability, intradaily variability, cosinor, resting HR, night dip, HR per step (DHRPS). Next: a person-day store and a small phenotype algebra, so the agents can define new constructs | `codas/aou/phenotypes.py`; M6 | partial |
| 11 | Hypotheses that are mechanism-first, comparative, quantitative and falsifiable: "X predicts D beyond Y, with this shape, robust to a lag, not associated with this negative control" | A hypothesis template: the incumbent it must beat, direction, shape, magnitude, mechanism with citations, and falsification predictions. Plus a deterministic validator and one review | M4 | M4 |
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
