# How top-tier studies discover and validate biomarkers: exemplars, AI discovery systems and methods

Compiled from abstracts, indexes and search summaries (no full text could be opened); a detail no source confirmed is marked UNVERIFIED. See README.md for how far to trust it.

77 citations: 32 discovery and validation exemplars, 8 AI-driven discovery systems, 37 methods papers and guidelines.

**Limits.** No full text could be opened (every page fetch failed), so details come from abstracts and search summaries; a method detail that could not be seen is marked UNVERIFIED, and so is a DOI that comes from memory rather than from a search result (shown as plain text). Two planned checks (the All of Us 2019 NEJM program paper and the All of Us genomics paper) were not run and are left out.

**Corrections to common citations.** Gadd et al. 2024 is in *Nature Aging*, not *Nature Medicine*. Khurshid et al. is *Circulation* 2022 (online 2021). The Ioannidis & Panagiotou JAMA DOI is 10.1001/jama.2011.713.

## 1. Discovery and validation exemplars

Abbreviations: MT = multiple-testing control; Val = validation; ΔC = change in C-index; DR10 = detection rate at 10% false-positive rate; PRS = polygenic risk score.

| # | Citation | Cohort / target / marker | Discovery & MT | Validation | Key analyses |
|---|---|---|---|---|---|
| 1 | Sun BB et al. Plasma proteomic associations with genetics and health in the UK Biobank. *Nature* 2023;622:329–338. [doi](https://doi.org/10.1038/s41586-023-06592-6) | UK Biobank Pharma Proteomics Project (UKB-PPP), 54,219 people; Olink Explore 3072, 2,923 proteins | Genome-wide protein-QTL (pQTL) mapping; 14,287 primary associations at 3,760 loci, 81% novel | Discovery/replication split by ancestry: UNVERIFIED | Source of cis-pQTL instruments for causal tests; disease-association details UNVERIFIED |
| 2 | Eldjarn GH et al. Large-scale plasma proteomics comparisons through genetics and disease associations. *Nature* 2023;622:348–358. [doi](https://doi.org/10.1038/s41586-023-06563-x) | >50k UKB (Olink; British/Irish, African, South Asian ancestry) vs 36k Icelanders (SomaScan); 1,514 measured on both | pQTL and disease associations compared across platforms | Cross-platform, cross-ancestry | Platforms correlate only modestly; cis-pQTL support 72% (Olink) vs 43% (SomaScan). A marker can be specific to one platform |
| 3 | Ferkingstad E et al. Large-scale integration of the plasma proteome with genetics and disease. *Nat Genet* 2021. DOI UNVERIFIED (10.1038/s41588-021-00978-w) | 35,559 Icelanders; 4,907 aptamers | 18,084 pQTLs (19% rare variants); 257,490 protein–trait associations across 373 traits | — | About 12% of GWAS-Catalog lead variants are in high LD with pQTLs; 938 drug-target genes |
| 4 | Gadd DA et al. Blood protein assessment of leading incident diseases and mortality in the UK Biobank. *Nat Aging* 2024;4:939–948. [doi](https://doi.org/10.1038/s43587-024-00655-7) | UKB 47,600; 1,468 proteins; 23 age-related diseases + mortality | Cox model per protein; 3,209 associations (963 proteins, 21 outcomes); preprint used Bonferroni P<5.4×10⁻⁶ | Random train/test split; scores only for outcomes with ≥150 cases (preprint) | Elastic-net Cox "ProteinScores": 6 improved 10-year AUC beyond age, sex and 24 lifestyle/clinical/physical factors; the type 2 diabetes score beat both PRS and HbA1c |
| 5 | Carrasco-Zanini J et al. Proteomic signatures improve risk prediction for common and rare diseases. *Nat Med* 2024;30:2489–2498. [doi](https://doi.org/10.1038/s41591-024-03142-z) | UKB-PPP 41,931; ~3,000 proteins; 218 diseases (81–6,038 cases); 10-year incidence | LASSO across 200 subsamples, then sparse regularised Cox; 1,000× bootstrap | External: EPIC-Norfolk for 6 diseases (N=295–1,116) | 5–20 proteins improved prediction for 67 diseases (median ΔC 0.07); beat clinical-assay models for 52; median DR10 45.5% vs 25%; single-cell RNA-seq showed 4 of 5 myeloma predictors expressed specifically in plasma cells |
| 6 | Guo Y et al. Plasma proteomic profiles predict future dementia in healthy adults. *Nat Aging* 2024;4:247–260. [doi](https://doi.org/10.1038/s43587-023-00565-0) | UKB 52,645; 1,417 incident dementia cases over 14.1 years; 1,463 proteins | Proteome-wide Cox screen, then prioritised GFAP, NEFL, GDF15, LTBP2 | Details UNVERIFIED | GFAP + demographics AUC 0.891; GFAP HR 2.32; GFAP and NEFL diverge ≥10 years before diagnosis; GFAP and LTBP2 specific to dementia |
| 7 | Walker KA et al. Large-scale plasma proteomic analysis identifies proteins and pathways associated with dementia risk. *Nat Aging* 2021;1:473–489. [doi](https://doi.org/10.1038/s43587-021-00064-0) | ARIC; 4,877 proteins; n=4,110 (428 incident cases) | Proteome-wide association; 38 proteins pass Bonferroni | 16 of 38 also predicted dementia from plasma drawn about 20 years earlier | Pathway enrichment (immune, lipid, haemostasis); Mendelian randomisation (MR) implicated SVEP1 and angiostatin in Alzheimer's disease |
| 8 | Papier K et al. Identifying proteomic risk factors for cancer using prospective and exome analyses of 1463 circulating proteins and risk of 19 cancers in the UK Biobank. *Nat Commun* 2024. [doi](https://doi.org/10.1038/s41467-024-48017-6) | UKB (N UNVERIFIED); 19 cancers + 9 subsites | 618 protein–cancer associations (MT method UNVERIFIED) | 107 survived a >7-year lag | 29 supported by cis-pQTL or exome-wide protein genetic scores; 4 had both lag and genetic support (e.g., CD74 with non-Hodgkin lymphoma) |
| 9 | Argentieri MA et al. Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations. *Nat Med* 2024. [doi](https://doi.org/10.1038/s41591-024-03164-7) | UKB 45,441; 2,897 proteins | Machine-learning age clock (204 proteins) | China Kadoorie Biobank (3,977), FinnGen (1,990) | Predicts 18 diseases and mortality; a 20-protein subset keeps 91% of accuracy |
| 10 | Zheng J et al. Phenome-wide Mendelian randomization mapping the influence of the plasma proteome on complex diseases. *Nat Genet* 2020;52:1122–1131. DOI UNVERIFIED (10.1038/s41588-020-0682-6) | 1,002 proteins × 225 phenotypes | Two-sample MR + colocalisation | — | 31.5% of MR-supported associations lacked colocalisation support; 111 putatively causal effects; pairs with both kinds of support were more often approved drug targets |
| 11 | Buergel T et al. Metabolomic profiles predict individual multidisease outcomes. *Nat Med* 2022;28:2309–2320. [doi](https://doi.org/10.1038/s41591-022-01980-3) | UKB 117,981; 168 NMR markers; 24 conditions | Neural-network "metabolomic state" | Whitehall II, Rotterdam, Leiden Longevity, PROSPER | Added information beyond clinical variables for 8 diseases (e.g., type 2 diabetes, dementia, heart failure); decision curves supported usefulness |
| 12 | Julkunen H et al. Atlas of plasma NMR biomarkers for health and disease in 118,461 individuals from the UK Biobank. *Nat Commun* 2023;14:604. [doi](https://doi.org/10.1038/s41467-023-36231-7) | UKB; 249 NMR measures; >700 diseases | Hypothesis-free atlas (models and thresholds UNVERIFIED) | — | Associations extend beyond cardiometabolic disease; association patterns are shared across diseases, so specificity has to be tested; the platform vendor is among the authors |
| 13 | Wang TJ et al. Metabolite profiles and the risk of developing diabetes. *Nat Med* 2011;17:448–453. [doi](https://doi.org/10.1038/nm.2307) | Framingham, 2,422 normoglycaemic people, 201 incident type 2 diabetes over 12 years | LC-MS profiling | Independent cohort (identity UNVERIFIED) | Three branched-chain/aromatic amino acids: >5-fold risk, top vs bottom quartile |
| 14 | Wang Z et al. Gut flora metabolism of phosphatidylcholine promotes cardiovascular disease. *Nature* 2011;472:57–63. [doi](https://doi.org/10.1038/nature09922) | Clinical cohorts (sizes UNVERIFIED) | Untargeted metabolomics found choline, TMAO and betaine | Independent large cohort | Mouse and germ-free experiments established the mechanism; the standard template from scan to mechanism |
| 15 | Würtz P et al. Metabolite profiling and cardiovascular event risk: a prospective study of 3 population-based cohorts. *Circulation* 2015;131:774–785. DOI UNVERIFIED | FINRISK 7,256 (discovery); SABRE, BWHHS (replication) | NMR panel | 2 external cohorts; mass-spectrometry cross-check | Per-SD HRs after lipid adjustment: phenylalanine 1.18, MUFA 1.17, omega-6 0.89; the score improved calibration but not discrimination |
| 16 | Zhou Y et al. A foundation model for generalizable disease detection from retinal images. *Nature* 2023;622:156–163. [doi](https://doi.org/10.1038/s41586-023-06555-x) | RETFound, self-supervised on 1.6M unlabelled retinal images | Foundation model | Internal/external (partly verified) | Better incident heart failure and myocardial infarction prediction with fewer labels |
| 17 | Poplin R et al. Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning. *Nat Biomed Eng* 2018;2(3):158. [doi](https://doi.org/10.1038/s41551-018-0195-0) | Trained on 284,335 | Deep learning | Two validation sets (12,026; 999) | Major adverse cardiac events AUC 0.70; the model relied on the optic disc and vessels |
| 18 | Attia ZI et al. An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction. *Lancet* 2019. [doi](https://doi.org/10.1016/S0140-6736(19)31721-0) | Mayo; 180,922 patients | CNN | Held-out test set | AUC 0.87 from one ECG; 0.90 from multiple ECGs |
| 19 | Khurshid S et al. ECG-based deep learning and clinical risk factors to predict atrial fibrillation. *Circulation* 2022;145:122–133. [doi](https://doi.org/10.1161/CIRCULATIONAHA.121.057480) | MGH training; BWH and UKB external | ECG-AI, the CHARGE-AF score, and both combined | 2 external sets | Combined AUROC 0.838/0.777/0.746 vs CHARGE-AF 0.802/0.752/0.732: the two are complementary |
| 20 | Yao X et al. AI-enabled electrocardiograms for identification of patients with low ejection fraction: a pragmatic, randomized clinical trial. *Nat Med* 2021. DOI UNVERIFIED | 22,641 patients; 348 clinicians randomised | Cluster RCT (EAGLE) | Prospective, randomised | Increased new low-EF diagnoses; reaches the clinical-outcomes rung |
| 21 | Denny JC et al. Systematic comparison of phenome-wide association study of electronic medical record data and genome-wide association study data. *Nat Biotechnol* 2013;31:1102–1110. [doi](https://doi.org/10.1038/nbt.2749) | 13,835 people; 3,144 SNPs × 1,358 phenotypes | PheWAS | Separate cohort of 7,406 | Recovered 66% of adequately powered known associations, which validates the pipeline; 63 new pleiotropic signals |
| 22 | Patel CJ et al. An Environment-Wide Association Study (EWAS) on type 2 diabetes mellitus. *PLoS One* 2010;5:e10746. [doi](https://doi.org/10.1371/journal.pone.0010746) | NHANES; 266 exposures | GWAS-style MT | Independent survey waves | Odds ratio per SD 1.7 for heptachlor epoxide; included as the canonical exposome-wide design (not a top venue) |
| 23 | Emerging Risk Factors Collaboration. C-reactive protein, fibrinogen, and cardiovascular disease prediction. *NEJM* 2012;367:1310–1320. DOI UNVERIFIED | 52 cohorts, 246,669 people | Pre-specified addition of markers | Individual-participant meta-analysis | ΔC +0.0039 (CRP); NRI 1.52%; about 1 extra event prevented per 400–500 people screened |
| 24 | Wang TJ et al. Multiple biomarkers for the prediction of first major cardiovascular events and death. *NEJM* 2006;355:2631–2639. DOI UNVERIFIED | Framingham 3,209; 10 biomarkers | Multimarker | — | Marginal improvement beyond conventional factors; a standard cautionary result |
| 25 | Master H et al. Association of step counts over time with the risk of chronic disease in the All of Us Research Program. *Nat Med* 2022;28:2301–2308. [doi](https://doi.org/10.1038/s41591-022-02012-w) | All of Us 6,042 (73% female, 84% white); Fitbit; median 4 years | Phenome-wide phecode screen (Bonferroni line ≈3.2×10⁻⁵), then time-varying Cox | None; authors ask for more diverse samples | Dose–response: linear for obesity and depression; plateau around 8,000–9,000 steps for diabetes and hypertension |
| 26 | Zheng NS et al. Sleep patterns and risk of chronic disease as measured by long-term monitoring with commercial wearable devices in the All of Us Research Program. *Nat Med* 2024;30:2648–2656. [doi](https://doi.org/10.1038/s41591-024-03155-8) | All of Us 6,785; median 4.5 years | Discovery PheWAS + time-varying Cox, adjusted for 10 covariates including time-varying steps | 112 stratified analyses; cells <20 suppressed | REM/deep sleep inversely related to atrial fibrillation, flagged as a possible artefact of heart-rate-based sleep staging; J-shaped duration effects; robust to adjustment for sleep apnoea |
| 27 | Schalkamp AK et al. Wearable movement-tracking data identify Parkinson's disease years before clinical diagnosis. *Nat Med* 2023;29:2048–2056. [doi](https://doi.org/10.1038/s41591-023-02440-2) | UKB ~104k accelerometry; 153 diagnosed PD | Machine learning vs genetic, lifestyle, biochemistry and symptom models | Internal | AUPRC 0.14 (diagnosed), 0.07 (prodromal) vs 0.01–0.03 for other data types; distinguished PD from osteoarthritis |
| 28 | Yang Y et al. Artificial intelligence-enabled detection and assessment of Parkinson's disease using nocturnal breathing signals. *Nat Med* 2022. DOI UNVERIFIED | 757 PD, 6,914 controls | Deep model on breathing | Details UNVERIFIED | Accuracy 80–86% from one night, 95% from 12 nights (press reports); severity estimated |
| 29 | Perez MV et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *NEJM* 2019;381:1909–1917. DOI UNVERIFIED | 419,297 enrolled | Irregular-pulse algorithm | Prospective, ECG-patch confirmation | PPV 0.84; 0.52% of participants notified |
| 30 | Omberg L et al. Remote smartphone monitoring of Parkinson's disease and individual response to therapy. *Nat Biotechnol* 2022;40:480–487. DOI UNVERIFIED | mPower smartphone study | Smartphone tasks | In-clinic subset | Tapping AUC 0.8; correlation with clinical severity r=0.71 |
| 31 | Giancardo L et al. Computer keyboard interaction as an indicator of early Parkinson's disease. *Sci Rep* 2016;6:34468. | 42 early PD vs 43 controls | Key hold-time model | Data public on PhysioNet | AUC 0.81 vs 0.75 for clinical finger-tapping |
| 32 | Patten et al. The All of Us Research Program's wearables dataset. *Nat Med* 2026. [doi](https://doi.org/10.1038/s41591-026-04352-3) | >59k Fitbit users, 14 years; 46% also have EHR and genomics | Data descriptor | — | Main resource for your system |

## 2. AI-driven discovery systems and how they were evaluated

| Citation | How it was evaluated |
|---|---|
| Swanson K et al. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. *Nature* 2025. [doi](https://doi.org/10.1038/s41586-025-09442-9) | Wet-lab tests of 92 designs; 2 improved variant binding |
| Gottweis J et al. Towards an AI co-scientist. [arXiv:2502.18864](https://arxiv.org/abs/2502.18864) (2025). Companion: Penadés JR et al., *Cell* 2025;188:6654–6665, [doi](https://doi.org/10.1016/j.cell.2025.08.018) | Multi-agent generate–debate–rank tournament; in-vitro (AML) and organoid (liver fibrosis) validation; reproduced an unpublished mechanism. A peer-reviewed version of the co-scientist paper itself: UNVERIFIED |
| Huang K et al. Automated hypothesis validation with agentic sequential falsifications. ICML 2025, [PMLR](https://proceedings.mlr.press/v267/huang25n.html) | POPPER: agents design falsification tests with sequential Type-I error control; matched human scientists roughly 10× faster |
| Majumder BP et al. DiscoveryBench: towards data-driven discovery with large language models. ICLR 2025, [arXiv](https://arxiv.org/abs/2407.01725v1) | 264 real tasks taken from published papers; best system ≈25% |
| Si C, Yang D, Hashimoto T. Can LLMs generate novel research ideas? A large-scale human study with 100+ NLP researchers. ICLR 2025, [arXiv](https://arxiv.org/pdf/2409.04109) | Blind expert review: LLM ideas rated more novel but less feasible; LLM self-evaluation unreliable |
| Ludwig J, Mullainathan S. Machine learning as a tool for hypothesis generation. *QJE* 2024;139:751–827. [doi](https://doi.org/10.1093/qje/qjad055) | Turns a predictive model into human-interpretable hypotheses (details of the validation step UNVERIFIED) |
| Lu C et al. Towards end-to-end automation of AI research. *Nature* 2026;651:914–919. doi 10.1038/s41586-026-10265-5 ([UBC release](https://www.cs.ubc.ca/news/2026/03/ai-scientist)) | One generated paper passed first-round review at a workshop; known failures are weak rigour and hallucination |
| Kosmos: an AI scientist for autonomous discovery. [arXiv:2511.02824](https://arxiv.org/pdf/2511.02824) (authors UNVERIFIED) | Independent experts audited 102 statements: 79.4% accurate |

## 3. Methodology papers and guidelines (what each requires)

| Citation | What it requires |
|---|---|
| Pepe MS et al. Phases of biomarker development for early detection of cancer. *JNCI* 2001;93:1054–61. [doi](https://doi.org/10.1093/jnci/93.14.1054) | Five phases, from exploratory through population impact |
| Hlatky MA et al. Criteria for evaluation of novel markers of cardiovascular risk (AHA statement). *Circulation* 2009;119:2408–16. [doi](https://doi.org/10.1161/CIRCULATIONAHA.109.192278) | Representative population, enough events, value added beyond standard factors, both discrimination and accuracy (calibration) reported |
| McShane LM et al. REporting recommendations for tumour MARKer prognostic studies (REMARK). *Br J Cancer* 2005;93:387–391. [doi](https://doi.org/10.1038/sj.bjc.6602678) | Reporting standard for marker studies |
| Collins GS et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378 (DOI inferred from article ID: 10.1136/bmj-2023-078378) | 27-item checklist; replaces TRIPOD 2015 |
| von Elm E et al. The STROBE statement. *Lancet* 2007;370:1453–57. [doi](https://doi.org/10.1016/S0140-6736(07)61602-X) | Observational-study reporting (22 items) |
| Skrivankova VW et al. STROBE-MR. *JAMA* 2021;326:1614–21. [doi](https://doi.org/10.1001/jama.2021.18236) | Reporting standard for MR |
| Wolff RF et al. PROBAST. *Ann Intern Med* 2019;170:51–58. DOI UNVERIFIED (explanation paper [doi](https://doi.org/10.7326/M18-1377)) | Risk of bias across 4 domains, 20 signalling questions |
| Moons KGM et al. PROBAST+AI. *BMJ* 2025;388:e082505. [doi](https://doi.org/10.1136/bmj-2024-082505) | Separate development (16 questions) and evaluation (18 questions) parts |
| Pencina MJ et al. Evaluating the added predictive ability of a new marker. *Stat Med* 2008;27:157–172. [doi](https://doi.org/10.1002/sim.2929) | Defines NRI and IDI |
| Kerr KF et al. Net reclassification indices for evaluating risk prediction instruments: a critical review. *Epidemiology* 2014;25:114–21 | Report NRI for events and non-events separately; avoid category-free NRI |
| Pepe MS et al. Testing for improvement in prediction model performance. *Stat Med* 2013;32:1467–82. [doi](https://doi.org/10.1002/sim.5727) | Testing ΔAUC is redundant with testing the coefficient; estimate it instead |
| Pepe MS et al. Limitations of the odds ratio. *Am J Epidemiol* 2004;159:882–90. [doi](https://doi.org/10.1093/aje/kwh101) | An odds ratio of 3 gives only 25% sensitivity at 10% false positives |
| Cook NR. Use and misuse of the ROC curve in risk prediction. *Circulation* 2007;115:928–35. [doi](https://doi.org/10.1161/CIRCULATIONAHA.106.672402) | The C-statistic is insensitive; calibration also matters |
| Steyerberg EW et al. Assessing the performance of prediction models. *Epidemiology* 2010;21:128–38. [doi](https://doi.org/10.1097/EDE.0b013e3181c30fb2) | Framework combining traditional and newer measures |
| Van Calster B et al. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17:230. [doi](https://doi.org/10.1186/s12916-019-1466-7) | Calibration must be reported |
| Vickers AJ, Elkin EB. Decision curve analysis. *Med Decis Making* 2006;26:565–74. [doi](https://doi.org/10.1177/0272989X06295361); Vickers et al. *BMJ* 2016;352:i6. [doi](https://doi.org/10.1136/bmj.i6) | Net benefit across decision thresholds |
| Riley RD et al. Calculating the sample size required for developing a clinical prediction model. *BMJ* 2020;368:m441. [doi](https://doi.org/10.1136/bmj.m441) | Minimum sample size and events (pmsampsize) |
| Collins GS et al. Evaluation of clinical prediction models (part 1). *BMJ* 2024;384:e074819. [doi](https://doi.org/10.1136/bmj-2023-074819) | Internal, internal–external and external validation; heterogeneity and fairness |
| VanderWeele TJ, Ding P. Introducing the E-value. *Ann Intern Med* 2017;167:268–74. [doi](https://doi.org/10.7326/M16-2607) | Minimum confounder strength needed to explain away an effect |
| Lipsitch M et al. Negative controls. *Epidemiology* 2010;21:383–88. [doi](https://doi.org/10.1097/EDE.0b013e3181d61eeb); Schuemie MJ et al. Empirical CI calibration. *PNAS* 2018, doi 10.1073/pnas.1708282114 (as cited in [package docs](https://ohdsi.r-universe.dev/EmpiricalCalibration/doc/EmpiricalPCalibrationVignette.html)) | Detect bias; calibrate p-values and CIs empirically |
| Hernán MA, Robins JM. Using big data to emulate a target trial. *Am J Epidemiol* 2016;183:758–64. [doi](https://doi.org/10.1093/aje/kwv254) | Explicit time zero, eligibility and treatment strategies |
| Lawlor DA et al. Triangulation in aetiological epidemiology. *IJE* 2016;45:1866–86. [doi](https://doi.org/10.1093/ije/dyw314) | Combine methods whose biases are unrelated; predict each bias's direction |
| Ioannidis JPA. Why most discovered true associations are inflated. *Epidemiology* 2008;19:640–48. [doi](https://doi.org/10.1097/EDE.0b013e31818131e7); Ioannidis & Panagiotou. *JAMA* 2011;305:2200–10. [doi](https://doi.org/10.1001/jama.2011.713); Ioannidis & Bossuyt. Waste, leaks, and failures in the biomarker pipeline. *Clin Chem* 2017;63:963–72. [doi](https://doi.org/10.1373/clinchem.2016.254649) | In 29 of 35 highly cited biomarker claims, the later meta-analysis found a smaller effect; remedies are pre-registration and external validation |
| Patel CJ et al. Vibration of effects. *J Clin Epidemiol* 2015;68:1046–58. [doi](https://doi.org/10.1016/j.jclinepi.2015.05.029); Simonsohn U et al. Specification curve analysis. *Nat Hum Behav* 2020;4:1208–14. [doi](https://doi.org/10.1038/s41562-020-0912-z) | Show how robust an estimate is across model specifications |
| Giambartolomei C et al. Colocalisation test (coloc). *PLoS Genet* 2014;10:e1004383. [doi](https://doi.org/10.1371/journal.pgen.1004383); Burgess S et al. MR guidelines. *Wellcome Open Res* 2023;4:186. [doi](https://doi.org/10.12688/wellcomeopenres.15555.3) | Standards for colocalisation and MR |
| Austin PC et al. Competing risks. *Circulation* 2016;133:601–09. [doi](https://doi.org/10.1161/CIRCULATIONAHA.115.017719) | Competing-risk survival methods |
| Sterne JAC et al. Multiple imputation. *BMJ* 2009;338:b2393. [doi](https://doi.org/10.1136/bmj.b2393) | Multiple imputation over complete-case analysis |
| Altman DG, Royston P. The cost of dichotomising continuous variables. *BMJ* 2006;332:1080. [doi](https://doi.org/10.1136/bmj.332.7549.1080) | Do not split continuous variables in two |
| Fry A et al. UK Biobank representativeness. *Am J Epidemiol* 2017;186:1026–34 | Healthy-volunteer bias |
| Ritchie SC et al. NMR data quality control in UK Biobank. *Sci Data* 2023;10:64. [doi](https://doi.org/10.1038/s41597-023-01949-y) | Removal of technical variation |

## 4. Synthesis

### (a) Recipe for a novel biomarker, with suggested agent gates

1. **Protocol agent: freeze the plan before looking at outcomes.** Specify:
   - Population and time zero.
   - A wearable baseline window, with follow-up starting after it so there is no immortal time (target-trial logic).
   - The outcome set: phecodes with at least N incident cases, or a curated list like Gadd's 23 or Buergel's 24.
   - Exclusion of prevalent cases.
   - Covariate tiers: T0 age/sex/site; T1 plus lifestyle and socioeconomic status; T2 plus established risk factors or scores; T3 plus PRS.
   - The metrics, multiple-testing plan, data splits, and sample size (Riley criteria; Gadd scored only outcomes with ≥150 cases).
   - Log each hypothesis with a timestamp in a ledger, and use TRIPOD+AI and STROBE as templates.
2. **Data agent: quality control and exploration** (see section f).
3. **Discovery agent: hypothesis-free scan.**
   - Fit features (per SD) × outcomes with Cox models, using time-varying exposure where available (Master; Zheng). Run T0, then T2.
   - Control multiplicity across the whole grid with Bonferroni or Benjamini–Hochberg FDR, and publish the full grid (the atlas pattern).
   - Run discovery on one split and lock the holdout.
4. **Critic agent: remove artefacts.**
   - Reverse causation: exclude early events, show long-lag effects (Papier >7 years) and pre-diagnostic trajectories (Guo; Walker's 20-years-earlier samples).
   - Measurement-mechanism confounding: check whether the device algorithm partly builds the outcome into the feature (Zheng's atrial fibrillation / heart-rate staging caveat).
   - Negative controls with empirical calibration; E-value; specification-curve analysis.
   - Competing risks; proportional-hazards checks; splines for dose–response; subgroup heterogeneity.
5. **Validation agent: replicate.**
   - Holdout first, then internal–external validation leaving out one site or period (Collins 2024), then an external cohort (e.g., UKB accelerometry), then a cross-platform or cross-device check (Würtz's mass-spectrometry check; Eldjarn's platform disagreement).
   - Report replication effect sizes and expect them to shrink (Ioannidis).
6. **Stats agent: quantify effects.** Report HR per SD and per category, absolute risk, cumulative incidence, and population attributable fraction (PAF).
7. **Prediction agent: test incremental value.**
   - Use nested models against T2 (plus PRS): ΔC with bootstrap CI, DR at a fixed false-positive rate, AUPRC for rare outcomes, calibration, categorical NRI, and decision curves.
   - Compare head-to-head against the strongest established predictor (Gadd vs PRS and HbA1c; Khurshid vs CHARGE-AF).
   - Prefer sparse panels.
8. **Causal agent: triangulate.**
   - Use cis-instrument MR with colocalisation (in Zheng 2020, 31.5% of MR-supported results failed colocalisation) and bidirectional MR.
   - For behaviours, emulate a target trial with a sustained-strategy definition.
   - Then triangulate across methods (Lawlor).
9. **Biology agent: mechanism and specificity.** Use pathway enrichment (Walker), cell-type expression (Carrasco-Zanini), disease specificity (Guo; Schalkamp), and attribution for deep features (Poplin). The TMAO paper (Wang Z) is the full template: screen, independent cohort, then experiments.
10. **Reporting agent.** Report decision curves and number needed to screen (the CRP/fibrinogen collaboration: about 1 event prevented per 400–500 screened). State which ladder rung is still missing. Follow the checklists, apply the All of Us rule of no cells under 20, and release code.

Suggested default gates. These are my own synthesis, not taken from any one paper:
- **G1 Discovery:** FDR q<0.05 (or Bonferroni) and at least 100 events.
- **G2 Robustness:** same direction with CI excluding the null after T2 adjustment and a 2-year lag; E-value for the CI bound larger than the strongest measured confounder; negative controls null; at least 80% of specifications agree.
- **G3 Replication:** same direction, p<0.05, and I²<50%.
- **G4 Incremental value:** ΔC CI excludes 0, or net benefit is positive across the pre-specified thresholds; calibration slope 0.8–1.2.
- **G5 Support:** at least one line of causal or biological evidence.
- **G6 Novelty:** the novelty check (section d) passes.

### (b) Evaluation ladder

| Rung | Hlatky 2009 | Pepe 2001 | Evidence required | Possible in All of Us? |
|---|---|---|---|---|
| 0 Analytical validity | (implicit) | Phase 2, assay | Reproducibility and QC; agreement across platforms or devices | Partly |
| 1 Proof of concept | Marker differs by outcome status | Phase 1, exploratory | Effect sizes, prevalent-case PheWAS | Yes |
| 2 Prospective validation | Predicts future outcomes in independent populations | Phase 3, pre-diagnostic samples | Incident-disease Cox models, lag analyses, replication | Yes (external cohort needed) |
| 3 Incremental value | Adds to standard risk markers | — | ΔC, calibration, NRI, head-to-head comparisons | Yes |
| 4 Clinical utility | Shifts risk across treatment thresholds | Phase 4, prospective screening | Decision curves, number needed to screen; prospective PPV (Apple Heart Study) | Decision curves only |
| 5 Clinical outcomes | Improves outcomes in an RCT | Phase 5, population impact | A trial such as EAGLE | Propose only |
| 6 Cost-effectiveness | Justifies the cost | — | Decision-analytic modelling | Model only |

Most Nature Medicine papers in the tables reach rungs 2–3 and add decision curves. Würtz and the CRP/fibrinogen collaboration show that rung 3 can be negative or tiny even when associations are strong.

### (c) Metrics beyond p-values

| Metric | How to report it | Source / caution |
|---|---|---|
| HR per SD and per category | 95% CI, events and person-years, minimal and full models | Hlatky; Würtz; never dichotomise (Altman–Royston) |
| Dose–response | Restricted cubic splines plus a non-linearity test | Master's plateau; Zheng's J-shape |
| Absolute risk | Aalen–Johansen cumulative incidence, 10-year risk, risk difference, number needed to screen | Austin; CRP/fibrinogen collaboration |
| Discrimination | Harrell or Uno C, or time-dependent AUC; ΔC with bootstrap CI; AUPRC; DR at fixed false-positive rate | Carrasco-Zanini; Schalkamp; estimate rather than test (Pepe 2013) |
| Calibration | Intercept, slope, flexible calibration plot, observed/expected by decile, in validation data | Van Calster; Cook |
| Reclassification | Categorical NRI at clinical thresholds, events and non-events reported separately | Pencina; Kerr |
| Net benefit | Decision curve over a pre-specified threshold range | Vickers; Buergel |
| Classification reality check | Sensitivity at a fixed false-positive rate | Pepe 2004 (odds ratio 3 gives 25% sensitivity) |
| PAF | g-computation for a defined shift (e.g., everyone below 8,000 steps moved to 8,000), stated as conditional on causal assumptions | General practice; not seen used in these papers |
| E-value | For the point estimate and for the CI bound | VanderWeele–Ding |
| Heterogeneity | Interaction p-values, I² and Q across sites/subgroups, fairness by subgroup | Collins 2024 |
| Multiplicity | Number of tests and the threshold or q-value used | Walker; Master |

### (d) How novelty is established

- **Search prior evidence automatically:** literature, GWAS Catalog, and published atlases (Julkunen; UKB-PPP). Report the share of findings that are new, as Sun did (81% of pQTLs). An arXiv system, [TriAgent](https://arxiv.org/abs/2510.16080v1), labels candidates known or novel using literature retrieval.
- **Adjust for established markers** (Würtz adjusted for lipids) and quantify the increment against a familiar benchmark (the CRP/fibrinogen paper set its gains beside HDL cholesterol's ΔC of 0.0050).
- **Compare head-to-head** with PRS, HbA1c, CHARGE-AF, and other data types (Schalkamp).
- **Show orthogonality:** report correlation with known risk factors, and show the marker still adds value inside models that already contain them.
- **Show specificity across outcomes**, separating disease-specific from shared signals (Guo; Carrasco-Zanini). Julkunen's shared association patterns are a warning against generic "sickness" markers.
- **Replicate consistently** across cohorts, platforms and ancestries.

### (e) Hypothesis generation and pre-specification

- The strongest papers pre-specify the framework (outcome universe, case thresholds, splits, multiple-testing plan, comparators) rather than individual hits.
- Hits from a scan are then prioritised by:
  - Temporal ordering: lags and pre-diagnostic trajectories.
  - Genetics: MR, colocalisation, exome scores.
  - Biology: pathways and cell types.
  - Experiments: e.g., the TMAO mouse work.
- Lessons from AI-discovery systems:
  - Rank hypotheses with tournaments (co-scientist).
  - Validate by falsification with sequential Type-I error control (POPPER).
  - Use blind expert review (Si) and statement-level accuracy audits (Kosmos).
  - Require experimental or external confirmation (Virtual Lab).
  - Benchmark success is low (DiscoveryBench ≈25%), so keep the holdout locked until each ledger entry is frozen. My own suggestion beyond these sources is to control false discoveries across the whole hypothesis stream with online FDR or e-values.

### (f) Standard preprocessing and exploration

- **Cohort:** STROBE flow diagram, time zero, exclusion of prevalent and washout cases, censoring at end of EHR observation, and validated outcome definitions. Denny's positive-control replication (66%) is a model for checking the phenotype pipeline.
- **Wearables:**
  - Valid days defined as ≥10 hours wear and 100–45,000 steps (used in later All of Us studies; not confirmed for Master itself).
  - Keep "main sleep" and drop naps; a [JMIR 2025](https://doi.org/10.2196/71718) study reclassified 4.75% of sleep logs.
  - Test sensitivity to the averaging window, since a [2026 medRxiv preprint](https://www.medrxiv.org/content/10.64898/2026.01.29.26344899v1.full) shows it changes associations.
  - Record device and algorithm versions, and include within-person variability (irregularity) features.
- **Omics:** correct technical variation (Ritchie: preparation time, plate well, batch, drift); handle values below the limit of detection; log- or rank-inverse-normal transform, then standardise to SD units.
- **Outliers:** winsorise or apply physiological bounds, and rerun excluding outliers as a sensitivity check.
- **Missing data:** profile the patterns and use multiple imputation for covariates (Sterne). Treat missing wear time as potentially informative.
- **Distributions and shape:** QQ plots and splines; never dichotomise.
- **Collinearity:** correlation heatmaps and clustering, VIF, and penalised models (LASSO/elastic net).
- **Selection:** compare the cohort with the population (Fry). The All of Us wearable cohorts skew female, white and college-educated (71–73% female, 84% white, 71% college in Master and Zheng); consider inverse-probability weighting and say so in the limitations.
