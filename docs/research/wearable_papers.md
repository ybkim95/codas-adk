# How top studies discover and validate novel wearable biomarkers: an evidence base for CoDaS on All of Us

**What could and could not be checked.** No host could be reached to open a full text (nature.com, PMC and Europe PMC all failed), so every detail below comes from search-result abstracts, indexes and summaries, and the search budget ran out before every item could be checked.

- **[UNV]** means no search result confirmed that detail.
- **DOIs** are given only where a search result showed one. Otherwise I give the PubMed, PMC, publisher, preprint or UK Biobank (UKB) page I saw.
- **Key figures and tables** couldn't be checked for any paper, because I never saw a full text.
- **Oura, Garmin and Whoop:** I confirmed no study from these devices in the target venues.
- **Not yet verified:** Walmsley 2022, the CAPTURE-24 papers (Chan) and Yuan 2024.

---

## Part 1. Evidence table (51 papers)

### A. All of Us (Fitbit plus EHR)

| # | Paper (link) | Cohort, N, outcome | How the biomarker was derived, design, analysis | Validation and effect sizes |
|---|---|---|---|---|
| A1 | Master H et al. *Association of step counts over time with the risk of chronic disease in the All of Us Research Program.* Nat Med 2022;28:2301–08. [doi:10.1038/s41591-022-02012-w](https://www.nature.com/articles/s41591-022-02012-w) | Fitbit owners who linked their own device (BYOD); N=6,042; median age 56.7; median 4.0 y of monitoring; incident disease across 1,711 phecodes | **Exposure:** daily steps (median 7,731). **Data rules:** >6 months of data required; days <100 steps dropped; months with <15 valid days dropped (15–16% of months). **Design:** outcomes counted only after the first 6 months of monitoring. **Model:** Cox with monthly time-varying steps and time-varying wear time; adjusted for age, race, sex, CAD, cancer, BMI, SBP, education, smoking, alcohol. Exact multiple-testing and spline method [UNV]. | Results held across limited vs full models, wear-time adjustment, and baseline steps over the first 3 or 6 months. No external cohort. Inverse linear association for obesity, sleep apnea, GERD and MDD (>8,200 steps/d protective). Plateau at 8,000–9,000 steps for T2D and hypertension. A later author correction (doi:10.1038/s41591-023-02313-8) concerns the AoU rule against reporting counts <20. |
| A2 | Zheng NS et al. *Sleep patterns and risk of chronic disease as measured by long-term monitoring with commercial wearable devices in the All of Us Research Program.* Nat Med 2024;30:2648–56. [doi:10.1038/s41591-024-03155-8](https://www.nature.com/articles/s41591-024-03155-8) | N=6,785; median 4.5 y; ~6.5M nights; 71% female, 84% white; incident phecodes | **Exposure:** Fitbit sleep duration, REM and deep sleep %, sleep irregularity (exact definition [UNV]). **Data rules:** ≥6 months of data; one search snippet says people with >30% of nights <4 h were excluded. **Model:** discovery phenome-wide logistic regression (adjusted for age, sex and other covariates), then time-varying confirmation. | Each +1% REM: OR 0.86 (0.82–0.91) for AF and related rhythm abnormalities; deep sleep OR 0.87. Only the deep-sleep result held in time-varying models. Irregularity raised odds of obesity, hyperlipidemia, hypertension, MDD and GAD. J-shaped duration curves for hypertension, MDD and GAD. The authors note Fitbit sleep staging partly uses HR, a possible artifact. |
| A3 | Brittain EL et al. *Physical Activity and Incident Obesity Across the Spectrum of Genetic Risk for Obesity.* JAMA Netw Open 2024;7:e243821. [doi:10.1001/jamanetworkopen.2024.3821](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2816822) | N=3,124 without obesity at baseline, with genomic data; mean 8,326 steps/d; median >5 y; incident obesity | Mean daily steps combined with a BMI polygenic risk score (PRS). Obesity HR modeled across step counts at the 25th, 50th and 75th PRS percentiles, by baseline BMI. Valid-day rules [UNV]. | Obesity incidence rose from 13% to 43% from the lowest to highest PRS quartile. A 75th-percentile PRS needs +2,280 steps/d (11,020 total) to match the 50th. Matching the 25th percentile needs +3,460 to +6,350 steps/d at BMI 22–28. |
| A4 | Chen et al. *Daily Heart Rate per Step (DHRPS): a wearables metric associated with cardiovascular disease… All of Us.* JAHA 2025. [doi:10.1161/JAHA.124.036801](https://pmc.ncbi.nlm.nih.gov/articles/PMC12184247/) | N=6,947 BYOD; 5.8M person-days; prevalent cardiometabolic disease | **New metric:** mean daily HR ÷ daily steps (HR "cost" of activity). Cross-sectional design. | DHRPS associated more strongly than HR or steps alone. Fitness link checked against treadmill tests in only 21 people. Top quartile: about 2× T2D, 1.7× heart failure, 1.6× hypertension, 1.4× coronary atherosclerosis. No association with stroke or MI. |
| A5 | (Brittain, corresponding author) *Daily steps offset risks of sedentary behavior in the All of Us research program.* Nat Commun 2026. [doi:10.1038/s41467-026-71652-0](https://www.nature.com/articles/s41467-026-71652-0) | N=15,327; 11 chronic outcomes | Sedentary minutes plus steps. **Data rules:** valid day = ≥10 h wear, 100–45,000 steps, 0–1,440 sedentary min; months with ≤15 valid days excluded; diagnoses counted only ≥6 months after monitoring began. | Extra 1,700–5,500 steps/d offset the excess risk of 14 vs 8 h/d sedentary: obesity and MASLD 1,700; hypertension and sleep apnea 2,200; diabetes 5,300; COPD 5,500. No amount of steps offset the risk for CAD or heart failure. |
| A6 | Fulda et al. *11 million days of longitudinal wearable data reveal novel future health insights.* medRxiv 2026 (preprint). [medRxiv](https://www.medrxiv.org/content/10.64898/2026.01.29.26344899v1) | N=29,351; >700 incident and >1,300 prevalent EHR outcomes | Four metrics: steps, peak 1-min cadence, peak 30-min cadence, HR per step. Valid day = ≥16 h wear. Averaged over windows of 1 day, 1 week, 1 month, 6 months and 1 year. Phenome-wide scan. | Longer windows gave stronger, more stable associations, an argument against 7-day snapshots. |
| A7 | (Univ. of Minnesota) *Association of physical activity from wearable devices and chronic disease risk…* Commun Med 2026 [venue seen only on an aggregator, UNV]. [Preprint](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12083687/) | N=22,019 with ≥6 months of Fitbit data | Steps and sedentary time. Methods [UNV]. | +2,000 steps/d → lower hazard of obesity, T2D and MDD. More sedentary time → higher cardiometabolic risk. |
| A8 | Barua et al. *Fitbit-measured physical activity is inversely associated with incident atrial fibrillation among All of Us participants.* [medRxiv 2024](https://www.medrxiv.org/content/10.1101/2024.06.27.24308221v1); [AHA abstract](https://www.ahajournals.org/doi/10.1161/circ.150.suppl_1.4138564). Journal version [UNV] | N=6,086; 1 year of Fitbit data as exposure; 55 AF events (0.9%) over 5 y | Weekly MVPA (moderate-to-vigorous activity); Cox. | 2.5–5 h/week: ~60% lower AF risk; >5 h: ~65% lower. Few events. |
| A9 | *Associations Between Objective Sleep Characteristics From Wearable Physiologic Monitors and Incident Atrial Fibrillation.* JACC Adv 2025. [doi:10.1016/j.jacadv.2025.102539](https://www.jacc.org/doi/10.1016/j.jacadv.2025.102539). **Partly verified** (URL seen once) | N=52,952 AoU; median 935 days | Cox; nights with <3 h or >16 h total sleep excluded. | Authors and effect sizes [UNV]. |
| A10 | Stanford kidney-stone cohort, J Endourol 2026. [Abstract](https://www.urotoday.com/recent-abstracts/endourology-urolithiasis/stone-disease/170796-long-term-physical-activity-patterns-and-incident-kidney-stone-risk-a-wearable-based-survival-analysis-in-u-s-cohort.html) (not a target venue; useful design template) | N=9,376; 103 events | Exposure from a 180-day baseline window. Valid HR day = ≥600 min of wear. Adjusted Cox. | HR 0.92 (0.86–0.99) per +1,000 steps/d. Sedentary time not associated. |
| A11 | Patten, Preble, Master, et al. *The All of Us Research Program's wearables dataset.* Nat Med 2026. [doi:10.1038/s41591-026-04352-3](https://www.rti.org/publication/the-all-of-us-research-programs-wearables-dataset) | Data descriptor: >59,000 participants over 14 y; >39M step and 31M sleep observations | 46% of these participants also have EHR, physical measurements, genomics or surveys. The WEAR study handed out free Fitbits to broaden who contributes data. | Defines the resource and its representativeness strategy. |
| A12 | Holko M et al. *Wearable fitness tracker use in federally qualified health center patients…* npj Digit Med 2022;5:53. [doi:10.1038/s41746-022-00593-x](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9038923/) | 1,007 patients at federally qualified health centers (FQHCs) | Survey. | About 1 in 5 own a tracker; over half of non-owners want one; cost and lack of information are the barriers. This is direct evidence of selection bias in who has device data. |

Other AoU items found but not tabled:
- [JMIR 2025 Fitbit sleep preprocessing algorithm, doi:10.2196/71718](https://www.jmir.org/2025/1/e71718)
- [JMIR mHealth 2023 data-cleaning considerations](https://mhealth.jmir.org/2023/1/e45103) (AoU does not provide device model)
- [AoU rest-activity rhythm and biological aging preprint](https://www.medrxiv.org/content/10.1101/2025.09.26.25336772.full.pdf)
- [Breast-cancer case–control](https://escholarship.org/uc/item/9jd5j98q): cases 6,766 vs controls 7,248 steps/d; venue [UNV]
- MDD preprints: [arXiv 2602.16583](https://arxiv.org/pdf/2602.16583) and [arXiv 2608.06063](https://arxiv.org/pdf/2608.06063)

### B. UK Biobank accelerometry

| # | Paper (link) | Cohort, N, outcome | How the biomarker was derived, design, analysis | Validation and effect sizes |
|---|---|---|---|---|
| B1 | Doherty A et al. *Large scale population assessment of physical activity using wrist worn accelerometers: the UK Biobank Study.* PLoS One 2017. [doi:10.1371/journal.pone.0169649](https://doaj.org/article/ccd18cb32c254941b0dca48a1b08c10e) | >100,000; Axivity AX3 wrist device at 100 Hz for 7 days | **Calibration:** from stationary 10-s windows (SD <13 mg on all axes). **Non-wear:** stationary ≥60 min, filled with the same time-of-day average from other days. **Inclusion** (per tool documentation): ≥72 h of wear and data in every hour of the 24-h cycle. | The processing basis for all UKB rows below. |
| B2 | Willetts M et al. *Statistical machine learning of sleep and physical activity phenotypes from sensor data in 96,220 UK Biobank participants.* Sci Rep 2018;8:7961. [PMC5962537](https://pmc.ncbi.nlm.nih.gov/articles/PMC5962537) | 96,220 | Balanced random forest plus hidden Markov model, trained on 159,504 labeled minutes from 132 adults. | ~87% accuracy. Gives time in sleep, sedentary, light and MVPA. |
| B3 | Doherty A et al. *GWAS identifies 14 loci for device-measured physical activity and sleep duration.* Nat Commun 2018. [PMC6288145](https://pmc.ncbi.nlm.nih.gov/articles/PMC6288145) | 91,105 | GWAS of device-measured phenotypes, then two-sample Mendelian randomization (MR). | 14 loci (7 new); SNP heritability ~15%. MR: +1 SD activity → diastolic BP −0.91 mmHg; hypertension OR 0.84. |
| B4 | Strain T et al. *Wearable-device-measured physical activity and future health risk.* Nat Med 2020. [PubMed 32807930](https://pubmed.ncbi.nlm.nih.gov/32807930/) | 96,476; median 3.1 y; 732 deaths | Wrist acceleration converted to activity energy expenditure (PAEE, kJ/kg/d), modeled jointly with the share of energy from MVPA. | PAEE 15→20 kJ/kg/d (at 10% MVPA): 21% lower mortality. At a fixed 15 kJ/kg/d, 20% vs 10% MVPA: 30% lower. |
| B5 | Khurshid S et al. *Accelerometer-derived physical activity and risk of atrial fibrillation.* Eur Heart J 2021. [PMC8291334](https://pmc.ncbi.nlm.nih.gov/articles/PMC8291334) | 93,669 without AF; median 5.2 y; 2,338 AF events | Cox adjusted for age, sex and each CHARGE-AF score component. Wear and lag rules [UNV]. | ≥150 min/wk MVPA: AF HR 0.82 (0.75–0.89); stroke 0.76 (0.64–0.90). Device vs self-reported MVPA correlated only r=0.16, and self-report did not predict AF. |
| B6 | Khurshid S et al. *Wearable accelerometer-derived physical activity and incident disease.* npj Digit Med 2022. [doi:10.1038/s41746-022-00676-9](https://www.nature.com/articles/s41746-022-00676-9) | 96,244; median 6.3 y; 697 phecode diseases; self-report comparison in 456,374 | Phenome-wide Cox. FDR threshold [UNV]; the group's 2026 follow-up uses FDR 1%. | MVPA associated with 373 diseases (54%), 98% of them protective; HR 0.70–0.98 per +150 min/wk; all 16 disease categories. |
| B7 | Stamatakis E et al. *Association of wearable device-measured vigorous intermittent lifestyle physical activity with mortality.* Nat Med 2022;28:2521–29. [doi:10.1038/s41591-022-02100-x](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9800274/) | 25,241 self-reported non-exercisers; 6.9 y; 852 deaths; 62,344 exercisers for comparison | **New construct:** VILPA, brief (1–2 min) vigorous bouts during daily life. How it was classified from raw data [UNV]. | Median 3 bouts/d vs none: 38–40% lower all-cause and cancer mortality; 48–49% lower CVD mortality. |
| B8 | Schalkamp AK et al. *Wearable movement-tracking data identify Parkinson's disease years before clinical diagnosis.* Nat Med 2023. [doi:10.1038/s41591-023-02440-2](https://ukdri.ac.uk/publications/wearable-movement-tracking-data-identify-parkinsons-disease-years-clinical-diagnosis) | ~103,712; diagnosed PD, prodromal (diagnosed ≤7 y later), matched controls | ML on accelerometry compared head-to-head with genetics, lifestyle, blood biochemistry and prodromal-symptom models. | Accelerometry model won and separated PD from other movement-affecting conditions. Reported AUPRC 0.14±0.04 for diagnosed and 0.07±0.03 for prodromal. An external test in REM sleep behaviour disorder cohorts ([Kahl 2026](https://ora.ox.ac.uk/objects/uuid:abacb962-ae3d-4332-9cb7-50d35b4a3fed/files/rzp38wg009)) performed worse. |
| B9 | Windred DP et al. *Sleep regularity is a stronger predictor of mortality risk than sleep duration.* Sleep 2024. [UKB page](https://www.ukbiobank.ac.uk/publications/sleep-regularity-is-a-stronger-predictor-of-mortality-risk-than-sleep-duration-a-prospective-cohort-study/) | 60,977; >10M hours of data | Sleep Regularity Index (SRI): 100 = perfectly regular, 0 = random. Median 81.0 (IQR 73.8–86.3). | Top four SRI quintiles had 20–48% lower mortality than the bottom quintile. Regularity beat duration head-to-head. |
| B10 | del Pozo Cruz B et al. *Prospective associations of daily step counts and intensity with cancer and CVD…* JAMA Intern Med 2022. [doi:10.1001/jamainternmed.2022.4000](https://www.ukbiobank.ac.uk/publications/prospective-associations-of-daily-step-counts-and-intensity-with-cancer-and-cardiovascular-disease-incidence-and-mortality-and-all-cause-mortality/) | 78,500; median 7 y | Steps split into incidental (<40 steps/min) and purposeful (≥40 steps/min), plus peak-30 cadence. | Dose–response up to ~10,000 steps/d. Cadence added benefit beyond step volume. |
| B11 | Lyall LM et al. *Association of disrupted circadian rhythmicity with mood disorders… 91,105 participants.* Lancet Psychiatry 2018;5:507–14. [PubMed 29776774](https://pubmed.ncbi.nlm.nih.gov/29776774/) | 91,105; cross-sectional | Relative amplitude (most active 10 h vs least active 5 h). Adjusted for age, season, sex, ethnicity, deprivation, smoking, alcohol, education, mean acceleration, BMI, childhood trauma. | Lower relative amplitude: lifetime MDD OR 1.06, bipolar disorder OR 1.11; also linked to mood instability and neuroticism. A later GWAS found loci at NFASC and SLC25A17. |
| B12 | Khurshid S et al. *Accelerometer-derived "weekend warrior" physical activity and incident cardiovascular disease.* JAMA 2023;330:247–52. [UKB page](https://www.ukbiobank.ac.uk/publications/accelerometer-derived-weekend-warrior-physical-activity-and-incident-cardiovascular-disease/) | 89,573; 6.3 y | Pattern feature: ≥150 min/wk MVPA concentrated in 1–2 days vs spread out vs inactive. | Both active patterns similar to each other vs inactive: AF ~0.80, MI 0.69, HF 0.63, stroke 0.81. |
| B13 | Kany S, Al-Alusi MA, et al. *Associations of "weekend warrior" physical activity with incident disease and cardiometabolic health.* Circulation 2024. [UKB page](https://www.ukbiobank.ac.uk/publications/associations-of-weekend-warrior-physical-activity-with-incident-disease-and-cardiometabolic-health/) (DOI not seen) | ~89,500; 678 conditions | Phenome-wide comparison of activity patterns. | Weekend warrior: 264 of 267 associations protective (hypertension 0.77, diabetes 0.57, obesity 0.55, sleep apnea 0.57). Regular pattern: 205 of 209. No difference between the two patterns. |
| B14 | Ajufo E, …, Khurshid S. *Accelerometer-measured sedentary behavior and risk of future cardiovascular disease.* JACC 2024. [Summary](https://mdedge.com/cardiology/article/271636/acute-coronary-syndromes/sitting-more-10-hours-daily-ups-heart-disease-risk) | 89,530 | ML classifier with 4 classes (sleep, sedentary, light, MVPA). | >10.6 h/d sedentary: HF and CV-death risk persisted after adjusting for MVPA; MI and AF associations weakened. Follow-up in [JACC Adv 2026](https://www.jacc.org/doi/10.1016/j.jacadv.2026.102718): >700 conditions, FDR 1%. |
| B15 | Feng H et al. *Associations of timing of physical activity with all-cause and cause-specific mortality…* Nat Commun 2023. [doi:10.1038/s41467-023-36546-5](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9938683/) | 92,139; >7 y | Groups by MVPA timing: morning (>50% between 05:00 and 11:00), midday-afternoon, evening, mixed. | Midday-afternoon and mixed groups had lower mortality than morning. Effects were stronger in older, male, less active people and those with prior CVD. |
| B16 | Zhang et al. *Health risks and genetic architecture of objectively measured multidimensional sleep health.* Nat Commun 2025. [doi:10.1038/s41467-025-62338-0](https://link.springer.com/article/10.1038/s41467-025-62338-0) | 85,233; 7.9 y | **New composite:** "Unfavorable Sleep Profile" across timing, efficiency, duration, rhythmicity and regularity. Phenome-wide scan plus GWAS. | Linked to 76 of 526 incident outcomes. Genetic hits: MEIS1 (common variants), TTC1 (rare coding). Low SES upstream. Externally validated in MESA. |
| B17 | Shim J, Fleisch E, Barata F. *CosinorAge* (exact title [UNV]). npj Digit Med 2024. [DOAJ](https://doaj.org/article/375dac44d8904566a5b8fdcc84aad084) | ~80,000 adults in the UK and US | Biological age built from circadian rhythm (cosinor) parameters. | +1 y of CosinorAge: 8–12% higher mortality, 3–14% higher age-related disease incidence. Held in an independent cohort using a different device. Added information beyond activity alone. |
| B18 | Choi KW et al. *Assessment of bidirectional relationships between physical activity and depression among adults: a 2-sample Mendelian randomization study.* JAMA Psychiatry 2019;76:399–408. [UCL](https://discovery.ucl.ac.uk/id/eprint/10073831/) | Genetic instruments: accelerometer activity (91,084), self-reported activity (377,234), MDD (143,265) | Bidirectional two-sample MR. | Accelerometer activity → MDD OR 0.74 per SD (0.59–0.92). No reverse effect. Self-reported activity showed nothing. |

### C. Consumer wearables: disease detection and monitoring

| # | Paper (link) | Cohort, N | Biomarker and design | Validation and effect sizes |
|---|---|---|---|---|
| C1 | Perez MV et al. *Large-scale assessment of a smartwatch to identify atrial fibrillation.* NEJM 2019;381:1909–17. [doi:10.1056/NEJMoa1901183](https://escholarship.org/uc/item/5n08v244) | 419,297; median 117 days | Apple Watch pulse-irregularity algorithm (PPG tachograms), then telemedicine visit and ECG patch. | 0.52% notified. AF on 34% of 450 analyzable patches. PPV 0.84 for a notification during ECG wear. |
| C2 | Lubitz SA et al. *Detection of atrial fibrillation in a large population using wearable devices: the Fitbit Heart Study.* Circulation 2022. [doi:10.1161/CIRCULATIONAHA.122.060291](https://www.acc.org/Latest-in-Cardiology/Journal-Scans/2022/10/03/17/27/Detection-of-AF-in-a-Large) | 455,699 | Alert = 11 consecutive irregular 5-min pulse windows. | 1% flagged (4% of those ≥65). AF in 32.2% of 1,057 patched. PPV 98.2% (95.5–99.5) for a repeat alert during ECG monitoring. |
| C3 | Guo Y et al. *Mobile photoplethysmographic technology to detect atrial fibrillation.* JACC 2019;74:2365–75. [doi:10.1016/j.jacc.2019.08.019](https://livrepository.liverpool.ac.uk/3066654/) | 187,912 monitored ≥14 days (Huawei Heart Study) | PPG screening; flagged users confirmed clinically, by ECG or Holter. | 0.23% flagged; strong age gradient. PPV [UNV]. |
| C4 | Tison GH et al. *Passive detection of atrial fibrillation using a commercially available smartwatch.* JAMA Cardiol 2018. [doi:10.1001/jamacardio.2018.0136](https://pmc.ncbi.nlm.nih.gov/articles/5875390) | Health eHeart (~139M HR measurements) | Deep neural network on HR and step data. | Cardioversion cohort (n=51): C-statistic 0.97. Ambulatory self-reported AF: 0.72, showing the drop from lab to free-living conditions. |
| C5 | Mishra T et al. *Pre-symptomatic detection of COVID-19 from smartwatch data.* Nat Biomed Eng 2020. [doi:10.1038/s41551-020-00640-6](https://communities.springernature.com/posts/early-detection-of-covid-19-at-scale-using-wearables) | ~5,300 participants; 32 COVID cases | Deviations in resting HR (RHR), steps and sleep from each person's own baseline; two-tier real-time alert. Algorithm names [UNV]. | 81% of cases showed changes; 22 of 25 detected at or before symptom onset; 63% detectable pre-symptomatically in real time. |
| C6 | Quer G et al. *Wearable sensor data and self-reported symptoms for COVID-19 detection.* Nat Med 2021 (DOI not seen). [Summary](https://federallabs.org/communications/federal-lab-news/fitness-app-biometrics-help-detect-covid-19-in-ncats-funded-study) | DETECT study, 30,529; 54 positive vs 279 negative symptomatic | Within-person change in RHR, sleep and steps. | Symptoms plus sensors: AUC 0.80 (IQR 0.73–0.86) vs 0.71 for symptoms alone. RHR alone did not discriminate. |
| C7 | Alavi A et al. *Real-time alerting system for COVID-19 and other stress events using wearable data.* Nat Med 2022;28:175–84. [doi:10.1038/s41591-021-01593-2](https://www.medrxiv.org/content/10.1101/2021.06.13.21258795v1) | 3,246 (preprint count) | Prospective real-time alerts based on RHR. | Flagged 78% of pre-symptomatic or asymptomatic infections, median 3 days before symptoms. Stress, alcohol and travel also triggered alerts, a specificity limit. |
| C8 | Li X, Dunn J, et al. *Digital health: tracking physiomes and activity using wearable biosensors reveals useful health-related information.* PLoS Biol 2017. [doi:10.1371/journal.pbio.2001402](https://pmc.ncbi.nlm.nih.gov/articles/PMC5230763/) | 60 people; ~2B measurements (Stanford iPOP) | HR normalized for activity. Insulin-resistance classifier built from steps, daytime HR and the day–night HR gap. | Lyme disease case. Insulin-resistance signal in 20 people with glucose tests (12 insulin-resistant). Proof of concept only. |
| C9 | Radin JM et al. *Harnessing wearable device data to improve state-level real-time surveillance of influenza-like illness in the USA.* Lancet Digit Health 2020. [DOAJ](https://doaj.org/article/0b19ae2d635a4893af58fc85285cdef4) | 47,249 users in 5 states; 13.3M days | Population-level signal: share of users with raised RHR and longer sleep. | Improved state-level flu-like-illness estimates by 6–33% (press figure). |
| C10 | Avram R et al. *A digital biomarker of diabetes from smartphone-based vascular signals.* Nat Med 2020 (DOI not seen). [UCSF](https://www.ucsf.edu/news/2020/08/418256/smartphones-may-help-detect-diabetes) | 53,870; ~3M smartphone PPG recordings | 39-layer convolutional network. | AUC 0.766 primary; 0.740 in a separate contemporary cohort (7,806); 0.830 with age, sex, race and BMI added. Score tracks HbA1c. |
| C11 | Spathis D et al. *Longitudinal cardio-respiratory fitness prediction through wearables in free-living environments.* npj Digit Med 2022. [doi:10.1038/s41746-022-00719-1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9718831/) | Fenland 11,059; longitudinal 2,675; external 181 with measured VO2max | Neural network from wearable signals plus biomarkers. | r=0.82 in holdout; tracked 7-year change. Overestimated fitness in a less-fit external cohort (distribution shift). |
| C12 | Stehlik J et al. *Continuous wearable monitoring analytics predict heart failure hospitalization: the LINK-HF multicenter study.* Circ Heart Fail 2020. [Record](https://mdanderson.elsevierpure.com/en/publications/continuous-wearable-monitoring-analytics-predict-heart-failure-ho) | 100 veterans (98% male) | Chest patch; ML learns a personal baseline and alerts on deviations. | Sensitivity 76–88%, specificity 85%; median 6.5 days of warning. |
| C13 | Miller AC et al. *A wearable-based aging clock associates with disease and behavior.* Nat Commun 2025;16:9264. [Record](https://collaborate.princeton.edu/en/publications/a-wearable-based-aging-clock-associates-with-disease-and-behavior/) | Apple Heart & Movement Study, ~214,000; 149M person-days | Deep-learned age from wrist PPG ("PpgAge"). | Mean absolute error 2.43 y (healthy cohort). Age gap >6 y: 1.5–5× rates of heart disease, HF and diabetes; HR 1.40 for cardiac events. Rises during pregnancy. |
| C14 | Natarajan A et al. *Heart rate variability with photoplethysmography in 8 million individuals: a cross-sectional study* (journal [UNV]). [bioRxiv](https://www.biorxiv.org/node/913419) | >8M Fitbit users | Beat-to-beat intervals over 24 h; RMSSD, SDRR, LF power. | Normative curves by age, sex and activity. |

### D. Methodology papers that define wearable phenotypes

| # | Paper (link) | What it defines | Key numbers |
|---|---|---|---|
| D1 | Phillips AJK et al. *Irregular sleep/wake patterns are associated with poorer academic performance and delayed circadian and sleep/wake timing.* Sci Rep 2017. [doi:10.1038/s41598-017-03171-4](https://dspace.mit.edu/handle/1721.1/113615) | Introduces the SRI; its formula wasn't shown in the snippets I saw [UNV] | 61 students, 30 days. Irregular sleepers had later melatonin onset (DLMO) and lower light amplitude; SRI correlated with grades (r=0.37); sleep duration did not differ. |
| D2 | Van Someren EJW et al. Chronobiol Int 1999;16(4). [doi:10.3109/07420529908998724](https://rdrr.io/cran/nparACT/) | Interdaily stability (IS) and intradaily variability (IV); relative amplitude in the same software packages | IS runs 0–1 (synchronization to 24 h); IV is ~0 for a sine wave and ~2 for noise (fragmentation). Implemented in nparACT and GGIR. |
| D3 | Saint-Maurice PF et al. *Association of daily step count and step intensity with mortality among US adults.* JAMA 2020;323:1151–60. [doi:10.1001/jama.2020.1382](https://www.nih.gov/news-events/news-releases/higher-daily-step-count-linked-lower-all-cause-mortality) | Peak-30 and peak-1 cadence, bout cadence (NHANES, N=4,840) | 12,000 vs 4,000 steps: HR 0.35 (0.28–0.45). Peak-30 cadence null once total steps were adjusted (HR 0.90, 0.65–1.27). |
| D4 | Huang T, Mariani S, Redline S. *Sleep irregularity and risk of cardiovascular events: MESA.* JACC 2020;75:991–99. [doi:10.1016/j.jacc.2019.12.054](https://pubmed.ncbi.nlm.nih.gov/32138974/) | SD of sleep duration and sleep-onset time over 7 days | 1,992 people without CVD; most irregular sleepers had about 2× the risk of CVD events. |
| D5 | Cole CR et al. *Heart-rate recovery immediately after exercise as a predictor of mortality.* NEJM 1999. [PubMed 10536127](https://pubmed.ncbi.nlm.nih.gov/10536127/) | Heart-rate recovery ≤12 bpm at 1 min counts as abnormal | N=2,428; 6 y; unadjusted RR of death 4.0 (3.0–5.2). |
| D6 | Ben-Dov IZ et al. *Blunted heart rate dip during sleep and all-cause mortality.* Arch Intern Med 2007;167:2116–21. [Via citing letter](https://www.revespcardiol.org/en-consideration-of-nondipping-heart-rate-d-articulo-S1885585721003510) | Nocturnal HR dip | N=3,957; 7 y; 34% higher mortality per 10% smaller night-time HR drop. |
| D7 | Kohn et al. (Segal lab). *Phenome-wide associations of sleep characteristics in the Human Phenotype Project.* Nat Med 2025. [doi:10.1038/s41591-024-03481-x](https://pubmed.ncbi.nlm.nih.gov/39870817/) | 448 sleep features from home sleep apnea tests | N=6,366; sleep features predicted >15% of body characteristics across 15 of 16 body systems in held-out people. |

---

## Part 2. Synthesis

### (a) The common recipe, from hypothesis to validation

1. **Name the gap and the metric to beat.** Strong papers say what the current measure misses:
   - Self-report misses objective activity (B5: r=0.16, and only the device measure predicted AF; B18).
   - One-week snapshots are unstable (A6).
   - Sleep duration ignores regularity (B9, D4).
   - Volume ignores intensity, pattern and timing (B4, D3, B10, B12, B15).
   - Average guidance ignores genetic risk (A3).
2. **State the expected direction and shape in advance.** Master's stated hypothesis was that more steps over time lower incidence. They then tested for linear trends vs plateaus.
3. **Build the cohort with explicit data-sufficiency rules.** AoU defaults that recur across papers:
   - ≥6 months of data.
   - Valid day: ≥10 h of wear (≥600 HR-minutes) and 100–45,000 steps.
   - Valid month: ≥15 valid days.
   - Sensitivity check at ≥16 h of wear.
   - Sleep: main sleep period only; drop implausible nights (<3 h or >16 h); exclude people with >30% of nights under 4 h.
   - UKB adds calibration, non-wear imputation by time of day, and the ≥72 h / all-24-hours rule.
4. **Build features in tiers.** Start with step volume, then intensity (cadence, MVPA, VILPA), pattern and timing, sedentary time, sleep stages and regularity (SD, SRI), circadian rhythm (IS, IV, relative amplitude, cosinor), the coupling of heart and behaviour (HR per step, RHR deviation from baseline, night-time HR dip), and finally composites or learned features (USP, CosinorAge, PpgAge). Write every formula down, and choose the averaging window on purpose.
5. **Fix time zero and guard against reverse causation.** Set the index date, exclude prevalent disease, and count only diagnoses at least 6 months after monitoring starts (A1, A5), or measure exposure in a fixed baseline window (A10).
6. **Primary model.** Cox, with exposure that varies over time when monitoring spans years (A1). Adjust for a standard set: age, sex, race, education, smoking, alcohol, BMI, SBP, CAD, cancer and wear time. Show dose–response curves with interpretable units: per 1,000 or 2,000 steps, per 150 min/wk, per SD, per percentage point of a sleep stage.
7. **Scan broadly, then go deep.** Run a phenome-wide scan with FDR control (1,711, 697, 678 or 526 outcomes in A1, B6, B13, B16), then examine the top hits in detail.
8. **Show what the new marker adds.** Compare it head-to-head with the incumbent and with other data types (A4, B8, B9, B17, D3). For prediction, report AUROC or AUPRC, the C-index and calibration, on held-out and external data.
9. **Stress-test.** Compare limited vs full adjustment, wear-time adjustment, different baseline windows, time-varying vs averaged exposure, and stricter valid-day rules. Test subgroups and interactions: sex, age, BMI, PRS, prior CVD.
10. **Triangulate.** Check against ground truth (ECG patch, VO2max, melatonin timing, HbA1c, treadmill), use genetics (GWAS then MR: B3, B18), and replicate in another cohort or device (B16, B17, C10). Report when replication is weaker (Kahl 2026).
11. **Translate.** Give thresholds (8,200 steps; 10.6 h sedentary) and counterfactual offsets ("+2,280 steps/d"; "+1,700–5,500 steps/d").

### (b) What sets apart studies that found genuinely new biomarkers

- **A new construct built from raw data with a physiological story:** VILPA, DHRPS (HR cost of movement as a fitness proxy), PpgAge, CosinorAge, the USP, relative amplitude, SRI, the prodromal Parkinson's motor signature, the smartphone-PPG diabetes score, and RHR deviation from a personal baseline.
- **They beat the incumbent head-to-head**, not just "reach significance": SRI beat duration; DHRPS beat HR or steps alone; accelerometry beat genetics and blood tests for PD; CosinorAge added to activity; the PPG score added to demographics (AUC 0.83).
- **Breadth with control for multiple testing, plus a specificity check:** PD was separated from other movement disorders.
- **Ground-truth or external anchors:** MESA, a different-device cohort, a contemporary cohort, VO2max, ECG patches.
- **Genetic or causal triangulation:** the MEIS1/TTC1 and NFASC/SLC25A17 loci; MR for blood pressure and depression.
- **Within-person dynamics:** PpgAge rising in pregnancy; COVID signals measured against each person's own baseline.
- **Honest limits:** cadence was null once volume was adjusted (D3); accuracy fell from lab to daily life (C4); alerts lacked specificity (C7); the fitness model failed under distribution shift (C11); PD replication was weaker.

### (c) Checklist for a convincing paper

- **Tables**
  - Table 1 by exposure quartile, including wear time, monitoring length, and BYOD vs WEAR recruitment.
  - A feature dictionary: formulas, quality-control rules, averaging windows.
  - The main HR table: per unit and by quartile, minimal and full adjustment.
  - A sensitivity-analysis table.
  - A phenome-wide supplement: number of cases, HR, CI, p and q, with **AoU cells of 1–19 suppressed** (see the A1 correction).
- **Figures**
  - Participant flow diagram.
  - Feature distribution and reliability across windows (A6).
  - Spline dose–response curve with the exposure histogram underneath.
  - Phenome-wide Manhattan or volcano plot with the FDR line.
  - Cumulative-incidence curves.
  - Subgroup and interaction forest plot.
  - Incremental-value panel: ROC/PR curves, calibration, ΔC.
  - Feature trajectory in the years before diagnosis (B8).
- **Analyses**
  - Lagged, time-varying Cox; splines; FDR; interactions.
  - Head-to-head comparison with the incumbent.
  - Volume-adjusted intensity and pattern features.
  - External or temporal validation; MR where GWAS instruments exist.
  - Not seen in the summaries I could access, so add them as upgrades: competing risk of death, negative-control outcomes, E-values, population attributable fraction.
- **Metrics**
  - HR per SD and per natural unit; absolute risk; thresholds; steps needed to offset a risk.
  - AUROC/AUPRC, C-index, ΔC and NRI, calibration slope.
  - PPV and sensitivity for detection; MAE for aging clocks.

### (d) Pitfalls and how the studies guard against them

| Pitfall | What top studies did | Suggested CoDaS rule |
|---|---|---|
| Wear-time and adherence bias | Valid-day and valid-month rules; wear time as a time-varying covariate (A1); ≥16 h sensitivity check (A6); time-of-day imputation (B1) | Compute wear from HR-minute coverage; always adjust for it and rerun at a stricter threshold |
| Reverse causation | Exclude prevalent disease; 6-month lag; baseline window; time-varying models; MR showing no reverse effect (B18) | Default 6-month lag, with 12–24-month sensitivity checks; plot pre-diagnosis trajectories |
| Healthy-user and selection bias | BYOD skews female, white and college-educated (A1, A2); FQHC cost barriers (A12); free devices via WEAR (A11) | Adjust for SES; compare BYOD vs WEAR; state generalizability limits |
| EHR ascertainment bias | Phecode outcomes; EHR-linked subsets | Require EHR activity before and after the index date; test ≥2 codes; adjust for visit density [my addition]; be wary of outcomes the device itself detects (AF alerts) |
| Device or algorithm change | AoU doesn't record device model (JMIR 2023); Fitbit sleep staging uses HR (A2); validation across devices (B17) | Standardize within person; check for change-points; favour robust features (duration, efficiency) over sleep stages |
| Missing data | Drop months with little data; longer averaging windows (A6) | Report missingness by covariates; sensitivity with weighting or multiple imputation |
| Intensity confounded by volume | Cadence null after adjusting steps (D3); joint modelling (B4) | Always adjust intensity and pattern features for volume |
| Multiple testing | FDR 1% | Pre-define test families; report q-values |
| Overfitting and distribution shift | External cohorts (C10, C11) | Temporal and site-based splits within AoU |
| Low event counts | 55 AF events (A8); 103 kidney stones (A10) | Minimum-event rule; report CIs |
| Low specificity | Alerts triggered by stress and alcohol (C7) | Negative-control outcomes and disease-specificity tests |

### (e) How the strongest papers frame their hypotheses

They are:
- **Mechanism-first:** regularity as a marker of circadian disruption; HR per step as fitness; prodromal slowing of movement.
- **Comparative:** "X beyond, or instead of, Y."
- **Quantitative:** expected direction, shape and threshold.
- **Time-ordered:** exposure measured before the outcome.
- **Actionable:** a threshold, or the steps needed to offset a risk, overall and by genetic risk.
- **Falsifiable and specific:** they name which outcomes should *not* move.

A template CoDaS can fill in:
> "In adults without [D] at the index date, [feature with a physiological rationale] measured over [window] predicts [direction] incidence of [D], independent of [incumbent metric] and [confounders], with [expected shape], robust to a [lag], and not associated with [negative-control outcome]. If causal, [genetic instrument] should give a concordant effect."

### Suggested CoDaS defaults for AoU (my synthesis, not verified evidence)

Features computable from AoU's minute-level steps, minute-level HR and sleep segments:
- HR per step, daily and at minute level.
- RHR deviation from each person's baseline.
- Night-time HR dip (sleep vs wake HR).
- HR response to step bouts, as a free-living analogue of heart-rate recovery. This would be new and is unvalidated.
- Peak-1 and peak-30 cadence.
- Short, high-cadence bouts approximating VILPA. VILPA itself was defined on raw accelerometry, so this is an adaptation.
- Weekend-warrior pattern and MVPA timing.
- Sedentary minutes.
- IS, IV, relative amplitude and cosinor parameters from minute data.
- SRI from sleep segments; variability in sleep midpoint and duration.

Two constraints:
- **No HRV:** AoU's Fitbit tables appear to hold minute-level HR rather than beat-to-beat intervals, so HRV features like C14's probably can't be reproduced [UNV; check the current data release].
- **Default rules:** use the data-sufficiency, lag, model, FDR and small-count-suppression rules in steps 3–7 above.

---

**Additional sources consulted:**
- [AoU Fitbit considerations (Researcher Workbench support)](https://support.researchallofus.org/hc/en-us/articles/9651723386388)
- [ACC DHRPS press release](https://www.acc.org/about-acc/press-releases/2025/03/20/09/58/a-smarter-way-to-track-heart-health-with-your-smartwatch)
- [VUMC: steps offset sedentary risk](https://news.vumc.org/2026/04/08/adding-steps-offsets-risk-of-chronic-disease-study/)
- [Google blog on the AoU sleep study](https://blog.google/innovation-and-ai/technology/health/fitbit-study-sleep/)
- [UKB physical activity monitor documentation](https://biobank.ctsu.ox.ac.uk/ukb/ukb/docs/PhysicalActivityMonitor.pdf)
- [biobankAccAnalysis methods](https://biobankaccanalysis.readthedocs.io/en/latest/methods.html)
- [Epistemonikos record for A7](https://www.epistemonikos.org/en/documents/ede34f0b12c4744aad7cdeadfd493dd3138fcd1e)
- [Strain 2020, Cambridge repository](https://www.repository.cam.ac.uk/handle/1810/308030)
- [HealthDay: weekend warrior and 200 diseases](https://www.healthday.com/health-news/exercise-and-fitness/being-a-weekend-warrior-can-cut-your-odds-for-200-diseases)
- [NHLBI news on MESA sleep irregularity](https://www.nhlbi.nih.gov/news/2020/study-finds-irregular-sleep-patterns-double-risk-cardiovascular-disease-older-adults)
