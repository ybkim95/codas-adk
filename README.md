# CoDaS — AI Co-Data-Scientist

Give CoDaS a participant-level table and a research goal in plain language. A team of
[google-adk](https://google.github.io/adk-docs/) + Gemini agents profiles the data, grounds its
hypotheses in the literature, runs an iterative discovery loop, argues each candidate adversarially,
and writes a report you can audit. The agents plan, interpret, debate, and decide when to stop; a
deterministic engine computes every number. The model never invents a statistic.

Implements the method described in arXiv:[2604.14615](https://arxiv.org/abs/2604.14615).

**What this repository is.** The discovery engine (`codas_core`) and the agent graph (`codas_agents`).
It runs on *participant-level analysis tables* — one row per participant, with engineered features and
the clinical endpoint. Building those tables from raw sensor streams (for example, deriving nightly
sleep features from ~4.5M hourly rows) is a separate upstream step and is not part of this repository.
The clinical cohorts are governed data and are not redistributed here; checksums are below so you can
confirm a table you hold matches the analysis set.

## Reproducing the results

Install into a fresh virtualenv from the pinned lock. Do not install into a shared base environment —
a broken base `numpy`/`pandas`/`scikit-learn` is the usual reason imports fail.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --only-binary=:all: -r requirements-lock.txt   # exact tested versions, wheels only
pip install .
python -m pytest -q                                        # 102 tests
```

Run the engine on the analysis tables and it returns the headline effects to three decimals:

```bash
# edit scripts/paper_cohorts.example.json so each `csv` points at a cohort table, then:
python scripts/reproduce_paper_biomarkers.py --config scripts/paper_cohorts.example.json
```

| Effect | Engine |
|---|---|
| DWB — main sleep-duration variability vs PHQ-8 | ρ = +0.252 |
| WEAR-ME — C-reactive protein vs HOMA-IR | ρ = +0.393 |
| WEAR-ME — HDL cholesterol vs HOMA-IR | ρ = −0.380 |

The cardiovascular-fitness index (steps ÷ resting heart rate, ρ = −0.374) is one the interpreter
*constructs*: when it proposes that ratio, the engine screens and validates it like any other feature
(the discovery loop below). Effect sizes reproduce exactly on the tables; the count of validated
candidates depends on the declared exclusions and thresholds, so the script reports it as computed,
never as a fixed number.

## Where each part of the method lives

| Capability | Code |
|---|---|
| Agent roles, six-phase orchestrator | `codas_agents/agent.py`, `codas_agents/prompts.py` |
| Interpreter proposes a transform, engine evaluates it | `codas_agents/tools.py` (`propose_feature`), `codas_core/discovery.py` |
| Leakage guardrails (variable exclusion, construct overlap) | `codas_core/data.py`, `codas_core/validation.py` |
| Validation battery — eleven checks across four dimensions | `codas_core/validation.py` (`CANONICAL_BATTERY`) |
| Quality gates and numeric verification | `codas_core/quality_gates.py`, `codas_agents/numeric_audit.py` |
| Literature grounding of hypotheses and mechanisms | `codas_agents/tools.py` (`search_literature`) |
| Spearman, Benjamini–Hochberg FDR (α = 0.05), bootstrap, permutation | `codas_core/statistics.py` |

Every number in a report traces back to the engine. `tests/test_validation_golden.py` pins the whole
pipeline output by hash, so a refactor cannot quietly change a verdict.

## How it works

The Orchestrator runs six phases over one shared memory and one deterministic tool set. Its core is a
discovery loop that deepens the search each round until a GapChecker judges another round won't pay off.

```
   a CSV  +  a research goal in plain language
        │
        ▼
   ORCHESTRATOR  ·  shared memory  ·  deterministic tools
   │
   ├─ Phase A      profile data → choose target & roles → frame hypotheses (literature-grounded)
   │
   ├─ Phase B&C    DISCOVERY LOOP  ⟲  deepens each round
   │                 1. search      run a deeper discovery round
   │                 2. interpret   statistical ∥ ML  (parallel; may propose new features)
   │                 3. validate    critic ⇄ defender  (adversarial)
   │                 4. gapcheck    converged ? leave : iterate
   │
   └─ Phase D/E/F   mechanism → novelty → strategy → grounded report
        │
        ▼
   an auditable report   ⟲   optional human feedback for another pass
```

Each phase is a `SequentialAgent`, the loop is a `LoopAgent`, and the two interpreters run in a
`ParallelAgent`. Agents share one `session.state` and write results back through `output_key`, so a
whole run is a single object you can inspect. The engine (`codas_core`) is pure numpy/pandas/scipy/
scikit-learn — no LLM, no network — and applies no name-based rules: you name the target, it infers
nothing from column names.

## Running your own data

```bash
export GOOGLE_API_KEY=...
python examples/run_agent.py                                   # bundled sample
python examples/run_agent.py path/to/data.csv "which features predict <target>?"

uvicorn codas_service.app:app --port 8000                      # or as a service
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/discover` | Deterministic discovery for an explicit `target_column` (no key needed) |
| POST | `/v1/agent` | The agent pipeline chooses the target and roles; returns a `session_id` |
| POST | `/v1/agent/feedback` | Resume that session with feedback for another pass |

Auth is via `X-CoDaS-Agent-Key` (`CODAS_AGENT_API_KEYS`, comma-separated); see `.env.example`. Beyond
the test suite, `scripts/robustness_audit.py` and `scripts/scientific_validation.py` are offline scored
audits (no key, no data), and CI runs both alongside the tests.

## Data

The analysis tables are not shipped. To confirm a table you hold matches the one used here:

| Cohort | Table | N | SHA-256 |
|---|---|---:|---|
| DWB (PHQ-8) | `aggregated_data.csv` | 7,497 | `5cc72c19efdaa44d57ce7f1874f214b8b38e997536d90ccd93f6ac17dab2eb2f` |
| WEAR-ME (HOMA-IR) | `wearme_clean.csv` | 1,078 | `4cc4644bc167c0236a150fcf45cf4f36e9f447c34495195c4961ca539ceeca69` |

Verify with `sha256sum <table.csv>` (`shasum -a 256` on macOS).

## Deployment

`deploy.sh` deploys the service to Google Cloud Run, with Vertex AI for Gemini (no external key) and
IAM-gated access. Configure the block at the top of the script — `PROJECT_ID`, `REGION`,
`ALLOW_UNAUTHENTICATED` — and run `./deploy.sh`; it prints the service URL and a generated
`x-codas-agent-key`.
