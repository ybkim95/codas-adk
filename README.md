# CoDaS — AI Co-Data-Scientist

Give CoDaS a tabular CSV and a research goal in plain language. A team of
[google-adk](https://google.github.io/adk-docs/) + Gemini agents profiles the data, frames
hypotheses, runs an iterative deepening search, validates it adversarially, and returns a grounded,
auditable set of candidate predictors of your target.

The agents plan, interpret, debate, and decide when to stop. Every number in the report is computed
by the deterministic engine, not the model. The engine assumes no schema, feature names, or problem
domain.

## How it works

The **Orchestrator** coordinates six phases over one shared memory and one deterministic tool set.
Its heart is an **iterative discovery loop** that deepens the search each round until a *GapChecker*
judges that going further no longer pays off.

```
   a CSV  +  a research goal in plain language
        │
        ▼
   ORCHESTRATOR  ·  shared memory  ·  deterministic tools
   │
   ├─ Phase A      profile data → choose target & roles → frame hypotheses
   │
   ├─ Phase B&C    DISCOVERY LOOP  ⟲  repeats, deepening the search each round
   │                 1. search      run a deeper discovery round
   │                 2. interpret   statistical ∥ ML        (in parallel)
   │                 3. validate    critic ⇄ defender       (adversarial)
   │                 4. gapcheck    converged ? → leave loop : iterate again
   │
   └─ Phase D/E/F   mechanism → novelty → strategy → grounded report
        │
        ▼
   an auditable report   ⟲   optional human feedback for another iteration
```

Built on **google-adk**: each phase is a `SequentialAgent`, the discovery loop is a `LoopAgent`
(the GapChecker ends it via `escalate`), and the two interpreters run inside a `ParallelAgent`. All
agents share one `session.state`; each writes its result back via `output_key`, so an entire run is
a single auditable object.

- **Grounded** — the deterministic engine (Spearman + FDR screening, a validation battery, and
  statistical leakage guards) is the only source of numbers; the LLM never invents a statistic.
- **Domain-agnostic** — no hardcoded features, columns, or domain rules; you name the target.
- **Iterative** — the search deepens until returns diminish, and a reviewer can request another pass.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[all]"
export GOOGLE_API_KEY=...

# run the pipeline on the bundled sample, or on your own CSV + question
python examples/run_agent.py
python examples/run_agent.py path/to/data.csv "which features predict <target>?"

# or run it as a service
uvicorn codas_service.app:app --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/discover` | Deterministic discovery for an explicit `target_column` |
| POST | `/v1/agent` | The agent pipeline picks the target/roles; returns a `session_id` |
| POST | `/v1/agent/feedback` | Resume that session with feedback for another iteration |

API-key auth via `X-CoDaS-Agent-Key` (`CODAS_AGENT_API_KEYS`, comma-separated); see `.env.example`.

## Layout & tests

- **`codas_core/`** — the deterministic engine (numpy/pandas/scipy/scikit-learn only; no LLM, no network).
- **`codas_agents/`** — the google-adk graph (`agent.py`), deterministic tools (`tools.py`),
  prompts (`prompts.py`), session/memory (`runtime.py`), guardrails + logging (`callbacks.py`).
- **`codas_service/`** — the FastAPI surface exposing both layers.

```bash
pip install ".[all,dev]" && python -m pytest -q     # 72 tests: engine, six-phase graph, loop tools, robustness
python scripts/robustness_audit.py                   # scored audit: no-crash, determinism, stats, service, scale
python scripts/scientific_validation.py              # longitudinal/wearable science: pseudo-replication, autocorrelation,
                                                     #   confounding, within-vs-between, leakage, effect size, imbalance
python scripts/agent_robustness.py                   # live audit: orchestration, grounding integrity, prompt-injection
python scripts/loadtest.py                            # load + soak: throughput, latency percentiles, memory stability
python scripts/reproduce_paper_biomarkers.py A.csv B.csv   # runs the paper's discovery method on its cohorts
```

## Cloud Deployment (Google Cloud Run)

CoDaS is ready to be deployed as a private, secure service on Google Cloud Run. 

### 1. Configuration & Deployment
You can deploy the service to any Google Cloud project using the provided `deploy.sh` script.

Configure the deployment by editing the **Configuration Block** in `deploy.sh`, or by exporting the same names as environment variables:
* `PROJECT_ID`: The target GCP project. Leave empty to use the active `gcloud` project.
* `SERVICE_NAME`: The name of the Cloud Run service.
* `REGION`: The GCP region to deploy to (e.g., `us-central1`).
* `GOOGLE_GENAI_USE_VERTEXAI`: Set to `"TRUE"` to automatically create and configure a dedicated service account with Vertex AI permissions for Gemini (recommended). No external Gemini API key is then required.
* `ALLOW_UNAUTHENTICATED`: Set to `"no"` to block public access and restrict requests to authorized IAM identities.

To deploy:
```bash
chmod +x deploy.sh
./deploy.sh
```

### 2. Verifying the Deployment
Once deployed, the script will output the service URL and a generated service key. This key is the CoDaS service access key (`x-codas-agent-key`), which is separate from any Gemini credentials. You can verify that the service is running using `curl`.

#### Health Check
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     -H "x-codas-agent-key: <GENERATED_API_KEY>" \
     https://<YOUR-SERVICE-URL>.run.app/v1/health
```

#### Stateless Discovery (Deterministic Engine)
```bash
curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     -H "x-codas-agent-key: <GENERATED_API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{"csv": "participant_id,age,depression_score\n1,25,10\n2,34,12", "target_column": "depression_score"}' \
     https://<YOUR-SERVICE-URL>.run.app/v1/discover
```

## About

CoDaS implements the architecture described in the CoDaS paper
(arXiv:[2604.14615](https://arxiv.org/pdf/2604.14615)). A companion project packages the
deterministic engine as a reusable agent skill:
[codas-science-skills](https://github.com/ybkim95/codas-science-skills).
