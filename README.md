# CoDaS — AI Co-Data-Scientist for Association Discovery

CoDaS turns any tabular CSV plus an explicit target column into a rigorously
validated, auditable set of candidate predictors of that target. Its defining
design choice is the **separation of LLM reasoning from deterministic
statistics**: a Gemini-driven [google-adk](https://google.github.io/adk-docs/)
agent plans, profiles, and explains, but every reportable number is produced by
deterministic Python runners that the model cannot bypass or invent.

The engine is **domain-agnostic**. It makes no assumption about column names or
problem domain: it never infers a target, participant, time, or confounder role
from a column's name, and it contains no hardcoded feature lists or
dataset-specific rules. You name the target; the engine does the statistics.

## Why it is more than a wrapper

CoDaS encodes the methodology a careful data scientist would apply, as code:

- **Deterministic and reproducible** — given the same input and seed, the output
  is identical to the bit. No statistic is ever produced by the LLM.
- **Statistical (not name-based) leakage guards** — a candidate that reconstructs
  the target near-deterministically, separates a low-cardinality target almost
  perfectly, or is collinear with an excluded column is caught by gates that
  operate on the actual target's *values*, not on column names or domain
  keywords.
- **Internal validation battery** — each candidate is screened (Spearman + FDR)
  and stress-tested for replication, bootstrap stability, subgroup robustness, and
  confounder adjustment before it is reported as `validated`, `conditional`, or
  `rejected`.
- **Pseudo-replication aware** — when you declare a participant/unit column,
  repeated measures are handled with grouped splits and intraclass-correlation /
  effective-N corrections, not naive row sampling.
- **Grounded Fact Sheet** — every reportable number is assembled into a
  deterministic Fact Sheet that the report layer must cite, so prose can never
  drift from the computed evidence.
- **Integrity-enforcing model prompt** — the Gemini boundary refuses to weaken
  rigor (drop FDR, skip cross-validation, remove confounder adjustment,
  cherry-pick) and never calls an association causal, or validated for diagnosis
  or deployment.

## Architecture

```
            ┌───────────────────────────────────────────────────────────┐
            │ codas_service/  —  thin FastAPI surface (API-key, stateless)│
            │   /v1/discover   /v1/profile             /v1/agent          │
            └───────────────┬───────────────────────────────┬───────────┘
                            │ deterministic                  │ google-adk + Gemini
            ┌───────────────▼───────────────┐   ┌────────────▼────────────────┐
            │ codas_core/  (no LLM, no net)  │   │ codas_agents/  (LlmAgent×11) │
            │  data · statistics · validation│◄──│  SequentialAgent orchestrator│
            │  discovery · reporting         │tool│  tool-grounded, guardrailed │
            │  deps: numpy/pandas/scipy/sklearn   │  deps: google-adk/google-genai│
            └────────────────────────────────┘   └──────────────────────────────┘
```

- **`codas_core/`** — the deterministic engine. Pure Python; depends only on
  numpy, pandas, scipy, scikit-learn. Call it directly: `run_discovery(df, request)`.
- **`codas_agents/`** — the [google-adk](https://google.github.io/adk-docs/)
  layer: eleven `LlmAgent`s (scout, profiler, empirical, validation, defender,
  critic, mechanism, novelty, strategy, artifact, report) chained by a
  `SequentialAgent`. Each is tool-grounded — it calls the deterministic runners
  and may plan/explain/choose the target, but cannot fabricate numbers. The
  harness is split for review: tools in `agent.py`, prompts in `prompts.py`,
  session/execution in `runtime.py`, guardrails + logging in `callbacks.py`.
  Gemini is reached through `codas_core/gemini.py`.
- **`codas_service/`** — a thin FastAPI app exposing both layers over HTTPS with
  server-to-server API-key auth. Stateless: callers send data inline.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[all]"          # engine + agent + service; use `pip install .` for engine-only
```

**1. Call the engine directly (no LLM, no key needed):**

```python
from codas_core.discovery import DiscoveryRequest, run_discovery_from_csv

# You name the target; the engine assumes nothing about the columns.
report = run_discovery_from_csv(
    "examples/sample_dataset.csv",
    DiscoveryRequest(target_column="depression_score", top_k=5),
)
print(report.fact_sheet["ml_metric_name"], report.fact_sheet["ml_metric_value"])
for c in report.candidates:
    print(c.feature, c.verdict, round(c.rho, 3))
```

The bundled `examples/sample_dataset.csv` is one example (a wearable-health table)
— the engine treats it like any other CSV. See `tests/test_generalization.py`,
which runs the same engine on housing prices and on abstract `x1..x6` columns.

Or run the demo: `python examples/quickstart.py`.

**2. Run the HTTP service:**

```bash
uvicorn codas_service.app:app --port 8000
curl -s localhost:8000/v1/discover -H "Content-Type: application/json" --data "$(python3 -c '
import json; print(json.dumps({"csv": open("examples/sample_dataset.csv").read(),
                               "target_column": "depression_score", "top_k": 5}))')"
```

**3. Run the google-adk + Gemini pipeline** (needs a Gemini key). Here the LLM
chooses the target/roles from the schema and the task:

```bash
export GOOGLE_API_KEY=...        # see .env.example
curl -s localhost:8000/v1/agent -H "Content-Type: application/json" \
  --data '{"csv":"...","query":"find the strongest predictors of <your target>"}'
```

## HTTP API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | none | Liveness probe |
| GET | `/v1/health` | key | Readiness + whether Gemini is configured |
| POST | `/v1/profile` | key | Structural summary (dtypes, missingness, numeric columns) |
| POST | `/v1/discover` | key | Deterministic discovery for an explicit target → report |
| POST | `/v1/agent` | key | google-adk + Gemini pipeline (LLM picks the target/roles) |

`/v1/discover` requires `target_column`; optional `participant_id_column`,
`time_column`, `excluded_columns`, `confounder_columns` are honored if given and
simply not used if omitted (no name-based guessing). Auth: send
`X-CoDaS-Agent-Key: <key>` (keys in `CODAS_AGENT_API_KEYS`, comma-separated for
rotation). With no keys configured the service answers only localhost. CORS is
explicit-origin and credential-free by default.

## Two ways to integrate

1. **Host as an API (Docker / Cloud Run).** `docker build -t codas . && docker run -p 8080:8080 codas`,
   then call `/v1/discover` or `/v1/agent`. Best when consumers are remote.
2. **Vendor the engine as a library.** `codas_core` is a pure
   numpy/pandas/scipy/scikit-learn package with no web, Firebase, or ADK
   dependency, so a consumer can `from codas_core.discovery import run_discovery`
   and call it in-process. Best when the consumer is itself a Python program.

## Project layout

```
codas_core/      deterministic engine (data, statistics, validation, discovery, reporting, gemini)
codas_agents/    google-adk harness — tools (agent.py), prompts (prompts.py), session (runtime.py), guardrails (callbacks.py)
codas_service/   thin FastAPI service (app, API-key auth)
examples/        sample dataset + quickstart.py
tests/           engine determinism, multi-domain generalization, and service tests
```

## Tests

```bash
pip install ".[all,dev]" && python -m pytest -q
```

The suite includes `test_generalization.py`, which proves the engine runs on
datasets from unrelated domains (housing, abstract synthetic) with arbitrary
column names — no special-casing, no hardcoded features.

## Scientific scope and honesty

CoDaS reports **candidate** associations and an internal validation verdict.
Internal validation is hypothesis-generating; it is not external validation. CoDaS
never labels a finding causal, validated for diagnosis, or deployment-ready. Treat
results as a rigorously filtered starting point for confirmatory study.

## License

Not yet chosen. The repository ships with a proprietary "all rights reserved"
placeholder (`LICENSE`); pick a license before any external distribution.
