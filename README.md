# CoDaS — AI Co-Data-Scientist

CoDaS turns **any tabular CSV + a target column** into a rigorously validated, auditable set of
candidate predictors of that target. It separates LLM reasoning from statistics: a
[google-adk](https://google.github.io/adk-docs/) + Gemini agent plans and explains, but every
reported number is computed by deterministic Python that the model cannot bypass or invent.

The engine is **domain-agnostic** — it makes no assumption about column names or problem domain,
and contains no hardcoded features or dataset-specific rules. You name the target; it does the stats.

## Quick Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[all]"
python examples/quickstart.py          # deterministic discovery on a sample CSV — no API key needed
```

Call the engine directly (this is the whole API):

```python
from codas_core.discovery import DiscoveryRequest, run_discovery_from_csv

report = run_discovery_from_csv("examples/sample_dataset.csv",
                                DiscoveryRequest(target_column="depression_score", top_k=5))
print(report.fact_sheet["ml_metric_name"], report.fact_sheet["ml_metric_value"])
for c in report.candidates:
    print(c.feature, c.verdict, round(c.rho, 3))      # verdict ∈ validated|conditional|rejected
```

Run it as a service, or let the Gemini agent pick the target itself:

```bash
uvicorn codas_service.app:app --port 8000            # POST /v1/discover with {csv, target_column}
export GOOGLE_API_KEY=...                             # then /v1/agent runs the google-adk pipeline
```

## What makes it trustworthy

- **Deterministic & reproducible** — same input + seed ⇒ identical output; no number comes from the LLM.
- **Statistical leakage guards** — a feature that reconstructs the target (near-perfect proxy,
  collinear duplicate, concurrent measure) is caught from the data, not from column names.
- **Validation battery** — each candidate is screened (Spearman + FDR) then stress-tested for
  replication, bootstrap stability, subgroup robustness, and confounder adjustment.
- **Pseudo-replication aware** — declare a participant column and repeated measures get grouped
  splits + ICC corrections, not naive row sampling.
- **Grounded Fact Sheet** — every reportable number is assembled deterministically and cited by the report.

## Architecture

```
codas_service/   thin FastAPI surface (API-key auth, stateless)   /v1/discover /v1/profile /v1/agent
      │ deterministic                              │ google-adk + Gemini
codas_core/      the engine (no LLM, no network)   codas_agents/   LlmAgent ×11 + SequentialAgent
  numpy/pandas/scipy/scikit-learn only             tool-grounded; the LLM cannot invent numbers
```

- **`codas_core/`** — deterministic engine; depends only on numpy/pandas/scipy/scikit-learn.
- **`codas_agents/`** — google-adk harness: tools (`agent.py`), prompts (`prompts.py`), session
  (`runtime.py`), guardrails+logging (`callbacks.py`). Gemini is reached via `codas_core/gemini.py`.
- **`codas_service/`** — FastAPI app exposing both layers over HTTPS.

## HTTP API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | none | Liveness |
| POST | `/v1/profile` | key | Schema / missingness / numeric columns |
| POST | `/v1/discover` | key | Deterministic discovery for an explicit `target_column` |
| POST | `/v1/agent` | key | google-adk + Gemini pipeline (LLM picks the target/roles) |

Auth: `X-CoDaS-Agent-Key: <key>` (`CODAS_AGENT_API_KEYS`, comma-separated for rotation); unset ⇒
localhost-only. CORS is explicit-origin, credential-free. See `.env.example` for configuration.

## Two ways to integrate

1. **Host it** — `docker build -t codas . && docker run -p 8080:8080 codas`, then call `/v1/discover`.
2. **Vendor the engine** — `codas_core` is a pure scientific library with no web/cloud dependency:
   `from codas_core.discovery import run_discovery` and call it in-process.

## Tests

```bash
pip install ".[all,dev]" && python -m pytest -q     # 38 tests; CI runs them on py3.10–3.12
```

Coverage: determinism, cross-domain generalization, adversarial edge cases (never crash), real
clinical datasets, and the agent harness. `python scripts/benchmark_datasets.py` runs the engine on
real public datasets (breast-cancer, diabetes, penguins, wine, …) and fails on any crash.

## Scope

CoDaS reports **candidate** associations with an internal validation verdict — hypothesis-generating,
not external or clinical validation. It never labels a finding causal or deployment-ready.

## License

Proprietary placeholder (`LICENSE`) — choose a license before any external distribution.
