# CoDaS — AI Co-Data-Scientist

Give CoDaS **any tabular CSV and a research goal in plain language**. A team of
[google-adk](https://google.github.io/adk-docs/) + Gemini agents profiles the data, frames
hypotheses, runs an iterative deepening search, validates it adversarially, and returns a
rigorously grounded, auditable set of candidate predictors of your target.

The agents plan, interpret, debate, and decide when to stop. Every **number** in the report is
computed by deterministic Python the model cannot bypass or invent — and the engine is
domain-agnostic: no hardcoded features or column names, so it works on any CSV.

## How it works

The **Orchestrator** coordinates six phases over one shared memory and one deterministic tool set.
Its heart is an **iterative discovery loop** that deepens the search each round until a *GapChecker*
judges that going further no longer pays off.

```
   any CSV  +  a research goal in plain language
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
   an auditable report   ⟲   optional: human feedback steers one more iteration
```

Built on **google-adk**: each phase is a `SequentialAgent`, the discovery loop is a `LoopAgent`
(the GapChecker ends it via `escalate`), and the two interpreters run inside a `ParallelAgent`. All
agents share one `session.state`; each writes its result back via `output_key`, so an entire run is
a single auditable object.

- **Grounded** — the deterministic engine (Spearman + FDR screening, a validation battery, and
  statistical leakage guards) is the only source of numbers; the LLM never invents a statistic.
- **Domain-agnostic** — no hardcoded features, columns, or domain rules; you name the target.
- **Iterative** — the search deepens until returns diminish, and a reviewer can steer one more pass.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[all]"

# 1 — deterministic engine only, no API key needed
python examples/quickstart.py

# 2 — the full six-phase agent pipeline (prints the live phase flow, then the report)
export GOOGLE_API_KEY=...
python examples/run_agent.py

# 3 — or run it as a service
uvicorn codas_service.app:app --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/discover` | Deterministic discovery for an explicit `target_column` |
| POST | `/v1/agent` | The agent pipeline picks the target/roles; returns a `session_id` |
| POST | `/v1/agent/feedback` | Resume that session with feedback for one more iteration |

API-key auth via `X-CoDaS-Agent-Key` (`CODAS_AGENT_API_KEYS`, comma-separated); see `.env.example`.

## Layout & tests

- **`codas_core/`** — the deterministic engine (numpy/pandas/scipy/scikit-learn only; no LLM, no network).
- **`codas_agents/`** — the google-adk graph (`agent.py`), deterministic tools (`tools.py`),
  prompts (`prompts.py`), session/memory (`runtime.py`), guardrails + logging (`callbacks.py`).
- **`codas_service/`** — the FastAPI surface exposing both layers.

```bash
pip install ".[all,dev]" && python -m pytest -q     # 47 tests, incl. the six-phase graph + loop tools
```
