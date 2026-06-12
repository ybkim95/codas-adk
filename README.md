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
python scripts/agent_robustness.py                   # live audit: orchestration, grounding integrity, prompt-injection
```

## About

CoDaS implements the architecture described in the CoDaS paper
(arXiv:[2604.14615](https://arxiv.org/pdf/2604.14615)). A companion project packages the
deterministic engine as a reusable agent skill:
[codas-science-skills](https://github.com/ybkim95/codas-science-skills).
