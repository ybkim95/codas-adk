"""Google ADK agent graph for CoDaS.

The Orchestrator coordinates state transitions across six phases, with all agents sharing one memory
(``session.state``), one Fact Sheet, and one deterministic tool set:

    root_agent  (SequentialAgent, the Orchestrator)
    ├─ Phase A  data_understanding  (SequentialAgent)
    │    ├─ scout_agent ............ profiles, chooses target/roles, set_target -> memory
    │    └─ hypotheses_agent ....... biological anchoring: grounds priors in the literature (search_literature)
    ├─ Phase B&C  discovery_loop  (LoopAgent, iterates until the GapChecker escalates)
    │    ├─ search_agent ........... run_discovery_round -> appends a round to memory
    │    ├─ dual_track  (ParallelAgent)        # parallel statistical ∥ ML iterations
    │    │    ├─ statistical_interpreter ...... reads the round, interprets association evidence
    │    │    └─ ml_interpreter ............... reads the round, interprets predictive evidence
    │    ├─ critic_agent ........... adversarial validation (attack)
    │    ├─ defender_agent ......... adversarial validation (defend from evidence)
    │    └─ gapcheck_agent ......... check_convergence -> escalate ends the loop
    └─ Phase D/E/F  reporting  (SequentialAgent)
         ├─ mechanism_agent / novelty_agent ... literature-grounded deep research (search_literature)
         ├─ strategy_agent / artifact_agent
         └─ report_agent ........... publication-style summary, invites human feedback

Numbers come only from the deterministic tools in ``tools.py`` (engine = ``codas.core``). The LLMs
plan, profile, interpret, debate, decide when to stop, and write — they never invent a statistic.
Prompts live in ``prompts.py``; guardrails/logging in ``callbacks.py``; session execution in
``runtime.py``.
"""

from __future__ import annotations

import os
from typing import Any

# ADK 2.4 marks SequentialAgent / LoopAgent / ParallelAgent as deprecated in favour of Workflow, but
# — per ADK's own deprecation notice — "Workflow cannot yet be used as an LlmAgent sub-agent". CoDaS
# nests these as sub-agents (the loop and the phase pipelines sit inside the root SequentialAgent), so
# the Workflow replacement cannot express this graph yet. We therefore pin google-adk (see
# pyproject.toml / requirements-lock.txt) and keep the current, correct API until Workflow supports
# nested sub-agents. The corresponding DeprecationWarnings are filtered in the pytest config so a
# run's output is not mistaken for staleness.
from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.models import Gemini
from google.genai.types import HttpRetryOptions

from codas.agents import prompts
from codas.agents.callbacks import (
    after_tool_logger,
    before_model_logger,
    before_tool_guardrail,
    report_grounding_audit,
)
from codas.agents.tools import (
    check_convergence,
    preview_columns,
    profile_dataset,
    propose_feature,
    run_discovery,
    run_discovery_round,
    search_literature,
    set_target,
)
from codas.core.settings import load_local_env

load_local_env()

# Per-call resilience: every model call retries a transient backend error (overload / rate limit /
# internal) with exponential backoff + jitter, so a Gemini 503 spike no longer aborts a multi-agent
# run. This is cheaper and stronger than retrying the whole pipeline — only the failing call repeats,
# not the agents that already succeeded. The service boundary keeps a last-resort retry on top.
# Two model tiers: a Pro tier for research-intensive reasoning and
# code generation (design, adversarial debate, mechanism, reporting) and a Flash tier for the repeated
# lower-latency tasks in the discovery loop (interpretation, gap-checking). Set CODAS_GEMINI_MODEL to
# force a single model across every agent (useful for CI or a constrained quota).
_REASONING_MODEL_NAME = os.getenv("CODAS_GEMINI_REASONING_MODEL", "gemini-3.1-pro-preview")
_FAST_MODEL_NAME = os.getenv("CODAS_GEMINI_FAST_MODEL", "gemini-3-flash-preview")
if os.getenv("CODAS_GEMINI_MODEL"):  # single-model override applies to both tiers
    _REASONING_MODEL_NAME = _FAST_MODEL_NAME = os.environ["CODAS_GEMINI_MODEL"]
_MODEL_RETRY = HttpRetryOptions(
    attempts=int(os.getenv("CODAS_MODEL_RETRY_ATTEMPTS", "5")),
    initial_delay=1.0,
    max_delay=30.0,
    exp_base=2.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)
REASONING_MODEL = Gemini(model=_REASONING_MODEL_NAME, retry_options=_MODEL_RETRY)
FAST_MODEL = Gemini(model=_FAST_MODEL_NAME, retry_options=_MODEL_RETRY)
# The orchestrator's loop runs at most this many deepening rounds; the GapChecker usually stops it
# earlier via escalate once deeper search saturates. Bounded so a live run stays within the timeout.
MAX_DISCOVERY_ROUNDS = int(os.getenv("CODAS_MAX_DISCOVERY_ROUNDS", "3"))


def _agent(
    name: str,
    description: str,
    instruction: str,
    *,
    model: Gemini = FAST_MODEL,
    tools: list | None = None,
    output_key: str | None = None,
    after_agent: Any = None,
) -> LlmAgent:
    """Construct an LlmAgent with the shared logging/guardrail callbacks.

    ``model`` selects the tier: the Flash ``FAST_MODEL`` is the default for the repeated discovery-loop
    tasks; reasoning-heavy agents pass ``model=REASONING_MODEL`` (the Pro tier). ``output_key`` publishes
    the agent's response into shared memory so later agents (including the other branch of a
    ParallelAgent) can read it; ``tools`` wires the deterministic tool guardrail; ``after_agent`` runs a
    post-agent callback (used to grounding-audit the final report).
    """
    kwargs: dict[str, Any] = dict(
        name=name,
        model=model,
        description=description,
        instruction=instruction,
        before_model_callback=before_model_logger,
    )
    if tools:
        kwargs.update(
            tools=tools,
            before_tool_callback=before_tool_guardrail,
            after_tool_callback=after_tool_logger,
        )
    if output_key:
        kwargs["output_key"] = output_key
    if after_agent:
        kwargs["after_agent_callback"] = after_agent
    return LlmAgent(**kwargs)


# --- Phase A: Data Profiling + Literature Grounding --------------------------------------------

scout_agent = _agent(
    "scout_agent",
    "Profiles the dataset and records the target and participant/time/confounder roles in memory.",
    prompts.SCOUT,
    model=REASONING_MODEL,
    tools=[profile_dataset, preview_columns, set_target],
)
hypotheses_agent = _agent(
    "hypotheses_agent",
    "Grounds prior expectations in the scientific literature to orient the search (biological anchoring).",
    prompts.HYPOTHESES,
    model=REASONING_MODEL,
    tools=[search_literature],
    output_key="hypotheses",
)
data_understanding = SequentialAgent(
    name="data_understanding",
    description="Phase A: profile the dataset, fix the analysis design, and frame the hypotheses.",
    sub_agents=[scout_agent, hypotheses_agent],
)

# --- Phase B&C: Parallel Agentic Search + Adversarial Validation (iterative loop) ---------------

search_agent = _agent(
    "search_agent",
    "Runs the next deterministic discovery round, deepening the search and recording it in memory.",
    prompts.SEARCH,
    tools=[run_discovery_round],
)
statistical_interpreter = _agent(
    "statistical_interpreter",
    "Interprets association evidence and proposes physiological feature transformations to evaluate.",
    prompts.STATISTICAL_INTERPRETER,
    tools=[propose_feature],
    output_key="statistical_interpretation",
)
ml_interpreter = _agent(
    "ml_interpreter",
    "Interprets the held-out predictive (cross-validated model vs null) evidence of the latest round.",
    prompts.ML_INTERPRETER,
    output_key="ml_interpretation",
)
dual_track = ParallelAgent(
    name="dual_track",
    description="Parallel statistical ∥ machine-learning iterations over the same round.",
    sub_agents=[statistical_interpreter, ml_interpreter],
)
critic_agent = _agent(
    "critic_agent",
    "Adversarially attacks candidates for leakage, confounding, construct overlap, and overclaiming.",
    prompts.CRITIC,
    model=REASONING_MODEL,
    output_key="critique",
)
defender_agent = _agent(
    "defender_agent",
    "Defends candidates only when deterministic evidence supports retention; concedes otherwise.",
    prompts.DEFENDER,
    model=REASONING_MODEL,
    output_key="defense",
)
gapcheck_agent = _agent(
    "gapcheck_agent",
    "GapChecker: compares rounds and escalates to end the loop once deeper search saturates.",
    prompts.GAPCHECK,
    tools=[check_convergence],
)
discovery_loop = LoopAgent(
    name="discovery_loop",
    description="Phases B&C: iterate deepening parallel search and adversarial validation until the "
    "GapChecker signals convergence.",
    max_iterations=MAX_DISCOVERY_ROUNDS,
    sub_agents=[search_agent, dual_track, critic_agent, defender_agent, gapcheck_agent],
)

# --- Phase D/E/F: Mechanism + Novelty + Strategy + Artifacts + Report ---------------------------

mechanism_agent = _agent(
    "mechanism_agent",
    "Links surviving candidates to literature-grounded mechanisms and interpretation boundaries.",
    prompts.MECHANISM,
    model=REASONING_MODEL,
    tools=[search_literature],
    output_key="mechanisms",
)
novelty_agent = _agent(
    "novelty_agent",
    "Classifies candidates as established, supported, emerging, or unverified against the literature.",
    prompts.NOVELTY,
    model=REASONING_MODEL,
    tools=[search_literature],
    output_key="novelty",
)
strategy_agent = _agent(
    "strategy_agent",
    "Recommends accept / reinvestigate / deeper-analysis / human-feedback from the run evidence.",
    prompts.STRATEGY,
    model=REASONING_MODEL,
    output_key="strategy",
)
artifact_agent = _agent(
    "artifact_agent",
    "Assembles tables and a fact-sheet view from grounded outputs.",
    prompts.ARTIFACT,
    output_key="artifacts",
)
report_agent = _agent(
    "report_agent",
    "Drafts a publication-style summary from the Fact Sheet and invites human feedback.",
    prompts.REPORT,
    model=REASONING_MODEL,
    output_key="report",
    after_agent=report_grounding_audit,  # runtime grounding guardrail: logs/warns on unverified figures
)
reporting = SequentialAgent(
    name="reporting",
    description="Phases D-F: mechanism, novelty, strategy, artifacts, and the final report.",
    sub_agents=[mechanism_agent, novelty_agent, strategy_agent, artifact_agent, report_agent],
)

# --- The Orchestrator ---------------------------------------------------------------------------

root_agent = SequentialAgent(
    name="codas_orchestrator",
    description="CoDaS orchestrator: six-phase, memory-shared, leakage-guarded association discovery "
    "with an iterative deepening search loop and optional human feedback.",
    sub_agents=[data_understanding, discovery_loop, reporting],
)

__all__ = [
    "root_agent",
    "data_understanding",
    "discovery_loop",
    "reporting",
    # Re-exported deterministic tools (used by the offline harness tests and the simple service path).
    "profile_dataset",
    "preview_columns",
    "run_discovery",
]
