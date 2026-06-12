"""Google ADK agent graph for CoDaS — a faithful build of the paper's architecture.

The Orchestrator coordinates state transitions across six phases, with all agents sharing one memory
(``session.state``), one Fact Sheet, and one deterministic tool set:

    root_agent  (SequentialAgent, the Orchestrator)
    ├─ Phase A  data_understanding  (SequentialAgent)
    │    ├─ scout_agent ............ profiles, chooses target/roles, set_target -> memory
    │    └─ hypotheses_agent ....... states prior expectations to test
    ├─ Phase B&C  discovery_loop  (LoopAgent, iterates until the GapChecker escalates)
    │    ├─ search_agent ........... run_discovery_round -> appends a round to memory
    │    ├─ dual_track  (ParallelAgent)        # the paper's parallel statistical ∥ ML iterations
    │    │    ├─ statistical_interpreter ...... reads the round, interprets association evidence
    │    │    └─ ml_interpreter ............... reads the round, interprets predictive evidence
    │    ├─ critic_agent ........... adversarial validation (attack)
    │    ├─ defender_agent ......... adversarial validation (defend from evidence)
    │    └─ gapcheck_agent ......... check_convergence -> escalate ends the loop
    └─ Phase D/E/F  reporting  (SequentialAgent)
         ├─ mechanism_agent / novelty_agent / strategy_agent / artifact_agent
         └─ report_agent ........... publication-style summary, invites human feedback

Numbers come only from the deterministic tools in ``tools.py`` (engine = ``codas_core``). The LLMs
plan, profile, interpret, debate, decide when to stop, and write — they never invent a statistic.
Prompts live in ``prompts.py``; guardrails/logging in ``callbacks.py``; session execution in
``runtime.py``.
"""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent

from codas_agents import prompts
from codas_agents.callbacks import after_tool_logger, before_model_logger, before_tool_guardrail
from codas_agents.tools import (
    check_convergence,
    preview_columns,
    profile_dataset,
    run_discovery,
    run_discovery_round,
    set_target,
)
from codas_core.settings import load_local_env

load_local_env()

MODEL = os.getenv("CODAS_GEMINI_MODEL", "gemini-3.5-flash")
# The orchestrator's loop runs at most this many deepening rounds; the GapChecker usually stops it
# earlier via escalate once deeper search saturates. Bounded so a live run stays within the timeout.
MAX_DISCOVERY_ROUNDS = int(os.getenv("CODAS_MAX_DISCOVERY_ROUNDS", "3"))


def _agent(
    name: str,
    description: str,
    instruction: str,
    *,
    tools: list | None = None,
    output_key: str | None = None,
) -> LlmAgent:
    """Construct an LlmAgent with the shared logging/guardrail callbacks.

    ``output_key`` publishes the agent's response into shared memory so later agents (including the
    other branch of a ParallelAgent) can read it; ``tools`` wires the deterministic tool guardrail.
    """
    kwargs: dict[str, Any] = dict(
        name=name,
        model=MODEL,
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
    return LlmAgent(**kwargs)


# --- Phase A: Data Profiling + Literature Grounding --------------------------------------------

scout_agent = _agent(
    "scout_agent",
    "Profiles the dataset and records the target and participant/time/confounder roles in memory.",
    prompts.SCOUT,
    tools=[profile_dataset, preview_columns, set_target],
)
hypotheses_agent = _agent(
    "hypotheses_agent",
    "States the prior expectations the analysis will test, to orient the search.",
    prompts.HYPOTHESES,
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
    "Interprets the association (Spearman + FDR) evidence of the latest round.",
    prompts.STATISTICAL_INTERPRETER,
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
    description="The paper's parallel statistical ∥ machine-learning iterations over the same round.",
    sub_agents=[statistical_interpreter, ml_interpreter],
)
critic_agent = _agent(
    "critic_agent",
    "Adversarially attacks candidates for leakage, confounding, construct overlap, and overclaiming.",
    prompts.CRITIC,
    output_key="critique",
)
defender_agent = _agent(
    "defender_agent",
    "Defends candidates only when deterministic evidence supports retention; concedes otherwise.",
    prompts.DEFENDER,
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
    "Links surviving candidates to plausible mechanisms and interpretation boundaries.",
    prompts.MECHANISM,
    output_key="mechanisms",
)
novelty_agent = _agent(
    "novelty_agent",
    "Classifies candidates as established, supported, emerging, or unverified from supplied evidence.",
    prompts.NOVELTY,
    output_key="novelty",
)
strategy_agent = _agent(
    "strategy_agent",
    "Recommends accept / reinvestigate / deeper-analysis / human-feedback from the run evidence.",
    prompts.STRATEGY,
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
    output_key="report",
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
