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

import logging
import os
from typing import Any, AsyncGenerator

# ADK 2.4 marks SequentialAgent / LoopAgent / ParallelAgent as deprecated in favour of Workflow, but
# — per ADK's own deprecation notice — "Workflow cannot yet be used as an LlmAgent sub-agent". CoDaS
# nests these as sub-agents (the loop and the phase pipelines sit inside the root SequentialAgent), so
# the Workflow replacement cannot express this graph yet. We therefore pin google-adk (see
# pyproject.toml / requirements-lock.txt) and keep the current, correct API until Workflow supports
# nested sub-agents. The corresponding DeprecationWarnings are filtered in the pytest config so a
# run's output is not mistaken for staleness.
from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.models import BaseLlm, Gemini, LlmRequest, LlmResponse
from google.genai.types import HttpRetryOptions

from codas.aou.privacy import check_llm_backend
from codas.agents import prompts
from codas.agents.callbacks import (
    after_tool_logger,
    before_model_logger,
    before_tool_guardrail,
    report_grounding_audit,
)
from codas.agents.tools import (
    check_convergence,
    declare_confounders,
    exclude_features,
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
# A 429 on shared capacity reflects load across the service, not this project's quota, and can last
# minutes; the backoff (2, 4, 8, 16, 32, 60, 60 s, with jitter) rides out about three minutes of it
# before the call gives up and the next configured model is tried.
_MODEL_RETRY = HttpRetryOptions(
    attempts=int(os.getenv("CODAS_MODEL_RETRY_ATTEMPTS", "8")),
    initial_delay=float(os.getenv("CODAS_MODEL_RETRY_INITIAL_DELAY", "2")),
    max_delay=float(os.getenv("CODAS_MODEL_RETRY_MAX_DELAY", "60")),
    exp_base=2.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)
# Status codes after which another model may succeed where this one did not: capacity exhausted, or
# the model is not served here.
_FALLBACK_CODES = {404, 429}
LOGGER = logging.getLogger("codas.agents")


class FallbackLlm(BaseLlm):
    """Tries ``models`` in order: when one fails with a capacity or not-found error before it has
    produced any output (its own retries already spent), the same request goes to the next one.

    Any other error, or one raised after part of a response was streamed, is raised unchanged, so two
    models' answers are never spliced together. ``model`` is the first model's name.
    """

    models: list[BaseLlm]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        for i, llm in enumerate(self.models):
            request = llm_request.model_copy(update={"model": llm.model})
            produced = False
            try:
                async for response in llm.generate_content_async(request, stream=stream):
                    produced = True
                    yield response
                return
            except Exception as exc:  # noqa: BLE001 - re-raised unless another model can serve the call
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if produced or code not in _FALLBACK_CODES or i == len(self.models) - 1:
                    raise
                LOGGER.warning("model %s unavailable (%s); retrying the call on %s",
                               llm.model, code, self.models[i + 1].model)


def _single_model(name: str) -> BaseLlm:
    """Gemini by default; a ``claude-*`` name is served as Claude on Vertex AI (Model Garden), using
    GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION and requiring the ``anthropic[vertex]`` package."""
    if name.strip().lower().startswith("claude"):
        from google.adk.models.anthropic_llm import Claude

        return Claude(model=name.strip())
    return Gemini(model=name.strip(), retry_options=_MODEL_RETRY)


def _make_model(name: str, *fallbacks: str) -> BaseLlm:
    """``name``, falling back to ``fallbacks`` and then to CODAS_GEMINI_FALLBACK_MODELS (comma-separated)
    when it is out of capacity or not served."""
    extra = [m.strip() for m in os.getenv("CODAS_GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
    names = list(dict.fromkeys(n.strip() for n in (name, *fallbacks, *extra) if n and n.strip()))
    if len(names) == 1:
        return _single_model(names[0])
    return FallbackLlm(model=names[0], models=[_single_model(n) for n in names])


check_llm_backend()  # AoU mode: refuse to build the graph on a backend outside the workspace project
REASONING_MODEL = _make_model(_REASONING_MODEL_NAME, _FAST_MODEL_NAME)  # the fast tier backs up the reasoning tier
FAST_MODEL = _make_model(_FAST_MODEL_NAME)
# The orchestrator's loop runs at most this many deepening rounds; the GapChecker usually stops it
# earlier via escalate once deeper search saturates. Bounded so a live run stays within the timeout.
MAX_DISCOVERY_ROUNDS = int(os.getenv("CODAS_MAX_DISCOVERY_ROUNDS", "3"))


def _agent(
    name: str,
    description: str,
    instruction: str,
    *,
    model: BaseLlm = FAST_MODEL,
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
    tools=[search_literature, declare_confounders, exclude_features],
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
    tools=[declare_confounders, exclude_features],
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
