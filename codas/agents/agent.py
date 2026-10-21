"""Google ADK agent graph for CoDaS.

The Orchestrator coordinates state transitions across six phases, with all agents sharing one memory
(``session.state``), one Fact Sheet, and one deterministic tool set:

    root_agent  (SequentialAgent, the Orchestrator)
    ├─ Phase A  data_understanding  (SequentialAgent)
    │    ├─ scout_agent ............ profiles, chooses target/roles, set_target -> memory
    │    └─ hypotheses_agent ....... literature anchoring; registers a-priori hypotheses before any statistic
    ├─ Phase B&C  discovery_loop  (LoopAgent, up to CODAS_MAX_DISCOVERY_ROUNDS)
    │    ├─ search_agent ........... run_discovery_round -> appends a round to memory
    │    ├─ method_runner .......... deterministic: next statistical + ML pool methods on the candidates
    │    ├─ dual_track  (ParallelAgent)        # statistical ∥ ML interpretation, with analysis tools
    │    │    ├─ statistical_interpreter
    │    │    └─ ml_interpreter
    │    ├─ critic_agent ........... adversarial validation (attack, checks with tools)
    │    ├─ defender_agent ......... adversarial validation (defend from evidence, with tools)
    │    ├─ gap_finder ............. open questions -> next methods and constraints
    │    └─ gapcheck_agent ......... check_convergence -> escalate ends the loop
    └─ Phase D/E/F  reporting  (SequentialAgent)
         ├─ candidate_debate ....... per-candidate critic -> defender, recorded keep / caveat / concede
         ├─ mechanism_agent / novelty_agent ... literature-grounded deep research (search_literature)
         ├─ strategy_agent / artifact_agent
         ├─ report_agent -> report_reviewer -> report_reviser (numeric audit on the final text)
         └─ lessons_agent .......... lessons about the cohort's features for later outcomes

Every agent after the Scout sees only its own turn plus the memory keys its instruction names
(``include_contents="none"``), so prompts stay bounded however long the run.

Numbers come only from the deterministic tools in ``tools.py`` (engine = ``codas.core``). The LLMs
plan, profile, interpret, debate, decide when to stop, and write — they never invent a statistic.
Prompts live in ``prompts.py``; guardrails/logging in ``callbacks.py``; session execution in
``runtime.py``.
"""

from __future__ import annotations

import asyncio
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
    remember_goal,
    report_grounding_audit,
)
from codas.agents.tools import (
    ANALYSIS_TOOLS,
    check_convergence,
    declare_confounders,
    exclude_features,
    preview_columns,
    profile_dataset,
    propose_feature,
    register_hypotheses,
    run_discovery,
    run_discovery_round,
    run_dose_response,
    run_mixed_model,
    run_multivariable_model,
    run_nonlinear_model,
    run_subgroup_analysis,
    run_survival_screen,
    record_debate,
    record_gap_analysis,
    record_lessons,
    run_method,
    search_literature,
    set_target,
)
from codas.agents.runners import CandidateDebate, MethodRunner
from codas.core import methods
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
# A request that sends nothing for this long is abandoned (a stalled connection otherwise waits
# forever), retried once on the same model and then handed to the next one.
MODEL_TIMEOUT_S = float(os.getenv("CODAS_MODEL_TIMEOUT_S", "600"))
MODEL_TIMEOUT_RETRIES = int(os.getenv("CODAS_MODEL_TIMEOUT_RETRIES", "1"))
LOGGER = logging.getLogger("codas.agents")


class ModelTimeout(TimeoutError):
    code = 408


async def _within(agen: AsyncGenerator[LlmResponse, None], seconds: float) -> AsyncGenerator[LlmResponse, None]:
    """Yield from ``agen``, raising ModelTimeout when the next chunk takes longer than ``seconds``."""
    it = agen.__aiter__()
    try:
        while True:
            try:
                yield await asyncio.wait_for(it.__anext__(), timeout=seconds)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise ModelTimeout(f"no response within {seconds:.0f} s") from exc
    finally:
        await agen.aclose()


class FallbackLlm(BaseLlm):
    """Tries ``models`` in order: when one fails with a capacity or not-found error, or stays silent past
    MODEL_TIMEOUT_S (after MODEL_TIMEOUT_RETRIES more tries), before it has produced any output, the same
    request goes to the next one.

    Any other error, or one raised after part of a response was streamed, is raised unchanged, so two
    models' answers are never spliced together. ``model`` is the first model's name.
    """

    models: list[BaseLlm]
    timeout_s: float = MODEL_TIMEOUT_S
    timeout_retries: int = MODEL_TIMEOUT_RETRIES

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        for i, llm in enumerate(self.models):
            request = llm_request.model_copy(update={"model": llm.model})
            for attempt in range(self.timeout_retries + 1):
                produced = False
                try:
                    async for response in _within(llm.generate_content_async(request, stream=stream), self.timeout_s):
                        produced = True
                        yield response
                    return
                except ModelTimeout:
                    if produced:
                        raise
                    LOGGER.warning("model %s: %s (try %d of %d)", llm.model, "no response within "
                                   f"{self.timeout_s:.0f} s", attempt + 1, self.timeout_retries + 1)
                    if attempt < self.timeout_retries:
                        continue
                    if i == len(self.models) - 1:
                        raise
                    break
                except Exception as exc:  # noqa: BLE001 - re-raised unless another model can serve the call
                    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                    if produced or code not in _FALLBACK_CODES or i == len(self.models) - 1:
                        raise
                    break
            LOGGER.warning("model %s unavailable; retrying the call on %s", llm.model, self.models[i + 1].model)


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
    # Even a single model is wrapped, so every call has the response timeout.
    return FallbackLlm(model=names[0], models=[_single_model(n) for n in names])


check_llm_backend()  # AoU mode: refuse to build the graph on a backend outside the workspace project
REASONING_MODEL = _make_model(_REASONING_MODEL_NAME, _FAST_MODEL_NAME)  # the fast tier backs up the reasoning tier
FAST_MODEL = _make_model(_FAST_MODEL_NAME, _REASONING_MODEL_NAME)  # and the reasoning tier backs up the fast one
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
    before_agent: Any = None,
    include_contents: str = "none",
) -> LlmAgent:
    """Construct an LlmAgent with the shared logging/guardrail callbacks.

    ``model`` selects the tier: the Flash ``FAST_MODEL`` is the default for the repeated discovery-loop
    tasks; reasoning-heavy agents pass ``model=REASONING_MODEL`` (the Pro tier). ``output_key`` publishes
    the agent's response into shared memory so later agents (including the other branch of a
    ParallelAgent) can read it; ``tools`` wires the deterministic tool guardrail; ``after_agent`` runs a
    post-agent callback (used to grounding-audit the final report).
    """
    # Context management (the paper's event pruning between phases): an agent sees its own turn plus the
    # memory keys its instruction names, never the whole conversation, so prompts stay bounded.
    if include_contents == "none":
        instruction = prompts.TASK_CONTEXT + instruction
    kwargs: dict[str, Any] = dict(
        name=name,
        model=model,
        description=description,
        instruction=instruction,
        before_model_callback=before_model_logger,
        include_contents=include_contents,
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
    if before_agent:
        kwargs["before_agent_callback"] = before_agent
    return LlmAgent(**kwargs)


# --- Phase A: Data Profiling + Literature Grounding --------------------------------------------

scout_agent = _agent(
    "scout_agent",
    "Profiles the dataset and records the target and participant/time/confounder roles in memory.",
    prompts.SCOUT,
    model=REASONING_MODEL,
    tools=[profile_dataset, preview_columns, set_target],
    before_agent=remember_goal,
    include_contents="default",  # the first agent reads the user's task directly
)
hypotheses_agent = _agent(
    "hypotheses_agent",
    "Grounds prior expectations in the scientific literature to orient the search (biological anchoring).",
    prompts.HYPOTHESES,
    model=REASONING_MODEL,
    tools=[search_literature, declare_confounders, exclude_features, propose_feature, register_hypotheses],
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
    tools=[propose_feature, run_survival_screen, run_subgroup_analysis, run_dose_response, run_mixed_model, run_method],
    output_key="statistical_interpretation",
)
ml_interpreter = _agent(
    "ml_interpreter",
    "Interprets the held-out predictive (cross-validated model vs null) evidence of the latest round.",
    prompts.ML_INTERPRETER,
    tools=[run_multivariable_model, run_nonlinear_model],
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
    tools=[declare_confounders, exclude_features, run_subgroup_analysis, run_multivariable_model, run_dose_response,
           run_mixed_model, run_method],
    output_key="critique",
)
defender_agent = _agent(
    "defender_agent",
    "Defends candidates only when deterministic evidence supports retention; concedes otherwise.",
    prompts.DEFENDER,
    model=REASONING_MODEL,
    tools=[*ANALYSIS_TOOLS, run_method],
    output_key="defense",
)
method_runner = MethodRunner(
    name="method_runner",
    description="Deterministic runner: applies the next statistical and ML methods to the round's candidates.",
)
gap_finder = _agent(
    "gap_finder",
    "Identifies what the analysis leaves open and chooses the next methods from the statistical and ML pools.",
    prompts.GAP_FINDER + "\n".join(f"- {m}: {methods.describe(m)}" for pool in methods.available(True).values()
                                    for m in pool),
    model=REASONING_MODEL,
    tools=[record_gap_analysis],
    output_key="gap_analysis_text",
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
    sub_agents=[search_agent, method_runner, dual_track, critic_agent, defender_agent, gap_finder, gapcheck_agent],
)

# --- Phase D/E/F: Mechanism + Novelty + Strategy + Artifacts + Report ---------------------------

candidate_critic = _agent(
    "candidate_critic",
    "Attacks one final candidate at a time, checking its suspicions with the analysis tools.",
    prompts.CANDIDATE_CRITIC,
    model=REASONING_MODEL,
    tools=[*ANALYSIS_TOOLS, run_method],
    output_key="candidate_critique",
)
candidate_defender = _agent(
    "candidate_defender",
    "Answers the critique of one candidate from evidence and records keep / caveat / concede.",
    prompts.CANDIDATE_DEFENDER,
    model=REASONING_MODEL,
    tools=[*ANALYSIS_TOOLS, run_method, record_debate],
    output_key="candidate_defense",
)
candidate_debate = CandidateDebate(
    name="candidate_debate",
    description="Phase D: per-candidate critic-defender debate over the final validated and conditional candidates.",
    sub_agents=[candidate_critic, candidate_defender],
)
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
report_reviewer = _agent(
    "report_reviewer",
    "Checks the draft report against the fact sheet, rounds, analyses and debates; lists the issues.",
    prompts.REPORT_REVIEWER,
    model=REASONING_MODEL,
    output_key="report_review",
)
report_reviser = _agent(
    "report_reviser",
    "Revises the report to fix the reviewer's issues, keeping every engine number and verdict.",
    prompts.REPORT_REVISER,
    model=REASONING_MODEL,
    output_key="report",
    after_agent=report_grounding_audit,  # the numeric audit runs again on the revised text
)
lessons_agent = _agent(
    "lessons_agent",
    "Records methodological lessons about this cohort's features for later outcomes (cross-outcome memory).",
    prompts.LESSONS,
    tools=[record_lessons],
)
reporting = SequentialAgent(
    name="reporting",
    description="Phases D-F: mechanism, novelty, strategy, artifacts, and the final report.",
    sub_agents=[candidate_debate, mechanism_agent, novelty_agent, strategy_agent, artifact_agent, report_agent,
                report_reviewer, report_reviser, lessons_agent],
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
