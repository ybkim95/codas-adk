"""Deterministic (no-LLM) agents in the CoDaS graph.

``MethodRunner`` is the paper's deterministic runner for the method pools: each loop round, after the
discovery round, it applies the methods the gap finder asked for (or the next unexplored method of each
pool) to the round's candidates and writes the results to shared memory, where the interpreters, the
critic and the defender read them. It calls no model; every number it writes comes from
``codas.core.methods``.

``CandidateDebate`` is the paper's per-candidate debate: after the loop, each final validated or
conditional candidate is debated on its own by a critic and a defender, who can run the analysis
tools to settle a point; the defender closes with a recorded verdict (keep, caveat, concede). The
verdicts shape how candidates are interpreted and reported, never the engine's validation.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types

from codas.agents.tools import debate_candidates, run_method_batch

METHODS_PER_ROUND = int(os.getenv("CODAS_METHODS_PER_ROUND", "1"))
DEBATE_CANDIDATES = int(os.getenv("CODAS_DEBATE_CANDIDATES", "5"))


class MethodRunner(BaseAgent):
    """Runs the next statistical and ML methods on the latest round's candidates."""

    async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
        state = dict(ctx.session.state)
        try:
            # Off the event loop: the methods take minutes, and a blocked loop holds back the outcome's
            # time limit and every other coroutine until they finish.
            delta = await asyncio.to_thread(run_method_batch, state, METHODS_PER_ROUND)
            text = "Methods run this round:\n" + "\n".join(
                line for line in delta["analyses_digest"].splitlines()
                if f"round {max(len(state.get('rounds') or []) - 1, 0)} " in line) if delta["rounds"] else "No round yet."
        except Exception as exc:  # noqa: BLE001 - a failed method batch must not stop the loop
            delta, text = {"next_methods": []}, f"Method runner could not run: {type(exc).__name__}: {exc}"
        yield Event(author=self.name, invocation_id=ctx.invocation_id, branch=ctx.branch,
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                    actions=EventActions(state_delta=delta))


class CandidateDebate(BaseAgent):
    """Debates each final candidate in turn with its sub-agents (critic, then defender)."""

    async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
        debates = list(ctx.session.state.get("candidate_debates") or [])
        for candidate in debate_candidates(dict(ctx.session.state), DEBATE_CANDIDATES):
            yield self._event(ctx, f"Debating candidate {candidate['feature']}.",
                              {"debate_candidate": candidate, "candidate_critique": "", "candidate_defense": "",
                               "debate_verdict": {}})
            for agent in self.sub_agents:
                async for event in agent.run_async(ctx):
                    yield event
            state = ctx.session.state
            verdict = state.get("debate_verdict") or {}
            debates.append({"feature": candidate["feature"],
                            "verdict": verdict.get("verdict", "not recorded"), "reason": verdict.get("reason", ""),
                            "evidence": verdict.get("evidence", ""),
                            "critique": str(state.get("candidate_critique") or "")[:2000],
                            "defense": str(state.get("candidate_defense") or "")[:2000]})
            yield self._event(ctx, f"Verdict on {candidate['feature']}: {debates[-1]['verdict']}.",
                              {"candidate_debates": debates})

    def _event(self, ctx: Any, text: str, delta: dict[str, Any]) -> Event:
        return Event(author=self.name, invocation_id=ctx.invocation_id, branch=ctx.branch,
                     content=types.Content(role="model", parts=[types.Part(text=text)]),
                     actions=EventActions(state_delta=delta))
