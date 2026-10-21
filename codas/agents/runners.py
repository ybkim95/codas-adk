"""Deterministic (no-LLM) agents in the CoDaS graph.

``MethodRunner`` is the paper's deterministic runner for the method pools: each loop round, after the
discovery round, it applies the methods the gap finder asked for (or the next unexplored method of each
pool) to the round's candidates and writes the results to shared memory, where the interpreters, the
critic and the defender read them. It calls no model; every number it writes comes from
``codas.core.methods``.
"""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types

from codas.agents.tools import run_method_batch

METHODS_PER_ROUND = int(os.getenv("CODAS_METHODS_PER_ROUND", "1"))


class MethodRunner(BaseAgent):
    """Runs the next statistical and ML methods on the latest round's candidates."""

    async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
        state = dict(ctx.session.state)
        try:
            delta = run_method_batch(state, per_pool=METHODS_PER_ROUND)
            text = "Methods run this round:\n" + "\n".join(
                line for line in delta["analyses_digest"].splitlines()
                if f"round {max(len(state.get('rounds') or []) - 1, 0)} " in line) if delta["rounds"] else "No round yet."
        except Exception as exc:  # noqa: BLE001 - a failed method batch must not stop the loop
            delta, text = {"next_methods": []}, f"Method runner could not run: {type(exc).__name__}: {exc}"
        yield Event(author=self.name, invocation_id=ctx.invocation_id, branch=ctx.branch,
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                    actions=EventActions(state_delta=delta))
