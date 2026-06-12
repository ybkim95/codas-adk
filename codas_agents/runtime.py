"""Small Google ADK runtime boundary for product-facing agent turns.

A turn runs an agent over one session and returns its text plus a per-event trace. The session is the
shared memory: passing ``state`` seeds it (e.g. the dataset path), and passing a persistent
``session_service`` lets a later turn resume the SAME session with all prior state and events intact —
which is how the optional human-feedback re-entry continues a finished discovery instead of restarting
it. With no ``session_service`` the turn is stateless (a fresh in-memory session per call).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types


@dataclass
class AdkAgentResult:
    text: str
    events: list[dict[str, Any]] = field(default_factory=list)


def new_session_service() -> InMemorySessionService:
    """A process-persistent session store. The service holds one of these so sessions survive across
    requests (enabling feedback re-entry). Swap for a DatabaseSessionService in multi-instance prod."""
    return InMemorySessionService()


def _event_text(event: Any) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "\n".join(str(getattr(part, "text", "") or "").strip() for part in parts if getattr(part, "text", None)).strip()


def _event_summary(event: Any) -> dict[str, Any]:
    usage = getattr(event, "usage_metadata", None)
    return {
        "author": getattr(event, "author", None),
        "model_version": getattr(event, "model_version", None),
        "has_text": bool(_event_text(event)),
        "finish_reason": str(getattr(event, "finish_reason", "") or ""),
        "token_count": getattr(usage, "total_token_count", None) if usage else None,
    }


async def _ensure_session(
    session_service: BaseSessionService,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    state: dict[str, Any] | None,
) -> None:
    """Create the session if it does not already exist; otherwise leave it (and its memory) intact."""
    existing = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if existing is None:
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id, state=state or {}
        )


async def _run_adk_agent_text_async(
    agent: BaseAgent,
    prompt: str,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    state: dict[str, Any] | None = None,
    session_service: BaseSessionService | None = None,
) -> AdkAgentResult:
    service = session_service or InMemorySessionService()
    await _ensure_session(service, app_name=app_name, user_id=user_id, session_id=session_id, state=state)
    runner = Runner(app_name=app_name, agent=agent, session_service=service)
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        text = _event_text(event)
        if text:
            text_parts.append(text)
        events.append(_event_summary(event))
    return AdkAgentResult(text="\n\n".join(text_parts).strip(), events=events)


def run_adk_agent_text(
    agent: BaseAgent,
    prompt: str,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    state: dict[str, Any] | None = None,
    session_service: BaseSessionService | None = None,
) -> AdkAgentResult:
    """Run ``agent`` for one turn over ``session_id`` and return its text plus an event trace.

    ``state`` seeds a new session's shared memory (ignored if the session already exists). Pass a
    persistent ``session_service`` to resume an existing session — its prior state and events carry
    over, which is how human-feedback re-entry continues a discovery rather than restarting it.
    """
    return asyncio.run(
        _run_adk_agent_text_async(
            agent,
            prompt,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=state,
            session_service=session_service,
        )
    )
