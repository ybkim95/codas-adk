"""Small Google ADK runtime boundary for product-facing agent turns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


@dataclass
class AdkAgentResult:
    text: str
    events: list[dict[str, Any]] = field(default_factory=list)


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


async def _run_adk_agent_text_async(
    agent: BaseAgent,
    prompt: str,
    *,
    app_name: str,
    user_id: str,
    session_id: str,
    state: dict[str, Any] | None = None,
) -> AdkAgentResult:
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id, state=state or {})
    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)
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
) -> AdkAgentResult:
    return asyncio.run(
        _run_adk_agent_text_async(
            agent,
            prompt,
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=state,
        )
    )
