"""CoDaS agent pipeline: the full six-phase Orchestrator on the sample dataset.

Runs the google-adk + Gemini agents end to end and prints the phase/agent flow as it happens, then
the grounded final report. Needs a Gemini key:

    export GOOGLE_API_KEY=...
    python examples/run_agent.py

The dataset path is seeded into shared memory (session.state['csv_path']); the agents profile it,
choose the target and roles, iterate the deepening discovery loop, and write the report. Every
number in that report is computed by the deterministic engine, not the model.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Run from a checkout without installing: put the repo root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A snappy demo: cap the loop and the resamples. Remove these to run at full depth.
os.environ.setdefault("CODAS_MAX_DISCOVERY_ROUNDS", "2")
os.environ.setdefault("CODAS_ROUND_RESAMPLES", "200")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from codas_agents.agent import root_agent

SAMPLE = Path(__file__).resolve().parent / "sample_dataset.csv"
GOAL = "Discover which features predict depression_score; choose the target and roles yourself, iterate, and write a short report."


def _line(author: str, parts) -> str | None:
    """One readable line per event: a tool call/return, or the first line of an agent's message."""
    for part in parts:
        call = getattr(part, "function_call", None)
        result = getattr(part, "function_response", None)
        if call:
            return f"  {author:24s} → call {call.name}()"
        if result:
            return f"  {author:24s} ← {result.name} returned"
    text = " ".join((getattr(p, "text", "") or "").strip() for p in parts if getattr(p, "text", None)).strip()
    if text:
        return f"  {author:24s} : {text.splitlines()[0][:88]}"
    return None


async def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY to run the agent pipeline (the engine-only demo is examples/quickstart.py).")

    sessions = InMemorySessionService()
    await sessions.create_session(app_name="codas", user_id="demo", session_id="s", state={"csv_path": str(SAMPLE)})
    runner = Runner(app_name="codas", agent=root_agent, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=GOAL)])

    print(f"goal   : {GOAL}\n" + "-" * 96)
    final = ""
    async for event in runner.run_async(user_id="demo", session_id="s", new_message=message):
        parts = getattr(getattr(event, "content", None), "parts", None) or []
        line = _line(getattr(event, "author", "?"), parts)
        if line:
            print(line)
        if getattr(event, "author", "") == "report_agent":
            text = " ".join((getattr(p, "text", "") or "") for p in parts if getattr(p, "text", None)).strip()
            if text:
                final = text

    session = await sessions.get_session(app_name="codas", user_id="demo", session_id="s")
    rounds = session.state.get("rounds", [])
    print("-" * 96)
    print(f"discovery rounds run: {len(rounds)} (search deepened each round)")
    for r in rounds:
        print(f"  round {r['round']}: validated={r['validated_count']:<2} "
              f"{r['ml_metric_name']}={r['ml_metric_value']}  feature_budget={r['ratio_feature_budget']}")
    print("\n=== grounded report ===\n" + (final or "(no report text captured)"))


if __name__ == "__main__":
    asyncio.run(main())
