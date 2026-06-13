"""ADK guardrail and observability callbacks for CoDaS agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from codas_agents.grounding import engine_numbers, ungrounded_claims


LOGGER = logging.getLogger("codas.agents")
# Library logging pattern: a NullHandler so importing never warns, but propagation LEFT ON so the host
# app's logging config (uvicorn / Cloud Run capture root at INFO) actually surfaces the agent trace —
# model calls, tool start/end, and the grounding guardrail. (Previously propagate=False made all of
# this unreachable in production unless a handler was attached to "codas.agents" by name.)
LOGGER.addHandler(logging.NullHandler())
ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATA_ROOTS = [
    (ROOT / ".codas_runs").resolve(),
    (ROOT / "examples").resolve(),
]


def _is_allowed_path(value: str) -> bool:
    try:
        path = Path(value).expanduser().resolve()
    except OSError:
        return False
    return any(path == root or root in path.parents for root in ALLOWED_DATA_ROOTS)


def before_tool_guardrail(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """Prevent ADK tools from reading arbitrary local files."""
    csv_path = args.get("csv_path")
    if csv_path and not _is_allowed_path(str(csv_path)):
        LOGGER.warning("Blocked tool %s from accessing %s", getattr(tool, "name", tool), csv_path)
        return {
            "error": "Path is outside CoDaS allowed data roots.",
            "allowed_roots": [str(root) for root in ALLOWED_DATA_ROOTS],
        }
    LOGGER.info("Tool start: %s args=%s", getattr(tool, "name", tool), {k: v for k, v in args.items() if k != "csv_path"})
    return None


def after_tool_logger(tool: Any, args: dict[str, Any], tool_context: Any, tool_response: dict[str, Any]) -> None:
    """Log deterministic tool completion without mutating the response."""
    status = "error" if isinstance(tool_response, dict) and "error" in tool_response else "ok"
    LOGGER.info("Tool end: %s status=%s", getattr(tool, "name", tool), status)
    return None


def before_model_logger(callback_context: Any, llm_request: Any) -> None:
    """Log model calls and leave policy enforcement to deterministic tools."""
    LOGGER.info("Model call: agent=%s", getattr(callback_context, "agent_name", "unknown"))
    return None


def report_grounding_audit(callback_context: Any) -> None:
    """Runtime grounding guardrail (non-blocking): after the report is written, log how many of its
    statistic-figures trace to a Fact Sheet value and WARN on any that do not — a possible fabricated
    number. Observability, not enforcement: it never edits or blocks the report, and never raises.
    Shares its grounding logic with the offline audit via ``codas_agents.grounding``."""
    try:
        state = callback_context.state
        report = str(state.get("report") or "")
        if not report:
            return None
        engine_vals = engine_numbers(
            state.get("fact_sheet"),
            (state.get("latest_report", {}) or {}).get("candidates"),
            state.get("rounds"),
        )
        ungrounded, total = ungrounded_claims(report, engine_vals)
        if total == 0:
            return None
        LOGGER.info("grounding: %d/%d report statistic-figures verified against the Fact Sheet",
                    total - len(ungrounded), total)
        if ungrounded:
            LOGGER.warning("grounding: %d unverified figure(s) in the report (possible fabrication): %s",
                           len(ungrounded), ungrounded[:6])
    except Exception as exc:  # never break a run on the guardrail
        LOGGER.debug("grounding audit skipped: %s", exc)
    return None
