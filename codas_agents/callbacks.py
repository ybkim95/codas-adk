"""ADK guardrail and observability callbacks for CoDaS agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("codas.agents")
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False
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
