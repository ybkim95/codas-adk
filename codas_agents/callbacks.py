"""ADK guardrail and observability callbacks for CoDaS agents."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any


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


_STAT_CLAIM = re.compile(
    r"(auc|r2|r²|rho|ρ|spearman|p-value|p value|\bp\b|correlation|coefficient|variance|q=)"
    r"[^\d\n]{0,18}([-+]?\d*\.?\d+)", re.I)


def _engine_numbers(state: Any) -> set[float]:
    """Numbers the deterministic engine produced (+ legitimate derivations), rounded for matching."""
    vals: set[float] = set()

    def add(v: Any) -> None:
        if isinstance(v, (int, float)) and v == v:
            for k in (2, 3, 4):
                vals.add(round(float(v), k))

    fs = state.get("fact_sheet", {}) or {}
    for key in ("ml_metric_value", "rows", "columns", "candidate_features_screened",
                "internal_battery_passing_variants", "ml_metric_vs_null_p", "ml_pr_auc", "positive_rate"):
        add(fs.get(key))
    if isinstance(fs.get("ml_metric_value"), (int, float)):
        add(fs["ml_metric_value"] * 100)
    for c in (state.get("latest_report", {}) or {}).get("candidates", []) or []:
        for key in ("rho", "q_value", "p_value", "n"):
            add(c.get(key))
        if isinstance(c.get("rho"), (int, float)):
            add(c["rho"] ** 2); add(c["rho"] ** 2 * 100); add(abs(c["rho"]))
    return vals


def report_grounding_audit(callback_context: Any) -> None:
    """Runtime grounding guardrail (non-blocking): after the report is written, log how many of its
    statistic-figures trace to a Fact Sheet value and WARN on any that do not — a possible fabricated
    number. Observability, not enforcement: it never edits or blocks the report, and never raises."""
    try:
        state = callback_context.state
        report = str(state.get("report") or "")
        if not report:
            return None
        engine_vals = _engine_numbers(state)
        claims = [(m.group(1), float(m.group(2))) for m in _STAT_CLAIM.finditer(report)]
        if not claims:
            return None
        ungrounded = [
            (kw, v) for kw, v in claims
            if not any(abs(v - e) <= 0.011 for e in engine_vals)
            and not (1900 < v < 2100) and v not in (0, 1, 2, 3, 4, 5, 6, 10, 90, 95, 100)
        ]
        LOGGER.info("grounding: %d/%d report statistic-figures verified against the Fact Sheet",
                    len(claims) - len(ungrounded), len(claims))
        if ungrounded:
            LOGGER.warning("grounding: %d unverified figure(s) in the report (possible fabrication): %s",
                           len(ungrounded), ungrounded[:6])
    except Exception as exc:  # never break a run on the guardrail
        LOGGER.debug("grounding audit skipped: %s", exc)
    return None
