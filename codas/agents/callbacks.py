"""ADK guardrail and observability callbacks for CoDaS agents."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from codas.agents.grounding import engine_numbers, ungrounded_claims
from codas.agents.numeric_audit import verify_and_correct


LOGGER = logging.getLogger("codas.agents")
# Library logging pattern: a NullHandler so importing never warns, but propagation LEFT ON so the host
# app's logging config (uvicorn / Cloud Run capture root at INFO) actually surfaces the agent trace —
# model calls, tool start/end, and the grounding guardrail. (Previously propagate=False made all of
# this unreachable in production unless a handler was attached to "codas.agents" by name.)
LOGGER.addHandler(logging.NullHandler())
ROOT = Path(__file__).resolve().parents[2]  # repo root (this file is codas/agents/callbacks.py)
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


def _write_numeric_audit(payload: dict[str, Any]) -> None:
    """Persist the numeric verification result to a per-run audit file. The
    directory is CODAS_AUDIT_DIR or ``<repo>/.codas_runs``; failures are swallowed (best-effort)."""
    try:
        audit_dir = Path(os.getenv("CODAS_AUDIT_DIR", str(ROOT / ".codas_runs")))
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / f"numeric_audit_{uuid.uuid4().hex[:12]}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        LOGGER.info("numeric audit written: %s", path)
    except Exception as exc:  # never break a run on the audit file
        LOGGER.debug("numeric audit file skipped: %s", exc)


def report_grounding_audit(callback_context: Any) -> None:
    """Numeric verification & grounding guardrail. After the report is written:
    (1) correct count near-misses (sample sizes, feature/candidate/validation-check counts) toward the
    Fact Sheet ground truth; (2) verify remaining statistic-figures trace to an engine value and warn on
    any that do not (a possible fabrication); (3) log both to a per-run audit file. Corrections are
    conservative and count-only; it never blocks a run and never raises."""
    try:
        state = callback_context.state
        report = str(state.get("report") or "")
        if not report:
            return None
        fact_sheet = state.get("fact_sheet")

        corrected, corrections = verify_and_correct(report, fact_sheet)
        if corrections:
            state["report"] = corrected  # apply the fix so downstream consumers see corrected prose
            report = corrected
            LOGGER.info("numeric verification corrected %d count(s): %s", len(corrections), corrections[:6])

        engine_vals = engine_numbers(
            fact_sheet,
            (state.get("latest_report", {}) or {}).get("candidates"),
            state.get("rounds"),
        )
        ungrounded, total = ungrounded_claims(report, engine_vals)
        if total:
            LOGGER.info("grounding: %d/%d report statistic-figures verified against the Fact Sheet",
                        total - len(ungrounded), total)
            if ungrounded:
                LOGGER.warning("grounding: %d unverified figure(s) in the report (possible fabrication): %s",
                               len(ungrounded), ungrounded[:6])
                # Surface the unverified figures IN the report, not only in a log the reader may never
                # see, so a human reviewer cannot miss a statistic that did not come from the engine.
                flagged = ", ".join(f"{kw} {val:g}" for kw, val in ungrounded[:8])
                report = report + (
                    "\n\n---\n**Numeric verification.** These figures could not be traced to a value the "
                    f"deterministic engine produced and must be checked before use: {flagged}. Every "
                    "reportable statistic should come from the Fact Sheet; the engine does not compute "
                    "quantities such as hazard/odds ratios, sensitivity, or specificity."
                )
                state["report"] = report

        if corrections or total:
            _write_numeric_audit({
                "corrections": corrections,
                "grounding": {"verified": total - len(ungrounded), "total": total,
                              "ungrounded": [{"keyword": k, "value": v} for k, v in ungrounded[:20]]},
            })
    except Exception as exc:  # never break a run on the guardrail
        LOGGER.debug("grounding audit skipped: %s", exc)
    return None
