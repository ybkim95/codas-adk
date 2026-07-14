"""Deterministic tools for the CoDaS ADK agents.

Every reportable number is produced here, by the deterministic Python engine in ``codas_core``.
The agents may plan, profile, choose the target/roles, interpret, debate, and decide when to stop,
but they never invent statistics: a statistic exists only if one of these tools computed it.

The tools fall into two groups:

* **stateless** (``profile_dataset``, ``preview_columns``, ``run_discovery``) — pure functions of their
  arguments, used for one-shot profiling/discovery and unit-tested offline.
* **stateful** (``set_target``, ``run_discovery_round``, ``check_convergence``) — they read and write the
  shared session memory (``tool_context.state``) so the orchestrator's iterative loop has a single,
  auditable source of truth. ``run_discovery_round`` deepens the feature-engineering budget each round
  and appends a compact summary to ``state['rounds']``; ``check_convergence`` is the GapChecker — it
  compares the last two rounds and signals the LoopAgent to stop (``actions.escalate``) once an extra
  round of deeper search stops paying off.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from google.adk.tools import ToolContext

from codas_core import gemini
from codas_core.data import InsufficientDataError, profile_dataframe, read_csv_dataset
from codas_core.discovery import DiscoveryRequest, run_discovery_from_csv
from codas_core.statistics import safe_spearman

_PROPOSE_OPS = ("ratio", "product", "difference", "sum")


def _json_safe(obj: Any) -> Any:
    """Make a tool result strictly JSON-serializable for the ADK -> Gemini boundary.

    Real datasets carry missing values, so profiles and previews contain NaN — and NaN/Infinity are
    not valid JSON, so the model API rejects the tool response with a 400. Recursively map non-finite
    floats to None and numpy scalars / exotic types (Timestamps, etc.) to natives or strings.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):  # includes numpy.float64 (a float subclass)
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar -> python native, then re-check finiteness
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return str(obj)

# Per-round deepening schedule for the discovery loop. The GapChecker (``check_convergence``) decides,
# it decides "whether to pursue deeper feature engineering or move to validation"; concretely each
# extra round widens the engineered-ratio-feature budget and the reported candidate count, so a later
# round can surface structure a shallower one missed. Bounded so the loop stays within the timeout.
_RATIO_FEATURES_BASE = int(os.getenv("CODAS_ROUND_RATIO_BASE", "12"))
_RATIO_FEATURES_STEP = int(os.getenv("CODAS_ROUND_RATIO_STEP", "12"))
_RATIO_FEATURES_MAX = int(os.getenv("CODAS_ROUND_RATIO_MAX", "48"))
_TOPK_BASE = int(os.getenv("CODAS_ROUND_TOPK_BASE", "10"))
_TOPK_STEP = int(os.getenv("CODAS_ROUND_TOPK_STEP", "5"))
_ROUND_RESAMPLES = int(os.getenv("CODAS_ROUND_RESAMPLES", "300"))
# GapChecker thresholds: a round "pays off" only if it validates a new candidate OR lifts the held-out
# model metric by at least this much. Below that, deeper search is judged to have saturated.
_MIN_METRIC_GAIN = float(os.getenv("CODAS_CONVERGENCE_MIN_METRIC_GAIN", "0.01"))


def _resolve_csv_path(state_path: Any, arg_path: str | None) -> Path | None:
    raw = (arg_path or "").strip() or (str(state_path).strip() if state_path else "")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


# --- stateless tools ---------------------------------------------------------------------------

def profile_dataset(csv_path: str = "", target_column: str = "", tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Profile the dataset: schema, dtypes, missingness, row count, and the numeric columns (the
    candidate targets). The dataset path lives in shared memory, so call this with no arguments;
    pass ``csv_path`` only to profile a different file. Makes no name-based assumption about the
    outcome."""
    state_path = tool_context.state.get("csv_path") if tool_context is not None else None
    path = _resolve_csv_path(state_path, csv_path)
    if path is None:
        return {"error": "No dataset path in memory; seed state['csv_path'] or pass csv_path."}
    df = read_csv_dataset(path)
    return _json_safe(profile_dataframe(df, target_column=(target_column or None)).to_dict())


def preview_columns(csv_path: str = "", limit: int = 20, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Return a small preview of column names, dtypes, and the first few rows for orientation. The
    dataset path lives in shared memory, so call this with no path; pass ``csv_path`` only to preview
    a different file."""
    state_path = tool_context.state.get("csv_path") if tool_context is not None else None
    path = _resolve_csv_path(state_path, csv_path)
    if path is None:
        return {"error": "No dataset path in memory; seed state['csv_path'] or pass csv_path."}
    df = pd.read_csv(path, nrows=max(1, min(limit, 200)))
    return _json_safe({
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "preview_rows": df.head(3).to_dict(orient="records"),
    })


def run_discovery(
    csv_path: str,
    target_column: str,
    participant_id_column: str | None = None,
    time_column: str | None = None,
    excluded_columns_csv: str = "",
    confounder_columns_csv: str = "",
    top_k: int = 15,
) -> dict[str, Any]:
    """Run one deterministic association-discovery pass for an explicit target column.

    Roles are taken as given: the engine infers nothing from column names. Pass comma-separated
    column names for excluded/confounder columns; leave participant/time empty if not applicable.
    """
    request = DiscoveryRequest(
        target_column=target_column,
        participant_id_column=participant_id_column,
        time_column=time_column,
        excluded_columns=_split_csv(excluded_columns_csv),
        confounder_columns=_split_csv(confounder_columns_csv),
        top_k=top_k,
        validation_resamples=int(os.getenv("CODAS_VALIDATION_RESAMPLES", "1000")),
    )
    return _json_safe(run_discovery_from_csv(Path(csv_path).expanduser().resolve(), request).to_dict())


def search_literature(query: str, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Retrieve grounded scientific-literature context for a biomarker query (Phase A biological
    anchoring / Phase D deep research). Runs a Google-search-grounded Gemini call and returns a short
    evidence summary with its source list, so mechanism/novelty claims can be anchored in published
    work rather than the model's parametric memory.

    Returns ``{"grounded": bool, "summary": str, "sources": [...], "queries": [...]}``. When no Gemini
    key is configured the tool returns ``grounded=False`` with an empty summary — the caller must then
    reason from general principles and MUST NOT fabricate citations. Statistics are never sourced here;
    every reported number still comes only from the deterministic engine.
    """
    if not query or not query.strip():
        return {"grounded": False, "summary": "", "sources": [], "queries": [], "note": "empty query"}
    reply = gemini.generate_grounded_reply(
        query.strip(),
        context=("You are grounding a wearable-biomarker discovery finding in the scientific literature. "
                 "Summarise established evidence and mechanisms with citations; do not invent numeric "
                 "statistics — those come only from the deterministic engine."),
    )
    if not reply.configured:
        return {"grounded": False, "summary": "", "sources": [], "queries": [],
                "note": "Literature grounding unavailable (no Gemini key). Reason from general principles "
                        "and do not fabricate citations."}
    if reply.error or not reply.text:
        return {"grounded": False, "summary": "", "sources": [], "queries": [],
                "note": f"Literature grounding failed: {reply.error or 'empty response'}. Do not fabricate citations."}
    return _json_safe({
        "grounded": True,
        "summary": reply.text,
        "sources": reply.sources,
        "queries": reply.queries,
    })


# --- stateful tools (shared session memory) ----------------------------------------------------

def set_target(
    target_column: str,
    tool_context: ToolContext,
    participant_id_column: str = "",
    time_column: str = "",
    excluded_columns_csv: str = "",
    confounder_columns_csv: str = "",
) -> dict[str, Any]:
    """Record the chosen target column and participant/time/excluded/confounder roles in shared
    memory so every later round and agent uses the same analysis design. Call this once, after
    profiling, before running any discovery round. Leave a role empty if it does not apply."""
    tool_context.state["target_column"] = target_column
    tool_context.state["participant_id_column"] = participant_id_column.strip()
    tool_context.state["time_column"] = time_column.strip()
    tool_context.state["excluded_columns"] = _split_csv(excluded_columns_csv)
    tool_context.state["confounder_columns"] = _split_csv(confounder_columns_csv)
    return {
        "target_column": target_column,
        "participant_id_column": participant_id_column.strip() or None,
        "time_column": time_column.strip() or None,
        "excluded_columns": _split_csv(excluded_columns_csv),
        "confounder_columns": _split_csv(confounder_columns_csv),
        "note": "Analysis design recorded in shared memory. Run the first discovery round next.",
    }


def propose_feature(
    operation: str,
    feature_a: str,
    feature_b: str,
    tool_context: ToolContext,
    name: str = "",
) -> dict[str, Any]:
    """Propose a physiologically-motivated transformation of two existing features for the engine to
    evaluate (generative interpreters propose transformations which deterministic
    runners immediately evaluate — e.g. a steps/resting-heart-rate fitness index).

    ``operation`` is one of ratio, product, difference, sum, applied to two existing numeric columns
    (``feature_a``, ``feature_b``). There is no free-form expression evaluation. The proposal is
    registered so the NEXT discovery round screens and validates it with the SAME FDR correction and
    validation battery as every other feature — a proposal is evaluated, never trusted. An exploratory
    Spearman association is returned for immediate feedback; the authoritative verdict is the feature's
    entry in the validated candidate list, not this preview number.
    """
    op = (operation or "").strip().lower()
    if op not in _PROPOSE_OPS:
        return {"error": f"unsupported operation {operation!r}; use one of: {', '.join(_PROPOSE_OPS)}."}
    if not (feature_a or "").strip() or not (feature_b or "").strip():
        return {"error": "propose_feature needs two existing column names (feature_a, feature_b)."}
    label = (name or f"{feature_a}_{op}_{feature_b}").strip()
    proposals = list(tool_context.state.get("proposed_features") or [])
    if not any(p.get("name") == label for p in proposals):
        proposals.append({"op": op, "a": feature_a, "b": feature_b, "name": label})
    tool_context.state["proposed_features"] = proposals

    exploratory: dict[str, Any] | None = None
    path = _resolve_csv_path(tool_context.state.get("csv_path"), "")
    target = str(tool_context.state.get("target_column") or "").strip()
    if path is not None and target:
        try:
            df = read_csv_dataset(path)
            if all(c in df.columns for c in (feature_a, feature_b, target)):
                a = pd.to_numeric(df[feature_a], errors="coerce")
                b = pd.to_numeric(df[feature_b], errors="coerce")
                col = {"ratio": a / b.replace(0.0, math.nan), "product": a * b,
                       "difference": a - b, "sum": a + b}[op]
                rho, _, n = safe_spearman(col, df[target])
                if math.isfinite(rho):
                    exploratory = {"spearman_rho": round(float(rho), 4), "n": int(n)}
        except Exception:
            exploratory = None

    return _json_safe({
        "registered": label,
        "operation": op,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "exploratory_association": exploratory,
        "note": "Registered. The next discovery round screens and validates this feature with the full "
                "FDR correction and validation battery; the exploratory rho is a preview, not the verdict.",
        "proposed_features": [p["name"] for p in proposals],
    })


def _round_summary(round_index: int, ratio_features: int, top_k: int, report: dict[str, Any]) -> dict[str, Any]:
    fact_sheet = report.get("fact_sheet", {})
    passing = [
        {
            "feature": c.get("feature"),
            "rho": round(float(c["rho"]), 4) if isinstance(c.get("rho"), (int, float)) else None,
            "verdict": c.get("verdict"),
        }
        for c in report.get("candidates", [])
        if c.get("verdict") in {"validated", "conditional"}
    ]
    return {
        "round": round_index,
        "ratio_feature_budget": ratio_features,
        "top_k": top_k,
        "validated_count": int(fact_sheet.get("internal_battery_passing_variants") or 0),
        "evaluated_count": int(fact_sheet.get("internal_battery_evaluated_variants") or 0),
        "ml_metric_name": fact_sheet.get("ml_metric_name"),
        "ml_metric_value": fact_sheet.get("ml_metric_value"),
        "ml_above_chance": fact_sheet.get("ml_above_chance"),
        "passing_candidates": passing[:10],
        "warning_count": len(report.get("warnings", [])),
    }


def run_discovery_round(tool_context: ToolContext, csv_path: str = "", target_column: str = "") -> dict[str, Any]:
    """Run the next deterministic discovery round on the shared dataset and record it in memory.

    Reads the dataset path and the analysis design (target/roles) from shared memory — call
    ``set_target`` first. The round index is the number of rounds already completed; each successive
    round widens the engineered-feature budget and the candidate count, so the search deepens over the
    loop. A compact summary (validated count, held-out model metric, surviving candidates) is appended
    to ``state['rounds']`` for the interpreters, the critic/defender, and the GapChecker to reason over.
    Pass ``target_column`` only to override what ``set_target`` recorded.
    """
    path = _resolve_csv_path(tool_context.state.get("csv_path"), csv_path)
    if path is None:
        return {"error": "No dataset path in memory; seed state['csv_path'] or pass csv_path."}
    target = (target_column or "").strip() or str(tool_context.state.get("target_column") or "").strip()
    if not target:
        return {"error": "No target column set. Call set_target(target_column=...) first."}

    rounds: list[dict[str, Any]] = list(tool_context.state.get("rounds") or [])
    round_index = len(rounds)
    ratio_features = min(_RATIO_FEATURES_BASE + _RATIO_FEATURES_STEP * round_index, _RATIO_FEATURES_MAX)
    top_k = _TOPK_BASE + _TOPK_STEP * round_index

    request = DiscoveryRequest(
        target_column=target,
        participant_id_column=str(tool_context.state.get("participant_id_column") or "").strip() or None,
        time_column=str(tool_context.state.get("time_column") or "").strip() or None,
        excluded_columns=list(tool_context.state.get("excluded_columns") or []),
        confounder_columns=list(tool_context.state.get("confounder_columns") or []),
        proposed_features=list(tool_context.state.get("proposed_features") or []),
        top_k=top_k,
        max_ratio_features=ratio_features,
        validation_resamples=_ROUND_RESAMPLES,
    )
    # A bad target/role choice makes the engine raise rather than crash the loop; return a recoverable
    # error so the agent can revise the design and retry.
    try:
        report = _json_safe(run_discovery_from_csv(path, request).to_dict())
    except InsufficientDataError as exc:
        return {"error": f"Discovery could not run for target '{target}': {exc}", "round": round_index}
    except Exception as exc:  # noqa: BLE001 - surface any engine failure to the agent, never abort the run
        return {"error": f"Discovery failed for target '{target}': {type(exc).__name__}: {exc}", "round": round_index}

    summary = _round_summary(round_index, ratio_features, top_k, report)
    tool_context.state["rounds"] = rounds + [summary]
    tool_context.state["fact_sheet"] = report.get("fact_sheet", {})
    tool_context.state["latest_report"] = report
    return summary


def check_convergence(tool_context: ToolContext) -> dict[str, Any]:
    """GapChecker: decide whether the iterative search has saturated and the loop should stop.

    Compares the two most recent rounds in shared memory. The loop should continue while a deeper
    round still pays off — it validates a new candidate or lifts the held-out model metric by a
    meaningful margin. Once an extra round adds no validated candidate and no metric gain (or the
    dataset shows no signal at all), the search has converged: this sets ``actions.escalate`` so the
    LoopAgent exits and the pipeline moves on to mechanism, novelty, and reporting.
    """
    rounds: list[dict[str, Any]] = list(tool_context.state.get("rounds") or [])
    if not rounds:
        return {"converged": False, "reason": "No discovery round has run yet.", "rounds_completed": 0}

    current = rounds[-1]
    # A clear null after the first round: nothing validated and the held-out model is not above its
    # permutation null. Deeper feature engineering cannot manufacture signal that is not there, so stop.
    if len(rounds) == 1:
        if current["validated_count"] == 0 and not current.get("ml_above_chance"):
            tool_context.actions.escalate = True
            return {
                "converged": True,
                "reason": "No validated candidate and the model is not above chance after the first "
                "round; the dataset shows no detectable signal. Stopping.",
                "rounds_completed": 1,
            }
        return {
            "converged": False,
            "reason": "Signal present after the first round; run one deeper round to test whether it "
            "strengthens or saturates.",
            "rounds_completed": 1,
        }

    previous = rounds[-2]
    gain_validated = current["validated_count"] - previous["validated_count"]
    prev_metric = previous.get("ml_metric_value")
    curr_metric = current.get("ml_metric_value")
    same_metric = previous.get("ml_metric_name") == current.get("ml_metric_name")
    gain_metric = (
        float(curr_metric) - float(prev_metric)
        if same_metric and curr_metric is not None and prev_metric is not None
        else 0.0
    )
    converged = gain_validated <= 0 and gain_metric < _MIN_METRIC_GAIN
    if converged:
        tool_context.actions.escalate = True
        reason = (
            f"Deeper search saturated: no new validated candidate (Δ={gain_validated}) and the model "
            f"metric gain ({gain_metric:+.3f}) is below {_MIN_METRIC_GAIN}. Stopping the loop."
        )
    else:
        reason = (
            f"Deeper search still paying off (validated Δ={gain_validated}, metric Δ={gain_metric:+.3f}); "
            f"continue iterating."
        )
    return {
        "converged": converged,
        "reason": reason,
        "gain_validated": gain_validated,
        "gain_metric": round(gain_metric, 4),
        "rounds_completed": len(rounds),
    }
