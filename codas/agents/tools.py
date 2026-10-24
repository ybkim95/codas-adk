"""Deterministic tools for the CoDaS ADK agents.

Every reportable number is produced here, by the deterministic Python engine in ``codas.core``.
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

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from google.adk.tools import ToolContext

from codas.aou.privacy import MASK, SMALL_CELL_MAX, aou_mode, literature_search_enabled, mask_small_counts_in_text
from codas.core import gemini
from codas.core.data import InsufficientDataError, profile_dataframe, read_csv_dataset
from codas.core.discovery import DiscoveryRequest, run_discovery_from_csv
from codas.core.statistics import safe_spearman

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
# The battery is specified at 1,000 resamples, so a round runs at that depth by default rather than
# reporting a verdict earned at a third of it. Lower it via the env var when iterating locally.
_ROUND_RESAMPLES = int(os.getenv("CODAS_ROUND_RESAMPLES", "1000"))
# GapChecker thresholds: a round "pays off" only if it validates a new candidate OR lifts the held-out
# model metric by at least this much. Below that, deeper search is judged to have saturated.
_MIN_METRIC_GAIN = float(os.getenv("CODAS_CONVERGENCE_MIN_METRIC_GAIN", "0.01"))


def registration_open(state: Any) -> bool:
    """True until the first discovery round runs: hypotheses registered now are a priori, so no tool may
    return anything computed from the outcome (its association with a feature, or with an outcome proxy).
    The information barrier is enforced here, in the tools, not left to the prompts."""
    return not (state.get("rounds") if state is not None else None)


def _resolve_csv_path(state_path: Any, arg_path: str | None) -> Path | None:
    # In All of Us mode the dataset is fixed by the runner: a path an agent supplies is ignored, so no tool
    # call can reach the sealed share or a cache that holds every participant.
    raw = "" if aou_mode() else (arg_path or "").strip()
    raw = raw or (str(state_path).strip() if state_path else "")
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
    preview = {
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }
    if aou_mode():  # All of Us: no participant row ever reaches the model
        preview["note"] = "Row preview withheld (All of Us mode); use profile_dataset for aggregates."
    else:
        preview["preview_rows"] = df.head(3).to_dict(orient="records")
    return _json_safe(preview)


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


# Model calls a tool makes on its own, outside its agent's model (the literature search calls Gemini with
# Google Search), keyed by the tool call's id. The execution report takes each entry when the call ends and
# counts it for the agent that called the tool; at most _MAX_USAGE untaken entries are kept.
MODEL_USAGE: dict[str, list[dict[str, Any]]] = {}
_MAX_USAGE = 1000


def _record_model_usage(tool_context: Any, usage: dict[str, Any] | None) -> None:
    call_id = getattr(tool_context, "function_call_id", None)
    if not call_id or not usage:
        return
    MODEL_USAGE.setdefault(call_id, []).append(dict(usage))
    while len(MODEL_USAGE) > _MAX_USAGE:
        MODEL_USAGE.pop(next(iter(MODEL_USAGE)))


def take_model_usage(call_id: str | None) -> list[dict[str, Any]]:
    """The model calls the tool call ``call_id`` made on its own, removed from the record."""
    return MODEL_USAGE.pop(call_id, []) if call_id else []


def search_literature(query: str, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Retrieve grounded scientific-literature context for a biomarker query (Phase A biological
    anchoring / Phase D deep research). Runs a Google-search-grounded Gemini call and returns a short
    evidence summary with its source list, so mechanism/novelty claims can be anchored in published
    work rather than the model's parametric memory.

    Returns ``{"grounded": bool, "summary": str, "sources": [...], "queries": [...]}``. When no Gemini
    key is configured the tool returns ``grounded=False`` with an empty summary — the caller must then
    reason from general principles and MUST NOT fabricate citations. Statistics are never sourced here;
    every reported number still comes only from the deterministic engine.

    Every query goes to the search again: Google's grounding terms forbid caching or reusing its results.
    """
    if not query or not query.strip():
        return {"grounded": False, "summary": "", "sources": [], "queries": [], "note": "empty query"}
    if aou_mode():
        if not literature_search_enabled():
            return {"grounded": False, "summary": "", "sources": [], "queries": [],
                    "note": "Literature search is turned off for this All of Us run. Reason from general "
                            "principles and do not fabricate citations."}
        query = mask_small_counts_in_text(query)
    reply = gemini.generate_grounded_reply(
        query.strip(),
        context=("You are grounding a wearable-biomarker discovery finding in the scientific literature. "
                 "Summarise established evidence and mechanisms with citations; do not invent numeric "
                 "statistics — those come only from the deterministic engine."),
    )
    _record_model_usage(tool_context, getattr(reply, "usage", None))
    if not reply.configured:
        return {"grounded": False, "summary": "", "sources": [], "queries": [],
                "note": "Literature grounding unavailable (no Gemini key). Reason from general principles "
                        "and do not fabricate citations."}
    if reply.error or not reply.text:
        return {"grounded": False, "summary": "", "sources": [], "queries": [],
                "note": f"Literature grounding failed: {reply.error or 'empty response'}. Do not fabricate citations."}
    # `grounded` has to mean the search actually returned something, not merely that the model
    # replied. A search-grounded call can come back fluent and confident with an empty source list,
    # which is parametric recall wearing the costume of retrieval; reporting that as grounded is how
    # an unsourced claim ends up in a report that says it was literature-anchored.
    grounded = bool(reply.sources)
    return _json_safe({
        "grounded": grounded,
        "summary": reply.text,
        "sources": reply.sources,
        "queries": reply.queries,
        "note": None if grounded else
        "The search returned no sources, so this summary is the model's own recall, not retrieved "
        "literature. Treat it as a prior to be checked, attach no citations to it, and say in the "
        "report that literature grounding was unavailable.",
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
    path = _resolve_csv_path(tool_context.state.get("csv_path"), "")
    if path is not None and path.exists():  # names only, for agents that no longer see the profile
        tool_context.state["dataset_columns"] = list(pd.read_csv(path, nrows=0).columns)
    return {
        "target_column": target_column,
        "participant_id_column": participant_id_column.strip() or None,
        "time_column": time_column.strip() or None,
        "excluded_columns": _split_csv(excluded_columns_csv),
        "confounder_columns": _split_csv(confounder_columns_csv),
        "note": "Analysis design recorded in shared memory. Run the first discovery round next.",
    }


def declare_confounders(confounder_columns_csv: str, tool_context: ToolContext, reason: str = "") -> dict[str, Any]:
    """Add confounders to the recorded analysis design, without disturbing the rest of it.

    A confounder is a column that plausibly causes BOTH a candidate feature and the outcome, so a
    feature that merely tracks it looks predictive while explaining nothing. Naming one here makes
    every later round residualise candidates against it before judging them, which is what separates
    a genuine predictor from an artefact of the confounder.

    Recognising a confounder is a judgement about the world, not about the schema, so it belongs to
    the agent that reasons over domain knowledge rather than to the Scout, which is deliberately
    restricted to what the profile shows. Unlike ``set_target`` this merges: it keeps the target and
    the participant/time/excluded roles already recorded, and adds to any confounders already there.

    Names that are not columns, or that are already serving as the target or the participant/time
    column, are ignored and reported back rather than silently dropped. ``reason`` (why each is a
    common cause of the features and the outcome) is recorded in the round summaries the report reads.
    Adding a confounder revises the design, so the loop runs at least one more round with it.
    """
    if not str(tool_context.state.get("target_column") or "").strip():
        return {"error": "Call set_target first; there is no analysis design to add confounders to."}

    requested = _split_csv(confounder_columns_csv)
    if not requested:
        return {"error": "declare_confounders needs at least one column name."}

    reserved = {
        str(tool_context.state.get("target_column") or "").strip(),
        str(tool_context.state.get("participant_id_column") or "").strip(),
        str(tool_context.state.get("time_column") or "").strip(),
    }
    reserved.discard("")

    columns: set[str] | None = None
    path = _resolve_csv_path(tool_context.state.get("csv_path"), "")
    if path is not None:
        try:
            columns = {str(c) for c in read_csv_dataset(path).columns}
        except Exception:
            columns = None  # fall through: record the names rather than lose a correct one

    accepted = list(tool_context.state.get("confounder_columns") or [])
    ignored: list[str] = []
    added: list[str] = []
    for name in requested:
        if name in accepted:
            continue
        if name in reserved or (columns is not None and name not in columns):
            ignored.append(name)
            continue
        accepted.append(name)
        added.append(name)
    tool_context.state["confounder_columns"] = accepted
    if added and tool_context.state.get("rounds"):
        # Rounds already ran without these; the loop must test the adjusted design before stopping.
        tool_context.state["design_revision"] = int(tool_context.state.get("design_revision") or 0) + 1
    if added:
        log = list(tool_context.state.get("confounder_log") or [])
        log.append({"columns": added, "reason": (reason or "").strip(),
                    "after_round": len(tool_context.state.get("rounds") or [])})
        tool_context.state["confounder_log"] = log

    return {
        "confounder_columns": accepted,
        "ignored": ignored,
        "note": "Recorded. Every later discovery round residualises candidates against these before "
                "assigning a verdict; a candidate that survives only because it tracks a confounder "
                "will now be rejected.",
    }


def exclude_features(columns_csv: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
    """Remove features that leak the target from every later discovery round.

    Use this only for target leakage: a column that is not an independent measurement but is computed
    from, or encodes, the outcome itself (for example a derived quantity whose formula takes the
    outcome, or part of it, as an input, or a value recorded after the outcome). A merely confounded
    feature is not leakage; declare the confounder with declare_confounders instead. Weak features are
    not leakage either.

    The excluded columns are dropped before feature engineering, so ratios and products built from
    them disappear too, and any proposed feature that uses them is withdrawn. The exclusion and its
    reason are recorded in shared memory and in every later round summary, so the report can state
    what was removed and why. Names that are not columns, or that are the target or the participant/time
    column, are ignored and reported back.
    """
    requested = _split_csv(columns_csv)
    if not requested:
        return {"error": "exclude_features needs at least one column name."}
    if not (reason or "").strip():
        return {"error": "exclude_features needs a reason: say how the column encodes the outcome."}

    reserved = {
        str(tool_context.state.get(key) or "").strip()
        for key in ("target_column", "participant_id_column", "time_column")
    }
    reserved.discard("")
    columns: set[str] | None = None
    path = _resolve_csv_path(tool_context.state.get("csv_path"), "")
    if path is not None:
        try:
            columns = {str(c) for c in read_csv_dataset(path, nrows=5).columns}
        except Exception:
            columns = None

    excluded = list(tool_context.state.get("excluded_columns") or [])
    accepted, ignored = [], []
    for name in requested:
        if name in reserved or (columns is not None and name not in columns):
            ignored.append(name)
        elif name not in excluded:
            excluded.append(name)
            accepted.append(name)
    if not accepted:
        return {"excluded_columns": excluded, "ignored": ignored, "note": "Nothing new to exclude."}

    proposals = list(tool_context.state.get("proposed_features") or [])
    withdrawn = [p["name"] for p in proposals if p.get("a") in accepted or p.get("b") in accepted]
    tool_context.state["proposed_features"] = [p for p in proposals if p["name"] not in withdrawn]
    tool_context.state["excluded_columns"] = excluded
    log = list(tool_context.state.get("exclusion_log") or [])
    log.append({"columns": accepted, "reason": reason.strip(), "after_round": len(tool_context.state.get("rounds") or [])})
    tool_context.state["exclusion_log"] = log
    tool_context.state["design_revision"] = int(tool_context.state.get("design_revision") or 0) + 1
    return {
        "excluded_columns": excluded,
        "newly_excluded": accepted,
        "withdrawn_proposals": withdrawn,
        "ignored": ignored,
        "note": "Recorded. The next discovery round drops these columns and everything engineered from "
                "them; the loop runs at least one more round on the revised design.",
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
    state = tool_context.state
    blocked = {str(state.get("target_column") or "").strip(),
               *(state.get("excluded_columns") or []), *(state.get("confounder_columns") or []),
               *(state.get("context_columns") or [])}
    blocked.discard("")
    used = [c for c in (feature_a.strip(), feature_b.strip()) if c in blocked]
    if used:
        return {"error": f"{', '.join(used)} cannot be a feature component: the design uses it as the target, "
                         "excludes it, or keeps it for adjustment only. Propose a transformation of other columns."}
    label = (name or f"{feature_a}_{op}_{feature_b}").strip()
    proposals = list(tool_context.state.get("proposed_features") or [])
    if not any(p.get("name") == label for p in proposals):
        proposals.append({"op": op, "a": feature_a, "b": feature_b, "name": label})
    tool_context.state["proposed_features"] = proposals

    exploratory: dict[str, Any] | None = None
    path = _resolve_csv_path(tool_context.state.get("csv_path"), "")
    target = str(tool_context.state.get("target_column") or "").strip()
    withheld = registration_open(tool_context.state)  # a priori: no association with the outcome yet
    if path is not None and target and not withheld:
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
        "note": ("Registered. Its association with the outcome is withheld until the first discovery round, so "
                 "hypotheses registered now stay a priori." if withheld else
                 "Registered. The next discovery round screens and validates this feature with the full "
                 "FDR correction and validation battery; the exploratory rho is a preview, not the verdict."),
        "proposed_features": [p["name"] for p in proposals],
    })


def _round_summary(round_index: int, ratio_features: int, top_k: int, report: dict[str, Any]) -> dict[str, Any]:
    fact_sheet = report.get("fact_sheet", {})
    def adjusted(c: dict[str, Any]) -> float | None:
        test = next((t for t in c.get("tests") or [] if t.get("name") == "confounder_adjusted_robustness"
                     and t.get("applicable", True)), None)
        value = test.get("metric") if test else None
        return round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(value) else None

    passing = [
        {
            "feature": c.get("feature"),
            "rho": round(float(c["rho"]), 4) if isinstance(c.get("rho"), (int, float)) else None,
            "verdict": c.get("verdict"),
            # What an analyst needs to weigh a candidate: its FDR q, its confounder-adjusted rho, and
            # the checks it failed.
            "q": float(f"{c['q_value']:.3g}") if isinstance(c.get("q_value"), (int, float)) else None,
            "adjusted_rho": adjusted(c),
            "failed_checks": [t.get("name") for t in c.get("tests") or []
                              if t.get("applicable", True) and not t.get("passed")],
        }
        for c in report.get("candidates", [])
        if c.get("verdict") in {"validated", "conditional"}
    ]
    warnings = [str(w)[:300] for w in report.get("warnings", [])[:8]]
    if aou_mode():
        warnings = [mask_small_counts_in_text(w) for w in warnings]
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
        "warnings": warnings,
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
        context_columns=list(tool_context.state.get("context_columns") or []),
        proposed_features=list(tool_context.state.get("proposed_features") or []),
        top_k=top_k,
        max_ratio_features=ratio_features,
        validation_resamples=_ROUND_RESAMPLES,
        covariate_imputations=int(tool_context.state.get("covariate_imputations") or 0),
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
    # The design this round ran on, so the GapChecker can tell a saturated search from a revised one.
    summary["design_revision"] = int(tool_context.state.get("design_revision") or 0)
    summary["excluded_features"] = list(tool_context.state.get("excluded_columns") or [])
    if tool_context.state.get("exclusion_log"):
        summary["exclusion_log"] = list(tool_context.state["exclusion_log"])
    summary["confounders"] = list(tool_context.state.get("confounder_columns") or [])
    if tool_context.state.get("confounder_log"):
        summary["confounder_log"] = list(tool_context.state["confounder_log"])
    context = list(tool_context.state.get("context_columns") or [])
    if context:
        summary["adjustment_covariates"] = _coverage(path, context)
        imputations = int(tool_context.state.get("covariate_imputations") or 0)
        summary["missing_covariate_handling"] = (
            f"multiply imputed ({imputations} draws, pooled)" if imputations > 0
            else "rows lacking a value are dropped from the adjusted tests")
    tool_context.state["rounds"] = rounds + [summary]
    tool_context.state["fact_sheet"] = report.get("fact_sheet", {})
    tool_context.state["latest_report"] = report
    if not rounds:  # registration has just closed: what earlier outcomes taught may now be shown
        tool_context.state["cohort_lessons"] = list(tool_context.state.get("prior_lessons") or [])
    return summary


def _coverage(path: Path, columns: list[str]) -> dict[str, Any]:
    """Share of rows with a value for each adjustment covariate: what an agent weighs before declaring
    one (rows lacking it are dropped from adjusted tests unless missing values are imputed)."""
    try:
        frame = read_csv_dataset(path)
    except Exception:  # noqa: BLE001 - coverage is advisory; never fail the round over it
        return {}
    out: dict[str, Any] = {}
    for c in [c for c in columns if c in frame.columns]:
        have = int(frame[c].notna().sum())
        lacking = len(frame) - have
        if aou_mode() and (1 <= have <= SMALL_CELL_MAX or 1 <= lacking <= SMALL_CELL_MAX):
            # A fraction next to the row count would reveal a small count; describe it instead.
            out[c] = {"non_missing_fraction": f"{MASK} rows have a value" if have <= SMALL_CELL_MAX
                      else f"{MASK} rows lack a value"}
        else:
            out[c] = {"non_missing_fraction": round(have / len(frame), 3) if len(frame) else None}
    return out


# --- a-priori hypotheses -----------------------------------------------------------------------

_KNOWNNESS = ("established", "emerging", "unexplored")
_MAX_A_PRIORI = int(os.getenv("CODAS_MAX_A_PRIORI_HYPOTHESES", "8"))


def register_hypotheses(hypotheses_json: str, tool_context: ToolContext) -> dict[str, Any]:
    """Register falsifiable hypotheses BEFORE any discovery statistic exists. ``hypotheses_json`` is a
    JSON list of objects with: ``feature`` (an existing column, or a feature you registered with
    propose_feature), ``direction`` (positive/negative or 1/-1: the association you predict),
    ``construct`` (what the feature measures), ``mechanism``, ``citation`` (from search_literature), and
    ``knownness``: established (the literature already reports it), emerging (indirect or preliminary
    evidence) or unexplored (no evidence found). Each is pre-registered and tested on the sealed holdout
    whatever discovery finds, in its own family, so a hypothesis fixed before the data could steer it
    does not pay for the exploratory search. Registration closes when the first discovery round runs."""
    state = tool_context.state
    if state.get("rounds"):
        return {"error": "Registration is closed: a discovery round has already run, so new hypotheses would "
                         "no longer be a priori."}
    try:
        items = json.loads(hypotheses_json)
    except (TypeError, ValueError) as exc:
        return {"error": f"hypotheses_json is not valid JSON: {exc}"}
    if not isinstance(items, list):
        return {"error": "hypotheses_json must be a JSON list of hypothesis objects."}
    path = _resolve_csv_path(state.get("csv_path"), "")
    columns = set(read_csv_dataset(path).columns) if path is not None else set()
    proposals = {p.get("name") for p in state.get("proposed_features") or []}
    blocked = {str(state.get("target_column") or ""), *(state.get("excluded_columns") or []),
               *(state.get("context_columns") or []), *(state.get("confounder_columns") or [])}
    registered = list(state.get("a_priori_hypotheses") or [])
    accepted, rejected = [], []
    for item in items:
        item = item if isinstance(item, dict) else {}
        feature = str(item.get("feature") or "").strip()
        raw = str(item.get("direction", "")).strip().lower()
        direction = 1 if raw in {"1", "+1", "positive", "+"} else -1 if raw in {"-1", "negative", "-"} else 0
        knownness = str(item.get("knownness") or "").strip().lower()
        problem = ("it is the target, excluded, or adjustment-only" if feature in blocked
                   else "not a column or a proposed feature" if feature not in columns | proposals
                   else "direction must be positive or negative" if not direction
                   else f"knownness must be one of {', '.join(_KNOWNNESS)}" if knownness not in _KNOWNNESS
                   else "already registered" if any(h["feature"] == feature for h in registered)
                   else f"at most {_MAX_A_PRIORI} hypotheses" if len(registered) >= _MAX_A_PRIORI else None)
        if problem:
            rejected.append({"feature": feature, "reason": problem})
            continue
        entry = {"feature": feature, "direction": direction, "knownness": knownness,
                 **{k: str(item.get(k) or "")[:300] for k in ("construct", "mechanism", "citation")}}
        registered.append(entry)
        accepted.append(entry)
    state["a_priori_hypotheses"] = registered
    return _json_safe({"registered": accepted, "rejected": rejected, "total_registered": len(registered),
                       "note": "Each registered hypothesis is tested once on the sealed holdout in the a-priori family."})


# --- analysis tools: deeper, deterministic analyses an agent chooses (codas.core.analyses) ---------

_MAX_ANALYSIS_FEATURES = 12


def _sig(obj: Any, digits: int = 4) -> Any:
    """Round every float to ``digits`` significant digits (p-values keep their magnitude)."""
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}") if math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sig(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sig(v, digits) for v in obj]
    return obj


class _Refused(ValueError):
    pass


def _analysis_setup(tool_context: ToolContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The shared dataset and design: target, follow-up time, confounders, what is blocked, and how the
    candidates and proposals are defined."""
    state = tool_context.state
    path = _resolve_csv_path(state.get("csv_path"), "")
    target = str(state.get("target_column") or "").strip()
    if path is None or not target:
        raise _Refused("No dataset or target in memory yet; the Scout sets them first.")
    frame = read_csv_dataset(path)
    time = str(state.get("survival_time_column") or "").strip() or None
    if time and time not in frame.columns:
        time = None
    excluded = set(state.get("excluded_columns") or [])
    design = {
        "target": target, "time": time, "imputations": int(state.get("covariate_imputations") or 0),
        "confounders": [c for c in state.get("confounder_columns") or [] if c in frame.columns and c != time],
        "excluded": excluded,
        "blocked_features": {target, *excluded, *(state.get("context_columns") or []),
                             *(state.get("confounder_columns") or []), *([time] if time else [])},
        "definitions": {**{c.get("feature"): {"components": list(c.get("components") or [])}
                           for c in (state.get("latest_report") or {}).get("candidates") or []},
                        **{p.get("name"): {"op": p.get("op"), "a": p.get("a"), "b": p.get("b")}
                           for p in state.get("proposed_features") or []}},
        "candidates": [c.get("feature") for c in (state.get("latest_report") or {}).get("candidates") or []
                       if c.get("verdict") in {"validated", "conditional"}],
    }
    return frame, design


def _with_features(frame: pd.DataFrame, design: dict[str, Any], names: list[str]) -> pd.DataFrame:
    from codas.core.analyses import rebuild_feature

    if not names:
        raise _Refused("Name at least one feature (a column, a proposed feature, or a candidate from the round).")
    out = frame.copy()
    for name in names[:_MAX_ANALYSIS_FEATURES]:
        if name in design["blocked_features"]:
            raise _Refused(f"{name} is the target, the follow-up time, an excluded column, or an adjustment-only "
                           "column, so it cannot be analysed as a candidate feature.")
        series = rebuild_feature(frame, name, design["definitions"].get(name))
        if series is None:
            raise _Refused(f"Unknown feature {name!r}: not a column, a proposed feature, or a candidate of the latest round.")
        out[name] = series
    return out


def _covariate(frame: pd.DataFrame, design: dict[str, Any], name: str) -> str:
    if name not in frame.columns or name in design["excluded"] or name in {design["target"], design["time"]}:
        raise _Refused(f"{name!r} is not an available covariate (it is missing, excluded, the target or the follow-up time).")
    return name


def _record(tool_context: ToolContext, tool: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    result = _json_safe(_sig(result))
    tool_context.state["analyses"] = list(tool_context.state.get("analyses") or []) + [
        {"tool": tool, "args": args, "result": result,
         "confounders": list(tool_context.state.get("confounder_columns") or [])}]
    tool_context.state["analyses_counts"] = analyses_counts(tool_context.state["analyses"])
    return result


def _earlier(state: Any, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """The recorded result of the same analysis (tool, arguments and confounders) in this outcome, if any."""
    key = {"tool": tool, "args": args, "confounders": sorted(state.get("confounder_columns") or [])}
    for record in reversed(list(state.get("analyses") or [])):
        if {"tool": record.get("tool"), "args": record.get("args"),
                "confounders": sorted(record.get("confounders") or [])} == key:
            return record.get("result")
    return None


def _run_analysis(tool_context: ToolContext, tool: str, args: dict[str, Any], body: Any) -> dict[str, Any]:
    earlier = _earlier(tool_context.state, tool, args)
    if earlier is not None:  # identical analysis: return it rather than spend minutes recomputing it
        return {**earlier, "note": "Already run in this analysis with the same arguments and confounders; this is "
                                   "that result. Run something new instead of repeating it."}
    try:
        frame, design = _analysis_setup(tool_context)
        return _record(tool_context, tool, args, body(frame, design))
    except _Refused as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - an analysis failure is reported to the agent, never fatal
        return {"error": f"{tool} failed: {type(exc).__name__}: {exc}"}


def run_survival_screen(tool_context: ToolContext, features_csv: str = "") -> dict[str, Any]:
    """Time-to-event screen: each feature's Cox hazard ratio per SD, adjusted for the declared confounders
    (missing confounder values imputed), with BH q-values across the listed features. Use it for an
    incident outcome to rank candidates the way the sealed holdout will test them (the discovery round
    treats the event as binary and ignores follow-up time). Defaults to the latest round's candidates."""
    def body(frame, design):
        if not design["time"]:
            raise _Refused("This outcome has no follow-up time; use the discovery round's rank statistics instead.")
        names = _split_csv(features_csv) or design["candidates"]
        data = _with_features(frame, design, names)
        from codas.core.analyses import survival_screen

        rows = survival_screen(data, names[:_MAX_ANALYSIS_FEATURES], design["target"], design["time"],
                               design["confounders"], design["imputations"])
        return {"adjusted_for": design["confounders"], "results": rows}
    return _run_analysis(tool_context, "survival_screen", {"features": features_csv}, body)


def run_subgroup_analysis(feature: str, by: str, tool_context: ToolContext) -> dict[str, Any]:
    """The feature's association within each level of ``by`` (tertiles when it is continuous, e.g. age),
    adjusted for the other confounders, with Cochran's Q and I^2 for heterogeneity. Use it to test
    whether an effect is specific to a group (sex, age band, a comorbidity) or consistent across them."""
    def body(frame, design):
        data = _with_features(frame, design, [feature.strip()])
        from codas.core.analyses import subgroup_analysis

        return subgroup_analysis(data, feature.strip(), design["target"], _covariate(frame, design, by.strip()),
                                 design["confounders"], design["time"], design["imputations"])
    return _run_analysis(tool_context, "subgroup_analysis", {"feature": feature, "by": by}, body)


def run_multivariable_model(tool_context: ToolContext, features_csv: str = "", covariates_csv: str = "") -> dict[str, Any]:
    """Fit the features jointly with covariates (OLS, logistic or Cox, by outcome type): each feature's
    independent effect given the others, and the cross-validated incremental value over a model of the
    covariates alone (delta R^2, AUC or C-index with a bootstrap CI) — what the wearable features add
    beyond what is already known. Defaults: the latest round's candidates and the declared confounders."""
    def body(frame, design):
        names = _split_csv(features_csv) or design["candidates"]
        data = _with_features(frame, design, names)
        covariates = ([_covariate(frame, design, c) for c in _split_csv(covariates_csv)] if covariates_csv.strip()
                      else design["confounders"])
        from codas.core.analyses import multivariable_model

        return multivariable_model(data, names[:_MAX_ANALYSIS_FEATURES], design["target"], covariates,
                                   design["time"], design["imputations"])
    return _run_analysis(tool_context, "multivariable_model", {"features": features_csv, "covariates": covariates_csv},
                         body)


def run_nonlinear_model(tool_context: ToolContext, features_csv: str = "") -> dict[str, Any]:
    """Gradient boosting against the linear model on the features plus the declared confounders (held-out
    R^2 or AUC), held-out permutation importance, and the shape of each feature's partial dependence
    (increasing, decreasing, U-shaped, inverted U, non-monotone, flat). Use it when an association may
    be non-linear or a threshold, or to see which features carry the prediction."""
    def body(frame, design):
        names = _split_csv(features_csv) or design["candidates"]
        data = _with_features(frame, design, names)
        from codas.core.analyses import nonlinear_model

        out = nonlinear_model(data, names[:_MAX_ANALYSIS_FEATURES], design["target"], design["confounders"])
        out["note"] = ("time to event modelled as its event indicator (follow-up time not used)" if design["time"]
                       else "held-out cross-validated comparison")
        return out
    return _run_analysis(tool_context, "nonlinear_model", {"features": features_csv}, body)


def run_dose_response(feature: str, tool_context: ToolContext) -> dict[str, Any]:
    """Dose-response of one feature: a restricted cubic spline adjusted for the declared confounders, a test
    of non-linearity, and the effect at the 10th-90th percentiles against the median (mean difference,
    odds ratio or hazard ratio). Use it to tell a linear gradient from a threshold or a plateau."""
    def body(frame, design):
        data = _with_features(frame, design, [feature.strip()])
        from codas.core.analyses import spline_dose_response

        return spline_dose_response(data, feature.strip(), design["target"], design["confounders"], design["time"],
                                    design["imputations"])
    return _run_analysis(tool_context, "dose_response", {"feature": feature}, body)


def run_mixed_model(feature: str, tool_context: ToolContext) -> dict[str, Any]:
    """Within-person analysis on the repeated-measures panel (when the task provides one): every later
    measurement of the outcome paired with the wearable summary of the period before it. A linear mixed
    model with a random intercept per person splits the feature's effect into a within-person part (a
    person compared with themselves: stable differences between people such as adherence, body size or
    socioeconomic position cannot produce it) and a between-person part. A feature that only tracks a
    stable trait shows a between-person effect and a within-person effect near zero."""
    def body(frame, design):
        path = str(tool_context.state.get("panel_csv_path") or "")
        if not path or not Path(path).exists():
            raise _Refused("No repeated-measures panel is available for this outcome.")
        panel = read_csv_dataset(Path(path))
        name = feature.strip()
        if name in design["blocked_features"]:
            raise _Refused(f"{name} is the target or an adjustment-only column, not a candidate feature.")
        from codas.core.analyses import rebuild_feature
        from codas.core.mixed import mixed_model

        series = rebuild_feature(panel, name, design["definitions"].get(name))
        if series is None:
            raise _Refused(f"{name} cannot be built from the panel's columns (period means and SDs of the daily sources).")
        panel = panel.assign(**{name: series})
        return mixed_model(panel, name, design["target"], [c for c in ("age", "days_from_start") if c in panel.columns])
    return _run_analysis(tool_context, "mixed_model", {"feature": feature}, body)


ANALYSIS_TOOLS = [run_survival_screen, run_subgroup_analysis, run_multivariable_model, run_nonlinear_model,
                  run_dose_response, run_mixed_model]


# --- method pools: the deterministic runner and the gap finder's requests ------------------------

_METHOD_FEATURES = int(os.getenv("CODAS_METHOD_FEATURES", "8"))
_DIGEST_ITEMS = int(os.getenv("CODAS_ANALYSES_DIGEST_ITEMS", "12"))


def analyses_counts(records: list[dict[str, Any]]) -> str:
    """Every analysis tool run in this outcome with how many times, in order of first use: the digest
    shows only the latest analyses, so a report writer reading it alone could miss that a tool ran."""
    counts: dict[str, int] = {}
    for rec in records:
        counts[str(rec.get("tool", ""))] = counts.get(str(rec.get("tool", "")), 0) + 1
    return "; ".join(f"{tool} x{n}" for tool, n in counts.items())


def analyses_digest(records: list[dict[str, Any]], limit: int = _DIGEST_ITEMS) -> str:
    """The latest analyses as one line each: what the agents read instead of the raw results."""
    from codas.core.methods import record_headline

    lines = [f"- round {rec.get('round', '?')} {rec.get('tool', '')}: {record_headline(rec)}" for rec in records[-limit:]]
    return "\n".join(lines)


def _execute_method(frame: pd.DataFrame, design: dict[str, Any], request: dict[str, Any], default_features: list[str],
                    round_index: int) -> dict[str, Any]:
    """Run one method request on its features (or the default ones) and return the analysis record."""
    from codas.core import methods

    name = request["method"]
    features = [f for f in request.get("features") or default_features
                if f not in design["blocked_features"]][:_METHOD_FEATURES]
    try:
        data = _with_features(frame, design, features)
        options = {k: _covariate(frame, design, request[k]) for k in ("by", "mediator") if request.get(k)}
        result = methods.run(name, data, features, design["target"], design["confounders"], design["time"],
                             design["imputations"], **options)
    except _Refused as exc:
        result = {"method": name, "error": str(exc)}
    return {"tool": f"method:{name}", "round": round_index,
            "args": {k: v for k, v in request.items() if k != "method"}, "result": _json_safe(_sig(result))}


def _candidate_names(state: dict[str, Any], design: dict[str, Any]) -> list[str]:
    report = state.get("latest_report") or {}
    names = design["candidates"] or [c.get("feature") for c in sorted(
        report.get("candidates") or [], key=lambda c: -abs(float(c.get("rho") or 0)))]
    return [n for n in names if n and n not in design["blocked_features"]][:_METHOD_FEATURES]


def run_method(method: str, tool_context: ToolContext, features_csv: str = "", by: str = "",
               mediator: str = "") -> dict[str, Any]:
    """Run one method from the statistical or ML pool now (see the gap finder's list): e.g. mediation
    through a named ``mediator``, subgroup_stratified ``by`` a covariate, propensity_stratification,
    robust_regression, mutual_information, or an ML model with nested CV. Defaults to the latest round's
    candidates. Use it to settle a specific question with evidence."""
    from codas.core import methods

    if methods.pool_of(method) is None:
        return {"error": f"unknown method {method!r}; choose one of the statistical or ML pool methods."}
    try:
        frame, design = _analysis_setup(tool_context)
        request = {"method": method, **({"features": _split_csv(features_csv)} if features_csv.strip() else {}),
                   **({"by": by.strip()} if by.strip() else {}), **({"mediator": mediator.strip()} if mediator.strip() else {})}
        for key in ("by", "mediator"):
            if request.get(key):
                _covariate(frame, design, request[key])
        state = tool_context.state
        record = _execute_method(frame, design, request, _candidate_names(state, design),
                                 max(len(state.get("rounds") or []) - 1, 0))
    except _Refused as exc:
        return {"error": str(exc)}
    if aou_mode():
        from codas.aou.privacy import mask_small_counts

        record = mask_small_counts(record)
    state["analyses"] = list(state.get("analyses") or []) + [record]
    explored = list(state.get("methods_explored") or [])
    state["methods_explored"] = explored + ([method] if method not in explored else [])
    digest = analyses_digest(state["analyses"])
    state["analyses_digest"] = mask_small_counts_in_text(digest) if aou_mode() else digest
    state["analyses_counts"] = analyses_counts(state["analyses"])
    return record["result"]


def debate_candidates(state: dict[str, Any], cap: int = 5) -> list[dict[str, Any]]:
    """The final candidates to debate one by one, strongest first, each with its battery summary and the
    one-line results of every analysis that included it."""
    report = state.get("latest_report") or {}
    summary = {c.get("feature"): c for c in ((state.get("rounds") or [{}])[-1].get("passing_candidates") or [])}
    chosen = sorted((c for c in report.get("candidates") or [] if c.get("verdict") in {"validated", "conditional"}),
                    key=lambda c: -abs(float(c.get("rho") or 0)))[:cap]
    out = []
    for c in chosen:
        name = c.get("feature")
        evidence = [line for line in analyses_digest(
            [a for a in state.get("analyses") or [] if name in json.dumps(a.get("result") or {}, default=str)],
            limit=8).splitlines()]
        info = summary.get(name, {})
        out.append({"feature": name, "rho": c.get("rho"), "verdict": c.get("verdict"), "q": info.get("q", c.get("q_value")),
                    "adjusted_rho": info.get("adjusted_rho"), "failed_checks": info.get("failed_checks", []),
                    "evidence": evidence})
    return out


_DEBATE_VERDICTS = ("keep", "caveat", "concede")


def record_debate(feature: str, verdict: str, reason: str, evidence: str, tool_context: ToolContext) -> dict[str, Any]:
    """Close the debate on the candidate under discussion: ``verdict`` is keep (the evidence holds),
    caveat (it holds with a stated limitation) or concede (it fails a named check); ``reason`` in one or
    two sentences; ``evidence`` names the tool result relied on. This informs how the candidate is
    interpreted and reported; it does not change the engine's validation or the pre-registration."""
    current = (tool_context.state.get("debate_candidate") or {}).get("feature")
    if feature.strip() != current:
        return {"error": f"The candidate under debate is {current!r}; record the verdict for it."}
    if verdict.strip().lower() not in _DEBATE_VERDICTS:
        return {"error": f"verdict must be one of {', '.join(_DEBATE_VERDICTS)}."}
    entry = {"feature": current, "verdict": verdict.strip().lower(), "reason": reason.strip()[:600],
             "evidence": evidence.strip()[:300]}
    tool_context.state["debate_verdict"] = entry
    return {"recorded": entry}


_MAX_LESSONS = int(os.getenv("CODAS_MAX_LESSONS", "6"))


def record_lessons(lessons_json: str, tool_context: ToolContext) -> dict[str, Any]:
    """Record what a later analysis of OTHER outcomes in this same cohort should know. ``lessons_json`` is
    a JSON list of ``{"lesson": one sentence, "about": [column names], "evidence": what showed it}``:
    methodological facts about the data (a column that measures wear or data availability rather than
    physiology, one that scales with body size, a covariate mostly missing, an unstable feature family),
    from discovery evidence only. No outcome-specific findings."""
    try:
        items = json.loads(lessons_json)
    except (TypeError, ValueError) as exc:
        return {"error": f"lessons_json is not valid JSON: {exc}"}
    if not isinstance(items, list):
        return {"error": "lessons_json must be a JSON list of lessons."}
    columns = set(tool_context.state.get("dataset_columns") or [])
    lessons = list(tool_context.state.get("lessons") or [])
    recorded, rejected = [], []
    for item in items:
        item = item if isinstance(item, dict) else {}
        text = str(item.get("lesson") or "").strip()[:300]
        about = [str(c) for c in item.get("about") or []]
        unknown = [c for c in about if columns and c not in columns]
        problem = ("empty lesson" if not text else f"unknown columns: {', '.join(unknown)}" if unknown
                   else f"at most {_MAX_LESSONS} lessons" if len(lessons) >= _MAX_LESSONS else None)
        if problem:
            rejected.append({"lesson": text, "reason": problem})
            continue
        entry = {"lesson": text, "about": about, "evidence": str(item.get("evidence") or "")[:200]}
        lessons.append(entry)
        recorded.append(entry)
    tool_context.state["lessons"] = lessons
    return _json_safe({"recorded": recorded, "rejected": rejected})


def run_method_batch(state: dict[str, Any], per_pool: int = 1) -> dict[str, Any]:
    """Run the requested methods (``state['next_methods']``), then the next unexplored method of each pool,
    on the round's candidates. Returns the state changes: the analyses with the new records, the methods
    explored, the latest round marked with what ran, and the digest the agents read."""
    from codas.core import methods

    context = SimpleNamespace(state=state)
    frame, design = _analysis_setup(context)
    names = _candidate_names(state, design)
    explored = list(state.get("methods_explored") or [])
    target = pd.to_numeric(frame[design["target"]], errors="coerce").to_numpy(float)
    from codas.core.analyses import _is_binary

    requests = methods.next_methods(explored, list(state.get("next_methods") or []), bool(design["time"]),
                                    _is_binary(target), per_pool)
    round_index = max(len(state.get("rounds") or []) - 1, 0)
    records = [_execute_method(frame, design, request, names, round_index) for request in requests]
    ran = [request["method"] for request in requests]
    if aou_mode():
        from codas.aou.privacy import mask_small_counts

        records = mask_small_counts(records)
    analyses = list(state.get("analyses") or []) + records
    rounds = [dict(r) for r in state.get("rounds") or []]
    if rounds:
        rounds[-1]["methods_run"] = ran
    digest = analyses_digest(analyses)
    return {"analyses": analyses, "methods_explored": explored + [m for m in ran if m not in explored],
            "next_methods": [], "rounds": rounds, "method_pool_active": True,
            "analyses_digest": mask_small_counts_in_text(digest) if aou_mode() else digest,
            "analyses_counts": analyses_counts(analyses)}


def record_gap_analysis(gaps_json: str, tool_context: ToolContext) -> dict[str, Any]:
    """Record what the analysis still leaves open and what to run next. ``gaps_json`` is a JSON object:
    ``gaps`` (list of open questions), ``next_methods`` (list of ``{"method": ..., "by": covariate,
    "mediator": column, "features": [...]}`` drawn from the statistical and ML pools), and ``constraints``
    (list of rules for later rounds; they are added to the earlier ones, never replace them). The
    deterministic runner executes the requested methods at the start of the next round."""
    from codas.core import methods

    try:
        payload = json.loads(gaps_json)
    except (TypeError, ValueError) as exc:
        return {"error": f"gaps_json is not valid JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"error": "gaps_json must be a JSON object with gaps, next_methods and constraints."}
    state = tool_context.state
    try:
        frame, design = _analysis_setup(tool_context)
    except _Refused as exc:
        return {"error": str(exc)}
    accepted, rejected = [], []
    for item in payload.get("next_methods") or []:
        item = item if isinstance(item, dict) else {"method": str(item)}
        name = str(item.get("method") or "")
        try:
            if methods.pool_of(name) is None:
                raise _Refused(f"unknown method {name!r}")
            entry = {"method": name}
            for key in ("by", "mediator"):
                if item.get(key):
                    entry[key] = _covariate(frame, design, str(item[key]))
            if item.get("features"):
                entry["features"] = [str(f) for f in item["features"]][:_METHOD_FEATURES]
            accepted.append(entry)
        except _Refused as exc:
            rejected.append({"request": item, "reason": str(exc)})
    constraints = list(state.get("constraints") or [])
    constraints += [str(c)[:300] for c in payload.get("constraints") or [] if str(c)[:300] not in constraints]
    entry = {"round": max(len(state.get("rounds") or []) - 1, 0),
             "gaps": [str(g)[:300] for g in payload.get("gaps") or []], "next_methods": accepted}
    state["gap_analysis"] = list(state.get("gap_analysis") or []) + [entry]
    state["next_methods"] = accepted
    state["constraints"] = constraints
    return _json_safe({"accepted": accepted, "rejected": rejected, "constraints": constraints,
                       "available_methods": {pool: [f"{m}: {methods.describe(m)}" for m in names]
                                             for pool, names in methods.available(bool(design["time"])).items()}})


_JACCARD_STABLE = float(os.getenv("CODAS_CONVERGENCE_JACCARD", "0.9"))


def _jaccard(a: dict[str, Any], b: dict[str, Any]) -> float:
    x = {c.get("feature") for c in a.get("passing_candidates") or []}
    y = {c.get("feature") for c in b.get("passing_candidates") or []}
    return len(x & y) / len(x | y) if x | y else 1.0


def _method_pool_stop(state: dict[str, Any], rounds: list[dict[str, Any]]) -> str | None:
    """The paper's stop rules for a loop with method pools: both pools exhausted, or the candidate set
    unchanged (Jaccard above the threshold) for two consecutive rounds."""
    from codas.core import methods

    explored = list(state.get("methods_explored") or [])
    survival = bool(str(state.get("survival_time_column") or "").strip())
    if methods.exhausted(explored, survival) or methods.exhausted(explored, survival, binary=True):
        return "Both method pools are exhausted: every statistical and ML method has been run. Stopping."
    if len(rounds) >= 3 and all(_jaccard(rounds[i - 1], rounds[i]) > _JACCARD_STABLE for i in (-1, -2)):
        return (f"The candidate set has not changed for two rounds (Jaccard > {_JACCARD_STABLE}); further "
                "rounds would re-test the same candidates. Stopping.")
    return None


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
    # A revised design (features excluded as leakage after this round ran) has not been tested yet, so
    # the loop must not stop on the old design's numbers; the LoopAgent's iteration bound still applies.
    if int(tool_context.state.get("design_revision") or 0) > int(current.get("design_revision") or 0):
        return {
            "converged": False,
            "reason": "The analysis design changed after the latest round (features excluded or "
                      "confounders added); run one more round on the revised design before judging "
                      "convergence.",
            "rounds_completed": len(rounds),
        }
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

    if tool_context.state.get("method_pool_active"):
        stop = _method_pool_stop(tool_context.state, rounds)
        if stop:
            tool_context.actions.escalate = True
            return {"converged": True, "reason": stop, "rounds_completed": len(rounds)}

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
    if converged and tool_context.state.get("method_pool_active") and current.get("methods_run"):
        return {"converged": False, "gain_validated": gain_validated, "gain_metric": round(gain_metric, 4),
                "reason": f"No new validated candidate (Δ={gain_validated}), but this round ran new methods "
                          f"({', '.join(current['methods_run'])}); continue until the pools are exhausted or the "
                          "candidate set stops changing.", "rounds_completed": len(rounds)}
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
