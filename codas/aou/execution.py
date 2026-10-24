"""How each agent ran, and how the agents fed one another.

``ExecutionPlugin`` is an ADK plugin given to the runner's app. It sees every agent turn, model call and
tool call the graph makes, so it records exact timings (parallel branches included), the model that
actually served each call (a fallback is visible), its token counts and errors, which tools each agent
called, the model calls a tool made on its own (the literature search), and which shared-memory keys each
event wrote. ``execution_tables`` turns those records into what a reader needs to debug a run:

* per agent: turns, model calls, tool calls by tool, tool and model errors, active time, tokens (prompt,
  output, thoughts, cached, tool-use prompt), the models that served it and an estimated cost;
* phases: how long each container agent (a sequence, loop or parallel step) ran;
* the information flow: which agent wrote a shared-memory key that a later agent's instructions read;
* the design changes: who set the target, declared confounders, excluded columns, proposed features,
  registered hypotheses or recorded a gap analysis, and in which discovery round;
* a timeline of agent turns.

Nothing here holds participant data: only agent and tool names, times, token counts and memory-key names.
Costs are estimates at list prices (``codas.core.model_tiers``); the workspace bill is the authority.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from codas.core import model_tiers

# Tools whose calls change the analysis design that later agents work on.
DESIGN_TOOLS = ("set_target", "record_data_memo", "declare_confounders", "exclude_features", "propose_feature",
                "register_hypotheses", "record_gap_analysis")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\??\}")
_TOKENS = ("prompt", "output", "thoughts", "cached", "tool_prompt", "total")
LOGGER = logging.getLogger("codas.aou.execution")


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    fields = {"prompt": "prompt_token_count", "output": "candidates_token_count", "thoughts": "thoughts_token_count",
              "cached": "cached_content_token_count", "tool_prompt": "tool_use_prompt_token_count",
              "total": "total_token_count"}
    return {k: int(getattr(usage, v, None) or 0) for k, v in fields.items()} if usage is not None else {}


def reads_of(agent: Any) -> list[str]:
    """The shared-memory keys an agent's instructions read (their ``{key?}`` placeholders)."""
    instruction = getattr(agent, "instruction", "")
    return sorted(set(_PLACEHOLDER.findall(instruction))) if isinstance(instruction, str) else []


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {str(error)[:120]}"


def _quietly(callback: Any) -> Any:
    """Bookkeeping must never stop a run: a callback that fails records nothing and lets the run go on."""
    @functools.wraps(callback)
    async def wrapper(self: Any, **kwargs: Any) -> None:
        try:
            await callback(self, **kwargs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("execution record skipped in %s: %s", callback.__name__, exc)
        return None
    return wrapper


class ExecutionPlugin(BasePlugin):
    """Records every agent turn, model call and tool call; see the module docstring."""

    def __init__(self, name: str = "codas_execution") -> None:
        super().__init__(name=name)
        self.calls: list[dict[str, Any]] = []
        self.reads: dict[str, list[str]] = {}
        self._open: dict[Any, float] = {}
        self.started = time.time()

    def _close(self, key: Any) -> float:
        return self._open.pop(key, time.time())

    # --- agent turns ---------------------------------------------------------------------------
    @_quietly
    async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        self.reads.setdefault(agent.name, reads_of(agent))
        self._open[("agent", callback_context.invocation_id, agent.name, callback_context.branch)] = time.time()

    @_quietly
    async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
        start = self._close(("agent", callback_context.invocation_id, agent.name, callback_context.branch))
        self.calls.append({"kind": "agent", "agent": agent.name, "branch": callback_context.branch,
                           "container": bool(getattr(agent, "sub_agents", None)), "start": start, "end": time.time()})

    # --- model calls ---------------------------------------------------------------------------
    @staticmethod
    def _model_key(ctx: Any) -> tuple:
        return ("model", ctx.invocation_id, ctx.agent_name, ctx.branch)

    @_quietly
    async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
        self._open[self._model_key(callback_context)] = time.time()

    @_quietly
    async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
        if getattr(llm_response, "partial", False):
            return  # a streamed fragment; the complete response is recorded
        start = self._close(self._model_key(callback_context))
        self.calls.append({"kind": "model", "agent": callback_context.agent_name, "start": start, "end": time.time(),
                           "model": getattr(llm_response, "model_version", None),
                           "finish": str(getattr(llm_response, "finish_reason", "") or ""),
                           "error": getattr(llm_response, "error_code", None), **_usage(llm_response)})

    @_quietly
    async def on_model_error_callback(self, *, callback_context: Any, llm_request: Any, error: Exception) -> None:
        start = self._close(self._model_key(callback_context))
        self.calls.append({"kind": "model", "agent": callback_context.agent_name, "start": start, "end": time.time(),
                           "model": None, "error": _error_text(error)})

    # --- tool calls ----------------------------------------------------------------------------
    @_quietly
    async def before_tool_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any) -> None:
        self._open[("tool", tool_context.function_call_id)] = time.time()

    def _tool_record(self, tool: Any, tool_context: Any, error: Any) -> dict[str, Any]:
        from codas.agents.tools import take_model_usage

        start = self._close(("tool", tool_context.function_call_id))
        return {"kind": "tool", "agent": tool_context.agent_name, "tool": tool.name, "start": start,
                "end": time.time(), "error": error, "model_usage": take_model_usage(tool_context.function_call_id)}

    @_quietly
    async def after_tool_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any,
                                  result: dict[str, Any]) -> None:
        self.calls.append(self._tool_record(tool, tool_context, bool(isinstance(result, dict) and result.get("error"))))

    @_quietly
    async def on_tool_error_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any,
                                     error: Exception) -> None:
        self.calls.append(self._tool_record(tool, tool_context, _error_text(error)))

    # --- shared-memory writes ------------------------------------------------------------------
    @_quietly
    async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
        delta = getattr(getattr(event, "actions", None), "state_delta", None) or {}
        if delta:
            self.calls.append({"kind": "write", "agent": event.author, "keys": sorted(map(str, delta)),
                               "time": float(getattr(event, "timestamp", None) or time.time())})

    def snapshot(self) -> dict[str, Any]:
        # "calls", not "records": the privacy filter drops any key named "records" as row-level data.
        return {"calls": list(self.calls), "reads": dict(self.reads), "started": self.started}


def _row(agents: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return agents.setdefault(name, {"agent": name, "turns": 0, "active_s": 0.0, "model_calls": 0, "model_errors": 0,
                                    "tool_calls": 0, "tool_errors": 0, "tools": {}, **{k: 0 for k in _TOKENS},
                                    "search_queries": 0, "models": {}, "cost_usd": 0.0, "_priced": True})


def _add_usage(a: dict[str, Any], usage: dict[str, Any], model: str | None) -> None:
    for k in _TOKENS:
        a[k] += int(usage.get(k) or 0)
    if model:
        a["models"][model] = a["models"].get(model, 0) + 1
    tokens_in = int(usage.get("prompt") or 0) + int(usage.get("tool_prompt") or 0)
    tokens_out = int(usage.get("output") or 0) + int(usage.get("thoughts") or 0)
    cost = model_tiers.cost(model, tokens_in, tokens_out)
    if cost is None and (tokens_in or tokens_out):
        a["_priced"] = False
    a["cost_usd"] += cost or 0.0


def execution_tables(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Per-agent table, phases, information flow, design changes, timeline and totals from a plugin snapshot."""
    records, reads = snapshot.get("calls") or [], snapshot.get("reads") or {}
    starts = [r.get("start", r.get("time")) for r in records if r.get("start", r.get("time")) is not None]
    started = float(snapshot.get("started") or min(starts or [0.0]))
    agents: dict[str, dict[str, Any]] = {}
    for r in records:
        if r["kind"] == "agent" and not r.get("container"):
            a = _row(agents, r["agent"])
            a["turns"] += 1
            a["active_s"] += max(0.0, r["end"] - r["start"])
        elif r["kind"] == "model":
            a = _row(agents, r["agent"])
            a["model_calls"] += 1
            a["model_errors"] += int(bool(r.get("error")))
            _add_usage(a, r, r.get("model"))
        elif r["kind"] == "tool":
            a = _row(agents, r["agent"])
            a["tool_calls"] += 1
            a["tool_errors"] += int(bool(r.get("error")))
            a["tools"][r["tool"]] = a["tools"].get(r["tool"], 0) + 1
            for usage in r.get("model_usage") or []:  # model calls the tool made on its own
                _add_usage(a, usage, usage.get("model"))
                a["search_queries"] += int(usage.get("search_queries") or 0)
    for a in agents.values():
        a["active_s"] = round(a["active_s"], 1)
        a["cost_usd"] = round(a["cost_usd"], 4) if a.pop("_priced") else None

    turns = sorted((r for r in records if r["kind"] == "agent" and not r.get("container")), key=lambda r: r["start"])
    phases = [{"agent": r["agent"], "start_s": round(r["start"] - started, 1), "end_s": round(r["end"] - started, 1)}
              for r in sorted((r for r in records if r["kind"] == "agent" and r.get("container")),
                              key=lambda r: r["start"])]
    # Information flow: a key written by one agent and read by a later turn of another agent.
    writes = sorted((r for r in records if r["kind"] == "write"), key=lambda r: r["time"])
    flow: dict[tuple[str, str, str], int] = {}
    for turn in turns:
        for key in reads.get(turn["agent"], []):
            writer = next((w["agent"] for w in reversed(writes) if w["time"] <= turn["start"] and key in w["keys"]),
                          None)
            if writer and writer != turn["agent"]:
                flow[(writer, key, turn["agent"])] = flow.get((writer, key, turn["agent"]), 0) + 1
    # Design changes, each placed in the discovery round during which it was made (0 = before the first).
    rounds_done = sorted(r["end"] for r in records
                         if r["kind"] == "tool" and r["tool"] == "run_discovery_round" and not r.get("error"))
    changes = [{"agent": r["agent"], "tool": r["tool"], "round": sum(1 for t in rounds_done if t <= r["start"]),
                "at_s": round(r["start"] - started, 1), "error": bool(r.get("error"))}
               for r in sorted((r for r in records if r["kind"] == "tool"), key=lambda r: r["start"])
               if r["tool"] in DESIGN_TOOLS]
    timeline = [{"agent": t["agent"], "branch": t.get("branch"), "start_s": round(t["start"] - started, 1),
                 "end_s": round(t["end"] - started, 1)} for t in turns]
    first = {}
    for t in timeline:
        first.setdefault(t["agent"], t["start_s"])
    rows = sorted(agents.values(), key=lambda a: (first.get(a["agent"], float("inf")), a["agent"]))
    ends = [r["end"] for r in records if r["kind"] in ("agent", "model", "tool")]
    totals = {"agents": len(rows), "model_calls": sum(a["model_calls"] for a in rows),
              "model_errors": sum(a["model_errors"] for a in rows),
              "tool_calls": sum(a["tool_calls"] for a in rows), "tool_errors": sum(a["tool_errors"] for a in rows),
              **{k: sum(a[k] for a in rows) for k in _TOKENS if k != "total"},
              "tokens": sum(a["total"] for a in rows), "search_queries": sum(a["search_queries"] for a in rows),
              "models": _merge(a["models"] for a in rows),
              "cost_usd": (None if any(a["cost_usd"] is None for a in rows)
                           else round(sum(a["cost_usd"] for a in rows), 2)),
              "wall_s": round(max(ends) - started, 1) if ends else 0.0}
    return {"agents": rows, "phases": phases,
            "flow": [{"writer": w, "key": k, "reader": r, "times": n} for (w, k, r), n in sorted(flow.items())],
            "design_changes": changes, "timeline": timeline, "totals": totals}


def flow_by_writer(flow: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The information flow with one row per (writer, key) and its readers, for display."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for f in flow:
        grouped.setdefault((f["writer"], f["key"]), []).append(f"{f['reader']} ({f['times']})")
    return [{"writer": w, "key": k, "read by (turns)": ", ".join(readers)} for (w, k), readers in grouped.items()]


def _merge(counts: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in counts:
        for k, v in c.items():
            out[k] = out.get(k, 0) + v
    return out


def totals_text(totals: dict[str, Any]) -> str:
    cost = ("unknown (a model without a listed price)" if totals.get("cost_usd") is None
            else f"about ${totals['cost_usd']:,.2f}")
    models = ", ".join(f"{k} x{v}" for k, v in (totals.get("models") or {}).items()) or "none"
    return (f"{totals.get('agents', 0)} agents; {totals.get('model_calls', 0)} model calls "
            f"({totals.get('model_errors', 0)} failed); {totals.get('tool_calls', 0)} tool calls "
            f"({totals.get('tool_errors', 0)} failed); {totals.get('tokens', 0):,} tokens "
            f"({totals.get('prompt', 0):,} prompt, {totals.get('output', 0):,} output, "
            f"{totals.get('thoughts', 0):,} thinking); {totals.get('wall_s', 0) / 60:.1f} min; models served: "
            f"{models}; model cost {cost} at list prices.")


def execution_markdown(tables: dict[str, Any]) -> str:
    """The execution report: totals, the per-agent table, phase durations, design changes and the
    information flow between agents."""
    lines = ["## Agent execution", "", totals_text(tables.get("totals") or {}), "",
             "| agent | turns | active min | model calls | errors | tool calls (by tool) | tokens | thinking | "
             "models served | cost (USD) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in tables.get("agents") or []:
        tools = ", ".join(f"{k} x{v}" for k, v in sorted(a["tools"].items(), key=lambda kv: -kv[1])) or "-"
        models = ", ".join(f"{k} x{v}" for k, v in a["models"].items()) or "-"
        cost = "" if a["cost_usd"] is None else f"{a['cost_usd']:.3f}"
        lines.append(f"| {a['agent']} | {a['turns']} | {a['active_s'] / 60:.1f} | {a['model_calls']} | "
                     f"{a['model_errors'] + a['tool_errors']} | {a['tool_calls']} ({tools}) | {a['total']:,} | "
                     f"{a['thoughts']:,} | {models} | {cost} |")
    if tables.get("phases"):
        lines += ["", "**Phases** (container agents, minutes from the start):", ""]
        lines += [f"- {p['agent']}: {p['start_s'] / 60:.1f} to {p['end_s'] / 60:.1f}" for p in tables["phases"]]
    if tables.get("design_changes"):
        lines += ["", "**Design changes** (who changed what the later agents work on, and in which round):", "",
                  "| round | minute | agent | tool |", "|---|---|---|---|"]
        lines += [f"| {c['round']} | {c['at_s'] / 60:.1f} | {c['agent']} | {c['tool']}"
                  f"{' (error)' if c['error'] else ''} |" for c in tables["design_changes"]]
    if tables.get("flow"):
        lines += ["", "**Information flow** (a memory key one agent wrote, and the later agents whose instructions "
                  "read it, with how many of their turns did):", "", "| writer | key | read by (turns) |",
                  "|---|---|---|"]
        lines += [f"| {f['writer']} | {f['key']} | {f['read by (turns)']} |" for f in flow_by_writer(tables["flow"])]
    return "\n".join(lines) + "\n"
