"""Full agent trace of one CoDaS run: every agent's text, tool call and tool response, in order.

The recorder sees the same events the ADK runner yields, so a tool response is the one the model
received (in AoU mode already sanitised by ``after_tool_logger``); everything written is masked again
for counts of 1-20. Phases and discovery rounds are read from the agent graph itself: a phase is a
direct child of the root agent, and a new round starts whenever the first agent of a loop speaks
after another agent of that loop. Nothing here knows which agents or datasets exist.

Outputs: ``<stem>_trace.jsonl`` (one event per line), ``<stem>_trace.md`` (readable, grouped by
phase and round) and a phase-coverage table showing which agents ran in each phase and round.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from codas.aou.privacy import mask_small_counts, mask_small_counts_in_text

_MD_VALUE_LIMIT = 1500  # characters of a tool payload shown in the readable trace (the JSONL keeps all)


def _leaves(node: Any) -> list[Any]:
    children = list(getattr(node, "sub_agents", None) or [])
    if not children:
        return [node]
    return [leaf for child in children for leaf in _leaves(child)]


def _nodes(node: Any) -> list[Any]:
    """The node and every agent under it (a container agent can emit events of its own)."""
    return [node] + [n for child in getattr(node, "sub_agents", None) or [] for n in _nodes(child)]


def _is_loop(node: Any) -> bool:
    return hasattr(node, "max_iterations")


class TraceRecorder:
    """Collects masked events from one run of ``root_agent`` and summarises phase coverage."""

    def __init__(self, root_agent: Any = None, stream_path: str | Path | None = None):
        self.events: list[dict[str, Any]] = []
        # Events are also appended here as they happen, so a run that dies midway still leaves its trace.
        self.stream_path = Path(stream_path) if stream_path else None
        if self.stream_path:
            self.stream_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream_path.write_text("", encoding="utf-8")
        self.phase_of: dict[str, str] = {}
        self.loop_of: dict[str, str] = {}
        self.loop_first: dict[str, str] = {}
        self.phase_agents: dict[str, list[str]] = {}
        self._round: dict[str, int] = {}
        self._last_in_loop: dict[str, str] = {}
        if root_agent is not None:
            self.bind(root_agent)

    def bind(self, root_agent: Any) -> None:
        """Read phases and loops from the agent graph (done by the runner once the graph is imported)."""
        for phase in getattr(root_agent, "sub_agents", None) or [root_agent]:
            leaves = [leaf.name for leaf in _leaves(phase)]
            self.phase_agents[phase.name] = leaves
            for node in _nodes(phase):
                self.phase_of[node.name] = phase.name
            for loop in self._loops(phase):
                names = [leaf.name for leaf in _leaves(loop)]
                self.loop_first[loop.name] = names[0]
                for name in names:
                    self.loop_of[name] = loop.name

    def _loops(self, node: Any) -> list[Any]:
        found = [node] if _is_loop(node) else []
        for child in getattr(node, "sub_agents", None) or []:
            found += self._loops(child)
        return found

    def record(self, event: Any) -> None:
        start = len(self.events)
        self._record(event)
        if self.stream_path and len(self.events) > start:
            with self.stream_path.open("a", encoding="utf-8") as fh:
                for e in self.events[start:]:
                    fh.write(json.dumps(e, default=str) + "\n")

    def _record(self, event: Any) -> None:
        author = str(getattr(event, "author", "") or "")
        loop = self.loop_of.get(author)
        round_no = None
        if loop:
            if author == self.loop_first[loop] and self._last_in_loop.get(loop) not in (None, author):
                self._round[loop] = self._round.get(loop, 1) + 1
            self._round.setdefault(loop, 1)
            self._last_in_loop[loop] = author
            round_no = self._round[loop]
        base = {"time": float(getattr(event, "timestamp", None) or time.time()), "author": author,
                "phase": self.phase_of.get(author, ""), "round": round_no}
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            if getattr(part, "thought", None):
                continue
            call, response = getattr(part, "function_call", None), getattr(part, "function_response", None)
            if call is not None:
                self.events.append({**base, "kind": "tool_call", "name": call.name,
                                    "args": mask_small_counts(dict(call.args or {}))})
            elif response is not None:
                self.events.append({**base, "kind": "tool_response", "name": response.name,
                                    "response": mask_small_counts(_plain(response.response))})
            elif getattr(part, "text", None):
                self.events.append({**base, "kind": "text", "text": mask_small_counts_in_text(part.text)})

    def coverage(self) -> dict[str, Any]:
        """For each phase, the agents that produced output; for each loop round, which loop agents ran.

        ``missing`` lists agents of a phase that never produced anything, and loop agents absent from
        a completed round, so a reader can see whether every step of the design actually ran.
        """
        spoke: dict[str, set[str]] = {}
        per_round: dict[int, set[str]] = {}
        for e in self.events:
            spoke.setdefault(e["phase"], set()).add(e["author"])
            if e["round"] is not None:
                per_round.setdefault(e["round"], set()).add(e["author"])
        phases = {p: {"ran": [a for a in agents if a in spoke.get(p, set())],
                      "silent": [a for a in agents if a not in spoke.get(p, set())]}
                  for p, agents in self.phase_agents.items()}
        loop_agents = [a for a in self.loop_of]
        rounds = {r: {"ran": [a for a in loop_agents if a in seen], "silent": [a for a in loop_agents if a not in seen]}
                  for r, seen in sorted(per_round.items())}
        missing = [f"{p}: {a}" for p, v in phases.items() for a in v["silent"]]
        # The last round may legitimately stop early (the convergence check ends the loop), so only
        # earlier rounds count as missing steps.
        for r, v in list(rounds.items())[:-1]:
            missing += [f"round {r}: {a}" for a in v["silent"]]
        return {"phases": phases, "rounds": rounds, "missing": missing}

    def coverage_markdown(self) -> str:
        cov = self.coverage()
        lines = ["| phase | agents that produced output | silent |", "|---|---|---|"]
        for p, v in cov["phases"].items():
            lines.append(f"| {p} | {', '.join(v['ran']) or '-'} | {', '.join(v['silent']) or '-'} |")
        if cov["rounds"]:
            lines += ["", "| round | loop agents that ran | silent |", "|---|---|---|"]
            for r, v in cov["rounds"].items():
                lines.append(f"| {r} | {', '.join(v['ran']) or '-'} | {', '.join(v['silent']) or '-'} |")
        lines += ["", ("Every agent produced output." if not cov["missing"]
                       else "Steps without output: " + "; ".join(cov["missing"]))]
        return "\n".join(lines)

    _STATE = ("events", "phase_of", "loop_of", "loop_first", "phase_agents")

    def snapshot(self) -> dict[str, Any]:
        """The recorded events and graph mapping, to restore the recorder after a later step failed."""
        return {k: getattr(self, k) for k in self._STATE}

    @classmethod
    def restore(cls, data: dict[str, Any]) -> "TraceRecorder":
        recorder = cls()
        for k in cls._STATE:
            setattr(recorder, k, data.get(k) or type(getattr(recorder, k))())
        return recorder

    def write(self, out_dir: Path, stem: str) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = out_dir / f"{stem}_trace.jsonl"
        with jsonl.open("w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, default=str) + "\n")
        md = out_dir / f"{stem}_trace.md"
        md.write_text(self.markdown(stem), encoding="utf-8")
        return {"trace_jsonl": str(jsonl), "trace_md": str(md)}

    def markdown(self, stem: str) -> str:
        lines = [f"# Agent trace: {stem}", "", "Counts of 1-20 are masked. Tool payloads are shortened here; "
                 "the .jsonl file has them in full.", "", "## Phase coverage", "", self.coverage_markdown()]
        heading = None
        for e in self.events:
            here = (e["phase"], e["round"])
            if here != heading:
                heading = here
                lines += ["", f"## {e['phase'] or 'other'}" + (f", round {e['round']}" if e["round"] else ""), ""]
            if e["kind"] == "text":
                lines += [f"**{e['author']}**", "", e["text"].strip(), ""]
            elif e["kind"] == "tool_call":
                lines.append(f"- `{e['author']}` → `{e['name']}({_short(e['args'])})`")
            else:
                lines.append(f"- `{e['name']}` returned: {_short(e['response'])}")
        return "\n".join(lines) + "\n"


def _plain(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _short(value: Any) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= _MD_VALUE_LIMIT else text[:_MD_VALUE_LIMIT] + " …"
