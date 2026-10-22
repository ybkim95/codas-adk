"""Tables and figures the notebook shows at each step, built only from masked, aggregate outputs.

Every function reads what a step already produced (the environment snapshot, the schema check, the
feasibility rows, the run summaries and the files each run wrote) and returns a pandas DataFrame or a
chart for ``display``; ``fit`` shows a table narrow enough for the page. Nothing here reads a
participant-level table. Figures are made with
``matplotlib.figure.Figure`` (not pyplot), so a notebook shows each one once, where it is displayed.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codas.aou.privacy import MASK

# Light-mode palette (validated categorical order) and text inks.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ad"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


class Chart:
    """A figure that a notebook shows as an image. A bare matplotlib Figure made without pyplot has no
    display formatter in some Jupyter setups and prints as ``<Figure size ...>``; a PNG always renders."""

    def __init__(self, figure: Any):
        self.figure = figure

    def _repr_png_(self) -> bytes:
        import io

        buffer = io.BytesIO()
        self.figure.savefig(buffer, format="png", dpi=110, facecolor=SURFACE)
        return buffer.getvalue()

    def _repr_html_(self) -> str:
        """The same PNG, scaled down to the output column (a bare PNG keeps its pixel width and is cut
        off at the edge of a narrow column or a printed page). Jupyter prefers this over the PNG."""
        data = base64.b64encode(self._repr_png_()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%;height:auto" alt="chart">'

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        self.figure.savefig(*args, **kwargs)


def _figure(width: float, height: float):
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(which="both", colors=INK_2, labelsize=8)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# --- fitting tables to the page ------------------------------------------------------------------

_CODE_SPAN = re.compile(r"(`[^`]*`)")


def _breakable(text: str) -> str:
    """Escaped text with a line-break opportunity after each underscore, so long identifiers wrap
    instead of pushing a table past the edge of the page. Numbers and other text are unchanged."""
    return html.escape(text).replace("_", "_<wbr>")


class Fitted:
    """A table shown to fit a notebook column or a printed page: no index, a slightly smaller font, and
    identifiers that wrap at underscores. ``table`` is the DataFrame itself."""

    def __init__(self, table: pd.DataFrame):
        self.table = table

    def _repr_html_(self) -> str:
        cells = self.table.astype(object).where(self.table.notna(), "")
        # Text reads left to right (Jupyter right-aligns every cell); numbers keep their alignment.
        cells = cells.apply(lambda col: col.map(  # Series.map works in every pandas version
            lambda v: f'<div style="text-align:left">{_breakable(v)}</div>' if isinstance(v, str) else v))
        cells.columns = [_breakable(str(c)) for c in cells.columns]
        body = cells.to_html(index=False, escape=False, border=0)
        return f'<div style="max-width:100%;font-size:0.85em">{body}</div>'


def fit(table: pd.DataFrame) -> Fitted:
    return Fitted(table)


def wrap_tables(markdown: str) -> str:
    """Markdown whose table rows let identifiers wrap at underscores; code spans and prose are left as
    written."""
    lines = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("|"):
            line = "".join(part if part.startswith("`") else part.replace("_", "_<wbr>")
                           for part in _CODE_SPAN.split(line))
        lines.append(line)
    return "\n".join(lines)


# --- setup and checks -------------------------------------------------------------------------

def environment_table(env: dict[str, Any], **extra: Any) -> pd.DataFrame:
    """The run's environment (``job.environment()``) plus any extra settings, one row per item."""
    rows = [(k, v) for k, v in extra.items()]
    rows += [("code commit", env.get("commit")), ("analysis code version", env.get("analysis_code")),
             ("python", env.get("python")), ("CPUs", env.get("cpus")),
             ("memory free / total (GB)", f"{env.get('memory_available_gb')} / {env.get('memory_total_gb')}"),
             ("disk free (GB)", env.get("disk_free_gb"))]
    rows += [(f"package {k}", v) for k, v in (env.get("packages") or {}).items()]
    rows += [(k, v) for k, v in (env.get("settings") or {}).items()]
    return pd.DataFrame(rows, columns=["setting", "value"]).astype({"value": str})


def schema_table(spec: Any, present: pd.DataFrame) -> pd.DataFrame:
    """For each wearable table the spec reads: the columns it needs, how many the CDR has, any missing."""
    have = set(zip(present["table_name"].astype(str), present["column_name"].astype(str)))
    rows = []
    for s in spec.wearables:
        needed = [s.date_column, *s.columns] + ([s.filter_column] if s.filter_column else [])
        missing = [c for c in needed if (s.table, c) not in have]
        rows.append({"table": s.table, "prefix": s.prefix, "kind": "daily" + (" (optional)" if s.optional else ""),
                     "columns needed": len(needed), "found": len(needed) - len(missing),
                     "missing": ", ".join(missing) or "-"})
    for p in getattr(spec, "phenotypes", []):
        needed = [c for c in p.needed() if c != "person_id"]
        missing = [c for c in needed if (p.table, c) not in have]
        rows.append({"table": p.table, "prefix": p.prefix, "kind": p.kind.replace("_", " ") + " (optional)",
                     "columns needed": len(needed), "found": len(needed) - len(missing),
                     "missing": ", ".join(missing) or "-"})
    return pd.DataFrame(rows)


def concept_table(items: list[tuple[str, Any]], concept_rows: list[dict[str, Any]],
                  problems: dict[str, str | None]) -> pd.DataFrame:
    """One row per (role, item, concept id): the concept's name and domain in this CDR and whether the
    item passed the check. ``items`` pairs a role label with an object that has name/concept_ids."""
    by_id = {int(r["concept_id"]): r for r in concept_rows}
    rows = []
    for role, item in items:
        for cid in item.concept_ids:
            found = by_id.get(int(cid), {})
            rows.append({"role": role, "name": item.name, "concept id": cid,
                         "name in this CDR": found.get("concept_name", "(not found)"),
                         "domain": found.get("domain_id", "-"), "check": problems.get(item.name) or "ok"})
    return pd.DataFrame(rows)


# --- feasibility ------------------------------------------------------------------------------

def feasibility_figure(rows: list[dict[str, Any]], chosen: list[str]):
    """Participants per lab outcome and cases per diagnosis; outcomes selected for CoDaS in blue.
    Masked counts (1-20) are drawn as zero and labelled."""
    groups = [("measurement", "participants with the lab value", "n_participants"),
              ("condition", "diagnoses: participants with the diagnosis (cases)", "n_cases")]
    present = [g for g in groups if any(r["kind"] == g[0] for r in rows)]
    if not present:
        return None
    heights = [max(1.5, 0.32 * sum(r["kind"] == g[0] for r in rows) + 0.8) for g in present]
    fig = _figure(8, sum(heights))
    axes = fig.subplots(len(present), 1, height_ratios=heights, squeeze=False)[:, 0]
    for ax, (kind, title, key) in zip(axes, present):
        part = sorted([r for r in rows if r["kind"] == kind], key=lambda r: _num(r.get(key)) or 0)
        values = [_num(r.get(key)) or 0 for r in part]
        colors = [BLUE if r["name"] in chosen else GRAY for r in part]
        ax.barh(range(len(part)), values, color=colors, height=0.6)
        ax.set_yticks(range(len(part)), [r.get("label", r["name"]) for r in part])
        top = max(values or [1]) or 1
        for i, (r, v) in enumerate(zip(part, values)):
            text = MASK if r.get(key) == MASK else f"{int(v):,}"
            ax.text(v + top * 0.01, i, text, va="center", fontsize=7, color=INK_2)
        ax.set_xlim(0, top * 1.15)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        _style(ax)
    fig.suptitle("Feasibility screen (blue = selected for CoDaS)", x=0.01, ha="left", fontsize=10, color=INK)
    return Chart(fig)


# --- runs ---------------------------------------------------------------------------------------

def _out_dir(result: dict[str, Any]) -> Path:
    return Path(result["summary_path"]).parent


def _trace(out: Path, stem: str) -> pd.DataFrame:
    path = out / f"{stem}_trace.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def literature_table(result: dict[str, Any]) -> pd.DataFrame | None:
    """Per outcome (failed ones too, from their trace): how many literature searches the agents ran, how
    many came back with sources, and what the others returned. Shows whether search works here."""
    out = _out_dir(result)
    rows = []
    for r in result.get("runs") or []:
        trace = _trace(out, r["cohort"])
        if trace.empty or not {"kind", "name"} <= set(trace.columns):
            continue
        replies = [x.get("result", x) if isinstance(x, dict) and "grounded" not in x else x
                   for x in trace.loc[(trace["kind"] == "tool_response") & (trace["name"] == "search_literature"),
                                      "response"]]
        replies = [x for x in replies if isinstance(x, dict)]
        if not replies:
            continue
        outcomes: dict[str, int] = {}
        for x in replies:
            what = ("sources returned" if x.get("grounded") else
                    str(x.get("note") or x.get("error") or "no sources returned"))[:160]
            outcomes[what] = outcomes.get(what, 0) + 1
        rows.append({"outcome": r["cohort"], "searches": len(replies),
                     "with sources": sum(bool(x.get("grounded")) for x in replies),
                     "what came back": "; ".join(f"{k} (x{v})" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1]))})
    return pd.DataFrame(rows) if rows else None


def run_minutes(out: Path, stem: str) -> float | None:
    trace = _trace(out, stem)
    if trace.empty or "time" not in trace:
        return None
    return round((trace["time"].max() - trace["time"].min()) / 60, 1)


def _metric(text: Any) -> Any:
    """'r2=0.0309205187' -> 'r2 = 0.031'."""
    if isinstance(text, str) and "=" in text:
        name, value = text.split("=", 1)
        try:
            return f"{name} = {float(value):.3f}"
        except ValueError:
            return text
    return text


def runs_table(result: dict[str, Any]) -> pd.DataFrame:
    """One row per outcome: how it ended and its headline numbers (masked)."""
    out = _out_dir(result)
    rows = []
    for r in result.get("runs") or []:
        stem = r["cohort"]
        conf = _read(out / f"{stem}_confirmation.json")
        rows.append({
            "outcome": stem, "status": "failed" if r.get("failed") else "finished",
            "participants (discovery)": r.get("rows"), "rounds": r.get("rounds"),
            "validated (last round)": r.get("validated_count"), "held-out model": _metric(r.get("ml_metric")),
            "replicated on holdout": r.get("replication"), "robust": r.get("robust"),
            "negative-control flags": sum(bool(h.get("falsification_failed")) for h in conf.get("hypotheses") or []),
            "minutes": run_minutes(out, stem), "error": r.get("failed") or "",
        })
    return pd.DataFrame(rows)


def timing_table(result: dict[str, Any]) -> pd.DataFrame:
    """Minutes each outcome spent in each phase (from the agent trace): where the time went."""
    out = _out_dir(result)
    frames = []
    for r in result.get("runs") or []:
        trace = _trace(out, r["cohort"])
        if trace.empty:
            continue
        trace = trace.sort_values("time")
        trace["phase"] = trace["phase"].replace("", "other")
        trace["next"] = trace["time"].shift(-1).fillna(trace["time"].iloc[-1])
        spent = (trace["next"] - trace["time"]).groupby(trace["phase"], sort=False).sum() / 60
        frames.append(spent.rename(r["cohort"]))
    return pd.DataFrame(frames).fillna(0).round(1) if frames else pd.DataFrame()


def timing_figure(result: dict[str, Any]):
    table = timing_table(result)
    if table.empty:
        return None
    palette = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig = _figure(8, 0.45 * len(table) + 1.4)
    ax = fig.subplots()
    left = np.zeros(len(table))
    for i, phase in enumerate(table.columns):
        ax.barh(table.index, table[phase], left=left, color=palette[i % len(palette)], height=0.6,
                edgecolor=SURFACE, linewidth=1, label=phase)
        left += table[phase].to_numpy()
    for y, total in enumerate(left):
        ax.text(total, y, f"  {total:.1f} min", va="center", fontsize=7, color=INK_2)
    ax.set_xlim(0, (left.max() or 1) * 1.15)
    ax.set_xlabel("minutes", fontsize=8, color=INK_2)
    ax.legend(fontsize=7, frameon=False, ncol=min(4, len(table.columns)), loc="lower left",
              bbox_to_anchor=(0, 1.0))
    fig.suptitle("Time per phase and outcome", x=0.01, ha="left", fontsize=10, color=INK)
    _style(ax)
    return Chart(fig)


def candidates_figure(result: dict[str, Any], stem: str):
    """The final round's surviving candidates for one outcome: Spearman rho, validated in blue."""
    state = _read(_out_dir(result) / f"{stem}_state.json")
    rounds = state.get("rounds") or []
    cands = [c for c in (rounds[-1].get("passing_candidates") if rounds else []) or [] if _num(c.get("rho")) is not None]
    if not cands:
        return None
    cands = sorted(cands, key=lambda c: c["rho"])
    fig = _figure(8, 0.3 * len(cands) + 1.0)
    ax = fig.subplots()
    ax.barh(range(len(cands)), [c["rho"] for c in cands], height=0.6,
            color=[BLUE if c.get("verdict") == "validated" else GRAY for c in cands])
    ax.set_yticks(range(len(cands)), [c["feature"][:60] for c in cands])
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Spearman rho with the outcome (discovery share)", fontsize=8, color=INK_2)
    ax.set_title(f"{stem}: candidates in the last round", loc="left", fontsize=9, color=INK)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=BLUE, label="validated"), Patch(color=GRAY, label="conditional")], fontsize=7,
              frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1))
    _style(ax)
    return Chart(fig)


def cohort_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    table = _read(_out_dir(result) / f"{stem}_state.json").get("cohort_table")
    return pd.DataFrame(table["rows"], columns=table["columns"]) if table else None


def confirmation_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    """The sealed-holdout test of each pre-registered hypothesis (a-priori family first, then the
    exploratory candidates) and its falsification results, including the wear-adherence check."""
    conf = _read(_out_dir(result) / f"{stem}_confirmation.json")
    if not conf:
        return None
    rows = []
    for h in conf.get("hypotheses") or []:
        ci = h.get("ci95") or [None, None]
        effect = (f"HR/SD {h.get('hr_per_sd')}" if h.get("test") == "cox" else
                  f"rho {h.get('rho')}" if "rho" in h else "")
        q = h.get("q_used", h.get("q_ledger", h.get("q")))
        family = h.get("family", "exploratory").replace("_", "-")
        adherence = h.get("adherence_adjusted")
        rows.append({"feature": h["feature"],
                     "family": family + (f" ({h['knownness']})" if h.get("knownness") else ""),
                     "discovery rho": h.get("discovery_rho"), "holdout effect": effect,
                     "95% CI": "" if ci[0] is None else f"{ci[0]} to {ci[1]}",
                     "q": None if not isinstance(q, float) or not np.isfinite(q) else round(q, 4),
                     "status": h.get("status"), "robust": h.get("robust"),
                     "holds after adherence": None if adherence is None else adherence.get("holds"),
                     "within-person": _within_text(h.get("within_person")),
                     "failed checks": ", ".join(h.get("falsification_failed") or []) or "-"})
    columns = ["feature", "family", "discovery rho", "holdout effect", "95% CI", "q", "status", "robust",
               "holds after adherence", "within-person", "failed checks"]
    return pd.DataFrame(rows, columns=columns)  # keeps its columns when nothing was pre-registered


def _within_text(fit: dict[str, Any] | None) -> str:
    if not fit:
        return ""
    if "error" in fit:
        return "n/a"
    w = fit["within"]
    return f"{w['estimate']:.3g} ({w['ci95'][0]:.3g} to {w['ci95'][1]:.3g}), p {w['p']:.2g}"


def hypotheses_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    """The hypotheses the agents registered from the literature before any statistic existed."""
    registered = _read(_out_dir(result) / f"{stem}_state.json").get("a_priori_hypotheses") or []
    if not registered:
        return None
    return pd.DataFrame([{"feature": h.get("feature"), "predicted": "positive" if h.get("direction", 0) > 0 else "negative",
                          "knownness": h.get("knownness"), "construct": h.get("construct"),
                          "mechanism": h.get("mechanism"), "citation": h.get("citation")} for h in registered])


def analyses_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    """Every method and analysis run, by round: the method pools' runs (``method:...``) and the analyses
    the agents chose (survival screen, subgroups, multivariable, non-linear, dose-response, mixed model)."""
    from codas.core.methods import record_headline

    ran = _read(_out_dir(result) / f"{stem}_state.json").get("analyses") or []
    if not ran:
        return None
    return pd.DataFrame([{"round": a.get("round"), "analysis": a.get("tool"),
                          "arguments": ", ".join(f"{k}={v}" for k, v in (a.get("args") or {}).items() if v),
                          "result": record_headline(a)} for a in ran])


def debates_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    """The per-candidate debate: each final candidate's verdict (keep, caveat, concede) and why."""
    debates = _read(_out_dir(result) / f"{stem}_state.json").get("candidate_debates") or []
    if not debates:
        return None
    return pd.DataFrame([{k: d.get(k) for k in ("feature", "verdict", "reason", "evidence")} for d in debates])


def lessons_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    """Lessons this outcome's agents recorded for later outcomes on the same cohort."""
    lessons = _read(_out_dir(result) / f"{stem}_state.json").get("lessons") or []
    if not lessons:
        return None
    return pd.DataFrame([{"lesson": x.get("lesson"), "about": ", ".join(x.get("about") or []),
                          "evidence": x.get("evidence")} for x in lessons])


_HR_TICKS = (0.25, 0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0)


def forest_figure(result: dict[str, Any]):
    """Holdout effects with 95% CIs for every pre-registered candidate across outcomes (filled = replicated)."""
    out = _out_dir(result)
    items = {"cox": [], "rank": []}
    names = [r["cohort"] for r in result.get("runs") or []]
    shared = os.path.commonprefix(names)
    shared = shared[:shared.rfind("_") + 1]  # the part every outcome name starts with, e.g. a run prefix
    for r in result.get("runs") or []:
        conf = _read(out / f"{r['cohort']}_confirmation.json")
        for h in conf.get("hypotheses") or []:
            ci = h.get("ci95") or [None, None]
            if ci[0] is None:
                continue
            kind, est = ("cox", h.get("hr_per_sd")) if h.get("test") == "cox" else ("rank", h.get("rho"))
            if _num(est) is not None:
                items[kind].append((f"{r['cohort'][len(shared):]}: {h['feature']}"[:70], est, ci[0], ci[1], bool(h.get("replicated"))))
    panels = [(k, v) for k, v in items.items() if v]
    if not panels:
        return None
    heights = [0.3 * len(v) + 0.9 for _, v in panels]
    fig = _figure(8, sum(heights))
    axes = fig.subplots(len(panels), 1, height_ratios=heights, squeeze=False)[:, 0]
    for ax, (kind, rows) in zip(axes, panels):
        for i, (label, est, lo, hi, filled) in enumerate(rows):
            ax.plot([lo, hi], [i, i], color=BLUE if filled else GRAY, linewidth=2)
            ax.plot(est, i, "o", markersize=7, color=BLUE if filled else GRAY,
                    markerfacecolor=BLUE if filled else SURFACE, markeredgewidth=1.5)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        null = 1.0 if kind == "cox" else 0.0
        ax.axvline(null, color=INK_2, linewidth=0.8, linestyle="--")
        if kind == "cox":
            from matplotlib.ticker import FixedLocator, NullLocator

            ax.set_xscale("log")
            lo = min(min(r[2] for r in rows), 1.0)
            hi = max(max(r[3] for r in rows), 1.0)
            ticks = [t for t in _HR_TICKS if lo / 1.05 <= t <= hi * 1.05] or [1.0]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlabel("hazard ratio per SD" if kind == "cox" else "rank correlation", fontsize=8, color=INK_2)
        _style(ax)
    fig.suptitle("Sealed holdout: pre-registered candidates (filled = replicated)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    return Chart(fig)


# --- background job ------------------------------------------------------------------------------

def heartbeat_table(log_path: str | Path) -> pd.DataFrame:
    path = Path(log_path).with_name("explore_heartbeat.jsonl")
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def log_digest_table(log_path: str | Path, top: int = 10, min_count: int = 3) -> pd.DataFrame | None:
    """The lines a job log repeated most (at least ``min_count`` times), with counts and first/last time:
    what flooded a run's output."""
    from codas.aou.job import log_digest

    rows = [r for r in log_digest(log_path, top=top) if r["count"] >= min_count]
    if not rows:
        return None
    table = pd.DataFrame(rows)[["count", "first", "last", "line"]]
    table["line"] = table["line"].str.slice(0, 300)
    return table.rename(columns={"count": "times", "line": "log line (time stamp removed)"})


def heartbeat_figure(log_path: str | Path):
    """The job's memory over time (process and machine) and which outcome it was on."""
    beats = heartbeat_table(log_path)
    if beats.empty:
        return None
    fig = _figure(8, 2.6)
    ax = fig.subplots()
    ax.plot(beats["elapsed_min"], beats["rss_gb"], color=BLUE, linewidth=2, label="job process memory (GB)")
    ax.plot(beats["elapsed_min"], beats["memory_available_gb"], color=ORANGE, linewidth=2,
            label="machine memory free (GB)")
    for outcome, part in beats.groupby("outcome", sort=False):
        if outcome:
            ax.axvline(part["elapsed_min"].min(), color=GRAY, linewidth=0.8)
            ax.text(part["elapsed_min"].min(), 1.01, f" {outcome}", fontsize=7, color=INK_2,
                    transform=ax.get_xaxis_transform(), va="bottom")
    ax.set_xlabel("minutes since the job started", fontsize=8, color=INK_2)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Background job: memory over time (one point per heartbeat)", x=0.01, ha="left", fontsize=10,
                 color=INK)
    _style(ax)
    return Chart(fig)
