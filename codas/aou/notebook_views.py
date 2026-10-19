"""Tables and figures the notebook shows at each step, built only from masked, aggregate outputs.

Every function reads what a step already produced (the environment snapshot, the schema check, the
feasibility rows, the run summaries and the files each run wrote) and returns a pandas DataFrame or a
matplotlib Figure for ``display``. Nothing here reads a participant-level table. Figures are made with
``matplotlib.figure.Figure`` (not pyplot), so a notebook shows each one once, where it is displayed.
"""

from __future__ import annotations

import json
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
        rows.append({"table": s.table, "prefix": s.prefix, "columns needed": len(needed),
                     "found": len(needed) - len(missing), "missing": ", ".join(missing) or "-"})
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
    ax.set_title(f"{stem}: candidates in the last round (blue = validated, gray = conditional)", loc="left",
                 fontsize=9, color=INK)
    _style(ax)
    return Chart(fig)


def cohort_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    table = _read(_out_dir(result) / f"{stem}_state.json").get("cohort_table")
    return pd.DataFrame(table["rows"], columns=table["columns"]) if table else None


def confirmation_table(result: dict[str, Any], stem: str) -> pd.DataFrame | None:
    """The sealed-holdout test of each pre-registered candidate and its falsification results."""
    conf = _read(_out_dir(result) / f"{stem}_confirmation.json")
    if not conf:
        return None
    rows = []
    for h in conf.get("hypotheses") or []:
        ci = h.get("ci95") or [None, None]
        effect = (f"HR/SD {h.get('hr_per_sd')}" if h.get("test") == "cox" else
                  f"rho {h.get('rho')}" if "rho" in h else "")
        q = h.get("q_ledger", h.get("q"))
        rows.append({"feature": h["feature"], "discovery rho": h.get("discovery_rho"), "holdout effect": effect,
                     "95% CI": "" if ci[0] is None else f"{ci[0]} to {ci[1]}",
                     "q": None if not isinstance(q, float) or not np.isfinite(q) else round(q, 4),
                     "status": h.get("status"), "robust": h.get("robust"),
                     "dose-response": (h.get("dose_response") or {}).get("monotone"),
                     "failed checks": ", ".join(h.get("falsification_failed") or []) or "-"})
    return pd.DataFrame(rows)


def forest_figure(result: dict[str, Any]):
    """Holdout effects with 95% CIs for every pre-registered candidate across outcomes (filled = replicated)."""
    out = _out_dir(result)
    items = {"cox": [], "rank": []}
    for r in result.get("runs") or []:
        conf = _read(out / f"{r['cohort']}_confirmation.json")
        for h in conf.get("hypotheses") or []:
            ci = h.get("ci95") or [None, None]
            if ci[0] is None:
                continue
            kind, est = ("cox", h.get("hr_per_sd")) if h.get("test") == "cox" else ("rank", h.get("rho"))
            if _num(est) is not None:
                items[kind].append((f"{r['cohort']}: {h['feature']}"[:70], est, ci[0], ci[1], bool(h.get("replicated"))))
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
            from matplotlib.ticker import FuncFormatter

            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v, 1) in (0.5, 0.7, 1.5, 2.0, 3.0)
                                                       else ""))
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
