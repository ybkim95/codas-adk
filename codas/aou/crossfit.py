"""Two-way confirmation: each half of the participants discovers, the other confirms.

The split seals half the participants (``cohort.holdout_bucket``). A run with ``discovery_share = "A"`` (the
default) lets the agents see share A and tests what they register on share B; a run with ``"B"`` swaps the roles
(``cohort.sealed_condition``): it discovers on B and tests on A. Each direction has its own run folder, phenome
scan, lessons and holdout ledger (the ledger key names the share), so nothing one direction learned reaches the
other's agents, and each direction's verdicts keep their own error control. A hypothesis that replicates in both
directions has two independent confirmations; this module puts the two side by side.

What the two directions cannot hide is what people learned from the first: a design changed after reading the
first direction's results may favour hypotheses that worked there. The report says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SHARES = ("A", "B")
FILE = "crossfit_summary.md"


def run_folder(out: Path, share: str) -> Path:
    """The run folder of a direction: ``out`` itself for share A, a sibling ``<out>_share_b`` for share B."""
    out = Path(out)
    base = out.with_name(out.name[: -len("_share_b")]) if out.name.endswith("_share_b") else out
    return base if share == "A" else base.with_name(base.name + "_share_b")


def _confirmations(folder: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(Path(folder).glob("*_confirmation.json")):
        try:
            out[path.name[: -len("_confirmation.json")]] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _effect(row: dict[str, Any]) -> str:
    if row.get("hr_per_sd") is not None:
        ci = row.get("ci95") or ["?", "?"]
        return f"HR {row['hr_per_sd']} ({ci[0]}-{ci[1]})"
    if row.get("rho") is not None:
        ci = row.get("ci95") or ["?", "?"]
        return f"rho {row['rho']} ({ci[0]}-{ci[1]})"
    return row.get("status") or "-"


def _verdict(row: dict[str, Any]) -> str:
    if row.get("replicated"):
        return "replicated" + (" (robust)" if row.get("robust") else "")
    if row.get("reproduced"):
        return "reproduced (reference)"
    return row.get("status") or "not replicated"


def summary(out: Path) -> dict[str, Any]:
    """Every outcome tested in both directions, and each feature tested in either, side by side."""
    a, b = _confirmations(run_folder(out, "A")), _confirmations(run_folder(out, "B"))
    outcomes = []
    for stem in sorted(set(a) & set(b)):
        rows_a = {r["feature"]: r for r in a[stem].get("hypotheses") or [] if "same_direction" in r}
        rows_b = {r["feature"]: r for r in b[stem].get("hypotheses") or [] if "same_direction" in r}
        features = []
        for feature in sorted(set(rows_a) | set(rows_b)):
            ra, rb = rows_a.get(feature), rows_b.get(feature)
            both = bool(ra and rb and (ra.get("replicated") or ra.get("reproduced"))
                        and (rb.get("replicated") or rb.get("reproduced")) and ra.get("direction") == rb.get("direction"))
            features.append({"feature": feature,
                             "a_to_b": None if ra is None else {"effect": _effect(ra), "verdict": _verdict(ra)},
                             "b_to_a": None if rb is None else {"effect": _effect(rb), "verdict": _verdict(rb)},
                             "both_directions": both})
        outcomes.append({"outcome": stem, "features": features,
                         "confirmed_both_ways": [f["feature"] for f in features if f["both_directions"]]})
    return {"outcomes": outcomes, "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a))}


def markdown(result: dict[str, Any]) -> str:
    if not result.get("outcomes"):
        return ""
    lines = ["## Two-way confirmation", "",
             "Each half discovered and the other confirmed: A to B (agents on share A, tested on share B) and B to A. "
             "Each direction has its own scan, lessons and ledger, and keeps its own error control; a feature "
             "replicated both ways has two independent confirmations. Designs changed after reading the first "
             "direction may still favour what worked there.", "",
             "| outcome | feature | A to B | B to A | both ways |", "|---|---|---|---|---|"]
    for o in result["outcomes"]:
        for f in o["features"]:
            cell = lambda d: "-" if d is None else f"{d['effect']}: {d['verdict']}"  # noqa: E731
            lines.append(f"| {o['outcome']} | {f['feature']} | {cell(f['a_to_b'])} | {cell(f['b_to_a'])} | "
                         f"{'yes' if f['both_directions'] else '-'} |")
    return "\n".join(lines) + "\n"


def write(out: Path) -> Path | None:
    """Write the summary beside share A's reports when both directions have tested an outcome."""
    result = summary(out)
    text = markdown(result)
    if not text:
        return None
    path = run_folder(out, "A") / FILE
    path.write_text(text, encoding="utf-8")
    return path
