"""Pre-registration and one-shot confirmation on the sealed holdout.

After the agents finish on the discovery table, the surviving candidates are frozen with everything
needed to test them: the feature's definition, the direction seen in discovery, the confounders and
exclusions of the final design, and the outcome. The record is hashed and written before the holdout
table is opened. Confirmation then rebuilds each feature on the holdout, tests it once, and corrects
for the number of pre-registered hypotheses (Benjamini-Hochberg). The agents never see the holdout or
these results; they are appended to the report as a deterministic section.

Test per outcome type:
* time to event (an incident condition): Cox model of the event on the standardised feature plus the
  confounders; hazard ratio per SD, its 95% CI and the E-value;
* anything else: partial Spearman correlation with the confounders (plain Spearman without any).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codas.aou.privacy import mask_small_counts
from codas.core.statistics import partial_spearman, safe_spearman
from codas.core.survival import cox_ph, e_value

_OPS = {"ratio": lambda a, b: a / b.replace(0.0, np.nan), "product": lambda a, b: a * b,
        "difference": lambda a, b: a - b, "sum": lambda a, b: a + b}


def _digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def preregister(state: dict[str, Any], out_dir: Path, stem: str, *, survival_time: str | None = None,
                max_hypotheses: int = 10) -> dict[str, Any]:
    """Freeze the final round's validated candidates as hypotheses and write ``<stem>_prereg.json``."""
    report = state.get("latest_report") or {}
    candidates = [c for c in report.get("candidates") or [] if c.get("verdict") == "validated"]
    candidates.sort(key=lambda c: -abs(float(c.get("rho") or 0)))
    proposed = {p.get("name"): p for p in state.get("proposed_features") or []}
    hypotheses = []
    for c in candidates[:max_hypotheses]:
        name = c["feature"]
        definition = ({"op": proposed[name]["op"], "a": proposed[name]["a"], "b": proposed[name]["b"]}
                      if name in proposed else {"components": list(c.get("components") or [])})
        hypotheses.append({"feature": name, "definition": definition, "direction": 1 if c["rho"] > 0 else -1,
                           "discovery_rho": round(float(c["rho"]), 4), "discovery_q": c.get("q_value")})
    record = {
        "stem": stem, "target": state.get("target_column"), "survival_time": survival_time,
        "confounders": list(state.get("confounder_columns") or []),
        "excluded": list(state.get("excluded_columns") or []),
        "hypotheses": hypotheses, "alpha": 0.05, "correction": "benjamini-hochberg",
        "registered_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    record["sha256"] = _digest(record)
    (out_dir / f"{stem}_prereg.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _feature(frame: pd.DataFrame, hypothesis: dict[str, Any]) -> pd.Series | None:
    """Rebuild a pre-registered feature on new rows from its recorded definition."""
    name, definition = hypothesis["feature"], hypothesis["definition"]
    num = lambda col: pd.to_numeric(frame[col], errors="coerce")  # noqa: E731
    if "op" in definition:
        if {definition["a"], definition["b"]} <= set(frame.columns):
            return _OPS[definition["op"]](num(definition["a"]), num(definition["b"]))
        return None
    if name in frame.columns:
        return num(name)
    parts = definition.get("components") or []
    if len(parts) == 2 and set(parts) <= set(frame.columns):
        a, b = num(parts[0]), num(parts[1])
        if f"{parts[0]}_over_{parts[1]}" == name:
            return a / b.replace(0.0, np.nan)
        if f"{parts[0]}_x_{parts[1]}" == name:  # the engine's product of standardised components
            return ((a - a.mean()) / a.std(ddof=0)) * ((b - b.mean()) / b.std(ddof=0))
    return None


def _bh(p: list[float]) -> list[float]:
    arr = np.asarray(p, dtype=float)
    q = np.full(len(arr), np.nan)
    ok = np.isfinite(arr)
    if ok.any():
        vals = arr[ok]
        order = np.argsort(vals)
        ranked = vals[order] * len(vals) / (np.arange(len(vals)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(len(vals))
        out[order] = np.clip(ranked, 0, 1)
        q[ok] = out
    return q.tolist()


def confirm(prereg: dict[str, Any], holdout_csv: Path, out_dir: Path) -> dict[str, Any]:
    """Test every pre-registered hypothesis once on the holdout; write ``<stem>_confirmation.json``."""
    if prereg.get("sha256") != _digest(prereg):
        raise ValueError("The pre-registration record was modified after it was registered.")
    frame = pd.read_csv(holdout_csv)
    target, time_col = prereg["target"], prereg.get("survival_time")
    # The follow-up time is the Cox model's time axis, never a covariate of it.
    confounders = [c for c in prereg["confounders"] if c in frame.columns and c != time_col]
    cov = frame[confounders].apply(pd.to_numeric, errors="coerce") if confounders else None
    rows = []
    for h in prereg["hypotheses"]:
        x = _feature(frame, h)
        row: dict[str, Any] = {"feature": h["feature"], "direction": h["direction"], "discovery_rho": h["discovery_rho"]}
        if x is None or x.notna().sum() < 30:
            row.update(status="not testable on the holdout", p=float("nan"))
            rows.append(row)
            continue
        if time_col:
            z = (x - x.mean()) / x.std(ddof=0)
            design = np.column_stack([z.to_numpy(float)] + ([cov.to_numpy(float)] if cov is not None else []))
            fit = cox_ph(pd.to_numeric(frame[time_col], errors="coerce"), pd.to_numeric(frame[target], errors="coerce"),
                         design)
            hr, lo, hi = fit.hazard_ratio(0)
            row.update(test="cox", hr_per_sd=round(hr, 4), ci95=[round(lo, 4), round(hi, 4)], p=float(fit.p[0]),
                       n=fit.n, events=fit.events, e_value=round(e_value(hr), 3),
                       same_direction=bool(np.sign(fit.coef[0]) == h["direction"]))
        else:
            y = pd.to_numeric(frame[target], errors="coerce")
            if cov is not None:
                rho, p, n = partial_spearman(x.to_numpy(float), y.to_numpy(float), cov.to_numpy(float))
            else:
                rho, p, n = safe_spearman(x.to_numpy(float), y.to_numpy(float))
            row.update(test="partial_spearman" if cov is not None else "spearman", rho=round(float(rho), 4),
                       p=float(p), n=int(n), same_direction=bool(np.sign(rho) == h["direction"]))
        rows.append(row)
    for row, q in zip(rows, _bh([r["p"] for r in rows])):
        row["q"] = q
        if "same_direction" in row:
            row["replicated"] = bool(row["same_direction"] and np.isfinite(q) and q <= prereg["alpha"])
            row["status"] = "replicated" if row["replicated"] else "not replicated"
    result = mask_small_counts({"prereg_sha256": prereg["sha256"], "holdout_rows": int(len(frame)),
                                "confounders": confounders, "hypotheses": rows,
                                "replicated": sum(1 for r in rows if r.get("replicated")),
                                "tested": sum(1 for r in rows if "replicated" in r)})
    (out_dir / f"{prereg['stem']}_confirmation.json").write_text(json.dumps(result, indent=2, default=str),
                                                                 encoding="utf-8")
    return result


def confirmation_markdown(result: dict[str, Any]) -> str:
    """The deterministic replication section appended to the report."""
    lines = ["## Holdout replication (pre-registered, deterministic)", "",
             f"{result['replicated']} of {result['tested']} pre-registered candidates replicated on the sealed "
             f"holdout ({result['holdout_rows']} participants; BH-corrected across the pre-registered set; "
             f"adjusted for {', '.join(result['confounders']) or 'nothing'}). The agents never saw these data "
             f"or results. Pre-registration hash: `{result['prereg_sha256'][:16]}`.", "",
             "| feature | discovery rho | holdout effect | 95% CI | q | E-value | status |", "|---|---|---|---|---|---|---|"]
    for r in result["hypotheses"]:
        if r.get("test") == "cox":
            effect, ci, ev = f"HR/SD {r['hr_per_sd']}", f"{r['ci95'][0]}-{r['ci95'][1]}", r.get("e_value", "")
        elif "rho" in r:
            effect, ci, ev = f"rho {r['rho']}", "", ""
        else:
            effect, ci, ev = "", "", ""
        q = r.get("q")
        q = q if isinstance(q, float) else None
        lines.append(f"| {r['feature']} | {r['discovery_rho']} | {effect} | {ci} | "
                     f"{'' if q is None or not np.isfinite(q) else f'{q:.3g}'} | {ev} | {r.get('status', '')} |")
    return "\n".join(lines) + "\n"
