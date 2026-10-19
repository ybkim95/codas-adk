"""Pre-registration and one-shot confirmation on the sealed holdout.

After the agents finish on the discovery table, the surviving candidates are frozen with everything
needed to test them: the feature's definition, the direction seen in discovery, the confounders and
exclusions of the final design, the outcome, and the falsification tests to run. The record is hashed
and written before the holdout table is opened. Confirmation then rebuilds each feature on the
holdout, tests it once, and corrects for multiple testing (Benjamini-Hochberg). The agents never see
the holdout or these results; they are appended to the report as a deterministic section.

Test per outcome type:
* time to event (an incident condition): Cox model of the event on the standardised feature plus the
  confounders; hazard ratio per SD, its 95% CI and the E-value;
* anything else: partial Spearman correlation with the confounders (plain Spearman without any).

Falsification, for every candidate, on the same holdout and with the same confounders:
* negative-control outcomes (columns ``nc_<name>`` and ``nc_<name>_days`` of the holdout table):
  conditions the exposure should not cause. An association (p below 0.05 divided by the number of
  controls tested, so a null candidate is flagged at most 5% of the time) flags the candidate, since
  it points to confounding or to differences in health-care contact rather than to the outcome;
* dose-response: the effect across quintiles of the feature should change monotonically in the
  pre-registered direction;
* lag sensitivity (time to event): dropping the first ``sensitivity_lag_days`` of follow-up, where
  undiagnosed disease could have changed the baseline wearable data, should keep the direction.
A candidate is ``robust`` when it replicates and passes every falsification test that could run.

Positive controls (columns with a known association with the outcome) are tested the same way, to
show that the confirmation can detect a real effect on this holdout.

Every hypothesis tested on a holdout is appended to a ledger next to the outputs
(``holdout_ledger.jsonl``), keyed by which participants the holdout holds. Replication is judged on
the BH q-value across every hypothesis ever tested on that holdout, so re-running the analysis, or
running other outcomes on the same participants, cannot buy extra chances at a false replication.
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
NC_PREFIX = "nc_"
LEDGER_NAME = "holdout_ledger.jsonl"
DEFAULT_SENSITIVITY_LAG_DAYS = 365
_MIN_ROWS = 30
_MIN_CONTROL_EVENTS = 10  # a negative control with fewer events cannot show an association
_QUANTILES = 5
_MONOTONE_RHO = 0.8  # rank correlation between quantile and effect that counts as monotone


def _digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def preregister(state: dict[str, Any], out_dir: Path, stem: str, *, survival_time: str | None = None,
                max_hypotheses: int = 10, negative_controls: list[str] | tuple[str, ...] = (),
                positive_controls: list[str] | tuple[str, ...] = (),
                sensitivity_lag_days: int = DEFAULT_SENSITIVITY_LAG_DAYS) -> dict[str, Any]:
    """Freeze the final round's validated candidates as hypotheses and write ``<stem>_prereg.json``.

    ``negative_controls`` are control names (the holdout has ``nc_<name>`` and ``nc_<name>_days``),
    ``positive_controls`` column names. The record also counts the tests the discovery ran, so the
    report can say how many looks preceded the registration."""
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
    rounds = state.get("rounds") or []
    record = {
        "stem": stem, "target": state.get("target_column"), "survival_time": survival_time,
        "confounders": list(state.get("confounder_columns") or []),
        "excluded": list(state.get("excluded_columns") or []),
        "hypotheses": hypotheses, "alpha": 0.05, "correction": "benjamini-hochberg",
        "negative_controls": list(negative_controls), "positive_controls": list(positive_controls),
        "sensitivity_lag_days": int(sensitivity_lag_days),
        "discovery_rounds": len(rounds),
        "discovery_variants_evaluated": sum(int(r.get("evaluated_count") or 0) for r in rounds),
        "discovery_features_proposed": len(proposed),
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


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _cox(time: pd.Series, event: pd.Series, x: pd.Series, cov: pd.DataFrame | None) -> dict[str, Any]:
    """Cox HR per SD of ``x`` adjusted for ``cov``."""
    z = (x - x.mean()) / x.std(ddof=0)
    design = np.column_stack([z.to_numpy(float)] + ([cov.to_numpy(float)] if cov is not None else []))
    fit = cox_ph(time, event, design)
    hr, lo, hi = fit.hazard_ratio(0)
    return {"test": "cox", "hr_per_sd": round(hr, 4), "ci95": [round(lo, 4), round(hi, 4)], "p": float(fit.p[0]),
            "n": fit.n, "events": fit.events, "e_value": round(e_value(hr), 3), "sign": float(np.sign(fit.coef[0]))}


def _rank(x: pd.Series, y: pd.Series, cov: pd.DataFrame | None) -> dict[str, Any]:
    if cov is not None:
        rho, p, n = partial_spearman(x.to_numpy(float), y.to_numpy(float), cov.to_numpy(float))
    else:
        rho, p, n = safe_spearman(x.to_numpy(float), y.to_numpy(float))
    half = 1.959964 / np.sqrt(max(n - 3, 1))  # Fisher-z interval
    lo, hi = (float(np.tanh(np.arctanh(np.clip(rho, -0.999999, 0.999999)) + s * half)) for s in (-1, 1))
    return {"test": "partial_spearman" if cov is not None else "spearman", "rho": round(float(rho), 4),
            "ci95": [round(lo, 4), round(hi, 4)], "p": float(p), "n": int(n), "sign": float(np.sign(rho))}


def _effect(frame: pd.DataFrame, x: pd.Series, target: str, time_col: str | None,
            cov: pd.DataFrame | None) -> dict[str, Any]:
    if time_col:
        return _cox(_num(frame, time_col), _num(frame, target), x, cov)
    return _rank(x, _num(frame, target), cov)


def _dose_response(frame: pd.DataFrame, x: pd.Series, target: str, time_col: str | None,
                   cov: pd.DataFrame | None, direction: int) -> dict[str, Any] | None:
    """Effect by quantile of ``x``: log HR against the lowest quantile (adjusted) for a time to event,
    else the mean outcome; monotone when it rises (or falls) with the quantile as pre-registered."""
    bins = pd.qcut(x.rank(method="first"), _QUANTILES, labels=False) if x.notna().sum() >= _QUANTILES * _MIN_ROWS else None
    if bins is None:
        return None
    k = int(bins.max()) + 1
    if time_col:
        dummies = np.column_stack([(bins == j).astype(float) for j in range(1, k)])
        dummies[bins.isna().to_numpy()] = np.nan
        design = np.column_stack([dummies] + ([cov.to_numpy(float)] if cov is not None else []))
        fit = cox_ph(_num(frame, time_col), _num(frame, target), design)
        estimates = [0.0] + [float(c) for c in fit.coef[:k - 1]]
        label = "log HR vs lowest quantile"
    else:
        y = _num(frame, target)
        estimates = [float(y[bins == j].mean()) for j in range(k)]
        label = "mean outcome"
    if not all(np.isfinite(estimates)):
        return None
    trend, _p, _n = safe_spearman(np.arange(k, dtype=float), np.asarray(estimates))
    return {"quantiles": k, "measure": label, "estimates": [round(e, 4) for e in estimates],
            "monotone": bool(np.isfinite(trend) and trend * direction >= _MONOTONE_RHO)}


def _lag_sensitivity(frame: pd.DataFrame, x: pd.Series, target: str, time_col: str, cov: pd.DataFrame | None,
                     lag: int, direction: int) -> dict[str, Any] | None:
    """The Cox effect among those still at risk ``lag`` days into follow-up, with time counted from then."""
    time = _num(frame, time_col)
    keep = time > lag
    if keep.sum() < _MIN_ROWS:
        return None
    res = _cox(time[keep] - lag, _num(frame, target)[keep], x[keep], cov.loc[keep] if cov is not None else None)
    return {"lag_days": lag, **res, "same_direction": bool(res["sign"] == direction)}


def _negative_controls(frame: pd.DataFrame, x: pd.Series, cov: pd.DataFrame | None,
                       names: list[str], alpha: float) -> list[dict[str, Any]]:
    out = []
    for name in names:
        event_col, time_col = f"{NC_PREFIX}{name}", f"{NC_PREFIX}{name}_days"
        if not {event_col, time_col} <= set(frame.columns):
            out.append({"control": name, "status": "not in the holdout table"})
            continue
        events = _num(frame, event_col)
        if events.sum() < _MIN_CONTROL_EVENTS:
            out.append({"control": name, "status": "too few events to test"})
            continue
        res = _cox(_num(frame, time_col), events, x, cov)
        res.pop("e_value", None)
        out.append({"control": name, **res})
    tested = [nc for nc in out if "p" in nc]
    for nc in tested:  # Bonferroni across the controls that could be tested
        nc["status"] = "associated" if nc["p"] < alpha / len(tested) else "no association"
    return out


def _ledger_entries(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("key") == key:
            entries.append(entry)
    return entries


def ledger_summary(out_dir: Path, key: str, alpha: float = 0.05) -> dict[str, Any]:
    """Every hypothesis tested on the holdout ``key`` so far, BH-corrected together: how many were
    tested, from how many registrations, and which replicate on that pooled correction."""
    entries = _ledger_entries(Path(out_dir) / LEDGER_NAME, key)
    q = _bh([e["p"] for e in entries])
    hits = [f"{e['stem']}: {e['feature']}" for e, qv in zip(entries, q)
            if e.get("same_direction") and np.isfinite(qv) and qv <= alpha]
    # A feature registered by several runs is listed once, with how many registrations replicated it.
    replicated = [name if hits.count(name) == 1 else f"{name} (x{hits.count(name)})" for name in dict.fromkeys(hits)]
    return {"key": key, "hypotheses": len(entries), "registrations": len({e["prereg"] for e in entries}),
            "replicated": replicated}


def _test_hypothesis(frame: pd.DataFrame, h: dict[str, Any], prereg: dict[str, Any], target: str,
                     time_col: str | None, cov: pd.DataFrame | None) -> dict[str, Any]:
    row: dict[str, Any] = {"feature": h["feature"], "direction": h["direction"], "discovery_rho": h["discovery_rho"]}
    x = _feature(frame, h)
    if x is None or x.notna().sum() < _MIN_ROWS:
        row.update(status="not testable on the holdout", p=float("nan"))
        return row
    res = _effect(frame, x, target, time_col, cov)
    row.update({k: v for k, v in res.items() if k != "sign"}, same_direction=bool(res["sign"] == h["direction"]))
    row["dose_response"] = _dose_response(frame, x, target, time_col, cov, h["direction"])
    if time_col:
        row["lag_sensitivity"] = _lag_sensitivity(frame, x, target, time_col, cov,
                                                  int(prereg.get("sensitivity_lag_days") or 0), h["direction"])
    row["negative_controls"] = _negative_controls(frame, x, cov, list(prereg.get("negative_controls") or []),
                                                  prereg["alpha"])
    return row


def _falsification(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(tests passed, tests failed) among the falsification tests that ran for ``row``."""
    passed, failed = [], []
    dose = row.get("dose_response")
    if dose is not None:
        (passed if dose["monotone"] else failed).append("dose-response")
    lag = row.get("lag_sensitivity")
    if lag is not None:
        (passed if lag["same_direction"] else failed).append("lag")
    for nc in row.get("negative_controls") or []:
        if nc.get("status") == "associated":
            failed.append(f"negative control {nc['control']}")
        elif nc.get("status") == "no association":
            passed.append(f"negative control {nc['control']}")
    return passed, failed


def confirm(prereg: dict[str, Any], holdout_csv: Path, out_dir: Path, *, ledger_key: str | None = None) -> dict[str, Any]:
    """Test every pre-registered hypothesis once on the holdout; write ``<stem>_confirmation.json``.

    With ``ledger_key`` (which participants the holdout holds) the hypotheses are appended to the
    holdout ledger and replication is judged on the BH q-value across the whole ledger."""
    if prereg.get("sha256") != _digest(prereg):
        raise ValueError("The pre-registration record was modified after it was registered.")
    frame = pd.read_csv(holdout_csv)
    target, time_col = prereg["target"], prereg.get("survival_time")
    # The follow-up time is the Cox model's time axis, never a covariate of it.
    confounders = [c for c in prereg["confounders"] if c in frame.columns and c != time_col]
    cov = frame[confounders].apply(pd.to_numeric, errors="coerce") if confounders else None
    rows = [_test_hypothesis(frame, h, prereg, target, time_col, cov) for h in prereg["hypotheses"]]
    for row, q in zip(rows, _bh([r["p"] for r in rows])):
        row["q"] = q

    ledger = None
    if ledger_key is not None:
        path = Path(out_dir) / LEDGER_NAME
        prior = [e for e in _ledger_entries(path, ledger_key) if e.get("prereg") != prereg["sha256"]]
        mine = [{"key": ledger_key, "prereg": prereg["sha256"], "stem": prereg["stem"], "feature": r["feature"],
                 "p": r["p"], "same_direction": r.get("same_direction", False)}
                for r in rows if np.isfinite(r["p"])]
        pooled = _bh([e["p"] for e in prior] + [e["p"] for e in mine])[len(prior):]
        cumulative = {e["feature"]: q for e, q in zip(mine, pooled)}
        for row in rows:
            row["q_ledger"] = cumulative.get(row["feature"], float("nan"))
        with path.open("a", encoding="utf-8") as fh:
            for e in mine:
                fh.write(json.dumps(e, default=str) + "\n")
        ledger = {"key": ledger_key, "hypotheses": len(prior) + len(mine),
                  "registrations": len({e["prereg"] for e in prior}) + 1}

    for row in rows:
        if "same_direction" not in row:
            continue
        q = row["q_ledger"] if ledger is not None else row["q"]
        row["replicated"] = bool(row["same_direction"] and np.isfinite(q) and q <= prereg["alpha"])
        row["status"] = "replicated" if row["replicated"] else "not replicated"
        passed, failed = _falsification(row)
        row["falsification_passed"], row["falsification_failed"] = passed, failed
        row["robust"] = bool(row["replicated"] and not failed)

    positive = []
    for column in prereg.get("positive_controls") or []:
        if column not in frame.columns:
            continue
        others = [c for c in confounders if c != column]
        res = _effect(frame, _num(frame, column), target, time_col,
                      frame[others].apply(pd.to_numeric, errors="coerce") if others else None)
        res.pop("sign", None)
        positive.append({"control": column, **res, "detected": bool(res["p"] < prereg["alpha"])})

    result = mask_small_counts({
        "prereg_sha256": prereg["sha256"], "holdout_rows": int(len(frame)), "confounders": confounders,
        "hypotheses": rows, "positive_controls": positive, "ledger": ledger,
        "negative_controls": list(prereg.get("negative_controls") or []),
        "discovery_variants_evaluated": prereg.get("discovery_variants_evaluated"),
        "replicated": sum(1 for r in rows if r.get("replicated")),
        "robust": sum(1 for r in rows if r.get("robust")),
        "tested": sum(1 for r in rows if "replicated" in r)})
    (out_dir / f"{prereg['stem']}_confirmation.json").write_text(json.dumps(result, indent=2, default=str),
                                                                 encoding="utf-8")
    return result


def _fmt_q(q: Any) -> str:
    return f"{q:.3g}" if isinstance(q, float) and np.isfinite(q) else ""


def _effect_text(r: dict[str, Any]) -> tuple[str, str, Any]:
    if r.get("test") == "cox":
        return f"HR/SD {r['hr_per_sd']}", f"{r['ci95'][0]}-{r['ci95'][1]}", r.get("e_value", "")
    if "rho" in r:
        ci = r.get("ci95")
        return f"rho {r['rho']}", (f"{ci[0]}-{ci[1]}" if ci else ""), ""
    return "", "", ""


def confirmation_markdown(result: dict[str, Any]) -> str:
    """The deterministic replication and falsification section appended to the report."""
    ledger = result.get("ledger")
    family = (f"BH-corrected across all {ledger['hypotheses']} hypotheses tested on this holdout so far "
              f"({ledger['registrations']} pre-registrations)" if ledger
              else "BH-corrected across the pre-registered set")
    looks = result.get("discovery_variants_evaluated")
    lines = ["## Holdout replication (pre-registered, deterministic)", "",
             f"{result['replicated']} of {result['tested']} pre-registered candidates replicated on the sealed "
             f"holdout ({result['holdout_rows']} participants; {family}; adjusted for "
             f"{', '.join(result['confounders']) or 'nothing'}), and {result.get('robust', 0)} also passed every "
             f"falsification test that could run. The agents never saw these data or results"
             + (f"; discovery evaluated {looks} candidate variants before registering" if looks else "")
             + f". Pre-registration hash: `{result['prereg_sha256'][:16]}`.", "",
             "| feature | discovery rho | holdout effect | 95% CI | q | E-value | status | dose-response | "
             "falsification failed |", "|---|---|---|---|---|---|---|---|---|"]
    for r in result["hypotheses"]:
        effect, ci, ev = _effect_text(r)
        q = r.get("q_ledger", r.get("q"))
        dose = r.get("dose_response")
        dose_text = "" if dose is None else ("monotone" if dose["monotone"] else "not monotone")
        status = r.get("status", "") + (" (robust)" if r.get("robust") else "")
        lines.append(f"| {r['feature']} | {r['discovery_rho']} | {effect} | {ci} | {_fmt_q(q)} | {ev} | {status} | "
                     f"{dose_text} | {', '.join(r.get('falsification_failed') or []) or '-'} |")
    if result.get("negative_controls"):
        lines += ["", "Negative-control outcomes (conditions the exposures should not cause; an association "
                  "points to confounding or to differences in health-care contact): "
                  + ", ".join(result["negative_controls"]) + "."]
        shown = [(r["feature"], nc) for r in result["hypotheses"] for nc in r.get("negative_controls") or []
                 if nc.get("test") == "cox"]
        if shown:
            lines += ["", "| feature | negative control | HR/SD | 95% CI | p |", "|---|---|---|---|---|"]
            lines += [f"| {f} | {nc['control']} | {nc['hr_per_sd']} | {nc['ci95'][0]}-{nc['ci95'][1]} | "
                      f"{nc['p']:.3g} |" for f, nc in shown]
    if result.get("positive_controls"):
        lines += ["", "Positive controls (known associations, tested the same way to show the holdout can detect "
                  "a real effect):", "", "| control | holdout effect | 95% CI | p | detected |", "|---|---|---|---|---|"]
        for pc in result["positive_controls"]:
            effect, ci, _ev = _effect_text(pc)
            lines.append(f"| {pc['control']} | {effect} | {ci} | {pc['p']:.3g} | {'yes' if pc['detected'] else 'no'} |")
    return "\n".join(lines) + "\n"
