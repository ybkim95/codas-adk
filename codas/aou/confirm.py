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
* wear adherence (negative-control exposure): the effect re-estimated with the days-with-data
  columns added to the confounders should keep its direction with a CI that excludes the null. A
  feature that only tracks how much a participant wears the device loses it; the days columns' own
  associations with the outcome are reported as the benchmark.
A candidate is ``robust`` when it replicates and passes every falsification test that could run.

Missing confounder values are multiply imputed when the pre-registration asks for it (chained
equations with the exposure and outcome as auxiliaries, pooled by Rubin's rules;
``codas.core.imputation``); the complete-case estimate is kept alongside for comparison. Negative
controls and dose-response stay complete-case.

Positive controls (columns with a known association with the outcome) are tested the same way, to
show that the confirmation can detect a real effect on this holdout.

Every hypothesis tested on a holdout is appended to a ledger next to the outputs
(``holdout_ledger.jsonl``), keyed by which participants the holdout holds. Replication is judged on
the BH q-value across every hypothesis ever tested on that holdout, so re-running the analysis, or
running other outcomes on the same participants, cannot buy extra chances at a false replication.

Hypotheses the agents registered from the literature before any statistic existed (``a_priori``
family, with their predicted direction and how established each is) are tested whatever discovery
found and judged on BH across that family alone: they were fixed before the data could steer them,
so they do not pay for the exploratory search. Discovery's validated candidates form the
``exploratory`` family and are judged on the ledger.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codas.aou.privacy import mask_identifiers, mask_small_counts
from codas.core.analyses import rebuild_feature
from codas.core.imputation import pooled_cox, pooled_partial_spearman_ci
from codas.core.statistics import safe_spearman
from codas.core.survival import cox_ph, e_value

NC_PREFIX = "nc_"
LEDGER_NAME = "holdout_ledger.jsonl"
DEFAULT_SENSITIVITY_LAG_DAYS = 365
_MIN_ROWS = 30
_MIN_CONTROL_EVENTS = 10  # a negative control with fewer events cannot show an association
_QUANTILES = 5
_MONOTONE_RHO = 0.8  # rank correlation between quantile and effect that counts as monotone
_SEED = 17


def _digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def preregister(state: dict[str, Any], out_dir: Path, stem: str, *, survival_time: str | None = None,
                max_hypotheses: int = 10, negative_controls: list[str] | tuple[str, ...] = (),
                positive_controls: list[str] | tuple[str, ...] = (),
                sensitivity_lag_days: int = DEFAULT_SENSITIVITY_LAG_DAYS,
                availability: list[str] | tuple[str, ...] = (), covariate_imputations: int = 0) -> dict[str, Any]:
    """Freeze the final round's validated candidates as hypotheses and write ``<stem>_prereg.json``.

    ``negative_controls`` are control names (the holdout has ``nc_<name>`` and ``nc_<name>_days``),
    ``positive_controls`` column names, ``availability`` the days-with-data columns used for the
    adherence check, ``covariate_imputations`` how many times missing confounder values are imputed. The record also counts the tests the discovery ran, so the
    report can say how many looks preceded the registration."""
    report = state.get("latest_report") or {}
    every = {c["feature"]: c for c in report.get("candidates") or []}
    candidates = [c for c in every.values() if c.get("verdict") == "validated"]
    candidates.sort(key=lambda c: -abs(float(c.get("rho") or 0)))
    proposed = {p.get("name"): p for p in state.get("proposed_features") or []}

    def definition(name: str) -> dict[str, Any]:
        if name in proposed:
            return {"op": proposed[name]["op"], "a": proposed[name]["a"], "b": proposed[name]["b"]}
        return {"components": list((every.get(name) or {}).get("components") or [])}

    def seen(name: str) -> dict[str, Any]:
        c = every.get(name) or {}
        rho = c.get("rho")
        return {"discovery_rho": round(float(rho), 4) if isinstance(rho, (int, float)) else None,
                "discovery_q": c.get("q_value")}

    hypotheses = []
    for h in (state.get("a_priori_hypotheses") or [])[:max_hypotheses]:
        hypotheses.append({"feature": h["feature"], "definition": definition(h["feature"]),
                           "direction": 1 if int(h["direction"]) > 0 else -1, **seen(h["feature"]),
                           "family": "a_priori", "knownness": h.get("knownness"), "construct": h.get("construct"),
                           "citation": h.get("citation")})
    registered = {h["feature"] for h in hypotheses}
    for c in [c for c in candidates if c["feature"] not in registered][:max_hypotheses]:
        name = c["feature"]
        hypotheses.append({"feature": name, "definition": definition(name), "direction": 1 if c["rho"] > 0 else -1,
                           **seen(name), "family": "exploratory"})
    rounds = state.get("rounds") or []
    record = {
        "stem": stem, "target": state.get("target_column"), "survival_time": survival_time,
        "confounders": list(state.get("confounder_columns") or []),
        "excluded": list(state.get("excluded_columns") or []),
        "hypotheses": hypotheses, "alpha": 0.05, "correction": "benjamini-hochberg",
        "negative_controls": list(negative_controls), "positive_controls": list(positive_controls),
        "sensitivity_lag_days": int(sensitivity_lag_days),
        "availability": list(availability), "covariate_imputations": int(covariate_imputations),
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
    return rebuild_feature(frame, hypothesis["feature"], hypothesis["definition"])


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


def _error(exc: Exception) -> str:
    return mask_identifiers(f"{type(exc).__name__}: {str(exc)[:200]}")


def _guarded(errors: dict[str, str], name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """``fn(*args)``, or None with the error recorded under ``name``: one numerical failure (a model that
    cannot be fitted on some subset) costs that check, never the other checks or the outcome."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded on the hypothesis and reported
        errors[name] = _error(exc)
        return None


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _cox(time: pd.Series, event: pd.Series, x: pd.Series, cov: pd.DataFrame | None, m: int = 0) -> dict[str, Any]:
    """Cox HR per SD of ``x`` adjusted for ``cov`` (missing ``cov`` values imputed ``m`` times)."""
    z = (x - x.mean()) / x.std(ddof=0)
    fit = pooled_cox(time, event, z, None if cov is None else cov.to_numpy(float), m=m, seed=_SEED)
    hr, (lo, hi) = fit["hr"], fit["ci95"]
    out = {"test": "cox", "hr_per_sd": round(hr, 4), "ci95": [round(lo, 4), round(hi, 4)], "p": float(fit["p"]),
           "n": fit["n"], "events": fit["events"], "e_value": round(e_value(hr), 3), "sign": float(np.sign(fit["coef"]))}
    if fit["imputations"]:
        out["imputations"] = fit["imputations"]
    return out


def _rank(x: pd.Series, y: pd.Series, cov: pd.DataFrame | None, m: int = 0) -> dict[str, Any]:
    if cov is not None:
        fit = pooled_partial_spearman_ci(x.to_numpy(float), y.to_numpy(float), cov.to_numpy(float), m=m, seed=_SEED)
        rho, p, n, (lo, hi) = fit["rho"], fit["p"], fit["n"], fit["ci95"]
    else:
        rho, p, n = safe_spearman(x.to_numpy(float), y.to_numpy(float))
        half = 1.959964 / np.sqrt(max(n - 3, 1))  # Fisher-z interval
        lo, hi = (float(np.tanh(np.arctanh(np.clip(rho, -0.999999, 0.999999)) + s * half)) for s in (-1, 1))
        fit = {"imputations": 0}
    out = {"test": "partial_spearman" if cov is not None else "spearman", "rho": round(float(rho), 4),
           "ci95": [round(lo, 4), round(hi, 4)], "p": float(p), "n": int(n), "sign": float(np.sign(rho))}
    if fit["imputations"]:
        out["imputations"] = fit["imputations"]
    return out


def _effect(frame: pd.DataFrame, x: pd.Series, target: str, time_col: str | None,
            cov: pd.DataFrame | None, m: int = 0) -> dict[str, Any]:
    if time_col:
        return _cox(_num(frame, time_col), _num(frame, target), x, cov, m)
    return _rank(x, _num(frame, target), cov, m)


def _excludes_null(res: dict[str, Any]) -> bool:
    lo, hi = res.get("ci95") or [np.nan, np.nan]
    null = 1.0 if res.get("test") == "cox" else 0.0
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > null or hi < null))


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
                     lag: int, direction: int, m: int = 0) -> dict[str, Any] | None:
    """The Cox effect among those still at risk ``lag`` days into follow-up, with time counted from then."""
    time = _num(frame, time_col)
    keep = time > lag
    if keep.sum() < _MIN_ROWS:
        return None
    res = _cox(time[keep] - lag, _num(frame, target)[keep], x[keep], cov.loc[keep] if cov is not None else None, m)
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
        try:
            res = _cox(_num(frame, time_col), events, x, cov)
        except Exception as exc:  # noqa: BLE001 - this control is reported as not computed
            out.append({"control": name, "status": "could not be computed", "error": _error(exc)})
            continue
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
    replicated = [f"{e['stem']}: {e['feature']}" for e, qv in zip(entries, q)
                  if e.get("same_direction") and np.isfinite(qv) and qv <= alpha]
    return {"key": key, "hypotheses": len(entries), "registrations": len({e["prereg"] for e in entries}),
            "replicated": replicated}


def _test_hypothesis(frame: pd.DataFrame, h: dict[str, Any], prereg: dict[str, Any], target: str,
                     time_col: str | None, cov: pd.DataFrame | None,
                     with_adherence: pd.DataFrame | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"feature": h["feature"], "direction": h["direction"], "discovery_rho": h["discovery_rho"],
                           "family": h.get("family", "exploratory")}
    for key in ("knownness", "construct", "citation"):
        if h.get(key) is not None:
            row[key] = h[key]
    x = _feature(frame, h)
    if x is None or x.notna().sum() < _MIN_ROWS:
        row.update(status="not testable on the holdout", p=float("nan"))
        return row
    m = int(prereg.get("covariate_imputations") or 0)
    errors: dict[str, str] = {}
    res = _guarded(errors, "effect", _effect, frame, x, target, time_col, cov, m)
    if res is None:  # the main estimate itself could not be computed: tested, not replicated
        row.update(p=float("nan"), same_direction=False, errors=errors)
        return row
    row.update({k: v for k, v in res.items() if k != "sign"}, same_direction=bool(res["sign"] == h["direction"]))
    if res.get("imputations"):
        cc = _guarded(errors, "complete_case", _effect, frame, x, target, time_col, cov, 0)
        if cc is not None:
            row["complete_case"] = {k: cc[k] for k in ("hr_per_sd", "rho", "ci95", "p", "n") if k in cc}
    if with_adherence is not None:
        adj = _guarded(errors, "adherence_adjusted", _effect, frame, x, target, time_col, with_adherence, m)
        if adj is not None:
            same = bool(adj["sign"] == h["direction"])
            row["adherence_adjusted"] = {**{k: v for k, v in adj.items() if k not in ("sign", "e_value")},
                                         "same_direction": same, "holds": same and _excludes_null(adj)}
    row["dose_response"] = _guarded(errors, "dose_response", _dose_response, frame, x, target, time_col, cov,
                                    h["direction"])
    if time_col:
        row["lag_sensitivity"] = _guarded(errors, "lag_sensitivity", _lag_sensitivity, frame, x, target, time_col,
                                          cov, int(prereg.get("sensitivity_lag_days") or 0), h["direction"], m)
    row["negative_controls"] = _guarded(errors, "negative_controls", _negative_controls, frame, x, cov,
                                        list(prereg.get("negative_controls") or []), prereg["alpha"]) or []
    if errors:
        row["errors"] = errors
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
    adherence = row.get("adherence_adjusted")
    if adherence is not None and np.isfinite(adherence.get("p", np.nan)):
        (passed if adherence["holds"] else failed).append("adherence adjustment")
    for nc in row.get("negative_controls") or []:
        if nc.get("status") == "associated":
            failed.append(f"negative control {nc['control']}")
        elif nc.get("status") == "no association":
            passed.append(f"negative control {nc['control']}")
    return passed, failed


def _within_person(panel: pd.DataFrame, h: dict[str, Any], target: str) -> dict[str, Any]:
    """The pre-registered feature's within- and between-person effects on the sealed repeated measurements."""
    from codas.core.mixed import mixed_model

    x = _feature(panel, h)
    if x is None:
        return {"error": "the feature cannot be built from the panel's period summaries"}
    data = panel.assign(**{h["feature"]: x})
    fit = mixed_model(data, h["feature"], target, [c for c in ("age", "days_from_start") if c in data.columns])
    if "error" not in fit:
        fit["same_direction"] = bool(np.sign(fit["within"]["estimate"]) == h["direction"])
    return fit


def confirm(prereg: dict[str, Any], holdout_csv: Path, out_dir: Path, *, ledger_key: str | None = None,
            panel_csv: Path | None = None) -> dict[str, Any]:
    """Test every pre-registered hypothesis once on the holdout; write ``<stem>_confirmation.json``.

    With ``panel_csv`` (the sealed repeated-measures panel) each hypothesis also gets its within-person
    effect from a mixed model; it is reported as evidence, not used to judge replication.

    With ``ledger_key`` (which participants the holdout holds) the hypotheses are appended to the
    holdout ledger and replication is judged on the BH q-value across the whole ledger."""
    if prereg.get("sha256") != _digest(prereg):
        raise ValueError("The pre-registration record was modified after it was registered.")
    frame = pd.read_csv(holdout_csv)
    target, time_col = prereg["target"], prereg.get("survival_time")
    # The follow-up time is the Cox model's time axis, never a covariate of it.
    confounders = [c for c in prereg["confounders"] if c in frame.columns and c != time_col]
    cov = frame[confounders].apply(pd.to_numeric, errors="coerce") if confounders else None
    m = int(prereg.get("covariate_imputations") or 0)
    # Wear adherence as a negative-control exposure: the days-with-data columns added to the confounders.
    availability = [c for c in prereg.get("availability") or [] if c in frame.columns and c not in confounders
                    and c != time_col]
    with_adherence = (frame[confounders + availability].apply(pd.to_numeric, errors="coerce")
                      if availability else None)
    rows = [_test_hypothesis(frame, h, prereg, target, time_col, cov, with_adherence) for h in prereg["hypotheses"]]
    if panel_csv is not None and Path(panel_csv).exists():
        panel = pd.read_csv(panel_csv)
        for row, h in zip(rows, prereg["hypotheses"]):
            try:
                row["within_person"] = _within_person(panel, h, target)
            except Exception as exc:  # noqa: BLE001 - evidence only; reported as not computed
                row["within_person"] = {"error": _error(exc)}
    for row, q in zip(rows, _bh([r["p"] for r in rows])):
        row["q"] = q
    for family in {r["family"] for r in rows}:  # BH within each family as well
        members = [r for r in rows if r["family"] == family]
        for row, q in zip(members, _bh([r["p"] for r in members])):
            row["q_family"] = q

    ledger = None
    if ledger_key is not None:
        path = Path(out_dir) / LEDGER_NAME
        prior = [e for e in _ledger_entries(path, ledger_key) if e.get("prereg") != prereg["sha256"]]
        mine = [{"key": ledger_key, "prereg": prereg["sha256"], "stem": prereg["stem"], "feature": r["feature"],
                 "p": r["p"], "same_direction": r.get("same_direction", False), "family": r["family"]}
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
        # A-priori hypotheses are judged within their own family; exploratory ones on the whole ledger.
        q = (row["q_family"] if row["family"] == "a_priori"
             else row["q_ledger"] if ledger is not None else row["q"])
        row["q_used"] = q
        row["replicated"] = bool(row["same_direction"] and np.isfinite(q) and q <= prereg["alpha"])
        row["status"] = ("test failed" if "effect" in (row.get("errors") or {})
                         else "replicated" if row["replicated"] else "not replicated")
        passed, failed = _falsification(row)
        row["falsification_passed"], row["falsification_failed"] = passed, failed
        row["robust"] = bool(row["replicated"] and not failed)

    positive = []
    for column in prereg.get("positive_controls") or []:
        if column not in frame.columns:
            continue
        others = [c for c in confounders if c != column]
        try:
            res = _effect(frame, _num(frame, column), target, time_col,
                          frame[others].apply(pd.to_numeric, errors="coerce") if others else None, m)
        except Exception as exc:  # noqa: BLE001
            positive.append({"control": column, "error": _error(exc), "detected": False})
            continue
        res.pop("sign", None)
        positive.append({"control": column, **res, "detected": bool(res["p"] < prereg["alpha"])})
    benchmark = []
    for column in availability:  # how strongly wear adherence alone tracks the outcome
        try:
            res = _effect(frame, _num(frame, column), target, time_col, cov, m)
        except Exception as exc:  # noqa: BLE001
            benchmark.append({"column": column, "error": _error(exc)})
            continue
        res.pop("sign", None)
        benchmark.append({"column": column, **res})

    result = mask_small_counts({
        "prereg_sha256": prereg["sha256"], "holdout_rows": int(len(frame)), "confounders": confounders,
        "hypotheses": rows, "positive_controls": positive, "ledger": ledger,
        "availability": availability, "adherence_benchmark": benchmark, "covariate_imputations": m,
        "negative_controls": list(prereg.get("negative_controls") or []),
        "discovery_variants_evaluated": prereg.get("discovery_variants_evaluated"),
        "replicated": sum(1 for r in rows if r.get("replicated")),
        "robust": sum(1 for r in rows if r.get("robust")),
        "tested": sum(1 for r in rows if "replicated" in r),
        "families": {f: {"tested": sum(1 for r in rows if r["family"] == f and "replicated" in r),
                         "replicated": sum(1 for r in rows if r["family"] == f and r.get("replicated")),
                         "robust": sum(1 for r in rows if r["family"] == f and r.get("robust"))}
                     for f in ("a_priori", "exploratory") if any(r["family"] == f for r in rows)}})
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
             f"{', '.join(result['confounders']) or 'nothing'}"
             + (f", missing confounder values multiply imputed ({result['covariate_imputations']} draws, "
                "complete-case estimates kept alongside)" if result.get("covariate_imputations") else "")
             + f"), and {result.get('robust', 0)} also passed every "
             f"falsification test that could run. The agents never saw these data or results"
             + (f"; discovery evaluated {looks} candidate variants before registering" if looks else "")
             + f". Pre-registration hash: `{result['prereg_sha256'][:16]}`.", ""]
    families = result.get("families") or {}
    groups = [("a_priori", "A-priori hypotheses (registered from the literature before any statistic existed; "
                           "direction as predicted; BH within this family of {n})"),
              ("exploratory", "Exploratory candidates (validated in discovery; " + family + ")")]
    for key, title in groups:
        members = [r for r in result["hypotheses"] if r.get("family", "exploratory") == key]
        if not members:
            continue
        if len(families) > 1 or key == "a_priori":
            counts = families.get(key, {})
            lines += [f"**{title.format(n=len(members))}**: {counts.get('replicated', 0)} of "
                      f"{counts.get('tested', 0)} replicated, {counts.get('robust', 0)} robust.", ""]
        lines += ["| feature | discovery rho | holdout effect | 95% CI | q | E-value | status | dose-response | "
                  "falsification failed |", "|---|---|---|---|---|---|---|---|---|"]
        for r in members:
            effect, ci, ev = _effect_text(r)
            q = r.get("q_used", r.get("q_ledger", r.get("q")))
            dose = r.get("dose_response")
            dose_text = "" if dose is None else ("monotone" if dose["monotone"] else "not monotone")
            status = r.get("status", "") + (" (robust)" if r.get("robust") else "")
            if r.get("knownness"):
                status += f"; {r['knownness']}"
            lines.append(f"| {r['feature']} | {r['discovery_rho']} | {effect} | {ci} | {_fmt_q(q)} | {ev} | {status} | "
                         f"{dose_text} | {', '.join(r.get('falsification_failed') or []) or '-'} |")
        lines.append("")
    failed_checks = [f"{r['feature']}: {', '.join(f'{k} ({v})' for k, v in r['errors'].items())}"
                     for r in result["hypotheses"] if r.get("errors")]
    if failed_checks:
        lines += ["Checks that could not be computed (numerical failure; reported, not counted as passed): "
                  + "; ".join(failed_checks) + ".", ""]
    adjusted = [r for r in result["hypotheses"] if r.get("adherence_adjusted")]
    if adjusted:
        lines += ["", "Wear adherence as a negative-control exposure: each effect re-estimated with "
                  + ", ".join(result.get("availability") or []) + " (days with wearable data) added to the "
                  "confounders. A feature that only tracks how much a participant wears the device loses its "
                  "effect here.", "", "| feature | adjusted for wear adherence | 95% CI | holds |", "|---|---|---|---|"]
        for r in adjusted:
            effect, ci, _ev = _effect_text(r["adherence_adjusted"])
            lines.append(f"| {r['feature']} | {effect} | {ci} | {'yes' if r['adherence_adjusted']['holds'] else 'no'} |")
        bench = "; ".join(f"{b['column']} {_effect_text(b)[0]} ({_effect_text(b)[1]})"
                          for b in result.get("adherence_benchmark") or [])
        if bench:
            lines += ["", f"Wear adherence alone, for comparison: {bench}."]
    within = [r for r in result["hypotheses"] if r.get("within_person") and "error" not in r["within_person"]]
    if within:
        lines += ["", "Within-person effects on the sealed repeated measurements (mixed model, random intercept per "
                  "person; each value paired with the wearable summary of the period before it). A feature that only "
                  "marks a stable trait of the people who have it shows a between-person effect and a within-person "
                  "effect near zero.", "",
                  "| feature | within-person (per SD) | 95% CI | p | between-person | people with repeats |",
                  "|---|---|---|---|---|---|"]
        for r in within:
            w, b = r["within_person"]["within"], r["within_person"]["between"]
            lines.append(f"| {r['feature']} | {w['estimate']:.3g} | {w['ci95'][0]:.3g}-{w['ci95'][1]:.3g} | {w['p']:.3g} | "
                         f"{b['estimate']:.3g} (p {b['p']:.3g}) | {r['within_person']['persons_with_repeats']} |")
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
