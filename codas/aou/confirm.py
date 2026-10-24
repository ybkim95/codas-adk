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
  associations with the outcome are reported as the benchmark. Days columns that are already
  confounders, or duplicate one, are left out; when none is left the check is reported as not counted
  rather than as passed.
A candidate is ``robust`` when it replicates and passes every falsification test that could run.

Missing confounder values are multiply imputed when the pre-registration asks for it (chained
equations with the exposure and outcome as auxiliaries, pooled by Rubin's rules;
``codas.core.imputation``); the complete-case estimate is kept alongside for comparison. Negative
controls and dose-response stay complete-case.

Positive controls (columns with a known association with the outcome) are tested the same way, to
show that the confirmation can detect a real effect on this holdout.

Every hypothesis tested on a holdout is appended to a ledger next to the outputs (``holdout_ledger.jsonl``,
``codas.aou.ledger``), keyed by which participants the holdout holds, with the verdict it got:

* a repeat of an earlier (outcome, feature) test, or any hypothesis about an outcome already tested on this
  holdout, is reported as non-confirmatory and spends nothing;
* novel hypotheses (emerging or unexplored a-priori hypotheses, and discovery's validated candidates) are
  judged by online false-discovery-rate control (LORD++) in ledger order, on the one-sided p in the
  pre-registered direction, so each verdict is final when made and re-running an analysis buys no new chance;
* established associations form a reference family, tested at 0.05 like positive controls and never counted
  as discoveries.

Searching the discovery share costs no holdout alpha; only these registered tests do.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codas.aou import ledger as ledger_book
from codas.aou.privacy import mask_identifiers, mask_small_counts
from codas.core import preprocess
from codas.core.analyses import rebuild_feature
from codas.core.imputation import pooled_cox, pooled_partial_spearman_ci
from codas.core.statistics import safe_spearman
from codas.core.survival import cox_ph, e_value

NC_PREFIX = "nc_"
LEDGER_NAME = ledger_book.LEDGER_NAME
# Status shown for each ledger role, by whether its test came out positive.
_ROLE_STATUS = {ledger_book.NOVEL: {True: "replicated", False: "not replicated"},
                ledger_book.REFERENCE: {True: "reference reproduced", False: "reference not reproduced"},
                ledger_book.NON_CONFIRMATORY: {True: "non-confirmatory", False: "non-confirmatory"}}
DEFAULT_SENSITIVITY_LAG_DAYS = 365
_MIN_ROWS = 30
_MIN_CONTROL_EVENTS = 10  # a negative control with fewer events cannot show an association
_QUANTILES = 5
_MONOTONE_RHO = 0.8  # rank correlation between quantile and effect that counts as monotone
# A registered hypothesis's own predictions (codas.core.hypotheses), scored on the holdout.
_TEMPLATE_FIELDS = ("incumbent", "shape", "magnitude", "predictions", "mechanism_steps", "alternatives", "mediators")
_THRESHOLD_SHARE = 0.6  # a threshold: one step between adjacent quantiles carries this share of the change
_U_SHARE = 0.25  # a U shape: both end quantiles sit this share of the range beyond the middle ones
_ATTENUATION_LIMIT = 0.5
_SEED = 17


def _digest(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _discovery_cox(frame: pd.DataFrame, h: dict[str, Any], target: str, time_col: str, confounders: list[str],
                   m: int) -> dict[str, Any]:
    """The feature's Cox hazard ratio on the discovery rows, adjusted as the holdout test will be."""
    x = _feature(frame, h)
    if x is None or x.notna().sum() < _MIN_ROWS:
        return {}
    cov = frame[confounders].apply(pd.to_numeric, errors="coerce") if confounders else None
    transform = h.get("transform")
    try:
        res = _cox(_num(frame, time_col), _num(frame, target), preprocess.apply(x, transform), cov, m,
                   scale=transform is None)
    except Exception:  # noqa: BLE001 - the direction then falls back to the screen's
        return {}
    return res if _finite(res) and res["sign"] != 0 else {}


def preregister(state: dict[str, Any], out_dir: Path, stem: str, *, survival_time: str | None = None,
                max_hypotheses: int = 10, negative_controls: list[str] | tuple[str, ...] = (),
                positive_controls: list[str] | tuple[str, ...] = (),
                sensitivity_lag_days: int = DEFAULT_SENSITIVITY_LAG_DAYS,
                availability: list[str] | tuple[str, ...] = (), covariate_imputations: int = 0,
                discovery_csv: str | Path | None = None, outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the final round's validated candidates as hypotheses and write ``<stem>_prereg.json``.

    ``negative_controls`` are control names (the holdout has ``nc_<name>`` and ``nc_<name>_days``),
    ``positive_controls`` column names, ``availability`` the days-with-data columns used for the
    adherence check, ``covariate_imputations`` how many times missing confounder values are imputed. The record also counts the tests the discovery ran, so the
    report can say how many looks preceded the registration.

    For a time to event, the discovery screen ranks candidates on the event flag, which ignores how
    long each participant was followed: a feature that lengthens follow-up can raise the flag while
    lowering the hazard. With ``discovery_csv`` (the rows the agents analysed) each exploratory
    candidate's direction is therefore taken from a Cox model on those rows, adjusted for the final
    confounders, as the holdout will test it; the screen's correlation is kept alongside and a
    disagreement is noted. A-priori hypotheses keep the direction they predicted.

    ``outcome`` (kind, concept ids, codes, value bounds) identifies the outcome in the holdout ledger, so a
    later registration about the same outcome is recognised whatever the run is called.

    With ``discovery_csv`` each hypothesis also gets its frozen transform (``codas.core.preprocess``: clipped
    at the discovery 0.5th-99.5th percentiles, logged when strongly skewed, scaled by the discovery SD), so a
    hazard ratio per SD means the same unit on the holdout as on the discovery share."""
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

    frame = pd.read_csv(discovery_csv) if discovery_csv and Path(discovery_csv).exists() else None
    target = state.get("target_column")
    adjust = [c for c in state.get("confounder_columns") or []
              if frame is not None and c in frame.columns and c != survival_time]

    def frozen(h: dict[str, Any]) -> dict[str, Any]:
        """The feature's transform fitted on the discovery rows (empty when it cannot be built there)."""
        x = _feature(frame, h) if frame is not None else None
        if x is None or x.notna().sum() < _MIN_ROWS:
            return {}
        transform = preprocess.fit(x)
        return {"transform": transform, "transform_text": preprocess.describe(transform)}

    def hazard(h: dict[str, Any]) -> dict[str, Any]:
        """The discovery Cox estimate for ``h`` (empty without a time to event or when it cannot be fitted)."""
        if frame is None or not survival_time or target not in frame.columns:
            return {}
        res = _discovery_cox(frame, h, target, survival_time, adjust, int(covariate_imputations))
        return {"discovery_hr": res["hr_per_sd"], "discovery_hr_ci95": res["ci95"], "sign": res["sign"]} if res else {}

    hypotheses = []
    for h in (state.get("a_priori_hypotheses") or [])[:max_hypotheses]:
        row = {"feature": h["feature"], "definition": definition(h["feature"]),
               "direction": 1 if int(h["direction"]) > 0 else -1, **seen(h["feature"]),
               "family": "a_priori", "knownness": h.get("knownness"), "construct": h.get("construct"),
               "citation": h.get("citation"), "citations": list(h.get("citations") or []),
               "declared_knownness": h.get("declared_knownness"),
               **{k: h[k] for k in _TEMPLATE_FIELDS if h.get(k)}}
        row.update(frozen(row))
        row.update({k: v for k, v in hazard(row).items() if k != "sign"})
        hypotheses.append(row)
    registered = {h["feature"] for h in hypotheses}
    for c in [c for c in candidates if c["feature"] not in registered][:max_hypotheses]:
        name = c["feature"]
        row = {"feature": name, "definition": definition(name), "direction": 1 if c["rho"] > 0 else -1,
               **seen(name), "family": "exploratory"}
        row.update(frozen(row))
        cox = hazard(row)
        if cox:
            sign = 1 if cox.pop("sign") > 0 else -1
            row.update(cox, direction_from="discovery Cox model")
            if sign != row["direction"]:
                row["direction_note"] = ("direction from the discovery Cox model; the binary screen disagreed "
                                         "(follow-up length differs with the feature)")
            row["direction"] = sign
        elif frame is not None and survival_time:
            row["direction_from"] = "discovery screen (the discovery Cox model could not be fitted)"
        hypotheses.append(row)
    rounds = state.get("rounds") or []
    record = {
        "stem": stem, "target": state.get("target_column"), "survival_time": survival_time,
        "outcome": dict(outcome) if outcome else None,
        "confounders": list(state.get("confounder_columns") or []),
        "excluded": list(state.get("excluded_columns") or []),
        "hypotheses": hypotheses, "alpha": 0.05,
        "correction": "online false-discovery-rate control (LORD++) for novel hypotheses; established ones as a "
                      "reference family at alpha; see codas.aou.ledger",
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


def _cox(time: pd.Series, event: pd.Series, x: pd.Series, cov: pd.DataFrame | None, m: int = 0,
         scale: bool = True) -> dict[str, Any]:
    """Cox HR per SD of ``x`` adjusted for ``cov`` (missing ``cov`` values imputed ``m`` times). With ``scale``
    off, ``x`` is already in frozen discovery-SD units and is used as it is."""
    z = (x - x.mean()) / x.std(ddof=0) if scale else x
    fit = pooled_cox(time, event, z, None if cov is None else cov.to_numpy(float), m=m, seed=_SEED)
    hr, (lo, hi) = fit["hr"], fit["ci95"]
    out = {"test": "cox", "hr_per_sd": round(hr, 4), "ci95": [round(lo, 4), round(hi, 4)], "p": float(fit["p"]),
           "n": fit["n"], "events": fit["events"], "e_value": round(e_value(hr), 3), "sign": float(np.sign(fit["coef"]))}
    if fit["imputations"]:
        out["imputations"] = fit["imputations"]
    if cov is not None and fit.get("dropped"):  # column 0 is the exposure; the rest are cov's columns
        out["dropped_covariates"] = [str(cov.columns[j - 1]) for j in fit["dropped"] if j >= 1]
    return out


def _finite(res: dict[str, Any] | None) -> bool:
    return res is not None and bool(np.isfinite(res.get("p", np.nan)))


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
            cov: pd.DataFrame | None, m: int = 0, scale: bool = True) -> dict[str, Any]:
    if time_col:
        return _cox(_num(frame, time_col), _num(frame, target), x, cov, m, scale)
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
                     lag: int, direction: int, m: int = 0, scale: bool = True) -> dict[str, Any] | None:
    """The Cox effect among those still at risk ``lag`` days into follow-up, with time counted from then."""
    time = _num(frame, time_col)
    keep = time > lag
    if keep.sum() < _MIN_ROWS:
        return None
    res = _cox(time[keep] - lag, _num(frame, target)[keep], x[keep], cov.loc[keep] if cov is not None else None, m,
               scale)
    return {"lag_days": lag, **res, "same_direction": bool(res["sign"] == direction)}


def _negative_controls(frame: pd.DataFrame, x: pd.Series, cov: pd.DataFrame | None,
                       names: list[str], alpha: float, scale: bool = True) -> list[dict[str, Any]]:
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
            res = _cox(_num(frame, time_col), events, x, cov, scale=scale)
        except Exception as exc:  # noqa: BLE001 - this control is reported as not computed
            out.append({"control": name, "status": "could not be computed", "error": _error(exc)})
            continue
        res.pop("e_value", None)
        if not _finite(res):
            out.append({"control": name, **res, "status": "could not be computed",
                        "error": "the hazard ratio is not finite (no usable rows or events after adjustment)"})
            continue
        out.append({"control": name, **res})
    tested = [nc for nc in out if "status" not in nc]
    for nc in tested:  # Bonferroni across the controls that could be tested
        nc["status"] = "associated" if nc["p"] < alpha / len(tested) else "no association"
    return out


def ledger_summary(out_dir: Path, key: str) -> dict[str, Any]:
    """Everything tested on the holdout ``key`` so far, with the verdicts frozen when each was made
    (``codas.aou.ledger.summary``)."""
    return ledger_book.summary(Path(out_dir), key)


def _test_hypothesis(frame: pd.DataFrame, h: dict[str, Any], prereg: dict[str, Any], target: str,
                     time_col: str | None, cov: pd.DataFrame | None,
                     with_adherence: pd.DataFrame | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"feature": h["feature"], "direction": h["direction"], "discovery_rho": h["discovery_rho"],
                           "family": h.get("family", "exploratory")}
    for key in ("knownness", "construct", "citation", "discovery_hr", "discovery_hr_ci95", "direction_from",
                "direction_note", "transform_text"):
        if h.get(key) is not None:
            row[key] = h[key]
    raw = _feature(frame, h)
    if raw is None or raw.notna().sum() < _MIN_ROWS:
        row.update(status="not testable on the holdout", p=float("nan"))
        return row
    # A Cox model reads the feature in frozen discovery-SD units (clipped and logged as registered); rank
    # statistics and quantile bins use the raw values, which monotone transforms leave unchanged.
    transform = h.get("transform") if time_col else None
    x, scale = (preprocess.apply(raw, transform), False) if transform else (raw, True)
    m = int(prereg.get("covariate_imputations") or 0)
    errors: dict[str, str] = {}
    res = _guarded(errors, "effect", _effect, frame, x, target, time_col, cov, m, scale)
    if res is None:  # the main estimate itself could not be computed: tested, not replicated
        row.update(p=float("nan"), same_direction=False, errors=errors)
        return row
    row.update({k: v for k, v in res.items() if k != "sign"}, same_direction=bool(res["sign"] == h["direction"]))
    if transform and transform.get("lo") is not None:  # do the extreme values drive the estimate?
        unclipped = _guarded(errors, "unclipped", _effect, frame, preprocess.apply(raw, transform, clip=False),
                             target, time_col, cov, m, False)
        if unclipped is not None:
            row["unclipped"] = {k: unclipped[k] for k in ("hr_per_sd", "ci95", "p") if k in unclipped}
    if res.get("imputations"):
        cc = _guarded(errors, "complete_case", _effect, frame, x, target, time_col, cov, 0, scale)
        if cc is not None:
            row["complete_case"] = {k: cc[k] for k in ("hr_per_sd", "rho", "ci95", "p", "n") if k in cc}
    if with_adherence is not None:
        adj = _guarded(errors, "adherence_adjusted", _effect, frame, x, target, time_col, with_adherence, m, scale)
        if adj is not None:
            same = bool(adj["sign"] == h["direction"])
            row["adherence_adjusted"] = {**{k: v for k, v in adj.items() if k not in ("sign", "e_value")},
                                         "same_direction": same, "holds": same and _excludes_null(adj)}
    incumbent = h.get("incumbent")
    if incumbent and incumbent in frame.columns and incumbent != h["feature"]:
        # How much of the effect the incumbent explains, given the other confounders: the effect without it
        # against the effect with it. When the agents already adjust for it, the main estimate is the latter.
        already = cov is not None and incumbent in cov.columns
        others = cov.drop(columns=[incumbent]) if already else cov
        others = others if others is not None and others.shape[1] else None
        added = _num(frame, incumbent).rename(incumbent).to_frame()
        without = (_guarded(errors, "incumbent_removed", _effect, frame, x, target, time_col, others, m, scale)
                   if already else res)
        adj = res if already else _guarded(errors, "incumbent_adjusted", _effect, frame, x, target, time_col,
                                           added if others is None else pd.concat([others, added], axis=1), m, scale)
        if without is not None and adj is not None:
            row["incumbent_adjusted"] = {**{k: v for k, v in adj.items() if k not in ("sign", "e_value")},
                                         "incumbent": incumbent, "same_direction": bool(adj["sign"] == h["direction"]),
                                         "attenuation": _attenuation(without, adj), "already_adjusted": already,
                                         "effect_without_excludes_null": _excludes_null(without)}
    row["dose_response"] = _guarded(errors, "dose_response", _dose_response, frame, raw, target, time_col, cov,
                                    h["direction"])
    if time_col:
        row["lag_sensitivity"] = _guarded(errors, "lag_sensitivity", _lag_sensitivity, frame, x, target, time_col,
                                          cov, int(prereg.get("sensitivity_lag_days") or 0), h["direction"], m, scale)
    row["negative_controls"] = _guarded(errors, "negative_controls", _negative_controls, frame, x, cov,
                                        list(prereg.get("negative_controls") or []), prereg["alpha"], scale) or []
    if errors:
        row["errors"] = errors
    return row


def _attenuation(main: dict[str, Any], adjusted: dict[str, Any]) -> float | None:
    """The share of the effect that adding a covariate removes, on the log-hazard (or correlation) scale:
    0 nothing, 1 all of it, above 1 a reversed sign."""
    key = "hr_per_sd" if "hr_per_sd" in main else "rho"
    a, b = main.get(key), adjusted.get(key)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if key == "hr_per_sd":
        if a <= 0 or b <= 0:
            return None
        a, b = float(np.log(a)), float(np.log(b))
    return round(1 - b / a, 3) if a and np.isfinite(a) and np.isfinite(b) else None


def _shape_holds(shape: str, dose: dict[str, Any] | None, direction: int) -> tuple[str, str]:
    """Whether the quantile pattern of the effect has the registered shape (an approximation from five
    quantiles: a spline would be finer)."""
    if not dose:
        return "not testable", "too few people for quantiles"
    e = list(dose["estimates"])
    seen = "quantiles " + ", ".join(f"{v:.3g}" for v in e)
    change = e[-1] - e[0]
    if shape == "monotone":
        return ("met" if dose["monotone"] else "failed"), seen
    if shape == "threshold":
        steps = [abs(b - a) for a, b in zip(e, e[1:])]
        ok = bool(change * direction > 0 and max(steps) >= _THRESHOLD_SHARE * abs(change) > 0)
        return ("met" if ok else "failed"), seen
    spread = max(e) - min(e)
    middle = e[1:-1]
    ends_high = min(e[0], e[-1]) - max(middle) >= _U_SHARE * spread > 0
    ends_low = min(middle) - max(e[0], e[-1]) >= _U_SHARE * spread > 0
    return ("met" if ends_high or ends_low else "failed"), seen


def _prediction(code: str, row: dict[str, Any]) -> tuple[str, str]:
    """(met / failed / not testable, what was seen) for one falsification prediction."""
    if code == "negative_controls_null":
        tested = [nc for nc in row.get("negative_controls") or [] if nc.get("status") in ("associated", "no association")]
        if not tested:
            return "not testable", "no negative control could be tested"
        hit = [nc["control"] for nc in tested if nc["status"] == "associated"]
        return ("failed", "associated: " + ", ".join(hit)) if hit else ("met", f"{len(tested)} controls null")
    if code == "survives_lag":
        lag = row.get("lag_sensitivity")
        if not lag or not _finite(lag):
            return "not testable", "too few people still at risk after the lag"
        ok = bool(lag["same_direction"] and _excludes_null(lag))
        return ("met" if ok else "failed"), f"after {lag['lag_days']} days: HR {lag.get('hr_per_sd', float('nan')):.3g}"
    if code == "adherence_robust":
        adh = row.get("adherence_adjusted")
        if not adh or not np.isfinite(adh.get("p", np.nan)):
            return "not testable", "wear adherence was not adjusted for separately"
        return ("met" if adh["holds"] else "failed"), "holds" if adh["holds"] else "lost after adjusting for wear"
    if code == "attenuation_below_half_after_incumbent":
        inc = row.get("incumbent_adjusted")
        if not inc or inc.get("attenuation") is None:
            return "not testable", "the incumbent-adjusted effect could not be estimated"
        if not inc.get("effect_without_excludes_null"):  # a share of an effect indistinguishable from none is noise
            return "not testable", "no effect on the holdout for the incumbent to attenuate"
        ok = bool(inc["same_direction"] and inc["attenuation"] < _ATTENUATION_LIMIT)
        return ("met" if ok else "failed"), f"{inc['incumbent']} removes {100 * inc['attenuation']:.0f}% of the effect"
    if code == "within_person_same_sign":
        wp = row.get("within_person")
        if not wp or "error" in wp:
            return "not testable", "no within-person estimate"
        return ("met" if wp.get("same_direction") else "failed"), f"within-person {wp['within']['estimate']:.3g}"
    return "not testable", "unknown prediction"


def _scorecard(h: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    """The registered predictions of ``h`` scored on its holdout row: the magnitude (does the 95% CI meet
    the predicted range), the shape, and each falsification prediction."""
    if "same_direction" not in row or not (h.get("magnitude") or h.get("shape") or h.get("predictions")):
        return None
    items = []
    mag = h.get("magnitude")
    if mag:
        lo, hi = row.get("ci95") or [np.nan, np.nan]
        if np.isfinite(lo) and np.isfinite(hi):
            result = "met" if lo <= mag["high"] and hi >= mag["low"] and row.get("same_direction") else "failed"
            items.append({"prediction": "magnitude", "result": result,
                          "seen": f"predicted {mag['low']:g}-{mag['high']:g}, holdout 95% CI {lo:.3g}-{hi:.3g}"})
        else:
            items.append({"prediction": "magnitude", "result": "not testable", "seen": "no finite estimate"})
    if h.get("shape"):
        result, seen = _shape_holds(h["shape"], row.get("dose_response"), int(h["direction"]))
        items.append({"prediction": f"shape: {h['shape']}", "result": result, "seen": seen})
    for code in h.get("predictions") or []:
        result, seen = _prediction(code, row)
        items.append({"prediction": code, "result": result, "seen": seen})
    tested = [i for i in items if i["result"] != "not testable"]
    return {"items": items, "met": sum(1 for i in tested if i["result"] == "met"), "testable": len(tested)}


def _shrinkage(row: dict[str, Any]) -> float | None:
    """Holdout over discovery effect on the log-hazard scale (1: as large as in discovery, 0: gone). For a
    candidate the discovery chose, the winner's curse makes it expected to fall below 1."""
    a, b = row.get("discovery_hr"), row.get("hr_per_sd")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a <= 0 or b <= 0 or a == 1:
        return None
    return round(float(np.log(b) / np.log(a)), 3)


def _falsification(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(tests passed, tests failed) among the falsification tests that ran for ``row``."""
    passed, failed = [], []
    dose = row.get("dose_response")
    if dose is not None:
        (passed if dose["monotone"] else failed).append("dose-response")
    lag = row.get("lag_sensitivity")
    if lag is not None and _finite(lag):
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


def audit(result: dict[str, Any]) -> list[str]:
    """What in this confirmation could not be computed or should not be trusted as is: estimates that are
    not finite, checks that ran for no hypothesis, covariates left out as collinear, positive controls
    the holdout did not detect. Shown above the replication table; the preflight stops on them."""
    flags: list[str] = []
    rows = result.get("hypotheses") or []
    missing = [r["feature"] for r in rows if "same_direction" in r and not np.isfinite(r.get("p", np.nan))]
    if missing:
        flags.append(f"holdout estimate could not be computed for {len(missing)} of {len(rows)} hypotheses: "
                     + ", ".join(missing))
    dropped = sorted({c for r in rows for c in r.get("dropped_covariates") or []})
    if dropped:
        flags.append("covariates left out of the Cox models as constant or collinear with other covariates: "
                     + ", ".join(dropped))
    controls = sorted({nc["control"] for r in rows for nc in r.get("negative_controls") or []})
    for control in controls:
        entries = [nc for r in rows for nc in r.get("negative_controls") or [] if nc["control"] == control]
        if entries and all(nc.get("status") == "could not be computed" for nc in entries):
            flags.append(f"negative control {control} could not be computed for any hypothesis")
    for pc in result.get("positive_controls") or []:
        if not np.isfinite(pc.get("p", np.nan)):
            flags.append(f"positive control {pc['control']} could not be computed")
        elif not pc.get("detected"):
            flags.append(f"positive control {pc['control']} was not detected: the holdout may lack power")
    reversed_ = [r["feature"] for r in rows if r.get("direction_note")]
    if reversed_:
        flags.append("pre-registered direction taken from the discovery Cox model because the binary screen "
                     "disagreed (the event flag ignores follow-up length): " + ", ".join(reversed_))
    failed = sorted({k for r in rows for k in (r.get("errors") or {})})
    if failed:
        flags.append("checks that raised an error for some hypotheses: " + ", ".join(failed))
    return flags


def _availability(frame: pd.DataFrame, declared: list[str], confounders: list[str],
                  time_col: str | None) -> tuple[list[str], str]:
    """The days-with-data columns that add something to the confounders, and what the check can show.

    A column with the same values as a confounder or as an earlier days column (two summaries of one
    wearable table count the same days) adds nothing: adjusting for it again leaves the estimate where
    it was, so the check would pass without testing anything."""
    present = [c for c in declared if c in frame.columns and c != time_col]
    values = {c: pd.to_numeric(frame[c], errors="coerce") for c in dict.fromkeys(present + confounders)}
    kept: list[str] = []
    for c in present:
        if c in confounders or any(values[c].equals(values[o]) for o in confounders + kept):
            continue
        kept.append(c)
    if kept:
        status = "tested: " + ", ".join(kept) + " added to the confounders"
    elif present:
        status = ("not counted: wear adherence is already in the adjustment set (" + ", ".join(present)
                  + " are confounders or duplicate one), so adding it again cannot change the estimate")
    else:
        status = "not tested: no days-with-data column in this table"
    return kept, status


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
    availability, adherence_status = _availability(frame, list(prereg.get("availability") or []), confounders, time_col)
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
    for row, h in zip(rows, prereg["hypotheses"]):
        card = _scorecard(h, row)
        if card is not None:
            row["scorecard"] = card
        shrink = _shrinkage(row)
        if shrink is not None:
            row["shrinkage"] = shrink
    # Which looks count, and how each is judged, is the ledger's decision (codas.aou.ledger): repeats and
    # outcomes already opened on this holdout are non-confirmatory, established associations are a reference
    # family, and novel hypotheses are judged by online FDR in ledger order with verdicts frozen when made.
    prior = ledger_book.load(out_dir, ledger_key) if ledger_key is not None else []
    again = {e["feature"]: e for e in prior if e.get("prereg") == prereg["sha256"] and "role" in e}
    if again and all(r["feature"] in again for r in rows):  # the same registration confirmed again
        entries = [again[r["feature"]] for r in rows]
    else:
        entries = ledger_book.decide(prior, prereg, rows, prereg.get("outcome"))
        if ledger_key is not None:
            ledger_book.append(out_dir, ledger_key, entries)
    ledger = ledger_book.summary(out_dir, ledger_key) if ledger_key is not None else None

    for row, entry in zip(rows, entries):
        if "same_direction" not in row:
            continue
        row["role"], row["p_one_sided"] = entry["role"], entry["p_one_sided"]
        if entry.get("reason"):
            row["role_reason"] = entry["reason"]
        if entry.get("level") is not None:
            row["test_level"] = entry["level"]
        positive = bool(entry.get("rejected"))
        row["replicated"] = positive and entry["role"] == ledger_book.NOVEL
        row["reproduced"] = positive and entry["role"] == ledger_book.REFERENCE
        row["status"] = ("test failed" if "effect" in (row.get("errors") or {})
                         else _ROLE_STATUS[entry["role"]][positive])
        passed, failed = _falsification(row)
        row["falsification_passed"], row["falsification_failed"] = passed, failed
        # Robust needs at least one falsification test that actually ran, and none that failed.
        row["robust"] = bool((row["replicated"] or row["reproduced"]) and passed and not failed)

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
        "audit": [],
        "prereg_sha256": prereg["sha256"], "holdout_rows": int(len(frame)), "confounders": confounders,
        "hypotheses": rows, "positive_controls": positive, "ledger": ledger,
        "availability": availability, "adherence_status": adherence_status, "adherence_benchmark": benchmark,
        "covariate_imputations": m,
        "negative_controls": list(prereg.get("negative_controls") or []),
        "discovery_variants_evaluated": prereg.get("discovery_variants_evaluated"),
        "replicated": sum(1 for r in rows if r.get("replicated")),
        "robust": sum(1 for r in rows if r.get("robust") and r.get("role") == ledger_book.NOVEL),
        "tested": sum(1 for r in rows if r.get("role") == ledger_book.NOVEL and "replicated" in r),
        "reference": {"tested": sum(1 for r in rows if r.get("role") == ledger_book.REFERENCE),
                      "reproduced": sum(1 for r in rows if r.get("reproduced")),
                      "robust": sum(1 for r in rows if r.get("robust") and r.get("role") == ledger_book.REFERENCE)},
        "non_confirmatory": sum(1 for r in rows if r.get("role") == ledger_book.NON_CONFIRMATORY),
        "predictions": {"met": sum(r["scorecard"]["met"] for r in rows if r.get("scorecard")),
                        "testable": sum(r["scorecard"]["testable"] for r in rows if r.get("scorecard"))}})
    result["audit"] = audit(result)
    (out_dir / f"{prereg['stem']}_confirmation.json").write_text(json.dumps(result, indent=2, default=str),
                                                                 encoding="utf-8")
    return result


def _fmt_q(q: Any) -> str:
    return f"{q:.3g}" if isinstance(q, float) and np.isfinite(q) else ""


def _test_text(r: dict[str, Any]) -> str:
    """The one-sided p in the pre-registered direction and the level it was judged at."""
    p, level = r.get("p_one_sided"), r.get("test_level")
    if not isinstance(p, float) or not np.isfinite(p):
        return ""
    return f"{p:.3g}" + (f" (level {level:.3g})" if isinstance(level, float) else "")


_ROLE_TITLES = {
    ledger_book.NOVEL: ("Novel hypotheses (emerging or unexplored a-priori hypotheses and exploratory candidates; "
                        "online false-discovery-rate control, LORD++, in ledger order: each test is judged at the "
                        "level shown, set only by earlier tests, so the verdict is final)"),
    ledger_book.REFERENCE: ("Reference associations (established in the literature; each tested at 0.05 like a "
                            "positive control and never counted as a discovery)"),
    ledger_book.NON_CONFIRMATORY: ("Not confirmatory (a repeat of an earlier test, or an outcome already tested on "
                                   "this holdout; reported for completeness, spends nothing)"),
}


def _effect_text(r: dict[str, Any]) -> tuple[str, str, Any]:
    if r.get("test") == "cox":
        return f"HR/SD {r['hr_per_sd']}", f"{r['ci95'][0]}-{r['ci95'][1]}", r.get("e_value", "")
    if "rho" in r:
        ci = r.get("ci95")
        return f"rho {r['rho']}", (f"{ci[0]}-{ci[1]}" if ci else ""), ""
    return "", "", ""


def _discovery_text(r: dict[str, Any]) -> str:
    """What discovery saw: the screen's correlation and, for a time to event, the Cox hazard ratio that
    set the pre-registered direction."""
    text = f"rho {r.get('discovery_rho')}"
    if r.get("discovery_hr") is not None:
        text += f"; HR {r['discovery_hr']}"
    return text


def reconciliation_markdown(result: dict[str, Any], engine_verdicts: dict[str, str]) -> str:
    """One row per pre-registered hypothesis: what the discovery engine decided, what the per-candidate
    debate concluded, and what the sealed holdout showed. It goes above the agents' narrative, which was
    written before the holdout was opened and so cannot know the last column."""
    lines = ["## Conclusions checked against the sealed holdout", "",
             "The agents' narrative below was written before the holdout was opened. Where it disagrees "
             "with this table (a candidate it calls failed that replicated, or the reverse), this table holds; "
             "the full holdout section follows the narrative.", "",
             "| feature | family | discovery engine | debate | holdout | p (one-sided, level) | robust |",
             "|---|---|---|---|---|---|---|"]
    for r in result.get("hypotheses") or []:
        family = r.get("family", "exploratory")
        engine = engine_verdicts.get(r["feature"]) or ("registered a priori" if family == "a_priori"
                                                       else "not in the final round")
        if family == "a_priori" and r["feature"] in engine_verdicts:
            engine = f"registered a priori; {engine}"
        effect, ci, _ev = _effect_text(r)
        holdout = r.get("status", "")
        if effect:
            holdout += f" ({effect}, {ci})"
        robust = "yes" if r.get("robust") else ("no" if "robust" in r else "")
        lines.append(f"| {r['feature']} | {family.replace('_', '-')} | {engine} | {r.get('debate') or '-'} | "
                     f"{holdout} | {_test_text(r)} | {robust} |")
    return "\n".join(lines) + "\n"


def confirmation_markdown(result: dict[str, Any]) -> str:
    """The deterministic replication and falsification section appended to the report."""
    ledger = result.get("ledger")
    looks = result.get("discovery_variants_evaluated")
    reference = result.get("reference") or {}
    lines = ["## Holdout replication (pre-registered, deterministic)", "",
             f"{result['replicated']} of {result['tested']} novel hypotheses replicated on the sealed holdout "
             f"({result['holdout_rows']} participants; online false-discovery-rate control across every novel "
             f"hypothesis tested on these participants; adjusted for {', '.join(result['confounders']) or 'nothing'}"
             + (f", missing confounder values multiply imputed ({result['covariate_imputations']} draws, "
                "complete-case estimates kept alongside)" if result.get("covariate_imputations") else "")
             + f"), and {result.get('robust', 0)} also passed every falsification test that could run. "
             + (f"{reference.get('reproduced', 0)} of {reference.get('tested', 0)} reference (established) "
                "associations were reproduced. " if reference.get("tested") else "")
             + (f"{result['non_confirmatory']} hypotheses are not confirmatory (repeats or an outcome already "
                "tested on this holdout). " if result.get("non_confirmatory") else "")
             + "The agents never saw these data or results"
             + (f"; discovery evaluated {looks} candidate variants before registering" if looks else "")
             + f". Pre-registration hash: `{result['prereg_sha256'][:16]}`.", ""]
    if ledger:
        lines += [f"The ledger for these participants holds {ledger['hypotheses']} hypotheses from "
                  f"{ledger['registrations']} registrations ({ledger.get('novel_tested', 0)} novel tests so far); "
                  f"the next novel test will be judged at level {ledger.get('next_level', float('nan')):.3g}.", ""]
    if result.get("audit"):
        lines += ["**Checks of this analysis** (read before the table):", "",
                  *[f"- {flag}" for flag in result["audit"]], ""]
    for role in (ledger_book.NOVEL, ledger_book.REFERENCE, ledger_book.NON_CONFIRMATORY):
        members = [r for r in result["hypotheses"] if r.get("role", ledger_book.NOVEL) == role]
        if not members:
            continue
        lines += [f"**{_ROLE_TITLES[role]}**", "",
                  "| feature | family | discovery | holdout effect | 95% CI | p (one-sided, level) | E-value | status | "
                  "dose-response | falsification failed |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in members:
            effect, ci, ev = _effect_text(r)
            dose = r.get("dose_response")
            dose_text = "" if dose is None else ("monotone" if dose["monotone"] else "not monotone")
            status = r.get("status", "") + (" (robust)" if r.get("robust") else "")
            if r.get("knownness"):
                status += f"; {r['knownness']}"
            if r.get("debate"):
                status += f"; debate: {r['debate']}"
            if r.get("role_reason"):
                status += f"; {r['role_reason']}"
            lines.append(f"| {r['feature']} | {r.get('family', 'exploratory').replace('_', '-')} | {_discovery_text(r)} | "
                         f"{effect} | {ci} | {_test_text(r)} | {ev} | {status} | {dose_text} | "
                         f"{', '.join(r.get('falsification_failed') or []) or '-'} |")
        lines.append("")
    cards = [r for r in result["hypotheses"] if r.get("scorecard")]
    if cards:
        total = result.get("predictions") or {}
        lines += [f"**Registered predictions on the holdout**: {total.get('met', 0)} of {total.get('testable', 0)} "
                  "predictions that could be tested were met (magnitude: the 95% CI meets the predicted range; "
                  "shape: from the effect by quantile; the falsification predictions from the checks below).", "",
                  "| feature | prediction | result | seen |", "|---|---|---|---|"]
        for r in cards:
            for item in r["scorecard"]["items"]:
                lines.append(f"| {r['feature']} | {item['prediction']} | {item['result']} | {item['seen']} |")
        lines.append("")
    shrunk = [r for r in result["hypotheses"] if r.get("shrinkage") is not None and r.get("family") == "exploratory"]
    if shrunk:
        lines += ["Shrinkage of the discovery candidates (holdout over discovery log hazard ratio; below 1 is "
                  "expected, since the search picked the largest effects): "
                  + "; ".join(f"{r['feature']} {r['shrinkage']:.2f}" for r in shrunk) + ".", ""]
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
    elif (result.get("adherence_status") or "").startswith("not counted"):
        lines += ["", f"Wear-adherence check {result['adherence_status']}."]
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
            lines += ["", "| feature | negative control | HR/SD | 95% CI | p | status |", "|---|---|---|---|---|---|"]
            lines += [f"| {f} | {nc['control']} | {nc['hr_per_sd']} | {nc['ci95'][0]}-{nc['ci95'][1]} | "
                      f"{nc['p']:.3g} | {nc.get('status', '')} |" for f, nc in shown]
    if result.get("positive_controls"):
        lines += ["", "Positive controls (known associations, tested the same way to show the holdout can detect "
                  "a real effect):", "", "| control | holdout effect | 95% CI | p | detected |", "|---|---|---|---|---|"]
        for pc in result["positive_controls"]:
            effect, ci, _ev = _effect_text(pc)
            lines.append(f"| {pc['control']} | {effect} | {ci} | {pc.get('p', float('nan')):.3g} | "
                         f"{'yes' if pc['detected'] else 'no'} |")
    return "\n".join(lines) + "\n"
