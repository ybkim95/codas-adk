"""What an analyst looks at before forming a hypothesis, and what the outcome shows once registration closes.

``blind_dossier`` describes the data without relating any candidate feature to the outcome, so it can be
read before hypotheses are registered:

* each candidate feature: coverage, mean and SD, quantiles, skew and excess kurtosis, the share of zeros
  and of extreme values (beyond 4 SD, and beyond 3 IQR from the quartiles), whether it is near-constant,
  and a natural unit (its interquartile range, rounded);
* feature families: groups whose members correlate at |rho| >= ``FAMILY_RHO`` with one another, with a
  representative (the best-covered member), so one construct is not tested many times under several names;
* wear dependence: each feature's strongest |rho| with the data-availability (days worn) columns;
* missingness: for each feature with missing values, how the people missing it differ in age, sex and wear
  (standardised mean differences), a sign that its missingness is not at random;
* adjustment covariates: univariate summaries only (a covariate can stand in for the outcome, so its
  correlations with the features would reveal the outcome's);
* the outcome on its own: its distribution, or for a time-to-event outcome the events, person-years,
  incidence per 1,000 person-years and censoring, overall and by sex and age band (``strata``).

``signal_map`` relates every candidate feature to the outcome on the discovery share, for after registration:
its association adjusted for age and sex (T0) and additionally for every adjustment covariate (T2), how much
of it the covariates explain (attenuation), whether being missing the feature is itself associated with the
outcome, and q-values across features.

Everything returned is aggregate. A quantile is shown only when at least ``MIN_TAIL`` people lie beyond it,
and a count or share resting on 1 to ``MIN_TAIL`` people is withheld (the All of Us dissemination rule).
Nothing here knows which dataset, outcome or feature it describes.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from codas.core.statistics import benjamini_hochberg, partial_spearman
from codas.core.survival import cox_ph

MIN_TAIL = 20  # people that must lie beyond a quantile, or behind a count, for it to be shown
FAMILY_RHO = 0.7
AGE_BANDS = (40, 60)  # age < 40, 40-59, >= 60
_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _round(value: float, digits: int = 4) -> float | None:
    return None if value is None or not np.isfinite(value) else float(f"{value:.{digits}g}")


def _shown_count(n: int, tail: int = MIN_TAIL) -> int | str:
    return f"<={tail}" if 1 <= n <= tail else int(n)


def _share(count: int, total: int, tail: int = MIN_TAIL) -> float | str | None:
    """count/total, or a mask when the count behind it is 1 to ``tail`` (or the rest is)."""
    if total <= 0:
        return None
    if 1 <= count <= tail or 1 <= total - count <= tail:
        return f"masked (<={tail} people)"
    return round(count / total, 4)


def natural_unit(iqr: float) -> float | None:
    """The interquartile range rounded to one significant figure: a unit a reader can picture."""
    if not np.isfinite(iqr) or iqr <= 0:
        return None
    return float(f"{iqr:.1g}")


def describe(values: pd.Series, tail: int = MIN_TAIL) -> dict[str, Any]:
    """One variable on its own; ``tail`` is the smallest count that may be shown (0 shows everything)."""
    x = pd.to_numeric(values, errors="coerce")
    present = x.dropna().to_numpy(dtype=float)
    n, missing = len(present), int(x.isna().sum())
    out: dict[str, Any] = {"n": _shown_count(n, tail), "missing": _share(missing, len(x), tail)}
    if n < max(2 * tail, 3):
        out["note"] = "too few values to describe"
        return out
    mean, sd = float(present.mean()), float(present.std(ddof=1))
    q = {p: float(np.quantile(present, p)) for p in _QUANTILES}
    varies = sd > 1e-12 * max(1.0, abs(mean))
    out.update(mean=_round(mean), sd=_round(sd), skew=_round(float(stats.skew(present))) if varies else None,
               excess_kurtosis=_round(float(stats.kurtosis(present))) if varies else None)
    out["quantiles"] = {f"p{int(p * 100)}": _round(v) for p, v in q.items() if min(p, 1 - p) * n >= tail}
    iqr = q[0.75] - q[0.25]
    zeros = int((present == 0).sum())
    beyond_sd = int((np.abs(present - mean) > 4 * sd).sum()) if sd > 0 else 0
    beyond_iqr = int(((present < q[0.25] - 3 * iqr) | (present > q[0.75] + 3 * iqr)).sum()) if iqr > 0 else 0
    top_share = float(pd.Series(present).value_counts(normalize=True).iloc[0])
    out.update(zeros=_share(zeros, n, tail), beyond_4sd=_share(beyond_sd, n, tail),
               beyond_3iqr=_share(beyond_iqr, n, tail),
               near_constant=bool(sd == 0 or top_share >= 0.95), natural_unit=natural_unit(iqr))
    return out


def _rho_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman", min_periods=3 * MIN_TAIL)


def families(frame: pd.DataFrame, features: list[str], threshold: float = FAMILY_RHO) -> list[dict[str, Any]]:
    """Groups of features linked by |rho| >= threshold (connected components), largest first, each with its
    representative: the member with the fewest missing values (then the first in column order)."""
    if len(features) < 2:
        return []
    rho = _rho_matrix(frame, features).abs()
    seen: set[str] = set()
    groups = []
    for start in features:
        if start in seen:
            continue
        group, stack = [], [start]
        seen.add(start)
        while stack:
            f = stack.pop()
            group.append(f)
            for g in features:
                if g not in seen and np.isfinite(rho.at[f, g]) and rho.at[f, g] >= threshold:
                    seen.add(g)
                    stack.append(g)
        if len(group) > 1:
            group.sort(key=features.index)
            coverage = {g: int(frame[g].notna().sum()) for g in group}
            sub = rho.loc[group, group].to_numpy()
            off = sub[~np.eye(len(group), dtype=bool)]
            groups.append({"representative": max(group, key=lambda g: (coverage[g], -features.index(g))),
                           "members": group, "min_abs_rho": _round(float(np.nanmin(off)), 3),
                           "max_abs_rho": _round(float(np.nanmax(off)), 3)})
    return sorted(groups, key=lambda g: -len(g["members"]))


def wear_dependence(frame: pd.DataFrame, features: list[str], availability: list[str]) -> dict[str, Any]:
    """Each feature's strongest |rho| with a data-availability column, and which one."""
    availability = [a for a in availability if a in frame.columns]
    if not availability:
        return {}
    rho = _rho_matrix(frame, list(dict.fromkeys(features + availability))).abs()
    out = {}
    for f in features:
        if f in availability:
            continue
        row = rho.loc[f, availability].dropna()
        if len(row):
            out[f] = {"max_abs_rho": _round(float(row.max()), 3), "with": str(row.idxmax())}
    return out


def _smd(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return _round((a.mean() - b.mean()) / pooled, 3) if pooled > 0 else None


def missingness(frame: pd.DataFrame, features: list[str], by: list[str], tail: int = MIN_TAIL) -> dict[str, Any]:
    """For each feature with missing values: how those missing it differ from those who have it, as a
    standardised mean difference in each ``by`` column. Shown only when both groups exceed ``tail``."""
    by = [b for b in by if b in frame.columns]
    out = {}
    for f in features:
        miss = frame[f].isna().to_numpy()
        n_miss = int(miss.sum())
        if n_miss <= max(tail, 1) or len(frame) - n_miss <= max(tail, 1):
            continue
        out[f] = {"missing": _share(n_miss, len(frame), tail),
                  **{f"smd_{b}": _smd(pd.to_numeric(frame[b], errors="coerce").to_numpy(dtype=float)[miss],
                                      pd.to_numeric(frame[b], errors="coerce").to_numpy(dtype=float)[~miss])
                     for b in by}}
    return out


def strata(frame: pd.DataFrame, *, age: str = "age", sex: str = "female") -> pd.Series:
    """The strata the outcome is summarised by before registration: age band x sex. A summary of the
    outcome within these strata says nothing about any feature."""
    band = pd.cut(pd.to_numeric(frame[age], errors="coerce"), [-np.inf, *AGE_BANDS, np.inf], right=False,
                  labels=[f"<{AGE_BANDS[0]}", *[f"{lo}-{hi - 1}" for lo, hi in zip(AGE_BANDS, AGE_BANDS[1:])],
                          f">={AGE_BANDS[-1]}"]) if age in frame.columns else pd.Series("all", index=frame.index)
    sexes = frame[sex].map({1: "female", 0: "male", 1.0: "female", 0.0: "male"}) if sex in frame.columns else "all"
    return band.astype(str) + " " + pd.Series(sexes, index=frame.index).astype(str)


def outcome_profile(frame: pd.DataFrame, target: str, survival_time: str | None = None, *, age: str = "age",
                    sex: str = "female", tail: int = MIN_TAIL) -> dict[str, Any]:
    """The outcome on its own, overall and within ``strata``."""
    y = pd.to_numeric(frame[target], errors="coerce")
    if not survival_time:
        out = {"kind": "value", "overall": describe(y, tail)}
        groups = strata(frame, age=age, sex=sex)
        out["by_stratum"] = {k: {"n": _shown_count(int(v.notna().sum()), tail),
                                 "mean": _round(float(v.mean())) if v.notna().sum() > max(tail, 1) else None}
                             for k, v in y.groupby(groups)}
        return out
    t = pd.to_numeric(frame[survival_time], errors="coerce") / 365.25

    def summary(mask: pd.Series) -> dict[str, Any]:
        events, years, n = int(y[mask].sum()), float(t[mask].sum()), int(mask.sum())
        enough = max(tail, 0)
        return {"participants": _shown_count(n, tail), "events": _shown_count(events, tail),
                "person_years": round(years) if n > enough else None,
                "per_1000_person_years": (_round(1000 * events / years, 3)
                                          if years > 0 and events > enough and n - events > enough else None),
                "median_follow_up_years": _round(float(t[mask].median()), 3) if n > enough else None,
                "censored": _share(n - events, n, tail)}

    ok = y.notna() & t.notna()
    groups = strata(frame, age=age, sex=sex)
    return {"kind": "time to event", "overall": summary(ok),
            "by_stratum": {k: summary(ok & (groups == k)) for k in sorted(groups[ok].unique())}}


def blind_dossier(frame: pd.DataFrame, *, target: str, features: list[str], survival_time: str | None = None,
                  covariates: Iterable[str] = (), availability: Iterable[str] = (), age: str = "age",
                  sex: str = "female", tail: int = MIN_TAIL) -> dict[str, Any]:
    """The outcome-blind dossier (see the module docstring). ``tail`` is the smallest count that may be
    shown: MIN_TAIL under a small-cell rule, 0 for data without one."""
    features = [f for f in features if f in frame.columns and f not in (target, survival_time)]
    covariates = [c for c in covariates if c in frame.columns and c not in features]
    availability = [a for a in availability if a in frame.columns]
    return {
        "participants": _shown_count(len(frame), tail),
        "features": {f: describe(frame[f], tail) for f in features},
        "families": families(frame, features),
        "wear_dependence": wear_dependence(frame, features, availability),
        "missingness": missingness(frame, features, [age, sex, *availability], tail),
        "covariates": {c: describe(frame[c], tail) for c in covariates},
        "outcome": outcome_profile(frame, target, survival_time, age=age, sex=sex, tail=tail),
    }


def digest(dossier: dict[str, Any], limit: int = 40) -> str:
    """A compact text of the dossier for the agents' instructions: the problems first."""
    feats = dossier.get("features") or {}
    lines = [f"Participants: {dossier.get('participants')}. Candidate features: {len(feats)}."]
    skewed = [f for f, d in feats.items() if isinstance(d.get("skew"), float) and abs(d["skew"]) > 2]
    constant = [f for f, d in feats.items() if d.get("near_constant")]
    extreme = [f for f, d in feats.items() if isinstance(d.get("beyond_4sd"), float) and d["beyond_4sd"] > 0.005]
    if constant:
        lines.append("Near-constant (little to learn from): " + ", ".join(constant[:limit]) + ".")
    if skewed:
        lines.append("Strongly skewed (|skew| > 2; consider a log or rank scale): " + ", ".join(skewed[:limit]) + ".")
    if extreme:
        lines.append("More than 0.5% of values beyond 4 SD: " + ", ".join(extreme[:limit]) + ".")
    fams = dossier.get("families") or []
    if fams:
        lines.append("Feature families (|rho| >= %.1f; each measures much the same thing): " % FAMILY_RHO + "; ".join(
            f"{g['representative']} ~ {', '.join(m for m in g['members'] if m != g['representative'])}"
            for g in fams[:limit]) + ".")
    wear = {f: d for f, d in (dossier.get("wear_dependence") or {}).items() if d["max_abs_rho"] >= 0.3}
    if wear:
        lines.append("Track how much the device was worn (|rho| >= 0.3 with a days-worn column): " + ", ".join(
            f"{f} ({d['max_abs_rho']})" for f, d in sorted(wear.items(), key=lambda kv: -kv[1]["max_abs_rho"])[:limit])
            + ".")
    miss = {f: d for f, d in (dossier.get("missingness") or {}).items()
            if any(isinstance(v, float) and abs(v) >= 0.2 for k, v in d.items() if k.startswith("smd_"))}
    if miss:
        lines.append("Missing not at random (those missing it differ by |SMD| >= 0.2): " + "; ".join(
            f"{f} (missing {d['missing']}; " + ", ".join(f"{k[4:]} {v}" for k, v in d.items()
                                                          if k.startswith("smd_") and v is not None) + ")"
            for f, d in list(miss.items())[:limit]) + ".")
    units = dossier.get("outcome_units") or {}
    if units.get("units"):
        listed = ", ".join(f"{u['unit']} ({u['share']})" for u in units["units"])
        lines.append(("The outcome is recorded in more than one unit, so its values are not all on one scale: "
                      if units.get("mixed") else "Outcome units: ") + listed + ".")
    reliability = {f: r["reliability"] for f, r in (dossier.get("reliability") or {}).items()
                   if isinstance(r.get("reliability"), (int, float))}
    weak = sorted((f for f, r in reliability.items() if r < 0.5), key=reliability.get)
    if weak:
        lines.append("Unreliable over the window (split-half reliability < 0.5: its first and second half barely "
                     "agree, so the value describes those weeks more than the person): "
                     + ", ".join(f"{f} ({reliability[f]})" for f in weak[:limit]) + ".")
    outcome = dossier.get("outcome") or {}
    overall = outcome.get("overall") or {}
    if outcome.get("kind") == "time to event":
        lines.append(f"Outcome: {overall.get('events')} events among {overall.get('participants')} participants over "
                     f"{overall.get('person_years')} person-years ({overall.get('per_1000_person_years')} per 1,000 "
                     f"person-years; median follow-up {overall.get('median_follow_up_years')} years).")
    elif overall:
        lines.append(f"Outcome: mean {overall.get('mean')}, SD {overall.get('sd')}, skew {overall.get('skew')}, "
                     f"quantiles {overall.get('quantiles')}.")
    return "\n".join(lines)


# --- after registration: the outcome-linked signal map --------------------------------------------------

def _filled(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Covariates for a screen: median-filled, plus a missing-value indicator for any with gaps."""
    blocks = []
    for c in columns:
        v = pd.to_numeric(frame[c], errors="coerce")
        blocks.append(v.fillna(v.median()).to_numpy(dtype=float))
        if v.isna().any():
            blocks.append(v.isna().to_numpy(dtype=float))
    return np.column_stack(blocks) if blocks else np.empty((len(frame), 0))


def _association(frame: pd.DataFrame, feature: str, target: str, survival_time: str | None,
                 covariates: np.ndarray) -> dict[str, Any]:
    x = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    if not survival_time:
        rho, p, n = partial_spearman(x, y, covariates)
        return {"effect": _round(rho, 3), "p": p, "n": n, "measure": "partial rank correlation"}
    t = pd.to_numeric(frame[survival_time], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & np.all(np.isfinite(covariates), axis=1)
    sd = float(np.std(x[ok], ddof=1)) if ok.sum() > 2 else 0.0
    if ok.sum() < 3 * MIN_TAIL or sd == 0 or y[ok].sum() < MIN_TAIL:
        return {"effect": None, "p": math.nan, "n": int(ok.sum()), "measure": "hazard ratio per SD"}
    fit = cox_ph(t[ok], y[ok], np.column_stack([(x[ok] - x[ok].mean()) / sd, covariates[ok]]))
    beta, se = float(fit.coef[0]), float(fit.se[0])
    p = float(2 * stats.norm.sf(abs(beta / se))) if np.isfinite(se) and se > 0 else math.nan
    return {"effect": _round(math.exp(beta), 3) if np.isfinite(beta) else None, "p": p, "n": int(ok.sum()),
            "measure": "hazard ratio per SD"}


def signal_map(frame: pd.DataFrame, *, target: str, features: list[str], survival_time: str | None = None,
               base: Iterable[str] = ("age", "female"), covariates: Iterable[str] = ()) -> dict[str, Any]:
    """Each feature's association with the outcome adjusted for ``base`` (T0) and additionally for
    ``covariates`` (T2), the attenuation between them, the association of being missing the feature with the
    outcome, and BH q-values (T2) across features. For after registration only."""
    base = [b for b in base if b in frame.columns]
    covariates = [c for c in covariates if c in frame.columns and c not in base]
    t0, t2 = _filled(frame, base), _filled(frame, base + covariates)
    rows = []
    for f in features:
        if f not in frame.columns:
            continue
        a0 = _association(frame, f, target, survival_time, t0)
        a2 = _association(frame, f, target, survival_time, t2) if covariates else a0
        row = {"feature": f, "measure": a0["measure"], "t0": a0["effect"], "t2": a2["effect"], "p_t2": a2["p"],
               "n": _shown_count(int(a2["n"]))}
        scale = (lambda e: math.log(e)) if survival_time else (lambda e: e)
        if a0["effect"] not in (None, 0) and a2["effect"] is not None and (not survival_time or a0["effect"] > 0):
            e0, e2 = scale(a0["effect"]), scale(a2["effect"])
            row["attenuation"] = _round(1 - e2 / e0, 3) if e0 else None
        missing = frame[f].isna()
        if MIN_TAIL < int(missing.sum()) < len(frame) - MIN_TAIL:
            indicator = frame.assign(_missing=missing.astype(float))
            row["missing_effect"] = _association(indicator, "_missing", target, survival_time, t0)["effect"]
        rows.append(row)
    q = benjamini_hochberg([r["p_t2"] if np.isfinite(r["p_t2"]) else 1.0 for r in rows])
    for r, qv in zip(rows, q):
        r["q_t2"] = _round(qv, 3)
        r["p_t2"] = _round(r["p_t2"], 3)
    return {"base": base, "covariates": covariates, "features": rows}


def signal_digest(signal: dict[str, Any], limit: int = 25) -> str:
    rows = [r for r in signal.get("features") or [] if r.get("q_t2") is not None]
    if not rows:
        return ""
    rows.sort(key=lambda r: (r["q_t2"], r["feature"]))
    measure = rows[0]["measure"]
    lines = [f"Signal map on the discovery share ({measure}; T0 adjusts for {', '.join(signal['base']) or 'nothing'}, "
             f"T2 also for {', '.join(signal['covariates']) or 'nothing more'}; attenuation = share of the T0 effect "
             "the covariates explain):"]
    for r in rows[:limit]:
        extra = f", attenuation {r['attenuation']}" if r.get("attenuation") is not None else ""
        miss = f", missing-indicator effect {r['missing_effect']}" if r.get("missing_effect") is not None else ""
        lines.append(f"- {r['feature']}: T0 {r['t0']}, T2 {r['t2']} (q {r['q_t2']}){extra}{miss}")
    return "\n".join(lines)
