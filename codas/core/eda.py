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

``audit_memo`` checks an analyst's memo on the dossier: each claim cites dossier entries, and every number it
states must be a value of those entries; ``issues`` lists the problems the dossier flags, so a memo's
coverage of them can be counted.

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
import re
from collections import Counter
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
# What the digest flags (and a memo is expected to notice).
SKEW_LIMIT = 2.0        # |skew| above this: strongly skewed
EXTREME_SHARE = 0.005   # more than this share of values beyond 4 SD
WEAR_RHO = 0.3          # |rho| with a days-worn column at or above this: tracks wear
MISSING_SMD = 0.2       # |standardised mean difference| between those missing a feature and the rest
RELIABILITY_FLAG = 0.5  # split-half reliability below this: unreliable over the window


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
    skewed = [f for f, d in feats.items() if isinstance(d.get("skew"), float) and abs(d["skew"]) > SKEW_LIMIT]
    constant = [f for f, d in feats.items() if d.get("near_constant")]
    extreme = [f for f, d in feats.items()
               if isinstance(d.get("beyond_4sd"), float) and d["beyond_4sd"] > EXTREME_SHARE]
    if constant:
        lines.append("Near-constant (little to learn from): " + ", ".join(constant[:limit]) + ".")
    if skewed:
        lines.append(f"Strongly skewed (|skew| > {SKEW_LIMIT:g}; consider a log or rank scale): "
                     + ", ".join(skewed[:limit]) + ".")
    if extreme:
        lines.append(f"More than {100 * EXTREME_SHARE:g}% of values beyond 4 SD: " + ", ".join(extreme[:limit]) + ".")
    fams = dossier.get("families") or []
    if fams:
        lines.append("Feature families (|rho| >= %.1f; each measures much the same thing): " % FAMILY_RHO + "; ".join(
            f"{g['representative']} ~ {', '.join(m for m in g['members'] if m != g['representative'])}"
            for g in fams[:limit]) + ".")
    wear = {f: d for f, d in (dossier.get("wear_dependence") or {}).items() if d["max_abs_rho"] >= WEAR_RHO}
    if wear:
        lines.append(f"Track how much the device was worn (|rho| >= {WEAR_RHO:g} with a days-worn column): " + ", ".join(
            f"{f} ({d['max_abs_rho']})" for f, d in sorted(wear.items(), key=lambda kv: -kv[1]["max_abs_rho"])[:limit])
            + ".")
    miss = {f: d for f, d in (dossier.get("missingness") or {}).items()
            if any(isinstance(v, float) and abs(v) >= MISSING_SMD for k, v in d.items() if k.startswith("smd_"))}
    if miss:
        lines.append(f"Missing not at random (those missing it differ by |SMD| >= {MISSING_SMD:g}): " + "; ".join(
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
    weak = sorted((f for f, r in reliability.items() if r < RELIABILITY_FLAG), key=reliability.get)
    if weak:
        lines.append(f"Unreliable over the window (split-half reliability < {RELIABILITY_FLAG:g}: its first and second half barely "
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


# --- the analyst's memo: every claim cites the dossier, and the numbers it states are checked -------------

# A number as a sentence states it (8,100 or 0.62 or 12%), not a digit inside a name such as p99 or hr10h.
_STATED = re.compile(r"(?<![\w.])(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(%?)(?!\w)")
_ABSENT = object()


def _lookup(dossier: dict[str, Any], path: str) -> Any:
    """The dossier entry at a dotted ``path`` (``features.<column>.skew``, ``families.<column>`` for the family
    holding that column, ``outcome.overall.events``), or ``_ABSENT``."""
    node: Any = dossier
    rest = [t for t in str(path or "").strip().split(".") if t]
    if not rest:
        return _ABSENT
    while rest:
        if isinstance(node, dict):
            for k in range(len(rest), 0, -1):  # a column name may itself hold a dot
                key = ".".join(rest[:k])
                if key in node:
                    node, rest = node[key], rest[k:]
                    break
            else:
                return _ABSENT
        elif isinstance(node, list):
            token = rest[0]
            hits = ([node[int(token)]] if token.isdigit() and int(token) < len(node) else
                    [g for g in node if isinstance(g, dict)
                     and (g.get("representative") == token or token in (g.get("members") or []))])
            if not hits:
                return _ABSENT
            node, rest = hits[0], rest[1:]
        else:
            return _ABSENT
    return node


def _canonical(dossier: dict[str, Any], path: str) -> str:
    """A cited path in the form ``issues`` uses: a family named by any member becomes its representative's."""
    parts = str(path or "").strip().split(".")
    if len(parts) >= 2 and parts[0] == "families":
        family = _lookup(dossier, ".".join(parts[:2]))
        if isinstance(family, dict) and family.get("representative"):
            parts[1] = family["representative"]
    return ".".join(parts)


def _values(node: Any) -> list[float]:
    """Every number in a dossier entry."""
    if isinstance(node, bool) or node is None:
        return []
    if isinstance(node, (int, float)):
        return [float(node)] if math.isfinite(node) else []
    if isinstance(node, dict):
        return [v for x in node.values() for v in _values(x)]
    if isinstance(node, (list, tuple)):
        return [v for x in node for v in _values(x)]
    return []


def _holds(text: str, percent: bool, values: list[float]) -> bool:
    """Whether a stated number is one of ``values`` to its stated precision (or within 2%); a percentage may
    also state a share."""
    x = float(text.replace(",", ""))
    tol = 0.5 * 10 ** -(len(text.split(".")[1]) if "." in text else 0)
    for target, t in [(x, tol)] + ([(x / 100, tol / 100)] if percent else []):
        if any(abs(target - v) <= t + 1e-9 or (v and abs(target - v) <= 0.02 * abs(v)) for v in values):
            return True
    return False


def check_claim(dossier: dict[str, Any], statement: str, fields: list[str]) -> tuple[str, str]:
    """``verified`` when every number in ``statement`` is a value of the cited ``fields`` (or the count of a
    cited list or table), ``cited`` when it states no number, else ``wrong value`` or ``unknown field``, with
    what did not check out."""
    if not fields:
        return "unknown field", "cite at least one entry of the data summary"
    values, missing = [], []
    for f in fields:
        node = _lookup(dossier, f)
        if node is _ABSENT:
            missing.append(f)
            continue
        values += _values(node)
        if isinstance(node, (list, dict)):
            values.append(float(len(node)))
    if missing:
        return "unknown field", "not in the data summary: " + ", ".join(missing)
    stated = _STATED.findall(statement)
    if not stated:
        return "cited", ""
    wrong = [n + p for n, p in stated if not _holds(n, bool(p), values)]
    if wrong:
        return "wrong value", ("the cited entries do not hold " + ", ".join(wrong) + " (write thresholds and "
                               "percentile names in words, and every other number as the summary gives it)")
    return "verified", ""


def issues(dossier: dict[str, Any]) -> list[dict[str, str]]:
    """The problems the dossier flags (those its digest lists), each with the entry that shows it."""
    out: list[dict[str, str]] = []

    def add(problem: str, subject: str, field: str) -> None:
        out.append({"problem": problem, "subject": subject, "field": field})

    for f, d in (dossier.get("features") or {}).items():
        if d.get("near_constant"):
            add("near-constant", f, f"features.{f}.near_constant")
        if isinstance(d.get("skew"), float) and abs(d["skew"]) > SKEW_LIMIT:
            add("strongly skewed", f, f"features.{f}.skew")
        if isinstance(d.get("beyond_4sd"), float) and d["beyond_4sd"] > EXTREME_SHARE:
            add("extreme values", f, f"features.{f}.beyond_4sd")
    for g in dossier.get("families") or []:
        add("one construct under several names", g["representative"], f"families.{g['representative']}")
    for f, d in (dossier.get("wear_dependence") or {}).items():
        if d.get("max_abs_rho", 0) >= WEAR_RHO:
            add("tracks wear", f, f"wear_dependence.{f}")
    for f, d in (dossier.get("missingness") or {}).items():
        if any(isinstance(v, float) and abs(v) >= MISSING_SMD for k, v in d.items() if k.startswith("smd_")):
            add("missing not at random", f, f"missingness.{f}")
    for f, r in (dossier.get("reliability") or {}).items():
        if isinstance(r.get("reliability"), (int, float)) and r["reliability"] < RELIABILITY_FLAG:
            add("unreliable over the window", f, f"reliability.{f}")
    if (dossier.get("outcome_units") or {}).get("mixed"):
        add("outcome in more than one unit", "outcome", "outcome_units")
    return out


def _covers(cited: str, field: str) -> bool:
    return cited == field or field.startswith(cited + ".") or cited.startswith(field + ".")


def audit_memo(dossier: dict[str, Any], memo: dict[str, Any], limit: int = 15) -> dict[str, Any]:
    """Each claim of ``memo`` ({"summary", "claims": [{"statement", "fields", "implication"}]}) checked
    against ``dossier``, with how many of the dossier's flagged problems the confirmed claims address."""
    rows = []
    for claim in memo.get("claims") or []:
        claim = claim if isinstance(claim, dict) else {"statement": str(claim)}
        given = claim.get("fields") if claim.get("fields") is not None else claim.get("field")
        fields = [str(f).strip() for f in (given if isinstance(given, list) else [given]) if f and str(f).strip()]
        statement = str(claim.get("statement") or "").strip()
        status, detail = check_claim(dossier, statement, fields)
        rows.append({"statement": statement[:400], "fields": fields,
                     "implication": str(claim.get("implication") or "")[:300], "status": status, "detail": detail})
    found = issues(dossier)
    cited = [_canonical(dossier, f) for r in rows if r["status"] in ("verified", "cited") for f in r["fields"]]
    missed = [i for i in found if not any(_covers(c, i["field"]) for c in cited)]
    return {"summary": str(memo.get("summary") or "")[:800], "claims": rows,
            "checked": dict(Counter(r["status"] for r in rows)), "issues": len(found),
            "noticed": len(found) - len(missed),
            "missed": [f"{i['problem']}: {i['subject']}" for i in missed[:limit]]}


def memo_text(audit: dict[str, Any]) -> str:
    """The memo as the hypothesis designer reads it: the summary and the claims the data summary confirms."""
    kept = [r for r in audit.get("claims") or [] if r["status"] in ("verified", "cited")]
    lines = [audit.get("summary") or ""] + [
        f"- {r['statement']} [{', '.join(r['fields'])}]" + (f" Implication: {r['implication']}" if r["implication"] else "")
        for r in kept]
    return "\n".join(line for line in lines if line)


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
        if covariates.shape[1]:
            rho, p, n = partial_spearman(x, y, covariates)
        else:  # nothing to adjust for (no base covariate): the plain rank correlation
            ok = np.isfinite(x) & np.isfinite(y)
            n = int(ok.sum())
            rho, p = stats.spearmanr(x[ok], y[ok]) if n > 2 else (math.nan, math.nan)
            rho, p = float(rho), float(p)
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
