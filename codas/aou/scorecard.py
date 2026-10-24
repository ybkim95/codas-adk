"""What a run did and how far its evidence reaches, read from its records rather than asserted.

* Recipe coverage: for each hypothesis tested on the sealed holdout, which steps of the analysis that published
  biomarker studies share it went through (the effect adjusted, in natural units and by quarter, absolute risk,
  dose-response, a lag, negative controls, wear adherence, the incumbent head to head, added predictive value,
  within-person change, consistency across strata, the sealed-holdout test, an E-value, the gain beyond a strong
  baseline of every established marker, the share a declared mediator carries, how specific the feature is
  across the phenome scan, the feature as a time-varying exposure, its course before the diagnosis, and its
  genetic triangulation by Mendelian randomization). Each step is ``yes`` (ran and gave a result, whatever the result), ``no``
  (applicable but missing, or could not be computed) or ``n/a`` (not applicable to this outcome or hypothesis),
  with what it showed; whether the result supports the hypothesis is the rung's to say.
* The evidence ladder (after Hlatky et al., Circulation 2009): 1, associated on the sealed holdout in the
  registered direction (nominal one-sided p < 0.05); 2, replicated under the ledger's error control with every
  falsification test that ran passed; 3, adding predictive value (a gain whose 95% interval excludes zero, of a
  size that is not negligible) beyond the strong baseline of the covariates and every established marker, or
  beyond the covariates when no marker is established. Clinical utility, outcomes and external validation lie
  beyond what one cohort's holdout can show and are named as the next steps.
* The scorecard: how well the data were understood, how the hypotheses were designed, and what the holdout
  showed, one measure per line.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

YES, NO, NA = "yes", "no", "n/a"
STEPS = [
    ("adjusted", "adjusted for confounders"),
    ("units", "per natural unit and by quarter"),
    ("absolute_risk", "absolute risk"),
    ("dose_response", "dose-response"),
    ("lag", "early events excluded (lag)"),
    ("negative_controls", "negative-control outcomes"),
    ("adherence", "wear adherence"),
    ("incumbent", "against the incumbent"),
    ("prediction", "added predictive value"),
    ("within_person", "within-person"),
    ("heterogeneity", "consistent across strata"),
    ("replication", "sealed-holdout test"),
    ("e_value", "E-value"),
    ("strong_baseline", "beyond the established markers"),
    ("mediation", "through a declared mediator"),
    ("specificity", "specific across the phenome"),
    ("time_varying", "time-varying exposure"),
    ("trajectory", "course before the diagnosis"),
    ("genetic", "genetic triangulation"),
]
RUNGS = {0: "not associated on the holdout", 1: "associated on the holdout",
         2: "replicated and robust to every falsification test that ran",
         3: "adds predictive value beyond the established markers"}
NEXT = {0: "a new hypothesis, or more participants",
        1: "replication under the ledger's error control, and the falsification tests",
        2: "added predictive value beyond the covariates and the established markers",
        3: "clinical utility (reclassification at decision thresholds) and replication in an external cohort"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _ci(values: Any) -> str:
    if not values or not all(_finite(v) for v in values):
        return ""
    return f" ({values[0]:.3g} to {values[1]:.3g})"


def _cells(row: dict[str, Any], confirmation: dict[str, Any], time_to_event: bool,
           incumbent_registered: Any = None, mediators_registered: Any = None,
           specificity: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Each step of the recipe for one tested hypothesis: [status, what it showed]. ``incumbent_registered`` is
    the marker the hypothesis was registered to beat (none for a discovery candidate), ``mediators_registered``
    the mediators it declared, ``specificity`` the phenome scan's profile of its feature (none when the scan
    did not screen it)."""
    sizes = row.get("effect_sizes") or {}
    cells: dict[str, list[str]] = {}
    confounders = confirmation.get("confounders") or []
    estimated = _finite(row.get("p"))
    cells["adjusted"] = ([YES, f"{len(confounders)} confounders" + (", multiply imputed" if row.get("imputations") else "")]
                         if estimated and confounders else [NO, "no estimate" if not estimated else "no confounder"])
    unit = sizes.get("hr_per_unit") if time_to_event else sizes.get("per_unit")
    quarter = sizes.get("quarter_hr") if time_to_event else sizes.get("top_vs_bottom_quarter")
    shown = ([f"per {sizes.get('unit')}: {unit:.3g}"] if _finite(unit) else []) + (["by quarter"] if quarter else [])
    cells["units"] = [YES, ", ".join(shown)] if shown else [NO, "not estimable"]
    if time_to_event:
        risk = sizes.get("cumulative_incidence") or []
        diff = sizes.get("risk_difference")
        cells["absolute_risk"] = ([YES, f"top minus bottom quarter {diff:+.3g}"] if _finite(diff) else
                                  [YES, "by quarter (difference masked)"] if any(r is not None for r in risk) else
                                  [NO, "masked or not estimable"])
    else:
        r2 = sizes.get("partial_delta_r2")
        cells["absolute_risk"] = [NA, "a value outcome" + (f"; variance explained {r2:.3g}" if _finite(r2) else "")]
    dose = row.get("dose_response")
    cells["dose_response"] = ([YES, "monotone" if dose.get("monotone") else "not monotone"] if dose
                              else [NO, "too few participants for quantiles"])
    if time_to_event:
        lag = row.get("lag_sensitivity")
        cells["lag"] = ([YES, "same direction" if lag.get("same_direction") else "direction changed"]
                        if lag and _finite(lag.get("p")) else [NO, "could not be computed"])
    else:
        cells["lag"] = [NA, "a value outcome"]
    controls = row.get("negative_controls") or []
    ran = [nc for nc in controls if nc.get("status") in ("no association", "associated")]
    if not confirmation.get("negative_controls"):
        cells["negative_controls"] = [NA, "none registered"]
    elif ran:
        bad = [nc["control"] for nc in ran if nc["status"] == "associated"]
        cells["negative_controls"] = [YES, f"{len(ran) - len(bad)} of {len(ran)} null" + (f"; associated: {', '.join(bad)}"
                                                                                       if bad else "")]
    else:
        cells["negative_controls"] = [NO, "too few events to test"]
    adherence = row.get("adherence_adjusted")
    status = str(confirmation.get("adherence_status") or "")
    if adherence and _finite(adherence.get("p")):
        cells["adherence"] = [YES, "holds" if adherence.get("holds") else "does not hold"]
    elif status.startswith("not counted"):
        cells["adherence"] = [YES, "already in the adjustment set"]
    else:
        cells["adherence"] = [NA, "no days-with-data column"] if status.startswith("not tested") else [NO, "not computed"]
    incumbent = row.get("incumbent_adjusted")
    if incumbent:
        att = incumbent.get("attenuation")
        cells["incumbent"] = [YES, f"{incumbent['incumbent']}: attenuation {att:.0%}" if _finite(att)
                              else f"{incumbent['incumbent']}: attenuation not testable"]
    elif str(incumbent_registered or "").strip().lower() in ("", "none", "-"):
        cells["incumbent"] = [NA, "no incumbent registered"]
    else:
        cells["incumbent"] = [NO, f"{incumbent_registered}: not computed (not a column of the holdout table?)"]
    gain = sizes.get("prediction") or {}
    cells["prediction"] = ([YES, f"{gain.get('metric')} gain {gain['gain']:+.3g}{_ci(gain.get('gain_ci95'))}"]
                           if _finite(gain.get("gain")) else [NO, "no frozen model"])
    within = row.get("within_person")
    cells["within_person"] = ([YES, f"within-person {within['within']['estimate']:.3g}"]
                              if within and "within" in within else
                              [NO, str(within.get("error"))[:60]] if within else [NA, "no repeated measurements"])
    varied = [h for h in row.get("heterogeneity") or [] if h.get("q_p") is not None]
    if varied:
        worst = max(varied, key=lambda h: h["i2"])
        cells["heterogeneity"] = [YES, f"{len(varied)} stratifiers; largest I^2 {worst['i2']:.2f} ({worst['by']})"]
    else:
        cells["heterogeneity"] = ([NO, "strata too small to estimate"] if row.get("heterogeneity")
                                  else [NA, "no stratifier registered"])
    if row.get("role") == "non_confirmatory":
        cells["replication"] = [NA, row.get("role_reason") or "not confirmatory"]
    else:
        cells["replication"] = [YES, "replicated" if row.get("replicated") else
                                "reproduced (reference)" if row.get("reproduced") else "not replicated"]
    if not time_to_event:
        cells["e_value"] = [NA, "a value outcome"]
    elif _finite(row.get("e_value")):
        bound = sizes.get("e_value_ci")
        cells["e_value"] = [YES, f"{row['e_value']:.3g}" + (f" (CI bound {bound:.3g})" if _finite(bound) else "")]
    else:
        cells["e_value"] = [NO, "not computed"]
    if _finite(gain.get("gain_beyond_strong")):
        cells["strong_baseline"] = [YES, f"{gain.get('metric')} gain beyond {', '.join(gain.get('baseline') or [])} "
                                         f"{gain['gain_beyond_strong']:+.3g}{_ci(gain.get('gain_beyond_strong_ci95'))}"]
    elif _finite(gain.get("gain")):
        cells["strong_baseline"] = [NA, "no established marker beyond the confounders"]
    else:
        cells["strong_baseline"] = [NO, "no frozen model"]
    mediation = row.get("mediation") or []
    if mediation:
        cells["mediation"] = [YES, "; ".join(f"{m['mediator']} carries " + (f"{m['explained']:.0%}" if _finite(m.get("explained"))
                                                                            else "an unknown share")
                                             for m in mediation)]
    else:
        cells["mediation"] = ([NO, "declared but not estimable on the holdout"] if mediators_registered
                              else [NA, "no mediator declared"])
    if specificity:
        breadth = specificity.get("breadth")
        cells["specificity"] = [YES, f"{specificity.get('specificity')}" + (f" (hit for {breadth:.0%} of outcomes)"
                                                                            if _finite(breadth) else "")]
    else:
        cells["specificity"] = [NA, "not screened by the phenome scan (a feature built in this run)"]
    tv, course = row.get("time_varying") or {}, row.get("trajectory") or {}
    if not time_to_event:
        cells["time_varying"] = [NA, "a value outcome"]
        cells["trajectory"] = [NA, "a value outcome"]
    else:
        cells["time_varying"] = ([YES, f"HR {tv['hr_per_sd']:.3g} per SD{_ci(tv.get('ci95'))}, lag {tv['lag_blocks']} blocks"]
                                 if _finite(tv.get("hr_per_sd")) else
                                 [NO, tv.get("status") or tv.get("error") or "not followed (no month store, or not a "
                                                                                 "daily source's mean)"])
        cells["trajectory"] = ([YES, f"onset: {course['onset']}"] if course.get("onset") not in (None, "not estimated")
                               else [NO, course.get("error") or ("too few cases in every window" if course
                                                                 else "not followed")])
    genetic = row.get("genetic") or {}
    if genetic.get("estimate") is not None:
        cells["genetic"] = [YES, f"MR {genetic['estimate']}{_ci(genetic.get('ci95'))}, F {genetic['f_statistic']}, "
                                 f"{genetic['variants']} variants"]
    elif genetic.get("error"):
        cells["genetic"] = [NO, str(genetic["error"])[:60]]
    else:
        cells["genetic"] = [NA, "no instruments for its construct, or no genomic data"]
    return cells


def rung(row: dict[str, Any]) -> int:
    """The highest rung of the evidence ladder the hypothesis reached (module doc)."""
    if not row.get("same_direction") or not _finite(row.get("p_one_sided")) or row["p_one_sided"] >= 0.05:
        return 0
    if not ((row.get("replicated") or row.get("reproduced")) and row.get("robust")):
        return 1
    gain = (row.get("effect_sizes") or {}).get("prediction") or {}
    key, size = (("gain_beyond_strong", "magnitude_beyond_strong") if gain.get("gain_beyond_strong") is not None
                 else ("gain", "magnitude"))
    lo = (gain.get(f"{key}_ci95") or [None])[0]
    if _finite(lo) and lo > 0 and gain.get(size) not in (None, "negligible"):
        return 3
    return 2


def coverage(confirmation: dict[str, Any] | None, time_to_event: bool,
             registered: list[dict[str, Any]] | None = None,
             specificity: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One record per hypothesis the holdout tested: its recipe cells and its rung. ``registered`` are the
    hypotheses as designed (their incumbents and mediators); ``specificity`` the phenome scan's profile of
    each feature it screened."""
    incumbents = {h.get("feature"): h.get("incumbent") for h in registered or []}
    mediators = {h.get("feature"): h.get("mediators") for h in registered or []}
    out = []
    for row in (confirmation or {}).get("hypotheses") or []:
        if "same_direction" not in row:
            continue
        level = rung(row)
        out.append({"feature": row["feature"], "family": row.get("family", "exploratory"),
                    "cells": _cells(row, confirmation, time_to_event, incumbents.get(row["feature"]),
                                    mediators.get(row["feature"]), (specificity or {}).get(row["feature"])),
                    "rung": level, "rung_label": RUNGS[level], "next": NEXT[level]})
    return out


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole}" + (f" ({100 * part / whole:.0f}%)" if whole else "")


def scorecard(state: dict[str, Any], time_to_event: bool) -> dict[str, Any]:
    """The outcome's scorecard and the recipe coverage of every tested hypothesis."""
    confirmation = state.get("confirmation") or {}
    records = coverage(confirmation, time_to_event, state.get("a_priori_hypotheses"), state.get("phenome_specificity"))
    lines: list[dict[str, str]] = []

    def add(section: str, measure: str, value: Any) -> None:
        if value not in (None, ""):
            lines.append({"section": section, "measure": measure, "value": str(value)})

    dossier = state.get("eda") or {}
    features = dossier.get("features") or {}
    reliability = [r.get("reliability") for r in (dossier.get("reliability") or {}).values() if _finite(r.get("reliability"))]
    add("data understanding", "candidate features profiled before any statistic", len(features) or None)
    add("data understanding", "features measured reliably over the window (split-half >= 0.5)",
        _share(sum(r >= 0.5 for r in reliability), len(reliability)) if reliability else None)
    quality = state.get("hypothesis_quality") or {}
    memo = quality.get("data_understanding") or {}
    if memo.get("claims"):
        add("data understanding", "memo claims the data summary confirms",
            _share(int(memo.get("verified", 0)) + int(memo.get("cited", 0)), int(memo["claims"])))
        add("data understanding", "problems in the data summary the memo noticed",
            _share(int(memo.get("problems_noticed", 0)), int(memo.get("problems_flagged", 0))))
    design = quality.get("design") or {}
    if design.get("registered"):
        add("design", "hypotheses registered before any statistic", design["registered"])
        add("design", "complete at the first submission",
            _share(int(design.get("accepted_first_time", 0)), int(design.get("features_tried", 0))))
        for key, label in (("novel_by_evidence", "new to the literature by counted evidence"),
                           ("on_a_constructed_feature", "on a feature or phenotype the agents constructed"),
                           ("with_incumbent", "naming the incumbent they must beat"),
                           ("magnitude_from_a_paper", "with a predicted magnitude from a paper")):
            if _finite(design.get(key)):
                add("design", label, f"{100 * design[key]:.0f}%")
    citations = state.get("citations") or {}
    if citations.get("used"):
        add("design", "citations among the papers retrieved in this run",
            _share(len(citations.get("verified") or []), len(citations["used"])))
    add("design", "literature service", state.get("literature_status"))
    if confirmation:
        add("holdout", "novel hypotheses replicated (online FDR across the ledger)",
            _share(int(confirmation.get("replicated", 0)), int(confirmation.get("tested", 0))))
        add("holdout", "of them robust to every falsification test that ran", confirmation.get("robust", 0))
        reference = confirmation.get("reference") or {}
        if reference.get("tested"):
            add("holdout", "established associations reproduced (reference family)",
                _share(int(reference.get("reproduced", 0)), int(reference["tested"])))
        predictions = confirmation.get("predictions") or {}
        if predictions.get("testable"):
            add("holdout", "registered predictions met", _share(int(predictions["met"]), int(predictions["testable"])))
        shrink = [r.get("shrinkage") for r in confirmation.get("hypotheses") or []
                  if r.get("family") == "exploratory" and _finite(r.get("shrinkage"))]
        if shrink:
            add("holdout", "median shrinkage of the discovery candidates (holdout over discovery)",
                f"{float(np.median(shrink)):.2f}")
    if records:
        counts = {k: sum(r["rung"] == k for r in records) for k in RUNGS}
        add("evidence", "hypotheses at each rung (0-3)", ", ".join(f"{k}: {v}" for k, v in counts.items()))
        top = max(records, key=lambda r: r["rung"])
        add("evidence", "highest rung reached", f"{top['rung']} ({top['rung_label']}): {top['feature']}")
        add("evidence", "next step for the strongest", top["next"])
        applicable = [c[0] for r in records for c in r["cells"].values() if c[0] != NA]
        add("depth", "recipe steps done where applicable", _share(applicable.count(YES), len(applicable)))
    return {"summary": lines, "hypotheses": records, "steps": [label for _, label in STEPS]}


def markdown(card: dict[str, Any]) -> str:
    """The scorecard, the recipe coverage and the evidence ladder as a section of the report."""
    if not card.get("summary"):
        return ""
    lines = ["## What the analysis covered, and how far the evidence reaches", "",
             "| | measure | value |", "|---|---|---|",
             *[f"| {s['section']} | {s['measure']} | {s['value']} |" for s in card["summary"]], ""]
    records = card.get("hypotheses") or []
    if records:
        lines += ["Recipe coverage (yes: ran and gave a result; no: applicable but missing; n/a: not applicable) and "
                  "the rung of the evidence ladder each hypothesis reached (1 associated on the holdout, 2 replicated "
                  "and robust, 3 adds predictive value beyond the established markers):", "",
                  "| feature | rung | " + " | ".join(label for _, label in STEPS) + " |",
                  "|---|---|" + "---|" * len(STEPS)]
        for r in records:
            lines.append(f"| {r['feature']} | {r['rung']} | " + " | ".join(r["cells"][key][0] for key, _ in STEPS) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"
