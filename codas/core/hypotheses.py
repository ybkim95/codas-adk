"""What a well-designed hypothesis states, and a deterministic check that it states it.

Strong biomarker papers put a hypothesis in one sentence: in people without the outcome at baseline, a feature
(measuring a construct) predicts the outcome in a stated direction, beyond the established marker it must beat
(the incumbent), with a stated shape and roughly a stated magnitude, through a mechanism with steps, robust to
the obvious alternative explanations, each of which has a test. Every part is a field here. ``validate``
refuses a hypothesis with a part missing or untestable, ``objective_problems`` one the outcome-blind data
summary shows cannot be tested well, so what reaches the sealed holdout can be scored there prediction by
prediction (``codas.aou.confirm``), not only as replicated or not. ``quality`` measures the whole design:
how often the template had to be corrected, what the hypotheses rest on, how an independent review rated
them and, after the holdout, which predictions held.

Nothing here names a dataset, an outcome or a feature: the columns, the outcome type and the citable papers
come from the caller.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

SHAPES = {
    "monotone": "the outcome changes steadily with the feature (a dose-response)",
    "threshold": "little change until the feature passes a level, then a clear change",
    "u_shaped": "both low and high values go with the worse outcome",
}
# Predictions the sealed-holdout confirmation can score, and what each says.
PREDICTIONS = {
    "negative_controls_null": "no association with the negative-control outcomes (rules out general confounding "
                              "and differences in health-care contact)",
    "survives_lag": "the effect holds among people still free of the outcome after the sensitivity lag (rules out "
                    "the outcome changing the feature before diagnosis)",
    "adherence_robust": "the effect holds after adjusting for how long the device was worn",
    "attenuation_below_half_after_incumbent": "adding the incumbent marker removes less than half of the effect",
    "within_person_same_sign": "within a person, periods with more of the feature go with the outcome in the "
                               "predicted direction (repeated lab values only)",
}
# The usual alternative explanations, each with the prediction that tests it.
ALTERNATIVES = {
    "wear": "adherence_robust",
    "reverse_causation": "survives_lag",
    "confounding": "negative_controls_null",
    "ascertainment": "negative_controls_null",
    "incumbent": "attenuation_below_half_after_incumbent",
    "between_person_confounding": "within_person_same_sign",
}
# The effect measure each outcome type is tested with, and the range a prediction may name.
MEASURES = {"hr_per_sd": (0.2, 5.0), "rho": (-0.8, 0.8)}
MIN_MECHANISM_STEPS = 2
# Objective reasons, read from the outcome-blind data summary, not to spend the holdout on a feature.
RELIABILITY_MIN = 0.5   # split-half reliability below this: the window's two halves barely agree
DUPLICATE_RHO = 0.9     # |rho| with the incumbent at or above this: the incumbent measured again
# The parts a refusal can concern, in template order.
PARTS = ("feature", "direction", "construct", "incumbent", "shape", "magnitude", "mechanism", "confounders",
         "alternatives", "predictions", "citation", "knownness", "data")
# An independent reviewer's rubric (1-5 each; for artefact risk 5 means little risk).
RUBRIC = ("specificity", "falsifiability", "plausibility", "novelty", "testability", "artefact_risk")


class Problem(str):
    """A refusal reason as the designer reads it, with the template part it concerns."""

    part: str

    def __new__(cls, part: str, text: str) -> "Problem":
        obj = super().__new__(cls, text)
        obj.part = part
        return obj


def _number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",")]
    return [str(v).strip() for v in value or [] if str(v).strip()]


def _steps(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("->", "\n").replace("→", "\n").splitlines()]
        return [p for p in parts if p]
    out = []
    for step in value or []:
        text = step.get("step") if isinstance(step, dict) else step
        cite = step.get("citation") if isinstance(step, dict) else ""
        text = f"{str(text or '').strip()} {str(cite or '').strip()}".strip()
        if text:
            out.append(text)
    return out


def validate(item: dict[str, Any], *, feature_ok: str | None, measure: str, columns: set[str],
             blocked: set[str], confounders: set[str], citable: set[str], cite_keys: Any,
             testable: set[str]) -> tuple[dict[str, Any] | None, list[Problem]]:
    """The hypothesis in normal form, or None and every problem found.

    ``feature_ok`` is the caller's verdict on the feature (None when it is a usable feature, else why not);
    ``measure`` the effect measure this outcome is tested with (``hr_per_sd`` or ``rho``); ``columns`` the
    table's columns; ``blocked`` columns no hypothesis may use (the target, excluded columns); ``confounders``
    the declared confounders; ``citable`` the citation keys retrieved in this run; ``cite_keys`` a function
    returning the keys in a text; ``testable`` the predictions this design can score."""
    problems: list[Problem] = []

    def problem(part: str, text: str) -> None:
        problems.append(Problem(part, text))

    if feature_ok:
        problem("feature", feature_ok)
    feature = str(item.get("feature") or "").strip()
    raw = str(item.get("direction", "")).strip().lower()
    direction = 1 if raw in {"1", "+1", "positive", "+"} else -1 if raw in {"-1", "negative", "-"} else 0
    if not direction:
        problem("direction", "direction must be positive or negative")
    construct = str(item.get("construct") or "").strip()
    if not construct:
        problem("construct", "name the construct the feature measures")

    incumbent = str(item.get("incumbent") or "").strip()
    if incumbent.lower() in {"", "none"}:
        incumbent = ""
    elif incumbent == feature:
        problem("incumbent", "the incumbent must be another column, the established marker the feature has to beat")
    elif incumbent not in columns or incumbent in blocked:
        problem("incumbent", f"incumbent {incumbent!r} is not a usable column (use 'none' when there is no "
                             "established marker to beat)")

    shape = str(item.get("shape") or "").strip().lower().replace("-", "_").replace(" ", "_")
    shape = {"linear": "monotone", "dose_response": "monotone", "j_shaped": "u_shaped"}.get(shape, shape)
    if shape not in SHAPES:
        problem("shape", f"shape must be one of {', '.join(SHAPES)}")

    mag = item.get("magnitude") if isinstance(item.get("magnitude"), dict) else {}
    low, high = _number(mag.get("low")), _number(mag.get("high"))
    source = str(mag.get("source") or "").strip()
    if str(mag.get("measure") or measure).strip() != measure:
        problem("magnitude", f"this outcome is tested as {measure}; give the magnitude in that measure")
    if low is None or high is None or low > high:
        problem("magnitude", f"magnitude needs numbers low <= high ({measure})")
    else:
        lo_ok, hi_ok = MEASURES[measure]
        null = 1.0 if measure == "hr_per_sd" else 0.0
        if low < lo_ok or high > hi_ok:
            problem("magnitude", f"a {measure} of {low}-{high} is outside the plausible {lo_ok}-{hi_ok}")
        elif direction and not ((direction > 0 and low > null) or (direction < 0 and high < null)):
            problem("magnitude", f"the magnitude range must lie on the predicted side of {null:g}")
    source_keys = list(cite_keys(source))
    if source.lower() != "analogy" and not (source_keys and all(k in citable for k in source_keys)):
        problem("magnitude", "the magnitude's source must be a retrieved paper's key or 'analogy'")

    steps = _steps(item.get("mechanism"))
    if len(steps) < MIN_MECHANISM_STEPS:
        problem("mechanism", f"the mechanism needs at least {MIN_MECHANISM_STEPS} steps "
                             "(feature -> intermediate -> outcome)")

    named_confounders = _names(item.get("confounders"))
    mediators = _names(item.get("mediators"))
    unknown = [c for c in named_confounders + mediators if c not in columns]
    if unknown:
        problem("confounders", f"not columns of the table: {', '.join(unknown)}")
    clash = [m for m in mediators if m in confounders or m in named_confounders]
    if clash:
        problem("confounders", f"{', '.join(clash)} cannot be both a mediator (left unadjusted) and a confounder")

    alternatives = []
    given = item.get("alternatives")
    for alt in (_names(given) if isinstance(given, str) else given or []):
        code = str(alt.get("explanation") if isinstance(alt, dict) else alt).strip().lower().replace(" ", "_")
        if code not in ALTERNATIVES:
            problem("alternatives", f"alternative explanation {code!r} must be one of {', '.join(ALTERNATIVES)}")
            continue
        alternatives.append({"explanation": code, "test": ALTERNATIVES[code]})
    predictions = list(dict.fromkeys(_names(item.get("predictions")) + [a["test"] for a in alternatives]))
    bad = [p for p in predictions if p not in PREDICTIONS]
    if bad:
        problem("predictions", f"unknown predictions {', '.join(bad)}; use {', '.join(PREDICTIONS)}")
    untestable = [p for p in predictions if p in PREDICTIONS and p not in testable]
    if untestable:
        problem("predictions", f"this design cannot score {', '.join(untestable)}")
    if "attenuation_below_half_after_incumbent" in predictions and not incumbent:
        problem("predictions", "a prediction about the incumbent needs an incumbent")
    if not [p for p in predictions if p in PREDICTIONS]:
        problem("predictions", "state at least one falsification prediction")

    if problems:
        return None, problems
    return {"feature": feature, "direction": direction, "construct": construct, "incumbent": incumbent,
            "shape": shape, "magnitude": {"measure": measure, "low": low, "high": high, "source": source},
            "mechanism_steps": steps, "confounders": named_confounders, "mediators": mediators,
            "alternatives": alternatives, "predictions": predictions,
            "relevance": str(item.get("relevance") or "")[:300]}, []


def objective_problems(entry: dict[str, Any], *, profile: dict[str, Any], incumbent_rho: float | None,
                       testable: set[str], wear_rho: float) -> list[Problem]:
    """What the outcome-blind data summary says against testing this (validated) hypothesis: a feature that is
    nearly constant or unreliable over the window, an incumbent it merely duplicates, or wear dependence
    without the test for it. ``profile`` holds the feature's ``near_constant``, ``reliability`` and
    ``wear`` ({max_abs_rho, with}) entries; ``incumbent_rho`` its rank correlation with the incumbent
    (None when not computed); ``wear_rho`` the wear dependence that calls for the adherence test."""
    out: list[Problem] = []
    if profile.get("near_constant"):
        out.append(Problem("data", "the feature is nearly constant in these data (one value in 95% of people or "
                                   "more), so it cannot separate people"))
    reliability = _number(profile.get("reliability"))
    if reliability is not None and reliability < RELIABILITY_MIN:
        out.append(Problem("data", f"its split-half reliability is {reliability:.2f} (< {RELIABILITY_MIN}): the "
                                   "first and second halves of the window barely agree, so it describes those weeks "
                                   "more than the person"))
    if incumbent_rho is not None and abs(incumbent_rho) >= DUPLICATE_RHO:
        out.append(Problem("data", f"it correlates with its incumbent {entry.get('incumbent')} at |rho| "
                                   f"{abs(incumbent_rho):.2f} (>= {DUPLICATE_RHO}): it is the incumbent measured "
                                   "again, not a new marker"))
    wear = profile.get("wear") or {}
    tracks = _number(wear.get("max_abs_rho"))
    if (tracks is not None and tracks >= wear_rho and "adherence_robust" in testable
            and "adherence_robust" not in entry.get("predictions", [])):
        out.append(Problem("data", f"it tracks how long the device was worn (|rho| {tracks:.2f} with "
                                   f"{wear.get('with')}); name the wear alternative so that adherence_robust "
                                   "tests it"))
    return out


def template_text(measure: str) -> str:
    """The template as the agents are told it."""
    return (
        "feature; direction (positive/negative); construct; incumbent (the established marker it must beat, a "
        "column, or 'none'); shape (" + ", ".join(SHAPES) + "); magnitude {measure: " + measure + ", low, high, "
        "source: a retrieved paper's key or 'analogy'}; mechanism (at least two steps, feature -> intermediate -> "
        "outcome, citing keys); confounders and mediators (columns; a mediator is not adjusted for); alternatives "
        "(explanations from " + ", ".join(ALTERNATIVES) + ", each tested by its prediction); predictions (from "
        + ", ".join(PREDICTIONS) + "); citation (keys); knownness (your own label); relevance")


def _share(items: list[Any], keep: Any) -> float | None:
    return round(sum(1 for x in items if keep(x)) / len(items), 3) if items else None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return round(sum(values) / len(values), 2) if values else None


def review_scores(items: Any, registered: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """An independent reviewer's rubric scores in normal form, and the problems with them: one entry per
    registered feature, each rubric item an integer 1-5."""
    out, problems = [], []
    for item in items if isinstance(items, list) else []:
        item = item if isinstance(item, dict) else {}
        feature = str(item.get("feature") or "").strip()
        if feature not in registered:
            problems.append(f"{feature or '(no feature)'} is not a registered hypothesis")
            continue
        scores = {}
        for key in RUBRIC:
            value = _number((item.get("scores") or item).get(key))
            if value is None or value != int(value) or not 1 <= value <= 5:
                problems.append(f"{feature}: {key} must be a whole number from 1 to 5")
                break
            scores[key] = int(value)
        else:
            out.append({"feature": feature, "scores": scores, "comment": str(item.get("comment") or "")[:400]})
    missing = [f for f in registered if f not in {r["feature"] for r in out}]
    if missing and not problems:
        problems.append("review every registered hypothesis: " + ", ".join(missing))
    return out, problems


def quality(attempts: list[dict[str, Any]], registered: list[dict[str, Any]], *,
            review: list[dict[str, Any]] | None = None, memo: dict[str, Any] | None = None,
            confirmation: dict[str, Any] | None = None, proposed: Any = ()) -> dict[str, Any]:
    """How good the design was, measured rather than asserted. ``design``: what the template refused and how
    often, and what the registered hypotheses rest on; ``data_understanding``: how much of the analyst's memo
    the data summary confirms and how many of its flagged problems the memo noticed; ``review``: an independent
    reviewer's rubric means (secondary: models judge their own kind unreliably); ``holdout``: replication and
    the registered predictions met, once the sealed holdout was tested. ``proposed`` names the features the
    agents constructed themselves."""
    first: dict[str, bool] = {}
    for a in attempts or []:
        first.setdefault(a.get("feature", ""), bool(a.get("accepted")))
    refused = Counter(part for a in attempts or [] if not a.get("accepted") for part in set(a.get("parts") or []))
    regs = list(registered or [])
    design = {
        "submitted": len(attempts or []), "features_tried": len(first), "registered": len(regs),
        "accepted_first_time": sum(first.values()),
        "refused_parts": {p: refused[p] for p in PARTS if refused.get(p)},
        "with_incumbent": _share(regs, lambda h: h.get("incumbent")),
        "magnitude_from_a_paper": _share(regs, lambda h: str((h.get("magnitude") or {}).get("source") or "analogy")
                                         .lower() != "analogy"),
        "citing_retrieved_papers": _share(regs, lambda h: h.get("citations")),
        "novel_by_evidence": _share(regs, lambda h: h.get("knownness") in ("emerging", "unexplored")),
        "novelty_not_assessed": _share(regs, lambda h: h.get("knownness") == "not assessed"),
        "on_a_constructed_feature": _share(regs, lambda h: h.get("feature") in set(proposed or ())),
        "constructs": len({str(h.get("construct") or "").strip().lower() for h in regs}),
        "mean_mechanism_steps": _mean([len(h.get("mechanism_steps") or []) for h in regs]),
        "mean_predictions": _mean([len(h.get("predictions") or []) for h in regs]),
        "shapes": dict(Counter(h.get("shape") for h in regs if h.get("shape"))),
    }
    out: dict[str, Any] = {"design": design}
    if memo:
        checked = memo.get("checked") or {}
        claims = sum(checked.values())
        out["data_understanding"] = {
            "claims": claims, **checked,
            "confirmed_share": round((checked.get("verified", 0) + checked.get("cited", 0)) / claims, 3) if claims else None,
            "problems_flagged": memo.get("issues", 0), "problems_noticed": memo.get("noticed", 0)}
    if review:
        out["review"] = {"hypotheses": len(review),
                         **{k: _mean([r["scores"][k] for r in review if k in r.get("scores", {})]) for k in RUBRIC}}
    if confirmation:
        rows = [r for r in confirmation.get("hypotheses") or [] if r.get("family") == "a_priori"]
        tested = [r for r in rows if "same_direction" in r]
        items = [i for r in rows for i in (r.get("scorecard") or {}).get("items", [])]
        kinds: dict[str, dict[str, int]] = {}
        for i in items:
            kind = i["prediction"].split(":")[0]
            if i["result"] != "not testable":
                k = kinds.setdefault(kind, {"met": 0, "testable": 0})
                k["testable"] += 1
                k["met"] += i["result"] == "met"
        out["holdout"] = {"a_priori_tested": len(tested),
                          "replicated": sum(1 for r in tested if r.get("replicated")),
                          "reproduced": sum(1 for r in tested if r.get("reproduced")),
                          "predictions_met": sum(k["met"] for k in kinds.values()),
                          "predictions_testable": sum(k["testable"] for k in kinds.values()),
                          "by_prediction": kinds}
    return out


def quality_markdown(q: dict[str, Any]) -> str:
    """The quality measures as a short section of the report."""
    d = q.get("design") or {}
    lines = ["## Hypothesis quality", "",
             f"{d.get('registered', 0)} hypotheses registered from {d.get('submitted', 0)} submissions "
             f"({d.get('accepted_first_time', 0)} of {d.get('features_tried', 0)} features complete at the first "
             "submission)."]
    if d.get("refused_parts"):
        lines.append("Refusals by template part: " + ", ".join(f"{k} {v}" for k, v in d["refused_parts"].items()) + ".")
    pct = {k: d.get(k) for k in ("with_incumbent", "magnitude_from_a_paper", "citing_retrieved_papers",
                                 "novel_by_evidence", "novelty_not_assessed", "on_a_constructed_feature")}
    lines.append("Of the registered hypotheses: " + "; ".join(
        f"{k.replace('_', ' ')} {100 * v:.0f}%" for k, v in pct.items() if isinstance(v, (int, float))) +
        f"; {d.get('constructs', 0)} distinct constructs; on average {d.get('mean_mechanism_steps')} mechanism steps "
        f"and {d.get('mean_predictions')} falsification predictions.")
    u = q.get("data_understanding")
    if u:
        lines.append(f"Data-understanding memo: {u['claims']} claims, {u.get('verified', 0)} with numbers the data "
                     f"summary confirms, {u.get('cited', 0)} citing it without numbers, {u.get('wrong value', 0)} "
                     f"with a number it does not confirm, {u.get('unknown field', 0)} citing nothing it holds; "
                     f"{u['problems_noticed']} of {u['problems_flagged']} problems the summary flags were noticed.")
    r = q.get("review")
    if r:
        lines.append("Independent review (1-5; secondary evidence, since a model rates its own kind unreliably): "
                     + ", ".join(f"{k.replace('_', ' ')} {r[k]}" for k in RUBRIC if r.get(k) is not None) + ".")
    h = q.get("holdout")
    if h:
        lines.append(f"On the sealed holdout: {h['replicated']} of {h['a_priori_tested']} a-priori hypotheses "
                     f"replicated and {h['reproduced']} reproduced as established; {h['predictions_met']} of "
                     f"{h['predictions_testable']} testable predictions were met ("
                     + ", ".join(f"{k} {v['met']}/{v['testable']}" for k, v in h["by_prediction"].items()) + ").")
    return "\n".join(lines) + "\n"
