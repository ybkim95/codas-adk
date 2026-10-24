"""Single source of truth for report grounding: do a report's statistics trace to numbers the
deterministic engine actually produced?

Used by both the runtime guardrail (``callbacks.report_grounding_audit``) and the live audit
(``scripts/agent_robustness``) so the two can never drift apart. A statistic-claim that matches no
engine value (within rounding) and is not a benign structural number is a likely fabrication.
"""

from __future__ import annotations

import re
from typing import Any

# A number immediately preceded by a statistic keyword is a "claim" we must be able to trace. The list
# spans what the engine and the recorded analyses produce (auc, rho, p, q, r2, hazard ratios) AND metrics
# no analysis produced (odds ratios, sensitivity, specificity, F1, effect size): such a number traces to
# nothing and is flagged. Short ambiguous abbreviations (HR/OR/RR) are matched only when written as a
# statistic ("HR=4.2", "OR: 3.1") to avoid catching the English words.
STAT_CLAIM = re.compile(
    r"(auroc|auprc|auc|r2|r²|r-squared|rho|ρ|spearman|pearson|kendall|"
    r"p-?value|p value|\bp\b|q-?value|q=|correlation|coefficient|variance|"
    r"hazard ratio|odds ratio|risk ratio|relative risk|"
    r"sensitivity|specificity|\bppv\b|\bnpv\b|accuracy|precision|recall|\bf1\b|"
    r"cohen'?s ?d|effect size|"
    r"\b(?:hr|or|rr)\s*[=:])"
    r"[^\d\n]{0,18}([-+]?\d*\.?\d+)", re.I)

# Benign non-engine numbers a report legitimately contains: years (handled separately) and small
# structural integers (phase counts like "6 phases", "top 10"). 90/95/100 are deliberately NOT here:
# a keyword-preceded 90/95/100 (e.g. "sensitivity 95%") is far more likely a fabricated metric than a
# structural number, and a legitimate "95% CI" is not preceded by a captured keyword.
_BENIGN_INTS = frozenset({0, 1, 2, 3, 4, 5, 6, 10})
_MATCH_TOL = 0.011  # two-decimal rounding tolerance for matching a claim to an engine value


def _numbers(obj: Any):
    """Every number inside a nested result (dicts and lists), booleans excluded."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _numbers(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _numbers(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


def engine_numbers(fact_sheet: dict | None, candidates: list | None, rounds: list | None = None,
                   analyses: list | None = None) -> set[float]:
    """Every number the engine produced (+ legitimate derivations: rho^2 variance, percent forms), including
    every number in the recorded analyses (``state["analyses"]``: the method pools and the agents' analysis
    tools), each rounded to 2/3/4 dp so a report citing any reasonable precision matches."""
    vals: set[float] = set()

    def add(value: Any) -> None:
        if isinstance(value, (int, float)) and value == value:  # finite, not NaN
            for places in (2, 3, 4):
                vals.add(round(float(value), places))

    fs = fact_sheet or {}
    for key in ("ml_metric_value", "rows", "columns", "candidate_features_screened",
                "internal_battery_passing_variants", "ml_metric_vs_null_p", "ml_pr_auc",
                "ml_positive_rate", "positive_rate", "ml_metric_null_mean", "ml_metric_null_p95",
                "ml_null_draws"):
        add(fs.get(key))
    if isinstance(fs.get("ml_metric_value"), (int, float)):
        add(fs["ml_metric_value"] * 100)
    for rnd in rounds or []:
        add(rnd.get("ml_metric_value"))
        add(rnd.get("validated_count"))
    for cand in candidates or []:
        for key in ("rho", "q_value", "p_value", "n"):
            add(cand.get(key))
        if isinstance(cand.get("rho"), (int, float)):
            add(cand["rho"] ** 2)
            add(cand["rho"] ** 2 * 100)
            add(abs(cand["rho"]))
    for record in analyses or []:
        for value in _numbers((record or {}).get("result") if isinstance(record, dict) else None):
            add(value)
            add(abs(value))
            add(value * 100)
    return vals


def ungrounded_claims(report: str, engine_vals: set[float]) -> tuple[list[tuple[str, float]], int]:
    """Return (ungrounded statistic-claims, total statistic-claims) found in the report text.

    A claim is grounded if its number is within rounding of an engine value, a plausible year, or a
    benign structural integer. Anything else is surfaced as a possible fabrication.
    """
    claims = [(m.group(1), float(m.group(2))) for m in STAT_CLAIM.finditer(report)]
    ungrounded = [
        (keyword, value) for keyword, value in claims
        if not any(abs(value - e) <= _MATCH_TOL for e in engine_vals)
        and not (1900 < value < 2100)
        and value not in _BENIGN_INTS
    ]
    return ungrounded, len(claims)


# Claims that the held-out model beats chance. A number can be grounded while the sentence around it
# is wrong ("exceeds the permutation null (p = 0.2)"), which the figure check above cannot see.
_CHANCE_CLAIM = re.compile(
    r"(above[- ]chance|better than chance|(exceed|beat|outperform|surpass)\w*\s+(the\s+|its\s+)?"
    r"(permutation[- ])?null|statistically (stable|significant|reliable) (predictive )?signal)",
    re.I,
)
_NEGATION = re.compile(r"\b(not|no|never|fails?|failed|failing|below|without|cannot|neither|nor)\b|n't", re.I)


def chance_claim_conflict(report: str, fact_sheet: dict | None) -> str | None:
    """A correction note when the report says the model beats chance but the engine says it does not."""
    fs = fact_sheet or {}
    if fs.get("ml_above_chance") is not False:
        return None
    for match in _CHANCE_CLAIM.finditer(report or ""):
        sentence_start = max(report.rfind(".", 0, match.start()), report.rfind("\n", 0, match.start())) + 1
        if not _NEGATION.search(report[sentence_start:match.start()]):
            p = fs.get("ml_metric_vs_null_p")
            p_text = f" (permutation p = {p:.3g})" if isinstance(p, (int, float)) else ""
            return ("**Engine check.** The held-out model did not beat its permutation null"
                    f"{p_text}; statements above that it performs above chance are not supported by "
                    "the deterministic engine.")
    return None
