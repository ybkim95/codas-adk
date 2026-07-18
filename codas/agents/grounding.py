"""Single source of truth for report grounding: do a report's statistics trace to numbers the
deterministic engine actually produced?

Used by both the runtime guardrail (``callbacks.report_grounding_audit``) and the live audit
(``scripts/agent_robustness``) so the two can never drift apart. A statistic-claim that matches no
engine value (within rounding) and is not a benign structural number is a likely fabrication.
"""

from __future__ import annotations

import re
from typing import Any

# A number immediately preceded by a statistic keyword is a "claim" we must be able to trace.
STAT_CLAIM = re.compile(
    r"(auc|r2|r²|r-squared|rho|ρ|spearman|p-value|p value|\bp\b|correlation|coefficient|variance|q-value|q=)"
    r"[^\d\n]{0,18}([-+]?\d*\.?\d+)", re.I)

# Benign non-engine numbers a report legitimately contains: years, and small structural integers
# (phase counts, "6 phases", percentages like 90/95/100).
_BENIGN_INTS = frozenset({0, 1, 2, 3, 4, 5, 6, 10, 90, 95, 100})
_MATCH_TOL = 0.011  # two-decimal rounding tolerance for matching a claim to an engine value


def engine_numbers(fact_sheet: dict | None, candidates: list | None, rounds: list | None = None) -> set[float]:
    """Every number the engine produced (+ legitimate derivations: rho^2 variance, percent forms),
    each rounded to 2/3/4 dp so a report citing any reasonable precision matches."""
    vals: set[float] = set()

    def add(value: Any) -> None:
        if isinstance(value, (int, float)) and value == value:  # finite, not NaN
            for places in (2, 3, 4):
                vals.add(round(float(value), places))

    fs = fact_sheet or {}
    for key in ("ml_metric_value", "rows", "columns", "candidate_features_screened",
                "internal_battery_passing_variants", "ml_metric_vs_null_p", "ml_pr_auc",
                "ml_positive_rate", "positive_rate"):
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
