"""Numeric verification & consistency enforcement.

After a report is drafted, this pass cross-checks the *count* claims in the prose against the
deterministic ground truth in the Fact Sheet and:

  * CORRECTS a count that is a near-miss of a known value — a hallucinated sample size, feature count,
    prioritized-candidate count, or validation-check count — snapping it to the exact figure, while
    leaving exact and clearly-unrelated numbers untouched (a large discrepancy is reported by the
    grounding audit as a possible fabrication rather than silently rewritten);
  * returns every correction so the report callback can log it to a per-run audit file.

This upgrades the report guardrail from "detect and warn" to "detect and fix", bounding
hallucinated counts to values the engine actually produced. Effect sizes and p-values are handled by
the grounding audit (``codas.agents.grounding``); this module deliberately does not rewrite them,
since a legitimately rounded ρ is not a hallucination.
"""

from __future__ import annotations

import re
from typing import Any

# Each anchor pairs a Fact Sheet key (the ground truth) with a pattern that captures the number written
# next to a matching noun. The number may precede the noun ("7,497 participants") or follow a label
# ("N = 7,497"). Only counts with an unambiguous single ground-truth value are corrected; the number of
# *applicable* validation checks varies per candidate, so it is deliberately NOT auto-corrected here
# (a mis-stated check count is surfaced by the grounding audit instead).
_COUNT_ANCHORS: list[tuple[str, str]] = [
    ("rows", r"(\d[\d,]{3,})\s*(?:participant-observations|participants|observations|subjects|samples)\b"),
    ("rows", r"\bN\s*=\s*(\d[\d,]{3,})"),
    ("candidate_features_screened", r"(\d[\d,]{1,})\s*(?:candidate\s+)?features?\s+(?:were\s+)?screened"),
    ("internal_battery_passing_variants", r"(\d+)\s+(?:validated|battery-passing)\s+(?:candidates?|biomarkers?)"),
]

# A count within this relative tolerance of the ground truth (but not exact) is treated as a
# transcription slip and corrected; a larger gap is left for the grounding audit to flag.
_COUNT_REL_TOL = 0.05

# Never rewrite a number written in a citation/comparison or explicit-subset context: "a prior study of
# N=7,400" or "the top 2 of the validated biomarkers" are correct as written and must not be snapped to
# this run's figures. When such a cue precedes the number, the value is left as-is (the grounding audit
# still surfaces it if it is genuinely wrong).
_SKIP_CUES = re.compile(
    r"prior|previous|earlier|another|\bother\b|\bstud(?:y|ies)\b|et al|reference|compared|versus|\bvs\b|"
    r"whereas|unlike|cohort of|largest|strongest|\btop\b|of the|subset|highlight|focus on|\bbest\b|"
    r"leading|notable|only",
    re.I,
)


def _as_int(text: str) -> int | None:
    try:
        return int(text.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _near(claim: int, truth: int) -> bool:
    if truth <= 0:
        return False
    return 0 < abs(claim - truth) <= max(1, round(truth * _COUNT_REL_TOL))


def verify_and_correct(report: str, fact_sheet: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(corrected_report, corrections)``. Each correction records the noun/key, the written
    value, and the ground-truth value it was snapped to."""
    fs = fact_sheet or {}
    corrections: list[dict[str, Any]] = []

    def _apply(text: str, truth: int | None, pattern: str, key: str) -> str:
        if truth is None:
            return text

        def _sub(match: re.Match) -> str:
            token = next((g for g in match.groups() if g), None)
            claim = _as_int(token) if token else None
            if claim is None or not _near(claim, truth):
                return match.group(0)
            context = match.string[max(0, match.start() - 50):match.start()]
            if _SKIP_CUES.search(context):
                return match.group(0)  # citation/subset context: correct as written, leave untouched
            corrections.append({"key": key, "from": claim, "to": int(truth)})
            return match.group(0).replace(token, f"{truth:,}" if truth >= 1000 else str(truth))

        return re.sub(pattern, _sub, text)

    corrected = report
    for key, pattern in _COUNT_ANCHORS:
        truth = fs.get(key)
        corrected = _apply(corrected, int(truth) if isinstance(truth, (int, float)) else None, pattern, key)
    return corrected, corrections
