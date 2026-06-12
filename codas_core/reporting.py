"""Deterministic report assembly grounded in the CoDaS Fact Sheet."""

from __future__ import annotations

from collections import Counter

from .models import Candidate, DatasetProfile


def _status_label(verdict: str) -> str:
    if verdict == "validated":
        return "battery-passing"
    if verdict == "conditional":
        return "conditional"
    if verdict == "rejected":
        return "failed internal gate"
    return verdict or "not tested"


def build_fact_sheet(
    profile: DatasetProfile,
    candidates: list[Candidate],
    target_column: str,
    ml_metrics: dict[str, float | int | str | None],
    discovery_rounds: int,
    feature_count: int,
    battery_evaluated_count: int | None = None,
    battery_passing_count: int | None = None,
) -> dict[str, object]:
    verdicts = Counter(candidate.verdict for candidate in candidates)
    reported_battery_passing = verdicts.get("validated", 0)
    return {
        "target_column": target_column,
        "rows": profile.rows,
        "columns": profile.columns,
        "numeric_columns": len(profile.numeric_columns),
        "candidate_features_screened": feature_count,
        "reported_candidate_variants": len(candidates),
        "candidate_features_reported": len(candidates),
        "internal_battery_evaluated_variants": battery_evaluated_count or len(candidates),
        "internal_battery_passing_variants_evaluated": battery_passing_count if battery_passing_count is not None else reported_battery_passing,
        "reported_battery_passing_variants": reported_battery_passing,
        "internal_battery_passing_variants": reported_battery_passing,
        "conditional_candidates": verdicts.get("conditional", 0),
        "rejected_candidates": verdicts.get("rejected", 0),
        "discovery_rounds": discovery_rounds,
        "ml_metric_name": ml_metrics.get("metric_name"),
        "ml_metric_value": ml_metrics.get("metric_value"),
        "ml_feature_count": ml_metrics.get("feature_count"),
        "ml_cv_strategy": ml_metrics.get("cv_strategy"),
        "ml_metric_null_mean": ml_metrics.get("null_metric_mean"),
        "ml_metric_null_p95": ml_metrics.get("null_metric_p95"),
        "ml_metric_vs_null_p": ml_metrics.get("metric_vs_null_p"),
        "ml_above_chance": ml_metrics.get("above_chance"),
        "ml_n_samples": ml_metrics.get("n_samples"),
        "ml_low_confidence": ml_metrics.get("low_confidence"),
        "ml_pr_auc": ml_metrics.get("pr_auc"),
        "ml_positive_rate": ml_metrics.get("positive_rate"),
    }


def candidate_table(candidates: list[Candidate], limit: int | None = None) -> str:
    lines = [
        "| Feature variant | Internal status | Spearman rho | q-value | Pass rate | Evidence |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for candidate in (candidates if limit is None else candidates[:limit]):
        lines.append(
            "| {feature} | {verdict} | {rho:.3f} | {q:.3g} | {rate:.0%} | {evidence} |".format(
                feature=candidate.feature,
                verdict=_status_label(candidate.verdict),
                rho=candidate.rho,
                q=candidate.q_value,
                rate=candidate.pass_rate,
                evidence=candidate.evidence.replace("validation checks", "audit checks"),
            )
        )
    return "\n".join(lines)


def build_markdown_report(
    fact_sheet: dict[str, object],
    candidates: list[Candidate],
    warnings: list[str],
) -> str:
    lines = [
        "# CoDaS Report",
        "",
        "## Fact Sheet",
        "",
    ]
    for key, value in fact_sheet.items():
        # Round floats so the report doesn't show noise like 0.802005643204628 (a skeptical
        # reviewer reads 15-decimal values as sloppy); 4 significant figures is plenty here.
        if isinstance(value, float):
            value = f"{value:.4g}"
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Predictor Candidates",
        "",
        candidate_table(candidates),
        "",
        "## Interpretation Boundary",
        "",
        "These findings are internally audited, hypothesis-generating associations. They do not establish external validation, external replication, diagnostic utility, or causality.",
    ])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    return "\n".join(lines)
