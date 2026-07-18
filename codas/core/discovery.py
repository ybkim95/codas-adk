"""End-to-end deterministic association discovery pipeline."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
try:  # StratifiedGroupKFold needs sklearn >= 0.24; fall back to GroupKFold if absent
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    InsufficientDataError,
    build_analysis_frame,
    profile_dataframe,
    read_csv_dataset,
)

from .models import Candidate, DiscoveryReport
from .quality_gates import evaluate_quality_gates
from .reporting import build_fact_sheet, build_markdown_report
from .statistics import auc_against_target, autocorr_effective_n, benjamini_hochberg, cluster_effective_n, correlation_pvalue, intraclass_correlation, lag1_autocorr, safe_spearman, signed_direction, within_subject_two_stage
from .validation import ValidationConfig, validate_candidate

# Library logger: silent unless the host app configures handlers. Logs phase progress at INFO so a
# caller can trace and debug a run (engine-only; the agent layer logs separately in callbacks.py).
LOGGER = logging.getLogger("codas.engine")
LOGGER.addHandler(logging.NullHandler())


@dataclass
class DiscoveryRequest:
    target_column: str
    participant_id_column: str | None = None
    time_column: str | None = None
    excluded_columns: list[str] = field(default_factory=list)
    confounder_columns: list[str] = field(default_factory=list)
    top_k: int = 25
    fdr_alpha: float = 0.05  # Benjamini-Hochberg FDR control level (alpha = 0.05)
    max_ratio_features: int = 24
    validation_resamples: int = 1000
    random_state: int = 17
    # Columns the USER/PROFILE has declared to be CONCURRENT diagnostic tests of the SAME
    # condition as a (binary/categorical) diagnostic target — e.g. a second assay/imaging read
    # scored on the same visit. They are construct-circular / co-diagnostic, NOT independent
    # upstream risk factors; if they survive the battery they are ADVISED against (not
    # auto-excluded, since the user may intend them). Matched case-insensitively by exact name.
    concurrent_test_columns: list[str] = field(default_factory=list)
    # Columns the USER/PROFILE has declared to be measured AT OR AFTER the outcome window (e.g. a
    # forward-looking "next-week" aggregate, or any post-outcome reading). Including such a feature is
    # look-ahead / temporal leakage: it inflates apparent predictive power for the CURRENT outcome.
    # CoDaS cannot infer measurement timing from values, so these are caller-declared; when declared
    # they are EXCLUDED before screening (a hard guard) and the exclusion is disclosed. Matched by name.
    post_outcome_columns: list[str] = field(default_factory=list)
    # Physiologically-motivated transformations PROPOSED by the generative interpreters: they propose
    # transformations which the deterministic runners immediately evaluate. Each spec is
    # {"op": ratio|product|difference|sum, "a": col, "b": col, "name": str?}.
    # A proposed feature is materialised as an ordinary column BEFORE screening, so it is evaluated by
    # the SAME FDR screen and validation battery as every other candidate — the loop can surface a
    # composite (e.g. steps/resting-heart-rate) that the bounded default enumeration would miss, but it
    # can never bypass the statistical gates.
    proposed_features: list[dict] = field(default_factory=list)


def _rank_candidates(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    autocorr_correct: bool = False,
    cluster_groups: "pd.Series | None" = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    p_values: list[float] = []
    # Temporal pseudo-replication guard: when the rows are an autocorrelated time series with no
    # participant grouping, adjacent samples are NOT independent, so the raw row count vastly
    # overstates significance (a tiny correlation between two slowly-drifting series reads p<1e-3
    # at n=800 when the AR(1)-effective n is ~50). When asked, recompute each screening p-value at
    # the Pyper-Peterman effective n using the lag-1 autocorrelation of BOTH series; white-noise
    # features (r1~0) are unaffected, so only genuinely temporally-confounded pairs are deflated.
    # Caller guarantees `frame` is in time order when autocorr_correct is True.
    target_r1 = lag1_autocorr(frame[target_column].to_numpy()) if autocorr_correct else 0.0
    # Clustered pseudo-replication guard (the between-subject analog): when rows are repeated
    # measures retained at row level (participant grouping present, target varies within subject),
    # recompute each screening p-value at a design-effect effective n using the ICC of BOTH the
    # feature and the target. A per-subject TRAIT (high ICC) is deflated toward the cluster count;
    # a feature varying freely within subject (ICC~0) is barely deflated. Prevents 5875 rows from
    # 42 subjects being scored as 5875 independent observations.
    cluster_target_icc = 0.0; cluster_mbar = 1.0; cluster_k = 0
    if cluster_groups is not None:
        cluster_k = int(pd.Series(cluster_groups).nunique())
        if cluster_k >= 2:
            cluster_mbar = float(len(frame)) / cluster_k
            cluster_target_icc = intraclass_correlation(frame[target_column].to_numpy(), cluster_groups)
    use_cluster = cluster_groups is not None and cluster_k >= 2 and cluster_mbar > 1.5
    for feature in feature_columns:
        rho, p_value, n = safe_spearman(frame[feature], frame[target_column])
        if not np.isfinite(rho) or not np.isfinite(p_value) or n < 20:
            continue
        if autocorr_correct:
            feat_r1 = lag1_autocorr(frame[feature].to_numpy())
            n_eff = autocorr_effective_n(n, feat_r1, target_r1)
            if n_eff < n:
                p_value = correlation_pvalue(rho, n_eff)
        if use_cluster:
            feat_icc = intraclass_correlation(frame[feature].to_numpy(), cluster_groups)
            n_eff = cluster_effective_n(n, cluster_mbar, feat_icc, cluster_target_icc, cluster_k)
            if n_eff < n:
                p_value = correlation_pvalue(rho, n_eff)
        p_values.append(p_value)
        candidates.append(Candidate(
            feature=feature,
            rho=rho,
            p_value=p_value,
            q_value=1.0,
            n=n,
            direction=signed_direction(rho),
            score=abs(rho),
        ))
    q_values = benjamini_hochberg(p_values)
    for candidate, q_value in zip(candidates, q_values):
        candidate.q_value = q_value
        candidate.score = abs(candidate.rho) * (1.0 - min(q_value, 1.0))
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _is_binary_target(y: pd.Series) -> bool:
    return y.dropna().nunique() == 2




def _ml_benchmark(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    random_state: int,
    participant_id_column: str | None = None,
) -> dict[str, Any]:
    usable = [column for column in feature_columns if column in frame.columns]
    cols = usable + [target_column]
    if participant_id_column and participant_id_column in frame.columns and participant_id_column not in cols:
        cols = cols + [participant_id_column]
    subset = frame[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_column])
    if len(usable) == 0 or len(subset) < 30:
        return {"metric_name": None, "metric_value": None, "feature_count": len(usable), "n_samples": len(subset)}

    x = subset[usable]
    y = subset[target_column]
    binary = _is_binary_target(y)
    if binary:
        labels = pd.Categorical(y).codes
        if len(np.unique(labels)) < 2:
            return {"metric_name": "auc", "metric_value": None, "feature_count": len(usable)}
        yv = labels
        scoring = "roc_auc"
        metric_label = "auc"
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
    else:
        yv = y.to_numpy(dtype=float)
        scoring = "r2"
        metric_label = "r2"
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

    # Group-aware CV when participants repeat: no participant may appear in both the train
    # and test folds, otherwise a flexible model can memorize participants and inflate the
    # metric (the canonical train/test leakage). (After participant aggregation each
    # participant is a single row, so groups only engages on row-level repeated-measures data.)
    groups = None
    if participant_id_column and participant_id_column in subset.columns:
        g = subset[participant_id_column].astype(str)
        if 1 < g.nunique() < len(subset):
            groups = g.to_numpy()

    def _make_cv(seed: int):
        if groups is not None:
            n_splits = min(5, int(pd.unique(groups).size))
            if n_splits < 2:
                return None, "group_insufficient"
            if binary and StratifiedGroupKFold is not None:
                return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), "participant_grouped"
            return GroupKFold(n_splits=n_splits), "participant_grouped"  # GroupKFold is deterministic
        if binary:
            n_splits = int(min(5, np.bincount(labels).min()))
            if n_splits < 2:
                return None, None
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), "stratified"
        return KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=seed), "kfold"

    cv0, cv_strategy = _make_cv(random_state)
    if cv0 is None:
        return {"metric_name": metric_label, "metric_value": None, "feature_count": len(usable), "cv_strategy": cv_strategy}

    def _cv_mean(seed: int, labels_vec) -> float:
        cvx, _ = _make_cv(seed)
        if cvx is None:
            return float("nan")
        try:
            return float(np.nanmean(cross_val_score(model, x, labels_vec, groups=groups, scoring=scoring, cv=cvx)))
        except Exception:
            return float("nan")

    # A single CV estimate is high-variance at small n (one fold-shuffle can read AUC 0.86 on
    # pure noise). Stabilize the reported metric as the MEAN over repeated CV, and build a
    # permutation-null distribution the SAME way (label-shuffled). A chance-level model then
    # reads as chance instead of being surfaced as a finding.
    reps = 20 if len(subset) <= 500 else (10 if len(subset) <= 3000 else 4)
    rng = np.random.default_rng(random_state)
    real_draws = [v for v in (_cv_mean(random_state + i, yv) for i in range(reps)) if np.isfinite(v)]
    real = float(np.mean(real_draws)) if real_draws else None
    null_draws: list[float] = []
    for _ in range(reps):
        yp = rng.permutation(yv)
        v = _cv_mean(int(rng.integers(0, 1_000_000)), yp)
        if np.isfinite(v):
            null_draws.append(v)
    out: dict[str, Any] = {"metric_name": metric_label, "metric_value": real, "feature_count": len(usable), "cv_strategy": cv_strategy}
    # Small-sample reliability flag: a permutation null cannot catch a spurious in-sample fit
    # (at n=40 a chance feature-label correlation is genuinely present in THIS sample and the
    # model fits it). When n is small or n is not >> feature count, the metric is unstable and
    # optimistic, and must be reported as low-confidence, not as a finding.
    n_samples = int(len(subset))
    out["n_samples"] = n_samples
    out["low_confidence"] = bool(n_samples < 100 or n_samples < 10 * max(1, len(usable)))
    if real is not None and null_draws:
        null_arr = np.asarray(null_draws, dtype=float)
        out["null_metric_mean"] = float(np.nanmean(null_arr))
        out["null_metric_p95"] = float(np.nanpercentile(null_arr, 95))
        out["metric_vs_null_p"] = float((1 + int(np.nansum(null_arr >= real))) / (len(null_arr) + 1))
        out["above_chance"] = bool(real > out["null_metric_p95"])
    # For binary outcomes, also report prevalence + PR-AUC (average precision): ROC-AUC is
    # optimistic at low base rates, so a rare-event detector needs the precision-recall view.
    if binary and real is not None:
        out["positive_rate"] = float(np.mean(yv))
        try:
            cv_pr, _ = _make_cv(random_state)
            if cv_pr is not None:
                out["pr_auc"] = float(np.nanmean(cross_val_score(model, x, yv, groups=groups, scoring="average_precision", cv=cv_pr)))
        except Exception:
            pass
    # In-sample (train) metric for the overfitting quality gate: a large train-vs-CV gap
    # signals memorization. Fit once on all rows and score in-sample; None if the fit fails.
    out["train_metric_value"] = None
    if real is not None:
        try:
            fitted = model.fit(x, yv)
            if binary:
                scores = (fitted.predict_proba(x)[:, 1] if hasattr(fitted, "predict_proba")
                          else fitted.decision_function(x))
                out["train_metric_value"] = float(roc_auc_score(yv, scores))
            else:
                out["train_metric_value"] = float(fitted.score(x, yv))  # in-sample R^2
        except Exception:
            out["train_metric_value"] = None
    return out


def _subsample_for_screening(df: pd.DataFrame, target_column: str, max_rows: int, seed: int) -> pd.DataFrame:
    """Stratified (by target when low-cardinality) random subsample, so screening,
    validation and model fitting stay tractable within the interactive request timeout on
    very large datasets. Association estimates (Spearman rho, FDR, bootstrap stability) are
    statistically equivalent at this sample size."""
    if len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    if target_column in df.columns:
        y = df[target_column]
        n_unique = int(y.nunique(dropna=True))
        if 1 < n_unique <= 20:  # classification / low-cardinality target -> stratify
            frac = max_rows / len(df)
            picks: list[np.ndarray] = []
            for _, idx in df.groupby(target_column, dropna=False).groups.items():
                idx_arr = np.asarray(list(idx))
                k = max(1, int(round(len(idx_arr) * frac)))
                picks.append(rng.choice(idx_arr, size=min(k, len(idx_arr)), replace=False))
            sel = np.concatenate(picks)
            return df.loc[sel].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sel = rng.choice(len(df), size=max_rows, replace=False)
    return df.iloc[sel].reset_index(drop=True)


def _dedupe_columns(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Make column names unique so ``df[col]`` never returns a DataFrame.

    Duplicate column names make ``df[name]`` return a DataFrame instead of a Series, which crashes
    every downstream numeric op with a cryptic TypeError. The first occurrence keeps the original
    name; later duplicates get a ``.1`` / ``.2`` suffix.
    """
    if not pd.Index(df.columns).duplicated().any():
        return df
    seen: dict[Any, int] = {}
    renamed: list[Any] = []
    for column in df.columns:
        if column in seen:
            seen[column] += 1
            renamed.append(f"{column}.{seen[column]}")
        else:
            seen[column] = 0
            renamed.append(column)
    out = df.copy()
    out.columns = renamed
    warnings.append(
        "Duplicate column names were detected and made unique (suffixes .1/.2). Verify the source "
        "export — duplicate columns usually indicate a malformed file."
    )
    return out


def _normalize_target(df: pd.DataFrame, target_column: str, warnings: list[str]) -> pd.DataFrame:
    """Coerce a non-numeric target so discovery can run, or raise a clear boundary error.

    * numeric target -> unchanged.
    * mostly-numeric text (e.g. ``"1.0"``, ``"2"``) -> coerced to numbers.
    * exactly two distinct non-numeric values -> encoded to 0/1 (binary classification).
    * any other non-numeric target (3+ text categories, free text) -> InsufficientDataError, because
      correlating arbitrary category codes of a nominal label would be meaningless.
    """
    if target_column not in df.columns:
        return df  # downstream raises the explicit "not present" error
    series = df[target_column]
    if pd.api.types.is_numeric_dtype(series):
        return df
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().mean() >= 0.8:
        out = df.copy()
        out[target_column] = coerced
        return out
    uniques = list(pd.unique(series.dropna()))
    if len(uniques) == 2:
        out = df.copy()
        out[target_column] = series.map({uniques[0]: 0.0, uniques[1]: 1.0}).astype(float)
        warnings.append(
            f"Target '{target_column}' was non-numeric with two classes ({uniques[0]!r}, "
            f"{uniques[1]!r}); encoded to 0/1 for binary classification."
        )
        return out
    raise InsufficientDataError(
        f"Target '{target_column}' is non-numeric with {len(uniques)} distinct values. Provide a "
        "numeric outcome or a binary (two-class) categorical target; correlating arbitrary codes of "
        "a multi-class text label is not meaningful."
    )


_PROPOSED_FEATURE_OPS = ("ratio", "product", "difference", "sum")


def _materialize_proposed_features(df: pd.DataFrame, proposed: list[dict], warnings: list[str]) -> pd.DataFrame:
    """Realise generative-interpreter feature proposals as ordinary numeric columns.

    Each proposal names a safe binary operation over two existing numeric columns; there is no
    expression evaluation. Once materialised, the new column flows through the standard screening, FDR
    correction and validation battery like any other feature — so a proposal is evaluated, never
    trusted. Malformed or duplicate proposals are skipped with a disclosed warning.
    """
    if not proposed:
        return df
    out = df
    for spec in proposed:
        op = str(spec.get("op", "")).strip().lower()
        a, b = spec.get("a"), spec.get("b")
        name = str(spec.get("name") or f"{a}_{op}_{b}").strip()
        if op not in _PROPOSED_FEATURE_OPS:
            warnings.append(f"Proposed feature '{name}' skipped: unsupported operation '{op}'.")
            continue
        if a not in out.columns or b not in out.columns:
            missing = [c for c in (a, b) if c not in out.columns]
            warnings.append(f"Proposed feature '{name}' skipped: column(s) not found: {missing}.")
            continue
        if name in out.columns:
            continue  # already present (a prior round materialised it) — evaluate once
        col_a = pd.to_numeric(out[a], errors="coerce")
        col_b = pd.to_numeric(out[b], errors="coerce")
        if op == "ratio":
            col = col_a / col_b.replace(0.0, np.nan)
        elif op == "product":
            col = col_a * col_b
        elif op == "difference":
            col = col_a - col_b
        else:  # sum
            col = col_a + col_b
        if out is df:
            out = df.copy()
        out[name] = col.replace([np.inf, -np.inf], np.nan)
        warnings.append(
            f"Evaluated caller-proposed feature '{name}' = {a} {op} {b}; it is screened and validated "
            f"with the full battery (a proposal is evaluated, not trusted)."
        )
    return out


_COLLINEAR_THRESHOLD = 0.85  # |inter-feature rho| above this => same construct, reported but not
# double-counted (paper 2.5-6 intra-cluster overlap gate). Matches the construct-overlap hard gate.


def _demote_collinear(candidates: list[Candidate], frame: pd.DataFrame) -> list[str]:
    """Demote near-duplicate passing candidates to 'collinear_redundant' (mutates verdict).

    Among the validated/conditional candidates, if two features are near-duplicates of each other
    (|inter-feature rho| > _COLLINEAR_THRESHOLD) they measure the same construct; the weaker one (by
    |rho_target|) is redundant. Reporting both would inflate the effective hit count. Returns warnings.
    """
    passing = [c for c in candidates if c.verdict in {"validated", "conditional"}]
    if len(passing) < 2:
        return []
    feats = [c.feature for c in passing if c.feature in frame.columns]
    demoted: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for i, fi in enumerate(feats):
        for j, fj in enumerate(feats):
            if j <= i or fi in demoted or fj in demoted:
                continue
            try:
                r = float(frame[[fi, fj]].dropna().corr().iloc[0, 1])
            except Exception:
                r = 0.0
            if np.isfinite(r) and abs(r) > _COLLINEAR_THRESHOLD:
                rho_i = abs(next(c.rho for c in passing if c.feature == fi))
                rho_j = abs(next(c.rho for c in passing if c.feature == fj))
                demoted.add(fj if rho_i >= rho_j else fi)
                pairs.append((fi, fj))
    if not demoted:
        return []
    for candidate in candidates:
        if candidate.feature in demoted:
            candidate.verdict = "collinear_redundant"
    return [
        f"Near-duplicate features detected among validated candidates (|inter-feature ρ| > "
        f"{_COLLINEAR_THRESHOLD}): {', '.join(str(p) for p in pairs)}. The stronger feature in each "
        f"pair is retained; the weaker is demoted to 'collinear_redundant' to avoid double-counting a "
        f"single signal."
    ]


def _input_quality_warnings(df: pd.DataFrame, target_column: str) -> list[str]:
    """Methodological caveats about the INPUT itself (read-only; returns the warnings to append).

    Heavy outcome missingness biases the analysis toward the observed subgroup (only rows with an
    observed target are used); high feature missingness has the same risk; and a class-imbalanced
    binary target makes held-out ROC-AUC optimistic.
    """
    out: list[str] = []
    if target_column not in df.columns:
        return out
    feature_cols = [c for c in df.columns
                    if c != target_column and pd.api.types.is_numeric_dtype(df[c])]
    high_miss = [c for c in feature_cols if float(df[c].isna().mean()) > 0.30]
    if high_miss:
        rates = {c: round(float(df[c].isna().mean() * 100), 0) for c in high_miss[:4]}
        out.append(
            f"High feature missingness detected: {rates}. These features are only available for a "
            "subset of rows; if data is missing-not-at-random the associations reflect the observed "
            "subgroup and may not generalise. Verify the missingness mechanism before interpreting."
        )
    target_missing = float(df[target_column].isna().mean())
    if target_missing > 0.10:
        out.append(
            f"{target_missing * 100:.0f}% of the outcome '{target_column}' is missing; only rows with "
            "an observed outcome are analyzed. If missingness is informative (missing-not-at-random), "
            "the associations are biased toward the reporting subgroup. Confirm the mechanism first."
        )
    observed = df[target_column].dropna()
    if observed.nunique() == 2:
        minority = float(observed.value_counts(normalize=True).min())
        if minority < 0.15:
            out.append(
                f"Outcome is imbalanced ({minority * 100:.1f}% minority class); held-out ROC-AUC is "
                "optimistic at low prevalence — prioritize precision-recall (PR-AUC) and calibrate to "
                f"the {minority * 100:.1f}% base rate before interpreting predictive performance."
            )
    return out


def _effect_size_warnings(candidates: list[Candidate]) -> list[str]:
    """Flag passing candidates with a small effect (|rho| < 0.20, < 4% variance explained):
    statistical significance at large n does not imply practical importance."""
    notes = [
        f"'{c.feature}' (ρ={c.rho:.2f}, ~{round(float(c.rho) ** 2 * 100, 1)}% variance explained)"
        for c in candidates
        if c.verdict in {"validated", "conditional"} and np.isfinite(c.rho) and abs(c.rho) < 0.20
    ]
    if not notes:
        return []
    return [
        f"Small effect size: {'; '.join(notes)}. Statistical significance at large n does not imply "
        "practical importance — review effect magnitudes before acting on findings with |ρ| < 0.20."
    ]


def _combined_feature_attribution(candidates: list[Candidate], analysis) -> list[str]:
    """Attribute a two-component engineered feature (a/b or a×b) to the component carrying the signal
    when the other is ~null, so synthesis does not credit the null component. Mutates candidate.evidence.
    `analysis` is the AnalysisFrame from build_analysis_frame.
    """
    notes: list[str] = []
    for candidate in candidates:
        if candidate.verdict not in {"validated", "conditional"}:
            continue
        components = analysis.feature_components.get(candidate.feature)
        if not components or len(components) != 2:
            continue
        if not (("_over_" in candidate.feature) or ("_x_" in candidate.feature)):
            continue
        rhos: dict[str, float] = {}
        for component in components:
            if component in analysis.frame.columns and analysis.target_column in analysis.frame.columns:
                r, _, _ = safe_spearman(analysis.frame[component], analysis.frame[analysis.target_column])
                rhos[component] = abs(r) if np.isfinite(r) else 0.0
        if len(rhos) != 2:
            continue
        (strong, strong_rho), (weak, weak_rho) = sorted(rhos.items(), key=lambda kv: kv[1], reverse=True)
        # Signal carried by ONE component: the strong one has a real effect, the weak one is ~null and
        # substantially weaker (so the combo is essentially a rescaling of the strong component).
        if strong_rho >= 0.25 and weak_rho < 0.15 and weak_rho < 0.5 * strong_rho:
            candidate.evidence = (str(candidate.evidence or "").rstrip() + " " + (
                f"ATTRIBUTION: this combined feature's association is carried by '{strong}' "
                f"(|ρ|={strong_rho:.2f}); the other component '{weak}' does NOT independently associate "
                f"with the target (|ρ|={weak_rho:.2f}). Attribute the finding to '{strong}' alone — do "
                f"NOT report '{weak}' as a contributing predictor.")).strip()
            notes.append(f"'{candidate.feature}' → driven by '{strong}', not '{weak}'")
    if not notes:
        return []
    return [
        "COMBINED-FEATURE ATTRIBUTION: " + "; ".join(notes[:6]) + ". These ratio/interaction features "
        "validated, but their signal comes from ONE component; the other component is null and must not "
        "be reported as an independent predictor."
    ]


def _construct_circularity_advisory(candidates: list[Candidate], analysis, request: "DiscoveryRequest") -> list[str]:
    """Advisory for a diagnostic (low-cardinality) target: a passing feature that is a caller-declared
    concurrent test, OR that separates the classes near-perfectly, is likely (near-)concurrent with or
    downstream of the outcome, so its association is circular. Separation is judged on |rho|>=0.80 AND,
    for a binary target, on AUC>=0.90 — Spearman saturates on a binary outcome (a feature separating the
    classes at AUC 0.95 reads |rho|~0.78, under the 0.80 gate), so the AUC branch catches the near-copy
    band the correlation gate misses. Detection is value-based + caller-declared (NOT name-based).
    Mutates candidate.evidence.
    """
    observed = analysis.frame[analysis.target_column].dropna() if analysis.target_column in analysis.frame else pd.Series(dtype=float)
    if not (2 <= int(observed.nunique()) <= 10):
        return []
    is_binary = int(observed.nunique()) == 2
    target_arr = pd.to_numeric(analysis.frame[analysis.target_column], errors="coerce").to_numpy()
    declared = {str(c).strip().lower() for c in (request.concurrent_test_columns or []) if str(c).strip()}
    circular: list[tuple[Candidate, str]] = []
    for candidate in candidates:
        if candidate.verdict not in {"validated", "conditional"}:
            continue
        name = str(candidate.feature or "")
        sep_auc = float("nan")
        if is_binary and candidate.feature in analysis.frame.columns:
            _a, _ = auc_against_target(target_arr, pd.to_numeric(analysis.frame[candidate.feature], errors="coerce").to_numpy())
            sep_auc = max(_a, 1.0 - _a) if np.isfinite(_a) else float("nan")
        if name.strip().lower() in declared:
            kind = "declared CONCURRENT diagnostic test of the same condition"
        elif np.isfinite(candidate.rho) and abs(candidate.rho) >= 0.80:
            kind = (f"separates the classes near-perfectly (|ρ|={abs(candidate.rho):.2f}) — far stronger "
                    "than a typical upstream factor, consistent with a concurrent test / post-outcome measurement")
        elif np.isfinite(sep_auc) and sep_auc >= 0.90:
            kind = (f"separates the two classes very strongly (AUC={sep_auc:.2f}, below the exact-copy hard "
                    "gate) — far stronger than a typical upstream factor and not caught by the |ρ| gate "
                    "because Spearman saturates on a binary target; consistent with a concurrent test / co-measurement")
        else:
            continue
        circular.append((candidate, kind))
        candidate.evidence = (str(candidate.evidence or "").rstrip() + (
            f" CONSTRUCT-CIRCULARITY ADVISORY: '{name}' {kind} for the low-cardinality target "
            f"'{request.target_column}' — it is (near-)concurrent with or downstream of the outcome, so a "
            "strong association is circular and does not establish an independent, predictive, or causal "
            "predictor. Confirm the measurement timeline; exclude outcome-derived / co-measured variables.")).strip()
    if not circular:
        return []
    listed = "; ".join(
        f"'{c.feature}' (ρ={c.rho:.2f}; {kind})" if np.isfinite(c.rho) else f"'{c.feature}' ({kind})"
        for c, kind in circular[:6]
    )
    return [
        f"Possible CONSTRUCT-CIRCULARITY / POST-OUTCOME leakage for the low-cardinality target "
        f"'{request.target_column}': {listed}. These features passed the validation battery, but a "
        "(near-)concurrent or downstream feature's association is circular — it does NOT demonstrate an "
        "independent, predictive, or causal predictor. CoDaS does not auto-exclude them: confirm the "
        "measurement timeline and exclude outcome-derived / co-measured variables for a genuine claim."
    ]


# A validated association at or above this strength on time-structured data is worth a temporal-leakage
# check (it is in the band where a forward-looking / post-outcome feature typically lands).
_TEMPORAL_LEAKAGE_RHO = float(os.getenv("CODAS_TEMPORAL_LEAKAGE_RHO", "0.5"))


def _temporal_leakage_advisory(candidates: list[Candidate], request: "DiscoveryRequest") -> list[str]:
    """Soft, value+structure-based look-ahead-leakage advisory (NOT name-based, NOT auto-excluding).

    Look-ahead leakage is a timing problem the engine cannot read from values, so it cannot auto-detect
    a post-outcome feature. But it CAN flag the risk where it matters: on time-structured data (a time
    axis, or repeated measures), any strongly-associated surviving feature should be confirmed to be
    measured at or before the outcome window. Cross-sectional data has no look-ahead notion, so this
    stays silent there (no false-positive spam). Caller-declared post_outcome_columns are already
    excluded upstream, so they never reach this advisory.
    """
    if not (request.time_column or request.participant_id_column):
        return []
    strong = [
        c for c in candidates
        if c.verdict in {"validated", "conditional"} and np.isfinite(c.rho) and abs(c.rho) >= _TEMPORAL_LEAKAGE_RHO
    ]
    if not strong:
        return []
    listed = ", ".join(f"'{c.feature}' (ρ={c.rho:+.2f})" for c in strong[:8])
    return [
        f"TEMPORAL-LEAKAGE CHECK (time-structured data): {len(strong)} surviving feature(s) show a strong "
        f"association with '{request.target_column}': {listed}. CoDaS cannot infer when each column was "
        f"measured, so confirm none is recorded at or after the outcome window — a forward-looking or "
        f"post-outcome feature (e.g. a 'next-week' aggregate) produces look-ahead leakage that inflates "
        f"apparent predictive power. Declare any such column in post_outcome_columns (or excluded_columns) "
        f"and re-run before interpreting these as predictors of the current outcome."
    ]


@dataclass
class _ScreenResult:
    analysis: Any
    validation_pool: list[Candidate]
    screened_count: int
    cluster_screen_info: dict[str, Any]
    audit_log: list[str]


def _within_subject_diagnostic(screen_frame, analysis, cluster_groups, request, cluster_screen_info) -> list[str]:
    """Two-stage within-subject diagnostic for repeated measures: per-subject feature-outcome
    correlations tested across subjects (n = N_subjects, not pseudo-replicated). Mutates
    cluster_screen_info in place with the within-subject associations and returns any warnings.
    Extracted verbatim from _screen (pinned by the full-report golden test)."""
    results: list[str] = []
    # WITHIN-SUBJECT two-stage screen (only for row-level repeated measures with a participant id).
    # The cross-sectional cluster-corrected screen above answers "do subjects with higher X have higher
    # outcome?" (between-subject, n≈N_subjects). For a LONGITUDINAL predictor the scientific question is
    # within-subject ("as THIS subject's X rises, does their outcome rise?"), which the cross-sectional
    # screen cannot see. Compute each subject's own feature↔outcome correlation, then test the per-subject
    # correlations against zero across subjects (n = N_subjects) — confound-free and not pseudo-replicated.
    # Surfaced as an ADDITIVE diagnostic (does not alter the conservative cross-sectional verdicts).
    if cluster_groups is not None:
        _ws_rows = []
        _ws_p = []
        for feat in analysis.feature_columns:
            r = within_subject_two_stage(screen_frame[feat], screen_frame[analysis.target_column], cluster_groups)
            if r["n_subjects"] >= 8 and np.isfinite(r["within_rho_median"]):
                _ws_rows.append({"feature": feat, **r})
                _ws_p.append(r["p_value"])
        if _ws_rows:
            _ws_q = benjamini_hochberg(_ws_p)
            for row, q in zip(_ws_rows, _ws_q):
                row["q_value"] = float(q)
            _ws_rows.sort(key=lambda d: (d["q_value"], -abs(d["within_rho_median"])))
            # Flag near-perfect within-subject correlations as likely target leakage/components (the same
            # signature the cross-sectional construct gate rejects) so they are NOT highlighted as predictors.
            for d in _ws_rows:
                d["likely_leakage"] = bool(abs(d["within_rho_median"]) > 0.95)
            _ws_sig = [d for d in _ws_rows
                       if d["q_value"] <= request.fdr_alpha and 0.1 <= abs(d["within_rho_median"]) <= 0.95]
            cluster_screen_info["within_subject_associations"] = _ws_rows[:12]
            cluster_screen_info["within_subject_significant_count"] = len(_ws_sig)
            if _ws_sig:
                _top = "; ".join(f"{d['feature']} (within-ρ̃={d['within_rho_median']:.2f}, q={d['q_value']:.1e}, "
                                 f"{int(round(d['frac_consistent_sign']*100))}% of {d['n_subjects']} subjects same-sign)"
                                 for d in _ws_sig[:5])
                results.append(
                    f"WITHIN-SUBJECT diagnostic (two-stage; per-subject correlations tested across "
                    f"{_ws_rows[0]['n_subjects']} subjects, NOT pseudo-replicated): {len(_ws_sig)} feature(s) track "
                    f"the outcome within participants — {_top}. This is the longitudinal-tracking signal the "
                    f"cross-sectional screen cannot capture; treat as exploratory (not corrected for within-subject "
                    f"temporal autocorrelation) and confirm with mixed-effects + external data."
                )

    return results


def _screen(df: pd.DataFrame, request: "DiscoveryRequest", effective_participant: str | None,
            effective_time: str | None) -> tuple["_ScreenResult", list[str]]:
    """Screening phase: subsample very large data, build the analysis frame, apply the
    pseudo-replication guards (autocorrelation / cluster design-effect / within-subject), rank
    candidates (Spearman + Benjamini-Hochberg FDR), and select the validation pool. Returns the
    result and the warnings it produced. Raises InsufficientDataError on unusable input."""
    warnings: list[str] = []
    _auto_participant = False
    # Cap the rows used for screening/validation/model on very large datasets so the
    # synchronous discovery finishes within the interactive request timeout (Cloud Run
    # hard-kills the streaming request at its timeout, which otherwise leaves the UI stuck
    # on "Running"). Spearman screening + bootstrap validation are statistically
    # equivalent on a stratified sample of this size; profiling above still reflects full n,
    # and mixed-effects diagnostics re-read the full raw data separately.
    screen_cap = int(os.getenv("CODAS_SCREEN_MAX_ROWS", "40000"))
    analysis_df = df
    if len(df) > screen_cap:
        analysis_df = _subsample_for_screening(df, request.target_column, screen_cap, request.random_state)
        warnings.append(
            f"Dataset has {len(df):,} rows; candidate screening, the validation battery and model "
            f"fitting used a stratified random sample of {len(analysis_df):,} rows so the analysis "
            f"completes within the interactive timeout (association estimates are statistically "
            f"equivalent at this sample size). Dataset profiling reflects all {len(df):,} rows."
        )
    analysis = build_analysis_frame(
        analysis_df,
        target_column=request.target_column,
        participant_id_column=effective_participant,
        time_column=effective_time,
        excluded_columns=request.excluded_columns,
        confounder_columns=request.confounder_columns,
        max_ratio_features=request.max_ratio_features,
    )
    audit_log = list(analysis.audit_log)
    if not analysis.feature_columns:
        # Be honest about WHY: if repeated measures were aggregated to one row per participant
        # and too few participants remain, this is a sample-size limit (the scientifically correct
        # outcome of not pseudo-replicating), not a feature problem. Say so precisely.
        _aggregated = any("aggregat" in line.lower() for line in audit_log)
        _n_units = int(len(analysis.frame))
        if _aggregated and _n_units < 20:
            raise InsufficientDataError(
                f"After aggregating repeated measures to one row per participant, only {_n_units} "
                f"participants remain — too few for reliable discovery (≥20 recommended). The row-level "
                f"data is not independent, so analyzing it un-aggregated would inflate significance "
                f"(pseudo-replication). Collect more participants or analyze within-participant trajectories."
            )
        raise InsufficientDataError("No usable numeric candidate features were found after exclusions.")

    # Temporal pseudo-replication guard. Ungrouped rows with a time axis can be a single
    # autocorrelated series (n-of-1 / continuous monitoring): adjacent samples are not independent,
    # so the raw row count overstates significance — a small correlation between two slowly-drifting
    # series reads p<1e-3 at n=800 when the AR(1)-effective n is ~50. When detected, screen at the
    # autocorrelation-corrected effective n and disclose the limitation. Tightly gated (no participant
    # grouping + usable time axis + autocorrelated outcome) so cross-sectional / multi-subject data
    # is untouched.
    autocorr_correct = False
    screen_frame = analysis.frame
    if analysis.participant_id_column is None and effective_time and effective_time in analysis_df.columns:
        _tvals = pd.to_datetime(analysis_df[effective_time], errors="coerce")
        if int(_tvals.notna().sum()) < 20:
            _tvals = pd.to_numeric(analysis_df[effective_time], errors="coerce")
        if int(_tvals.notna().sum()) >= 20:
            _order = [i for i in _tvals.sort_values(kind="stable").index if i in analysis.frame.index]
            if len(_order) >= 20:
                _ordered = analysis.frame.loc[_order]
                _target_r1 = lag1_autocorr(_ordered[analysis.target_column].to_numpy())
                if abs(_target_r1) > 0.2:
                    autocorr_correct = True
                    screen_frame = _ordered
                    warnings.append(
                        f"Rows are a single time-ordered series with no participant grouping and the outcome "
                        f"is temporally autocorrelated (lag-1 r={_target_r1:.2f}); adjacent samples are NOT "
                        f"independent observations. Screening significance uses an autocorrelation-corrected "
                        f"effective sample size (Pyper–Peterman AR(1)), so a small correlation between two "
                        f"slowly-drifting series is not declared significant on the inflated row count. Treat "
                        f"associations as exploratory and validate on a held-out FUTURE window (forward-chaining) "
                        f"— a random train/test split leaks future into past on autocorrelated data."
                    )

    # Clustered pseudo-replication guard. When a participant/cluster id is present but the frame was
    # NOT collapsed to one row per participant (the outcome varies within subject, so build_analysis_frame
    # retained row-level data for mixed-effects), the raw row count over-states screening significance
    # (5875 rows from 42 subjects scored as 5875 independent obs). Screen at a design-effect effective n
    # (ICC-based) so per-subject TRAITS are deflated toward the cluster count while genuinely
    # within-subject-varying features are barely touched. Emit an ACCURATE warning describing what
    # actually happened (the prior unconditional "aggregated ... not inflated" note was false here).
    cluster_groups = None
    cluster_screen_info: dict[str, Any] = {}
    _pcol = analysis.participant_id_column
    if _pcol and _pcol in screen_frame.columns:
        _k = int(screen_frame[_pcol].nunique())
        _rows = int(len(screen_frame))
        if _k >= 2 and _rows > 1.5 * _k and not autocorr_correct:
            cluster_groups = screen_frame[_pcol]
            _mbar = _rows / _k
            _icc = intraclass_correlation(screen_frame[analysis.target_column].to_numpy(), cluster_groups)
            _eff = cluster_effective_n(_rows, _mbar, _icc, _icc, _k)
            cluster_screen_info = {
                "screen_cluster_id": _pcol, "screen_n_clusters": _k, "screen_rows": _rows,
                "screen_outcome_icc": round(float(_icc), 4), "screen_effective_n": int(_eff),
            }
            warnings.append(
                f"'{_pcol}' is a participant/cluster id with repeated measures ({_rows:,} rows from {_k} "
                f"participants, ~{_mbar:.0f} each) and the outcome varies WITHIN participant, so rows were "
                f"retained (not aggregated) to preserve within-participant signal. Screening significance is "
                f"computed at a cluster design-effect effective n (outcome ICC={_icc:.2f}) ≈ {_eff} — NOT the "
                f"inflated row count — so a per-subject trait is not declared significant on {_rows:,} "
                f"pseudo-replicated rows. The held-out model uses participant-grouped CV. Treat associations as "
                f"exploratory; confirm with the mixed-effects diagnostics (within-participant) and external data."
                + (" Pass the participant id explicitly to override." if _auto_participant else "")
            )
        elif _auto_participant and _k >= 2 and _rows <= 1.5 * _k:
            warnings.append(
                f"Detected '{_pcol}' as a participant/cluster id; repeated measures were aggregated to one row "
                f"per participant ({_k} participants) so per-row significance is not inflated. Pass it explicitly "
                f"to override."
            )
    ranked = _rank_candidates(screen_frame, analysis.target_column, analysis.feature_columns,
                              autocorr_correct=autocorr_correct, cluster_groups=cluster_groups)
    if not ranked:
        raise InsufficientDataError("No candidate feature had enough valid observations for screening.")

    warnings.extend(_within_subject_diagnostic(screen_frame, analysis, cluster_groups, request, cluster_screen_info))
    screened_count = len(ranked)
    validation_pool = [
        candidate for candidate in ranked if candidate.q_value <= request.fdr_alpha
    ]
    # FDR significance is a NON-WAIVABLE hard gate. When the screen finds 0 significant candidates,
    # we must NOT promote non-significant features into validation — doing so is equivalent to
    # p-hacking and GUARANTEES false positives on null datasets (any dataset will produce SOME
    # top-ranked features that then pass the internal validation battery simply because the battery
    # does not re-apply the FDR threshold). The previous fallback was removed: if q > fdr_alpha for every
    # feature, the discovery correctly reports "no significant predictors detected".
    # Exception: autocorrelation/cluster-corrected pools are already conservative — allow top_k
    # non-significant candidates there as exploratory candidates (they will be labeled conditional).
    if len(validation_pool) == 0 and (autocorr_correct or cluster_groups is not None):
        validation_pool = ranked[: max(1, min(len(ranked), request.top_k))]
        warnings.append(
            "FDR screen found no significant candidates at the corrected effective sample size; "
            "top-ranked features are included as exploratory (conditional at best). "
            "Interpret as hypothesis-generating only — external replication is required."
        )
    elif len(validation_pool) == 0:
        warnings.append(
            "FDR screen found no significant candidates (all q-values > fdr_alpha). "
            "No features meet the minimum significance threshold for predictor validation. "
            "This dataset does not provide sufficient evidence for any of the screened features. "
            "Consider: larger sample size, stronger signal, or a targeted a priori hypothesis."
        )
    validation_pool = validation_pool[: max(request.top_k, request.top_k * 2)]
    return _ScreenResult(analysis=analysis, validation_pool=validation_pool,
                         screened_count=screened_count, cluster_screen_info=cluster_screen_info,
                         audit_log=audit_log), warnings


def _ordinal_outcome_warnings(df, request) -> list[str]:
    results: list[str] = []
    # --- MED FIX: Ordinal outcome detection ---
    # When the target has a small number of unique integer values (2-6) the appropriate
    # statistical model is proportional-odds / ordinal regression, not plain OLS. Flag this
    # so the user knows to interpret R² with caution and consider an ordinal-specific method.
    if request.target_column in df.columns:
        _tgt_vals = df[request.target_column].dropna()
        _n_unique = int(_tgt_vals.nunique())
        _is_integer_like = (
            pd.api.types.is_integer_dtype(_tgt_vals)
            or (pd.api.types.is_float_dtype(_tgt_vals) and (_tgt_vals % 1 == 0).all())
        )
        if _is_integer_like and 2 < _n_unique <= 6:
            results.append(
                f"Target '{request.target_column}' appears to be ordinal (integer, {_n_unique} unique levels: "
                f"{sorted(_tgt_vals.unique().tolist()[:7])}). CoDaS treats it as continuous for screening "
                f"(Spearman correlation is rank-based and valid), but R² and the linear model are not optimal "
                f"for ordinal outcomes. For inference, consider proportional-odds (ordered logistic) regression "
                f"or a cumulative link model to respect the ordered-categorical structure."
            )

    return results


def _within_person_warnings(candidates, cluster_screen_info, analysis, request) -> list[str]:
    results: list[str] = []
    # --- MED FIX: Within-person signal surfacing ---
    # When the between-person aggregate analysis finds no validated predictors, check whether
    # the within-subject screening (done during cluster-based analysis) surfaced significant
    # within-person associations. If so, surface them in the report so the user is not left
    # with an empty result when the real signal is within-person deviation-from-baseline.
    _validated_count = sum(1 for c in candidates if c.verdict == "validated")
    if _validated_count == 0 and cluster_screen_info:
        _ws_rows = cluster_screen_info.get("within_subject_associations", [])
        _sig_within = [
            r for r in _ws_rows
            if float(r.get("q_value", 1)) < 0.10  # FDR corrected within-subject q
            and r.get("feature") not in (request.target_column,)
        ]
        if _sig_within:
            _within_summary = "; ".join(
                f"'{r['feature']}' (within-person ρ={float(r.get('within_rho_median', 0)):.3f}, "
                f"q={float(r.get('q_value', 1)):.2g})"
                for r in _sig_within[:5]
            )
            results.append(
                f"WITHIN-PERSON SIGNAL DETECTED despite no between-person validated predictors. "
                f"The between-person (aggregate) analysis found no significant associations, but the "
                f"within-person (repeated-measures) analysis found significant within-subject associations "
                f"for: {_within_summary}. "
                f"This pattern indicates that the true signal is a deviation-from-baseline rather than "
                f"an absolute-value predictor — common in repeated-measures studies with high inter-individual "
                f"variability. Inspect the mixed_effects_diagnostics.csv artifact for within-person "
                f"regression coefficients and confidence intervals."
            )
    elif _validated_count == 0 and analysis.participant_id_column:
        # Even without cluster-screen data: if we have repeated measures and no between-person
        # findings, add a generic advisory that within-person analysis may be more informative.
        results.append(
            "WITHIN-PERSON ANALYSIS ADVISORY: No significant between-person (aggregate) predictors were "
            "detected. Your dataset has repeated measurements per participant — the true signal may be a "
            "within-person deviation from baseline rather than an absolute-value difference between people. "
            "Inspect the mixed_effects_diagnostics.csv artifact in the Files panel: a significant "
            "fixed-effect coefficient there indicates a within-person association that the aggregate screen "
            "cannot detect."
        )

    return results


def _opaque_schema_warnings(df, request) -> list[str]:
    results: list[str] = []
    # --- MED FIX: Opaque/cryptic column schema warning ---
    # When column names carry no semantic content (e.g., v001, feat_3, col_7) the discovered
    # "predictor" cannot be mechanistically interpreted without a data dictionary. Flag this
    # so the user provides context before making any scientific claims.
    _opaque_patterns = (
        "^v[0-9]+$", "^feat[_-]?[0-9]+$", "^col[_-]?[0-9]+$", "^x[0-9]+$",
        "^f[0-9]+$", "^var[_-]?[0-9]+$", "^field[_-]?[0-9]+$",
    )
    _opaque_cols = []
    import re as _re
    for col in df.columns:
        if col in (request.target_column, request.participant_id_column, request.time_column):
            continue
        if any(_re.fullmatch(pat, str(col).lower().strip()) for pat in _opaque_patterns):
            _opaque_cols.append(col)
    if _opaque_cols and len(_opaque_cols) >= 3:
        results.append(
            f"OPAQUE SCHEMA: {len(_opaque_cols)} column names appear to be meaningless codes "
            f"({', '.join(_opaque_cols[:5])}...). Without a data dictionary the discovered associations "
            f"cannot be mechanistically interpreted — e.g., validating 'v003' as a predictor is only "
            f"meaningful if you know what v003 measures. "
            f"Please provide a data dictionary before interpreting or reporting these results as scientific findings."
        )

    return results


def _interaction_terms_warnings(df, request) -> list[str]:
    results: list[str] = []
    # --- H4 FIX: Interaction predictors caveat ---
    # CoDaS engineers per-feature summary statistics (mean, SD, min/max, CV, ratios) but does NOT
    # generate multiplicative cross-feature interaction terms. This means interaction-only predictors
    # (where the product or conditional effect of two features drives the outcome, not either
    # main effect) are structurally outside the current search space and will not be detected.
    # Surface this as a consistent limitation note so the boundary is explicit. This fires whenever
    # at least 2 numeric features exist (so interactions are even possible) — INCLUDING the
    # 0-validated-candidate case, which is precisely when an undetected interaction is the likely
    # explanation for a null result.
    _n_numeric_features = sum(
        1 for c in df.columns
        if c not in (request.target_column, request.participant_id_column, request.time_column)
        and pd.api.types.is_numeric_dtype(df[c])
    )
    if _n_numeric_features >= 2:
        results.append(
            "INTERACTION TERMS NOT TESTED: CoDaS screens univariate and per-family summary "
            "features (mean, SD, CV, ratios). Multiplicative interactions between distinct feature "
            "families (e.g., activity_index × sleep_index) are NOT generated and will not be detected. "
            "If no predictors were found but you expect a synergistic/conditional effect, the signal "
            "may live in an interaction the current screen cannot see — specify the interaction term "
            "explicitly or request a focused interaction analysis."
        )

    return results


def _assemble_report(df: pd.DataFrame, candidates: list[Candidate], validated: list[Candidate],
                     screen: "_ScreenResult", profile, request: "DiscoveryRequest",
                     warnings: list[str]) -> DiscoveryReport:
    """Reporting phase: ML benchmark on the top features, grounded Fact Sheet, methodological
    warnings, markdown report, and audit log -> DiscoveryReport. `profile` is the DatasetProfile;
    `screen` is the _ScreenResult from _screen()."""
    analysis = screen.analysis
    screened_count = screen.screened_count
    cluster_screen_info = screen.cluster_screen_info
    validation_pool = screen.validation_pool
    audit_log = screen.audit_log
    model_features = [
        candidate.feature
        for candidate in candidates
        if candidate.verdict in {"validated", "conditional"}
    ][:10]
    if not model_features and candidates:
        # Nothing passed the validation battery (e.g. every strong feature was flagged as a possible
        # restated-outcome / leakage). Still report a predictive-performance number from the top
        # screened candidates so the report is never blank; the per-candidate verdicts carry the
        # caveats. No-op when validated/conditional features exist, so the common path is unchanged.
        model_features = [candidate.feature for candidate in candidates][:10]
    ml_metrics = _ml_benchmark(
        analysis.frame, analysis.target_column, model_features, request.random_state,
        participant_id_column=analysis.participant_id_column,
    )
    fact_sheet = build_fact_sheet(
        profile=profile,
        candidates=candidates,
        target_column=request.target_column,
        ml_metrics=ml_metrics,
        discovery_rounds=1,
        feature_count=screened_count,
        battery_evaluated_count=len(validated),
        battery_passing_count=sum(candidate.verdict == "validated" for candidate in validated),
    )
    # Surface the clustered-screening correction in the fact sheet so the headline evidence is
    # auditable: a reader (or evaluator) can see the screening significance was computed at the
    # cluster design-effect effective n, not the inflated row count.
    if cluster_screen_info and isinstance(fact_sheet, dict):
        fact_sheet.update(cluster_screen_info)
    # Deterministic quality gates: record each gate's decision in the Fact Sheet so the reason a
    # result view was or was not surfaced is auditable. A single concise note
    # summarises any gate that would suppress its corresponding table.
    gates = evaluate_quality_gates(analysis.frame, model_features, ml_metrics, candidates)
    if isinstance(fact_sheet, dict):
        fact_sheet["quality_gates"] = gates
    if gates["triggered"]:
        details = "; ".join(
            f"{g['name']} ({g['detail']})" for g in gates["gates"] if g["triggered"]
        )
        warnings.append(
            f"Quality gates triggered — {details}. The corresponding result view(s) are flagged for "
            f"suppression per the output-suppression policy; interpret with caution."
        )
    warnings.extend(_ordinal_outcome_warnings(df, request))
    warnings.extend(_within_person_warnings(candidates, cluster_screen_info, analysis, request))
    warnings.extend(_opaque_schema_warnings(df, request))
    warnings.extend(_interaction_terms_warnings(df, request))
    markdown = build_markdown_report(fact_sheet, candidates, warnings)
    audit_log.extend([
        f"Screened {screened_count} candidate features with Spearman correlation and Benjamini-Hochberg FDR.",
        f"Audited {len(validation_pool)} candidate variants with the CoDaS internal validation battery.",
        "Generated Fact Sheet before report assembly.",
    ])
    LOGGER.info(
        "discovery done: target=%r screened=%d candidates=%d validated=%d metric=%s=%s",
        request.target_column, screened_count, len(candidates),
        sum(1 for c in candidates if c.verdict == "validated"),
        fact_sheet.get("ml_metric_name"), fact_sheet.get("ml_metric_value"),
    )
    return DiscoveryReport(
        profile=profile,
        candidates=candidates,
        fact_sheet=fact_sheet,
        audit_log=audit_log,
        warnings=warnings,
        markdown_report=markdown,
    )


def run_discovery(df: pd.DataFrame, request: DiscoveryRequest) -> DiscoveryReport:
    warnings: list[str] = []
    # Robustness: accept any DataFrame. Make column names unique and coerce a non-numeric target
    # before anything reads the frame, so malformed inputs give a clear result, never a crash.
    df = _dedupe_columns(df, warnings)
    df = _normalize_target(df, request.target_column, warnings)
    # +/-inf is never a valid measurement and would poison variance/correlation/scaling; treat it as
    # missing. Guarded so it is a no-op (no copy) on clean data.
    _numeric = df.select_dtypes(include="number")
    if _numeric.shape[1] and bool(np.isinf(_numeric.to_numpy(dtype="float64", na_value=np.nan)).any()):
        df = df.replace([np.inf, -np.inf], np.nan)
    # Generative-interpreter proposals: materialise proposed transformations as ordinary
    # columns so they enter the same screening + FDR + validation as every other feature.
    df = _materialize_proposed_features(df, request.proposed_features, warnings)
    LOGGER.info("discovery start: target=%r rows=%d cols=%d", request.target_column, len(df), df.shape[1])
    # Hard temporal-leakage guard: caller-declared post-outcome columns are excluded before screening
    # so a forward-looking / post-outcome feature can never be reported as a predictor of the current
    # outcome. (The engine cannot infer timing from values; this relies on the caller's declaration.)
    if request.post_outcome_columns:
        present = [c for c in request.post_outcome_columns if c in df.columns]
        if present:
            merged = list(dict.fromkeys(list(request.excluded_columns) + present))
            request = replace(request, excluded_columns=merged)
            warnings.append(
                f"Excluded {len(present)} caller-declared post-outcome column(s) before analysis to "
                f"prevent look-ahead / temporal leakage: {present}. A feature measured at or after the "
                f"outcome window is not a legitimate predictor of the current outcome."
            )
    warnings.extend(_input_quality_warnings(df, request.target_column))
    profile = profile_dataframe(
        df,
        target_column=request.target_column,
        participant_id_column=request.participant_id_column,
        time_column=request.time_column,
    )
    # NOTE: the engine performs NO name-based proxy/subscale exclusion. Construct overlap and
    # item->total leakage are caught STATISTICALLY downstream (construct-validity / near-determinism
    # hard gates operate on the actual target's values, not on column names). A caller who already
    # knows certain columns are proxies can still pass them in request.excluded_columns.

    # Roles come ONLY from the caller. profile_dataframe
    # is used ONLY for its structural facts; it never infers a role from column names.
    effective_participant = request.participant_id_column
    effective_time = request.time_column
    # NOTE: the accurate repeated-measures warning is emitted AFTER build_analysis_frame, once we
    # know whether the frame was actually aggregated to one row per participant (target constant
    # within subject) or retained at row level (target varies within subject -> cluster-corrected
    # screening). Emitting an unconditional "aggregated ... not inflated" claim here was FALSE for
    # within-subject-varying targets (e.g. longitudinal UPDRS): the rows stayed un-aggregated and
    # screening ran at the inflated row count while the warning claimed otherwise.
    if (not effective_participant) and request.target_column in df.columns:
        # No participant id declared. Warn if the rows look like un-keyed repeated measures so the
        # failure is loud, not silent (the correction still requires the caller to declare the id). A
        # STRING column that clusters many rows is flagged on its repeat structure; a NUMERIC integer
        # column is flagged only when the target is near-constant within its groups (a per-subject
        # trait/outcome), so an ordinary low-cardinality integer feature is not mistaken for an id.
        _n = int(len(df))
        _tgt_num = pd.to_numeric(df[request.target_column], errors="coerce")
        for _c in df.columns:
            if _c == request.target_column:
                continue
            _s = df[_c]
            _k = int(_s.nunique(dropna=True))
            if not (2 < _k < _n / 3 and (_n / max(_k, 1)) >= 3):
                continue
            _is_object = pd.api.types.is_object_dtype(_s)
            _is_int = (pd.api.types.is_numeric_dtype(_s)
                       and bool(np.all(np.mod(_s.dropna().to_numpy(dtype=float), 1.0) == 0.0)))
            if not (_is_object or _is_int):
                continue
            if _is_int and not _is_object:
                try:
                    _within = float(np.nanmedian(_tgt_num.groupby(_s).transform("std").to_numpy()))
                    _overall = float(_tgt_num.std())
                    if not (_overall > 0 and _within / _overall < 0.35):
                        continue  # target varies within groups -> ordinary feature, not an id
                except Exception:
                    continue
            warnings.append(
                f"Rows may be repeated measures: column '{_c}' has {_k} distinct values across "
                f"{_n:,} rows (~{_n / _k:.0f} rows each). If '{_c}' identifies participants/subjects, "
                f"set it as the participant id so per-row significance is not inflated (pseudo-replication)."
            )
            break
        # Undeclared temporal structure: with neither a participant nor a time column, a target that is
        # strongly autocorrelated in row order signals a time series scored at the raw (inflated) n.
        if not effective_time:
            _tv = _tgt_num.dropna().to_numpy()
            if _tv.size >= 20:
                _r1 = lag1_autocorr(_tv)
                if np.isfinite(_r1) and abs(_r1) > 0.3:
                    warnings.append(
                        f"The target is strongly autocorrelated in row order (lag-1 ρ={_r1:.2f}). If these "
                        f"rows are a time series, set time_column so significance is deflated to the "
                        f"autocorrelation-effective sample size (otherwise a spurious trend reads as significant)."
                    )
    screen, _screen_warnings = _screen(df, request, effective_participant, effective_time)
    warnings.extend(_screen_warnings)
    analysis = screen.analysis
    validation_pool = screen.validation_pool
    screened_count = screen.screened_count
    cluster_screen_info = screen.cluster_screen_info
    audit_log = screen.audit_log

    # Bound validation work so the run always finishes within the interactive request
    # timeout, regardless of dataset size: (1) scale resamples inversely with row count
    # so resamples x rows stays ~constant, and (2) stop validating further candidates once
    # a wall-clock budget is hit, finalizing gracefully with whatever passed. Together with
    # the row cap above and the per-candidate LOO cap, this is the hard guarantee that the
    # UI can never hang on "Running" waiting for a request the platform will kill.
    # The battery runs the requested resamples (1,000 by default, matching the paper's permutation and
    # bootstrap counts). The work budget only clamps pathologically large frames so a run still finishes
    # within the interactive timeout; at 8M it preserves the full 1,000 for cohorts up to ~8k rows
    # (every cohort in the paper) and only reduces beyond that.
    work_budget = int(os.getenv("CODAS_VALIDATION_WORK_BUDGET", "8000000"))
    eff_resamples = max(200, min(request.validation_resamples, work_budget // max(1, len(analysis.frame))))
    config = ValidationConfig(
        n_resamples=eff_resamples,
        random_state=request.random_state,
        fdr_alpha=request.fdr_alpha,
    )
    time_budget = float(os.getenv("CODAS_DISCOVERY_BUDGET_SECONDS", "300"))
    deadline = time.monotonic() + time_budget
    validated = []
    # Candidates are validated strongest-first, so each is residualized (paper 2.6-8) against the
    # biomarkers already validated ahead of it, requiring it to add signal beyond the established ones.
    validated_features: list[str] = []
    for candidate in validation_pool:
        result = validate_candidate(
            frame=analysis.frame,
            candidate=candidate,
            target_column=analysis.target_column,
            participant_id_column=analysis.participant_id_column,
            confounder_columns=analysis.confounder_columns,
            excluded_columns=analysis.excluded_columns,
            feature_components=analysis.feature_components,
            config=config,
            prior_validated_columns=validated_features,
        )
        validated.append(result)
        if result.verdict == "validated":
            validated_features.append(result.feature)
        if time.monotonic() > deadline:
            warnings.append(
                f"Validation time budget ({int(time_budget)}s) reached; validated "
                f"{len(validated)} of {len(validation_pool)} candidates and finalized to stay "
                f"within the interactive timeout (remaining candidates were not blocked, just deferred)."
            )
            break
    verdict_rank = {"validated": 0, "conditional": 1, "rejected": 2, "untested": 3}
    ordered_candidates = sorted(
        validated,
        key=lambda item: (verdict_rank.get(item.verdict, 9), -item.score, item.q_value),
    )
    candidates = ordered_candidates[: request.top_k]

    # Keep high-risk hard-gate failures visible in the audit trail even when many
    # stronger passing associations exist. The audit trail must show leakage and
    # construct-overlap failures, not just the successful shortlist.
    selected_features = {candidate.feature for candidate in candidates}
    hard_gate_failures = sorted(
        [
            candidate
            for candidate in validated
            if candidate.feature not in selected_features
            and any(test.hard_gate and test.applicable and not test.passed for test in candidate.tests)
        ],
        key=lambda item: abs(item.rho) if np.isfinite(item.rho) else 0.0,
        reverse=True,
    )
    if hard_gate_failures:
        candidates.extend(hard_gate_failures[:3])
        warnings.append("High-risk hard-gate failures were retained in the audit trail for inspection.")

    warnings.extend(_demote_collinear(candidates, analysis.frame))

    warnings.extend(_effect_size_warnings(candidates))

    warnings.extend(_combined_feature_attribution(candidates, analysis))

    warnings.extend(_construct_circularity_advisory(candidates, analysis, request))

    warnings.extend(_temporal_leakage_advisory(candidates, request))

    # NOTE: no NAME-based temporal/future-leakage advisory. The engine does not infer a measurement
    # timeline from column names. A caller who knows a feature is measured post-outcome or in a future
    # window can exclude it explicitly via request.excluded_columns.

    return _assemble_report(df, candidates, validated, screen, profile, request, warnings)


def run_discovery_from_csv(path: str | Path, request: DiscoveryRequest) -> DiscoveryReport:
    df = read_csv_dataset(path)
    return run_discovery(df, request)
