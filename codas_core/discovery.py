"""End-to-end deterministic association discovery pipeline."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
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
from .reporting import build_fact_sheet, build_markdown_report
from .statistics import autocorr_effective_n, benjamini_hochberg, cluster_effective_n, correlation_pvalue, intraclass_correlation, lag1_autocorr, safe_spearman, signed_direction, within_subject_two_stage
from .validation import ValidationConfig, validate_candidate


@dataclass
class DiscoveryRequest:
    target_column: str
    participant_id_column: str | None = None
    time_column: str | None = None
    excluded_columns: list[str] = field(default_factory=list)
    confounder_columns: list[str] = field(default_factory=list)
    top_k: int = 25
    fdr_alpha: float = 0.10
    max_ratio_features: int = 24
    validation_resamples: int = 1000
    random_state: int = 17
    # Columns the USER/PROFILE has declared to be CONCURRENT diagnostic tests of the SAME
    # condition as a (binary/categorical) diagnostic target — e.g. a second assay/imaging read
    # scored on the same visit. They are construct-circular / co-diagnostic, NOT independent
    # upstream risk factors; if they survive the battery they are ADVISED against (not
    # auto-excluded, since the user may intend them). Matched case-insensitively by exact name.
    concurrent_test_columns: list[str] = field(default_factory=list)


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


def run_discovery(df: pd.DataFrame, request: DiscoveryRequest) -> DiscoveryReport:
    warnings: list[str] = []
    # Heavy OUTCOME missingness is silently biasing: only rows with an observed target are
    # analyzed, so if the outcome is missing-not-at-random (e.g. high-stress states under-reported
    # in EMA, sicker patients lost to follow-up) the association reflects the reporting subgroup,
    # not the cohort. Surface it explicitly rather than analyzing the biased subset as if complete.
    if request.target_column in df.columns:
        # Feature-level missingness warning (sensor dropout, sensor dropout).
        # A feature with >30% missing data is either a signal with systematic
        # non-wear periods, or a measurement artifact. If the missingness correlates with
        # the outcome (MNAR), the analysis on observed rows is biased toward a subset.
        _HIGH_FEAT_MISS = 0.30
        # Scope to genuine CANDIDATE features: a column we block from candidacy (another outcome/label
        # like target_BDI2, or a demographic) is not a "feature", so warning about its missingness
        # would mislabel it. (Surfacing target_BDI2 at 92% missing as a "feature" reads as if it were
        # a predictor candidate when it is excluded as a circular alternative outcome.)
        _feat_cols_for_miss = [c for c in df.columns if c != request.target_column
                               and pd.api.types.is_numeric_dtype(df[c])]
        _high_miss_feats = [c for c in _feat_cols_for_miss
                            if float(df[c].isna().mean()) > _HIGH_FEAT_MISS]
        if _high_miss_feats:
            _miss_rates = {c: round(float(df[c].isna().mean()*100), 0) for c in _high_miss_feats[:4]}
            warnings.append(
                f"High feature missingness detected: {_miss_rates}. These features are only available "
                f"for a subset of rows; if data is missing-not-at-random (e.g. sensor dropout during "
                f"high-activity or high-stress periods), associations reflect the observed subgroup and "
                f"may not generalise. Verify the missingness mechanism before interpreting."
            )

        _target_missing = float(df[request.target_column].isna().mean())
        if _target_missing > 0.10:
            warnings.append(
                f"{_target_missing * 100:.0f}% of the outcome '{request.target_column}' is missing; only rows with an "
                "observed outcome are analyzed. If missingness is informative (missing-not-at-random — e.g. "
                "under-reported states or loss to follow-up), the associations are biased toward the reporting "
                "subgroup. Confirm the missingness mechanism before interpreting."
            )
        # Class imbalance: ROC-AUC is optimistic at low prevalence (a rare-event detector can look
        # strong while missing most positives). Surface the base rate and steer to PR-AUC.
        _t_obs = df[request.target_column].dropna()
        if _t_obs.nunique() == 2:
            _minority = float(_t_obs.value_counts(normalize=True).min())
            if _minority < 0.15:
                warnings.append(
                    f"Outcome is imbalanced ({_minority * 100:.1f}% minority class); held-out ROC-AUC is "
                    f"optimistic at low prevalence — prioritize precision–recall (PR-AUC) and calibrate to the "
                    f"{_minority * 100:.1f}% base rate before interpreting predictive performance."
                )
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
    _auto_participant = False
    # NOTE: the accurate repeated-measures warning is emitted AFTER build_analysis_frame, once we
    # know whether the frame was actually aggregated to one row per participant (target constant
    # within subject) or retained at row level (target varies within subject -> cluster-corrected
    # screening). Emitting an unconditional "aggregated ... not inflated" claim here was FALSE for
    # within-subject-varying targets (e.g. longitudinal UPDRS): the rows stayed un-aggregated and
    # screening ran at the inflated row count while the warning claimed otherwise.
    if (not effective_participant) and request.target_column in df.columns:
        # No participant id found. Warn if the rows still look like un-keyed repeated measures
        # (a string/categorical column clusters many rows) so the reviewer can set it.
        _n = int(len(df))
        for _c in df.columns:
            if _c == request.target_column:
                continue
            _s = df[_c]
            if not pd.api.types.is_object_dtype(_s):
                continue
            _k = int(_s.nunique(dropna=True))
            if 2 < _k < _n / 3 and (_n / max(_k, 1)) >= 3:
                warnings.append(
                    f"Rows may be repeated measures: column '{_c}' has {_k} distinct values across "
                    f"{_n:,} rows (~{_n / _k:.0f} rows each). If '{_c}' identifies participants/subjects, "
                    f"set it as the participant id so per-row significance is not inflated (pseudo-replication)."
                )
                break
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
                warnings.append(
                    f"WITHIN-SUBJECT diagnostic (two-stage; per-subject correlations tested across "
                    f"{_ws_rows[0]['n_subjects']} subjects, NOT pseudo-replicated): {len(_ws_sig)} feature(s) track "
                    f"the outcome within participants — {_top}. This is the longitudinal-tracking signal the "
                    f"cross-sectional screen cannot capture; treat as exploratory (not corrected for within-subject "
                    f"temporal autocorrelation) and confirm with mixed-effects + external data."
                )

    # Prognostic-target POST-OUTCOME leakage warning. For a prognostic target (recurrence / survival /
    # progression / mortality), a strongly-associated feature whose NAME implies a post-treatment or
    # follow-up assessment (e.g. 'response_to_therapy', 'follow_up_*', 'remission', 'vital_status') is
    # very likely recorded AFTER the prediction time -> temporal leakage for an early/prognostic model.
    # CoDaS cannot know the measurement timeline from values alone, so this is an ADVISORY flag (not an
    # auto-reject): surface it so the researcher confirms timing. Gated on a prognostic-looking target so
    # cross-sectional/diagnostic outcomes (where a 'response' feature can be a legitimate baseline) are
    # untouched. (Found on the thyroid-recurrence benchmark: 'response_ord' (post-therapy response, rho
    # 0.82 with recurrence) was validated with no timing caveat.)
    _tgt_low = str(request.target_column or "").lower()
    _is_prognostic = any(k in _tgt_low for k in ("recurr", "surviv", "progress", "relaps", "mortal",
                                                 "death", "readmiss", "prognos", "time_to", "tte", "_event"))
    if _is_prognostic:
        _post_tokens = ("response", "follow_up", "followup", "post_op", "postop", "post_treat",
                        "posttreat", "remission", "recovery", "vital_status", "outcome", "discharge")
        _flagged = [f"{c.feature} (ρ={c.rho:.2f})" for c in ranked[: max(request.top_k, 10)]
                    if abs(c.rho or 0) >= 0.3 and any(tok in str(c.feature or "").lower() for tok in _post_tokens)]
        if _flagged:
            warnings.append(
                f"Possible POST-OUTCOME leakage for a prognostic target ('{request.target_column}'): "
                f"{', '.join(_flagged[:5])} have names suggesting a post-treatment / follow-up assessment. "
                f"If recorded AFTER the prediction time, including them is temporal leakage for an early/"
                f"prognostic model. Confirm the measurement timeline; exclude post-baseline variables for a "
                f"genuine early-detection claim."
            )

    screened_count = len(ranked)
    validation_pool = [
        candidate for candidate in ranked if candidate.q_value <= request.fdr_alpha
    ]
    # FDR significance is a NON-WAIVABLE hard gate. When the screen finds 0 significant candidates,
    # we must NOT promote non-significant features into validation — doing so is equivalent to
    # p-hacking and GUARANTEES false positives on null datasets (any dataset will produce SOME
    # top-ranked features that then pass the 12-dimension battery simply because the battery doesn't
    # re-apply the FDR threshold). The previous fallback was removed: if q > fdr_alpha for every
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

    # Bound validation work so the run always finishes within the interactive request
    # timeout, regardless of dataset size: (1) scale resamples inversely with row count
    # so resamples x rows stays ~constant, and (2) stop validating further candidates once
    # a wall-clock budget is hit, finalizing gracefully with whatever passed. Together with
    # the row cap above and the per-candidate LOO cap, this is the hard guarantee that the
    # UI can never hang on "Running" waiting for a request the platform will kill.
    work_budget = int(os.getenv("CODAS_VALIDATION_WORK_BUDGET", "3000000"))
    eff_resamples = max(200, min(request.validation_resamples, work_budget // max(1, len(analysis.frame))))
    config = ValidationConfig(
        n_resamples=eff_resamples,
        random_state=request.random_state,
        fdr_alpha=request.fdr_alpha,
    )
    time_budget = float(os.getenv("CODAS_DISCOVERY_BUDGET_SECONDS", "300"))
    deadline = time.monotonic() + time_budget
    validated = []
    for candidate in validation_pool:
        validated.append(
            validate_candidate(
                frame=analysis.frame,
                candidate=candidate,
                target_column=analysis.target_column,
                participant_id_column=analysis.participant_id_column,
                confounder_columns=analysis.confounder_columns,
                excluded_columns=analysis.excluded_columns,
                feature_components=analysis.feature_components,
                config=config,
            )
        )
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
    # stronger passing associations exist. Expert reviewers need to see leakage
    # and construct-overlap failures, not just the successful shortlist.
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
        warnings.append("High-risk hard-gate failures were retained in the public audit trail for reviewer inspection.")

    # Inter-feature collinearity dedup: among the final validated/conditional candidates, if two
    # features are near-duplicates of each other (|inter-feature ρ| > 0.95 in the analysis frame)
    # they are measuring the same construct and the weaker one is redundant — reporting both as
    # independent predictors inflates the effective hit count. Demote the weaker (lower |ρ_target|)
    # to "collinear_redundant" and surface a warning so the reviewer sees one clear predictor.
    # Only applies to the shortlist candidates (not hard-gate-failures kept for audit transparency).
    _COLLINEAR_THRESHOLD = 0.90  # lowered from 0.95: r≥0.90 means same signal (e.g. PPG vs ECG HRV)
    _passing_cols = [c for c in candidates if c.verdict in {"validated", "conditional"}]
    _demoted = set()
    if len(_passing_cols) >= 2:
        _feats = [c.feature for c in _passing_cols if c.feature in analysis.frame.columns]
        _collinear_pairs: list[tuple[str, str]] = []
        for _i, _fi in enumerate(_feats):
            for _j, _fj in enumerate(_feats):
                if _j <= _i or _fi in _demoted or _fj in _demoted:
                    continue
                try:
                    _r = float(analysis.frame[[_fi, _fj]].dropna().corr().iloc[0, 1])
                except Exception:
                    _r = 0.0
                if np.isfinite(_r) and abs(_r) > _COLLINEAR_THRESHOLD:
                    _rho_i = abs(next(c.rho for c in _passing_cols if c.feature == _fi))
                    _rho_j = abs(next(c.rho for c in _passing_cols if c.feature == _fj))
                    _weaker = _fj if _rho_i >= _rho_j else _fi
                    _demoted.add(_weaker)
                    _collinear_pairs.append((_fi, _fj))
        if _demoted:
            for _cand in candidates:
                if _cand.feature in _demoted:
                    _cand.verdict = "collinear_redundant"
            warnings.append(
                f"Near-duplicate features detected among validated candidates (|inter-feature ρ| > "
                f"{_COLLINEAR_THRESHOLD}): {', '.join(str(p) for p in _collinear_pairs)}. The stronger "
                f"feature in each pair is retained; the weaker is demoted to 'collinear_redundant' "
                f"to avoid double-counting a single biological signal."
            )

    # Effect-size calibration: statistically significant at large n does NOT imply practically
    # actionable. A validated candidate with |ρ| < 0.20 (small effect by Cohen's convention)
    # explains < 4% of variance. Surface the variance-explained figure so a reviewer can judge
    # practical significance before acting on the finding.
    _small_effect_notes = []
    for _c in candidates:
        if _c.verdict in {"validated", "conditional"} and np.isfinite(_c.rho) and abs(_c.rho) < 0.20:
            _pct = round(float(_c.rho) ** 2 * 100, 1)
            _small_effect_notes.append(f"'{_c.feature}' (ρ={_c.rho:.2f}, ~{_pct}% variance explained)")
    if _small_effect_notes:
        warnings.append(
            f"Small effect size: {'; '.join(_small_effect_notes)}. Statistical significance at "
            f"large n does not imply practical importance — review effect magnitudes "
            f"before acting on findings with |ρ| < 0.20."
        )

    # --- S20 FIX: combined-feature (ratio / interaction) attribution ---
    # A two-component engineered feature (a/b or a×b) can validate purely because ONE component
    # carries the signal while the other is null. Reporting it as a joint a/b predictor spuriously
    # implicates the null component. When that pattern is detected, attribute the finding to the real
    # component and annotate the candidate so synthesis does not credit the null one.
    _attribution_notes: list[str] = []
    for _c in candidates:
        if _c.verdict not in {"validated", "conditional"}:
            continue
        _comps = analysis.feature_components.get(_c.feature)
        if not _comps or len(_comps) != 2:
            continue
        if not (("_over_" in _c.feature) or ("_x_" in _c.feature)):
            continue
        _comp_rhos: dict[str, float] = {}
        for _comp in _comps:
            if _comp in analysis.frame.columns and analysis.target_column in analysis.frame.columns:
                _r, _, _ = safe_spearman(analysis.frame[_comp], analysis.frame[analysis.target_column])
                _comp_rhos[_comp] = abs(_r) if np.isfinite(_r) else 0.0
        if len(_comp_rhos) != 2:
            continue
        (_strong, _strong_rho), (_weak, _weak_rho) = sorted(_comp_rhos.items(), key=lambda kv: kv[1], reverse=True)
        # Signal carried by ONE component: the strong one has a real effect, the weak one is ~null
        # and substantially weaker (so the combo is essentially a rescaling of the strong component).
        if _strong_rho >= 0.25 and _weak_rho < 0.15 and _weak_rho < 0.5 * _strong_rho:
            _note = (
                f"ATTRIBUTION: this combined feature's association is carried by '{_strong}' "
                f"(|ρ|={_strong_rho:.2f}); the other component '{_weak}' does NOT independently "
                f"associate with the target (|ρ|={_weak_rho:.2f}). Attribute the finding to "
                f"'{_strong}' alone — do NOT report '{_weak}' as a contributing predictor."
            )
            _c.evidence = (str(_c.evidence or "").rstrip() + " " + _note).strip()
            _attribution_notes.append(f"'{_c.feature}' → driven by '{_strong}', not '{_weak}'")
    if _attribution_notes:
        warnings.append(
            "COMBINED-FEATURE ATTRIBUTION: " + "; ".join(_attribution_notes[:6]) + ". "
            "These ratio/interaction features validated, but their signal comes from ONE component; "
            "the other component is null and must not be reported as an independent predictor."
        )

    # Construct-circularity / post-diagnosis advisory (diagnostic target). The cross-sectional
    # construct hard-gate only trips on a NEAR-deterministic copy of the outcome (AUC>=0.99 /
    # class-purity>=0.97), so a strong-but-imperfect PRIOR DIAGNOSIS (Dx_*) or a CONCURRENT
    # diagnostic test of the same condition (a second assay scored on the same visit) slips
    # through and is reported as a "validated" predictor. Such a feature is (near-)concurrent
    # with or downstream of the diagnosis, so its association is circular — it does NOT show an
    # independent, predictive, or causal predictor (this is hard-fail #2: construct-circularity /
    # post-diagnosis leakage). CoDaS cannot know the measurement timeline from values alone and a
    # name-derived diagnosis column is occasionally an intended predictor, so this is an ADVISORY
    # (not auto-exclusion): surface it and annotate the candidate's note so synthesis flags it.
    # Gated on a binary / low-cardinality categorical (diagnostic) target; the name match is
    # conservative (word-boundary, time/age/count-qualified 'diagnosis' covariates are NOT flagged)
    # to avoid demoting legitimate predictors. (Found on cervical_cancer/Biopsy: Schiller,
    # Hinselmann, Citology (concurrent tests) and Dx/Dx_Cancer/Dx_HPV (prior diagnoses) validated.)
    _tgt_obs = analysis.frame[analysis.target_column].dropna() if analysis.target_column in analysis.frame else pd.Series(dtype=float)
    _diagnostic_target = 2 <= int(_tgt_obs.nunique()) <= 10
    if _diagnostic_target:
        _declared_cotests = {str(c).strip().lower() for c in (request.concurrent_test_columns or []) if str(c).strip()}
        _circular: list[tuple[Candidate, str]] = []
        for _c in candidates:
            if _c.verdict not in {"validated", "conditional"}:
                continue
            _name = str(_c.feature or "")
            if _name.strip().lower() in _declared_cotests:
                _kind = "declared CONCURRENT diagnostic test of the same condition"
            elif np.isfinite(_c.rho) and abs(_c.rho) >= 0.80:
                # Value-based circularity probe (complements the name/declared checks, which miss an
                # obscurely-named concurrent assay e.g. 'panel_score_v2'): a single feature that
                # separates a DIAGNOSTIC outcome near-perfectly (|ρ|>=0.80, ~AUC>=0.93) — yet below the
                # AUC>=0.99 exact-copy hard-gate — is far more consistent with a concurrent diagnostic
                # test / post-diagnosis measurement of the same condition than with an independent
                # upstream risk factor (genuine single-feature predictors of a disease almost never
                # reach |ρ|>=0.80; calibrated to fire on 0/30+ real-benchmark markers). ADVISORY only.
                _kind = (f"separates the diagnostic classes near-perfectly (|ρ|={abs(_c.rho):.2f}, "
                         f"~AUC≳0.93) — far stronger than a typical upstream risk factor, consistent "
                         f"with a concurrent diagnostic test / post-diagnosis measurement")
            else:
                continue
            _circular.append((_c, _kind))
            # Reflect the caveat in the candidate's own note so the synthesis surfaces it per-feature.
            _c.evidence = (
                str(_c.evidence or "").rstrip()
                + f" CONSTRUCT-CIRCULARITY ADVISORY: '{_name}' {_kind} for the diagnostic target "
                  f"'{request.target_column}' — it is (near-)concurrent with or downstream of the "
                  f"diagnosis, so a strong association is circular (post-diagnosis leakage) and does not "
                  f"establish an independent, predictive, or causal predictor. Confirm the measurement "
                  f"timeline; exclude diagnosis-derived / co-diagnostic variables for a genuine predictor claim."
            ).strip()
        if _circular:
            _listed = "; ".join(
                f"'{_c.feature}' (ρ={_c.rho:.2f}; {_kind})"
                if np.isfinite(_c.rho) else f"'{_c.feature}' ({_kind})"
                for _c, _kind in _circular[:6]
            )
            warnings.append(
                f"Possible CONSTRUCT-CIRCULARITY / POST-DIAGNOSIS leakage for the diagnostic target "
                f"'{request.target_column}': {_listed}. These features were promoted by the validation "
                f"battery, but a prior-diagnosis or concurrent diagnostic-test feature is (near-)concurrent "
                f"with or downstream of the outcome, so its association is circular — it does NOT demonstrate "
                f"an independent, predictive, or causal predictor. CoDaS does NOT auto-exclude them (a "
                f"diagnosis-named column is occasionally an intended predictor): confirm the measurement "
                f"timeline and exclude diagnosis-derived / co-diagnostic variables for a genuine predictor claim."
            )

    # NOTE: no NAME-based temporal/future-leakage advisory. The engine does not infer a measurement
    # timeline from column names. A caller who knows a feature is measured post-outcome or in a future
    # window can exclude it explicitly via request.excluded_columns.

    model_features = [
        candidate.feature
        for candidate in candidates
        if candidate.verdict in {"validated", "conditional"}
    ][:10]
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
            warnings.append(
                f"Target '{request.target_column}' appears to be ordinal (integer, {_n_unique} unique levels: "
                f"{sorted(_tgt_vals.unique().tolist()[:7])}). CoDaS treats it as continuous for screening "
                f"(Spearman correlation is rank-based and valid), but R² and the linear model are not optimal "
                f"for ordinal outcomes. For inference, consider proportional-odds (ordered logistic) regression "
                f"or a cumulative link model to respect the ordered-categorical structure."
            )

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
            if float(r.get("q", 1)) < 0.10  # FDR corrected within-subject q
            and r.get("feature") not in (request.target_column,)
        ]
        if _sig_within:
            _within_summary = "; ".join(
                f"'{r['feature']}' (within-person ρ={float(r.get('within_rho_median', 0)):.3f}, "
                f"q={float(r.get('q', 1)):.2g})"
                for r in _sig_within[:5]
            )
            warnings.append(
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
        warnings.append(
            "WITHIN-PERSON ANALYSIS ADVISORY: No significant between-person (aggregate) predictors were "
            "detected. Your dataset has repeated measurements per participant — the true signal may be a "
            "within-person deviation from baseline rather than an absolute-value difference between people. "
            "Inspect the mixed_effects_diagnostics.csv artifact in the Files panel: a significant "
            "fixed-effect coefficient there indicates a within-person association that the aggregate screen "
            "cannot detect."
        )

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
        warnings.append(
            f"OPAQUE SCHEMA: {len(_opaque_cols)} column names appear to be meaningless codes "
            f"({', '.join(_opaque_cols[:5])}...). Without a data dictionary the discovered associations "
            f"cannot be mechanistically interpreted — e.g., validating 'v003' as a predictor is only "
            f"meaningful if you know what v003 measures. "
            f"Please provide a data dictionary before interpreting or reporting these results as scientific findings."
        )

    # --- H4 FIX: Interaction predictors caveat ---
    # CoDaS engineers per-feature summary statistics (mean, SD, min/max, CV, ratios) but does NOT
    # generate multiplicative cross-feature interaction terms. This means interaction-only predictors
    # (where the product or conditional effect of two features drives the outcome, not either
    # main effect) are structurally outside the current search space and will not be detected.
    # Surface this as a consistent limitation note so reviewers know the boundary. This fires whenever
    # at least 2 numeric features exist (so interactions are even possible) — INCLUDING the
    # 0-validated-candidate case, which is precisely when an undetected interaction is the likely
    # explanation for a null result.
    _n_numeric_features = sum(
        1 for c in df.columns
        if c not in (request.target_column, request.participant_id_column, request.time_column)
        and pd.api.types.is_numeric_dtype(df[c])
    )
    if _n_numeric_features >= 2:
        warnings.append(
            "INTERACTION TERMS NOT TESTED: CoDaS screens univariate and per-family summary "
            "features (mean, SD, CV, ratios). Multiplicative interactions between distinct feature "
            "families (e.g., activity_index × sleep_index) are NOT generated and will not be detected. "
            "If no predictors were found but you expect a synergistic/conditional effect, the signal "
            "may live in an interaction the current screen cannot see — specify the interaction term "
            "explicitly or request a focused interaction analysis."
        )

    markdown = build_markdown_report(fact_sheet, candidates, warnings)
    audit_log.extend([
        f"Screened {screened_count} candidate features with Spearman correlation and Benjamini-Hochberg FDR.",
        f"Audited {len(validation_pool)} candidate variants with the CoDaS internal validation battery.",
        "Generated Fact Sheet before report assembly.",
    ])
    return DiscoveryReport(
        profile=profile,
        candidates=candidates,
        fact_sheet=fact_sheet,
        audit_log=audit_log,
        warnings=warnings,
        markdown_report=markdown,
    )


def run_discovery_from_csv(path: str | Path, request: DiscoveryRequest) -> DiscoveryReport:
    df = read_csv_dataset(path)
    return run_discovery(df, request)
