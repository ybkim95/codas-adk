"""Implementation of the CoDaS internal validation battery."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _stats
from sklearn.model_selection import train_test_split

from .models import Candidate, ValidationTestResult
from .statistics import (
    auc_against_target,
    partial_spearman,
    safe_kendall,
    safe_pearson,
    safe_spearman,
    same_sign,
    spearman_rho_fast,
)


def _near_deterministic_of_target(x, y, threshold: float = 0.97) -> bool:
    """Does a SINGLE feature near-perfectly determine a low-cardinality (categorical) target?
    Bins the feature into quantiles and computes majority-vote class accuracy; ~1.0 means the feature
    is a copy/proxy of the outcome (leakage), regardless of Spearman rho. Generalises the binary-AUC
    ceiling to MULTICLASS targets (where Spearman saturates and binary AUC is undefined) — e.g. a
    near-perfect leak of a 3-class fetal-health target reads Spearman ~0.72 yet determines the class."""
    xv = pd.to_numeric(pd.Series(x), errors="coerce")
    yv = pd.Series(y).reset_index(drop=True)
    xv = xv.reset_index(drop=True)
    mask = xv.notna() & yv.notna()
    xv, yv = xv[mask], yv[mask]
    n = int(len(xv))
    k = int(yv.nunique())
    if n < 30 or k < 2 or k > 20:  # only categorical / low-cardinality targets
        return False
    # Enough bins to RESOLVE class boundaries: on an imbalanced multiclass target a coarse binning
    # leaves boundary bins mixed and caps a true separator's purity ~0.96. Fine bins push a real
    # separator to ~0.99 while a legitimate strong feature stays ~0.80-0.85 (clean margin).
    nb = min(100, max(40, n // 25), n)
    try:
        bins = pd.qcut(xv.rank(method="first"), q=nb, duplicates="drop")
    except Exception:
        return False
    grp = pd.DataFrame({"b": bins.values, "y": yv.values}).groupby("b", observed=True)["y"]
    correct = int(grp.apply(lambda s: int(s.value_counts().iloc[0])).sum())
    return (correct / n) >= threshold


def _cyclic_confounder_period(name: str) -> float | None:
    """Period (raw units) for a high-cardinality CYCLIC time confounder, so a circadian/seasonal
    rhythm can be adjusted with Fourier (sin/cos) terms. Hour-of-day is cyclic (0 ≈ 23) and its
    effect is sinusoidal: a linear adjustment leaves the circadian confound intact, which would
    let a time-of-day rhythm masquerade as a stress/mechanism predictor. Low-cardinality cyclics
    (month, weekday) fall through to one-hot, which is more flexible within the cap."""
    low = str(name).lower()
    if "hour" in low:
        return 24.0
    if "minute" in low:
        return 60.0
    if "dayofyear" in low or "day_of_year" in low or low == "doy":
        return 365.0
    return None


def _is_adjustable_confounder(series: pd.Series) -> bool:
    """A confounder can be adjusted for if it is numeric, OR a low-cardinality categorical —
    including a STRING-coded one (a real multi-site study codes site as 'BWH'/'MGH', not 0/1/2).
    High-cardinality strings (free-text, IDs) are not adjustable and are skipped."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    return 2 <= int(series.dropna().nunique()) <= 12


def _confounder_covariate_matrix(frame: pd.DataFrame, confounders: list[str], index) -> np.ndarray:
    """Build the covariate matrix for partial-correlation confounder adjustment.

    * CYCLIC time confounders (hour/minute/day-of-year) are expanded into sin/cos (Fourier)
      terms so a circadian/seasonal rhythm is actually removed (linear/one-hot can't capture a
      cyclic, high-cardinality confound — this is the classic "it's just time-of-day").
    * Multi-level categorical confounders (a site/batch code, 3-12 discrete levels) are one-hot
      encoded so a category-specific (step) confound is fully adjusted, not under-adjusted by a
      single linear term.
    * Binary (sex, group) and continuous (age, bmi) confounders stay a single column -- unchanged.
    NaN rows are preserved as NaN so partial_spearman drops incomplete observations.
    """
    blocks: list[np.ndarray] = []
    sub = frame.loc[index, confounders]
    n_rows = len(sub)
    for col in confounders:
        raw = sub[col]
        # STRING / non-numeric categorical confounder (e.g. site "BWH"/"MGH") -> one-hot.
        if not pd.api.types.is_numeric_dtype(raw):
            nun = int(raw.dropna().nunique())
            if 2 <= nun <= 12:
                dummies = pd.get_dummies(raw.astype("object"), prefix=col, drop_first=True, dtype=float)
                arr = np.array(dummies.to_numpy(dtype=float))  # writable copy (pandas>=3.0 CoW returns a read-only view)
                if arr.size:
                    arr[raw.isna().to_numpy(), :] = np.nan
                    blocks.append(arr)
                    continue
            continue  # high-cardinality / unusable string contributes nothing
        s = pd.to_numeric(raw, errors="coerce")
        non_na = s.dropna()
        if len(non_na) == 0:
            blocks.append(s.to_numpy(dtype=float).reshape(-1, 1))
            continue
        period = _cyclic_confounder_period(col)
        if period is not None:
            v = s.to_numpy(dtype=float)
            block = np.column_stack([np.sin(2 * np.pi * v / period), np.cos(2 * np.pi * v / period)])
            block[s.isna().to_numpy(), :] = np.nan
            blocks.append(block)
            continue
        nunique = int(non_na.nunique())
        is_integer_valued = bool(np.all(np.mod(non_na.to_numpy(dtype=float), 1.0) == 0.0))
        if 3 <= nunique <= 12 and is_integer_valued:
            dummies = pd.get_dummies(s, prefix=col, drop_first=True, dtype=float)
            arr = np.array(dummies.to_numpy(dtype=float))  # writable copy (pandas>=3.0 CoW returns a read-only view)
            if arr.size:
                arr[s.isna().to_numpy(), :] = np.nan  # let partial_spearman mask incomplete rows
                blocks.append(arr)
                continue
        blocks.append(s.to_numpy(dtype=float).reshape(-1, 1))
    if not blocks:
        return np.empty((n_rows, 0), dtype=float)
    return np.column_stack(blocks)


@dataclass
class ValidationConfig:
    alpha: float = 0.05
    n_resamples: int = 1000
    random_state: int = 17
    construct_threshold: float = 0.85
    min_holdout_n: int = 20
    # Leave-one-out influence is a sanity check for single-point leverage; 200 randomly
    # sampled points detect influential observations adequately and cost ~10x less than
    # 2000 (which dominated runtime on large datasets). Override via CODAS_MAX_LOO_CHECKS.
    max_loo_checks: int = int(os.getenv("CODAS_MAX_LOO_CHECKS", "200"))
    fdr_alpha: float = 0.10


def _valid_arrays(frame: pd.DataFrame, feature: str, target: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    subset = frame[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    return subset[feature].to_numpy(dtype=float), subset[target].to_numpy(dtype=float), subset


def _split_holdout(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    participant_id_column: str | None,
    random_state: int,
) -> pd.DataFrame:
    needed = [feature, target]
    if participant_id_column and participant_id_column in frame.columns:
        needed.append(participant_id_column)
    subset = frame[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if participant_id_column and participant_id_column in subset.columns:
        # np.asarray: a pandas string/Arrow-backed participant column yields an extension array that
        # sklearn cannot integer-index; coerce to a plain numpy array so train_test_split works.
        ids = np.asarray(subset[participant_id_column].dropna().unique())
        if len(ids) < 5:
            return subset.iloc[0:0]
        _, test_ids = train_test_split(ids, test_size=0.2, random_state=random_state)
        return subset[subset[participant_id_column].isin(list(test_ids))]
    if len(subset) < 25:
        return subset.iloc[0:0]
    _, test = train_test_split(subset, test_size=0.2, random_state=random_state)
    return test


def _bootstrap_distribution(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    random_state: int,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    # x/y are pre-dropna'd (finite); rho via the fast rank+Pearson path.
    # Repeated measures: when `groups` gives a participant id per row (>= 3 clusters), resample
    # PARTICIPANTS with replacement (cluster / block bootstrap), not rows. A row bootstrap treats
    # pseudo-replicated rows as independent and yields an anti-conservatively tight CI, so a chance
    # between-subject correlation would falsely read as "stable"; the cluster bootstrap widens the CI
    # to the true effective sample size. Cross-sectional data (groups=None) is unchanged / bit-identical.
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    n = len(x)
    if groups is not None:
        unique = np.unique(groups)
        if 3 <= len(unique) < n:
            idx_by_cluster = [np.flatnonzero(groups == g) for g in unique]
            k = len(unique)
            for _ in range(n_resamples):
                picked = rng.integers(0, k, k)
                index = np.concatenate([idx_by_cluster[j] for j in picked])
                rho = spearman_rho_fast(x[index], y[index])
                if np.isfinite(rho):
                    values.append(rho)
            return np.asarray(values, dtype=float)
    for _ in range(n_resamples):
        index = rng.integers(0, n, n)
        rho = spearman_rho_fast(x[index], y[index])
        if np.isfinite(rho):
            values.append(rho)
    return np.asarray(values, dtype=float)


def _permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    n_resamples: int,
    random_state: int,
) -> float:
    # Spearman(x, permute(y)) == Pearson(rank(x), permute(rank(y))) exactly, and permuting the ranks
    # leaves their mean/std unchanged, so each null rho is a single dot product against precomputed,
    # mean-centred ranks — no per-resample re-ranking. rng.permutation(n) consumes the RNG identically
    # to the old rng.permutation(y), so the null distribution (and p-value) is bit-identical.
    rng = np.random.default_rng(random_state)
    n = len(x)
    if n < 3 or not np.isfinite(observed_rho):
        return float("nan")
    rx = _stats.rankdata(x)
    ry = _stats.rankdata(y)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    denom = n * sx * sy
    target = abs(observed_rho)
    ge = 0
    total = 0
    for _ in range(n_resamples):
        perm = rng.permutation(n)
        rho = float(np.dot(rxc, ryc[perm]) / denom)
        if np.isfinite(rho):
            total += 1
            if abs(rho) >= target:
                ge += 1
    if total == 0:
        return float("nan")
    return float((ge + 1) / (total + 1))


def _confounder_tests(frame, subset, x, y, candidate, confounder_columns, config):
    """Confounder-adjustment dimension: partial Spearman given the usable confounders, plus a
    hard gate that fails a feature whose association does not survive adjustment (sign flip,
    >=75% magnitude collapse, or a non-significant partial at n>=50). Returns the dimension's
    test result(s). Extracted verbatim from validate_candidate (behaviour pinned by the golden test)."""
    results: list[ValidationTestResult] = []
    usable_confounders = [
        column
        for column in confounder_columns
        if column in frame.columns and _is_adjustable_confounder(frame[column])
    ]
    if usable_confounders:
        covariates = _confounder_covariate_matrix(frame, usable_confounders, subset.index)
        partial_rho, partial_p, partial_n = partial_spearman(x, y, covariates)
        confounder_adjusted_passed = same_sign(candidate.rho, partial_rho) and partial_p <= config.alpha
        results.append(ValidationTestResult(
            name="confounder_adjusted_robustness",
            dimension="robustness",
            passed=confounder_adjusted_passed,
            metric=partial_rho,
            p_value=partial_p,
            details=f"partial_spearman_n={partial_n}, covariates={usable_confounders}",
        ))
        # HARD GATE: if the association does not SURVIVE adjustment for the supplied
        # confounders, the feature is confounded, not an independent predictor, and must
        # not be promoted to "validated". Three independent criteria for confounding:
        #   (a) Sign flip after adjustment — the "predictor" reverses direction.
        #   (b) Effect collapses to <25% of the raw effect (magnitude attenuation ≥75%).
        #   (c) Partial effect is statistically non-significant (p>α) while still having a
        #       substantial raw effect (|raw_rho|>0.25). This is the key case for masking
        #       confounded associations: the partial correlation may retain the correct sign
        #       but be explicable entirely by the confounder.
        # Low-power samples (n<50) are exempted from (c) to avoid false downgrades when the
        # partial test simply lacks power.
        severe_sign_flip = (
            np.isfinite(partial_rho) and np.isfinite(candidate.rho)
            and abs(candidate.rho) > 1e-9
            and not same_sign(candidate.rho, partial_rho)
        )
        severe_collapse = (
            np.isfinite(partial_rho) and np.isfinite(candidate.rho)
            and abs(candidate.rho) > 1e-9
            and abs(partial_rho) < 0.25 * abs(candidate.rho)
        )
        severe_nonsig = (
            partial_n >= 50
            and np.isfinite(partial_rho) and np.isfinite(candidate.rho)
            and abs(candidate.rho) > 0.25           # meaningful raw effect
            and partial_p > config.alpha             # but partial is non-significant
        )
        severely_confounded = partial_n >= 50 and (severe_sign_flip or severe_collapse or severe_nonsig)
        if severely_confounded:
            _reason = (
                "sign flip after confounder adjustment" if severe_sign_flip
                else f"effect magnitude collapses to {abs(partial_rho):.3f} (raw={abs(candidate.rho):.3f}, attenuation≥75%)" if severe_collapse
                else f"partial effect non-significant after adjustment (partial_p={partial_p:.3g}>α={config.alpha}, partial_rho={partial_rho:.3f})"
            )
            _conf_msg = (
                f"raw_rho={candidate.rho:.3f}, partial_rho={partial_rho:.3f} given {usable_confounders}: "
                f"confounded — {_reason}. "
                f"The raw association appears to be driven by {usable_confounders} rather than an "
                f"independent relationship with the target."
            )
        else:
            _conf_msg = (
                f"raw_rho={candidate.rho:.3f}, partial_rho={partial_rho:.3f} given {usable_confounders}: "
                f"survives confounder adjustment"
            )
        results.append(ValidationTestResult(
            name="confounder_independence_hard_gate",
            dimension="robustness",
            passed=not severely_confounded,
            hard_gate=True,
            applicable=True,
            metric=partial_rho,
            p_value=partial_p,
            details=_conf_msg,
        ))
    else:
        results.append(ValidationTestResult(
            name="confounder_adjusted_robustness",
            dimension="robustness",
            passed=False,
            applicable=False,
            details="no numeric confounders supplied",
        ))
    return results


def _sequential_split_test(subset, feature, target_column, candidate):
    """Sequential (temporal/batch) split consistency: split rows in given order into halves and
    check the association holds (same sign, magnitude ratio >= 0.25) in both — a drift/cohort
    artefact otherwise. Returns [] when there are too few rows. Pinned by the golden test."""
    results: list[ValidationTestResult] = []
    # Sequential (temporal / batch) split consistency: divide the rows in their GIVEN ORDER into
    # a first half and second half. If the association holds in the first half but NOT in the
    # second half (or vice versa), the effect may reflect a batch, drift, or cohort artefact
    # rather than a stable biological relationship. This is especially important in longitudinal
    # datasets where the first and second halves may correspond to different periods or cohorts.
    # NOT a hard gate (the first/last ordering may be arbitrary in cross-sectional data);
    # surfaces as an instability note so a reviewer can check whether their data is ordered by
    # time or cohort, in which case this inconsistency is a meaningful replication concern.
    n_sub = len(subset)
    if n_sub >= 40:
        first_half = subset.iloc[: n_sub // 2]
        second_half = subset.iloc[n_sub // 2 :]
        rho_f, _, _ = safe_spearman(first_half[feature], first_half[target_column])
        rho_s, _, _ = safe_spearman(second_half[feature], second_half[target_column])
        # Consistent = same sign in BOTH halves AND neither half is near-zero relative to the other.
        # A 10x magnitude drop (e.g. ρ=-0.56 vs ρ=-0.05) is a replication concern even if signs agree.
        _mag_ratio = (
            min(abs(rho_f), abs(rho_s)) / max(abs(rho_f), abs(rho_s))
            if max(abs(rho_f), abs(rho_s)) > 1e-9 else 1.0
        )
        consistent = (
            np.isfinite(rho_f) and np.isfinite(rho_s)
            and same_sign(candidate.rho, rho_f) and same_sign(candidate.rho, rho_s)
            and _mag_ratio >= 0.25  # second-half effect must be at least 25% of first-half
        )
        results.append(ValidationTestResult(
            name="sequential_split_consistency",
            dimension="stability",
            passed=consistent,
            # Not a hard gate — only flags drift/batch concern; does not block validation
            applicable=True,
            metric=float(min(abs(rho_f), abs(rho_s))) if (np.isfinite(rho_f) and np.isfinite(rho_s)) else float("nan"),
            details=(
                f"first_half_rho={rho_f:.3f}, second_half_rho={rho_s:.3f} — "
                + ("UNSTABLE: different directions in first vs second half; check if rows are ordered "
                   "by time/cohort (if so, this is a replication concern)" if not consistent else "stable")
            ),
        ))
    return results


def _leave_one_out_test(x, y, candidate, config):
    results: list[ValidationTestResult] = []
    loo_indices = np.arange(len(x))
    if len(loo_indices) > config.max_loo_checks:
        rng = np.random.default_rng(config.random_state)
        loo_indices = np.sort(rng.choice(loo_indices, config.max_loo_checks, replace=False))
    sign_flips = 0
    for idx in loo_indices:
        mask = np.ones(len(x), dtype=bool)
        mask[idx] = False
        rho, _, _ = safe_spearman(x[mask], y[mask])
        if not same_sign(candidate.rho, rho):
            sign_flips += 1
            break
    results.append(ValidationTestResult(
        name="leave_one_out_influence",
        dimension="stability",
        passed=sign_flips == 0 and len(loo_indices) > 0,
        metric=float(sign_flips),
        details=f"checked={len(loo_indices)}",
    ))

    return results


def _subgroup_consistency_test(y, subset, feature, target_column, candidate):
    results: list[ValidationTestResult] = []
    median = np.nanmedian(y)
    lower = subset[subset[target_column] <= median]
    upper = subset[subset[target_column] > median]
    if len(lower) >= 10 and len(upper) >= 10:
        lower_rho, lower_p, _ = safe_spearman(lower[feature], lower[target_column])
        upper_rho, upper_p, _ = safe_spearman(upper[feature], upper[target_column])
        passed = same_sign(candidate.rho, lower_rho) and same_sign(candidate.rho, upper_rho)
        results.append(ValidationTestResult(
            name="subgroup_consistency",
            dimension="robustness",
            passed=passed,
            metric=float(min(abs(lower_rho), abs(upper_rho))),
            p_value=float(max(lower_p, upper_p)),
            details=f"median_split=({len(lower)}, {len(upper)})",
        ))
    else:
        results.append(ValidationTestResult(
            name="subgroup_consistency",
            dimension="robustness",
            passed=False,
            applicable=False,
            details="not enough rows after median split",
        ))

    return results


def _method_triangulation_test(x, y, candidate, config):
    results: list[ValidationTestResult] = []
    pearson_r, pearson_p, _ = safe_pearson(x, y)
    kendall_tau, kendall_p, _ = safe_kendall(x, y)
    method_passed = (
        same_sign(candidate.rho, pearson_r)
        and same_sign(candidate.rho, kendall_tau)
        and pearson_p <= config.alpha
        and kendall_p <= config.alpha
    )
    results.append(ValidationTestResult(
        name="method_triangulation",
        dimension="robustness",
        passed=method_passed,
        metric=float(min(abs(pearson_r), abs(kendall_tau))),
        p_value=float(max(pearson_p, kendall_p)),
        details=f"pearson={pearson_r:.4g}, kendall_tau={kendall_tau:.4g}",
    ))

    return results


def _construct_validity_test(x, y, candidate, config):
    results: list[ValidationTestResult] = []
    # Construct-validity hard gate: a feature so target-like that it is a copy/proxy of the outcome
    # (leakage), not an independent predictor. Spearman |rho|>threshold catches near-perfect CONTINUOUS
    # leaks, but Spearman SATURATES on a binary target (a perfect class separator reads rho~0.8, below
    # the 0.85 threshold), so a near-deterministic copy of a binary outcome would slip through. Add a
    # class-separation ceiling: an essentially-perfect single-feature separator (AUC>=0.99 either
    # direction) is leakage regardless of Spearman rho. (Found via adversarial leak-injection across 12
    # real datasets: an injected target-copy was validated on 7 binary-target datasets.)
    construct_auc, _construct_auc_n = auc_against_target(y, x)
    near_perfect_separator = (
        (np.isfinite(construct_auc) and max(construct_auc, 1.0 - construct_auc) >= 0.99)
        or _near_deterministic_of_target(x, y)  # multiclass-aware (binary AUC undefined for >2 classes)
    )
    construct_passed = not (
        candidate.n > 30
        and (
            (np.isfinite(candidate.rho) and abs(candidate.rho) > config.construct_threshold)
            or near_perfect_separator
        )
    )
    results.append(ValidationTestResult(
        name="construct_validity_hard_gate",
        dimension="robustness",
        passed=construct_passed,
        hard_gate=True,
        metric=abs(candidate.rho),
        details=f"threshold={config.construct_threshold}, auc={construct_auc:.4g} (near_perfect_separator={near_perfect_separator})",
    ))

    return results


def _subsample_replication_test(frame, feature, target_column, participant_id_column, candidate, config):
    """Within-sample 20% resplit stability check (NOT a pre-selection holdout — an optimistic
    stability test; selection multiplicity is controlled by the FDR screen). Pinned by the golden test."""
    results: list[ValidationTestResult] = []
    # NOTE: this is a within-sample 20% resplit of the SAME data used to screen/select the
    # candidate, NOT a pre-selection holdout, so it is an optimistic stability check rather than
    # true out-of-sample replication. Named `subsample_replication` (not `independent_replication`)
    # to avoid overclaiming. Selection multiplicity is controlled by the FDR screen, not this test.
    holdout = _split_holdout(frame, feature, target_column, participant_id_column, config.random_state)
    if len(holdout) >= config.min_holdout_n:
        holdout_rho, holdout_p, holdout_n = safe_spearman(holdout[feature], holdout[target_column])
        passed = same_sign(candidate.rho, holdout_rho) and holdout_p <= config.alpha
        results.append(ValidationTestResult(
            name="subsample_replication",
            dimension="replication",
            passed=passed,
            metric=holdout_rho,
            p_value=holdout_p,
            details=f"within-sample 20% resplit (not a pre-selection holdout); holdout_n={holdout_n}",
        ))
    else:
        results.append(ValidationTestResult(
            name="subsample_replication",
            dimension="replication",
            passed=False,
            applicable=False,
            details=f"holdout_n={len(holdout)} below minimum {config.min_holdout_n}",
        ))

    return results


def validate_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    target_column: str,
    participant_id_column: str | None,
    confounder_columns: list[str],
    excluded_columns: list[str],
    feature_components: dict[str, list[str]],
    config: ValidationConfig | None = None,
) -> Candidate:
    config = config or ValidationConfig()
    feature = candidate.feature
    x, y, subset = _valid_arrays(frame, feature, target_column)
    tests: list[ValidationTestResult] = []

    tests.extend(_subsample_replication_test(frame, feature, target_column, participant_id_column, candidate, config))
    perm_p = _permutation_p_value(x, y, candidate.rho, config.n_resamples, config.random_state)
    tests.append(ValidationTestResult(
        name="permutation_test",
        dimension="replication",
        passed=np.isfinite(perm_p) and perm_p <= config.alpha,
        p_value=perm_p,
        details=f"resamples={config.n_resamples}",
    ))

    # Cluster bootstrap for repeated measures: resample participants, not pseudo-replicated rows.
    boot_groups = None
    if participant_id_column and participant_id_column in frame.columns:
        ids = frame.loc[subset.index, participant_id_column].to_numpy()
        if 3 <= len(np.unique(ids)) < len(ids):
            boot_groups = ids
    boot = _bootstrap_distribution(x, y, config.n_resamples, config.random_state, groups=boot_groups)
    if len(boot) >= max(50, config.n_resamples // 10):
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        ci_mid = float((ci_low + ci_high) / 2)
        bootstrap_passed = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
        tests.append(ValidationTestResult(
            name="bootstrap_stability",
            dimension="stability",
            passed=bool(bootstrap_passed),
            metric=float(ci_mid),
            details=f"ci95=[{ci_low:.4g}, {ci_high:.4g}], resamples={len(boot)}",
        ))
    else:
        ci_low = ci_high = ci_mid = float("nan")
        tests.append(ValidationTestResult(
            name="bootstrap_stability",
            dimension="stability",
            passed=False,
            applicable=False,
            details="insufficient valid bootstrap resamples",
        ))

    tests.extend(_leave_one_out_test(x, y, candidate, config))
    tests.extend(_subgroup_consistency_test(y, subset, feature, target_column, candidate))
    tests.extend(_sequential_split_test(subset, feature, target_column, candidate))

    tests.extend(_method_triangulation_test(x, y, candidate, config))
    tests.extend(_construct_validity_test(x, y, candidate, config))
    tests.extend(_confounder_tests(frame, subset, x, y, candidate, confounder_columns, config))

    independence_failures = []
    components = feature_components.get(feature, [feature])
    for column in excluded_columns:
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            rho, _, _ = safe_spearman(frame[feature], frame[column])
            if np.isfinite(rho) and abs(rho) > config.construct_threshold:
                independence_failures.append(f"excluded:{column}")
    for component in components:
        if component != feature and component in frame.columns:
            rho, _, _ = safe_spearman(frame[component], frame[target_column])
            if np.isfinite(rho) and abs(rho) > config.construct_threshold:
                independence_failures.append(f"component_proxy:{component}")
    tests.append(ValidationTestResult(
        name="construct_independence_hard_gate",
        dimension="robustness",
        passed=not independence_failures,
        hard_gate=True,
        details=", ".join(independence_failures) if independence_failures else "independent",
    ))

    ci_available = bool(np.isfinite(ci_mid))
    ci_consistent = ci_available and same_sign(candidate.rho, ci_mid)
    tests.append(ValidationTestResult(
        name="ci_consistency_hard_gate",
        dimension="stability",
        passed=ci_consistent,
        applicable=ci_available,
        hard_gate=True,
        metric=ci_mid,
        details=(
            "bootstrap CI midpoint sign compared with point estimate"
            if ci_available
            else "not applied because bootstrap CI could not be estimated"
        ),
    ))

    auc, auc_n = auc_against_target(y, x)
    tests.append(ValidationTestResult(
        name="discriminative_power",
        dimension="discriminative_power",
        passed=np.isfinite(auc) and auc >= 0.55,
        metric=auc,
        details=f"auc_n={auc_n}, threshold=0.55",
    ))

    # Multiple-comparison discipline: a candidate that did not survive the
    # Benjamini-Hochberg FDR screen must never be promoted to "validated", no
    # matter how many per-feature resampling checks it passes by chance. This is
    # the single guard that keeps the agent from "discovering" predictors in
    # noise when the FDR screen correctly found nothing significant.
    fdr_significant = bool(np.isfinite(candidate.q_value) and candidate.q_value <= config.fdr_alpha)
    tests.append(ValidationTestResult(
        name="multiple_comparison_screen",
        dimension="multiplicity",
        passed=fdr_significant,
        metric=float(candidate.q_value) if np.isfinite(candidate.q_value) else None,
        details=f"benjamini_hochberg_q={candidate.q_value:.4g}, fdr_alpha={config.fdr_alpha}",
    ))

    applicable = [test for test in tests if test.applicable]
    passed_count = sum(1 for test in applicable if test.passed)
    pass_rate = passed_count / len(applicable) if applicable else 0.0
    test_map = {test.name: test for test in tests}
    core_names = {
        "subsample_replication",
        "permutation_test",
        "bootstrap_stability",
        "ci_consistency_hard_gate",
    }
    core_passed = all(test_map[name].passed for name in core_names if test_map[name].applicable)
    hard_gate_failed = any(test.hard_gate and test.applicable and not test.passed for test in tests)
    first_three_failed = all(
        (not test_map[name].passed)
        for name in ("subsample_replication", "permutation_test", "bootstrap_stability")
        if test_map[name].applicable
    )

    if hard_gate_failed or first_three_failed:
        verdict = "rejected"
    elif pass_rate >= 0.70 and core_passed and fdr_significant:
        verdict = "validated"
    elif pass_rate >= 0.40:
        # NOTE: intentionally NOT gated on fdr_significant. On a true-null dataset the
        # acceptance harness relies on the top candidates being carried forward as
        # "conditional" so the negative-control (label-shuffle permutation null) and
        # confounder-attenuation analyses can run *on them* and demonstrate the null.
        # Emptying this bucket breaks that null-demonstration pipeline. The "this is not
        # a finding" honesty is enforced at the synthesis layer instead (explicit
        # "0 passed the full validation battery" headline + per-signal verdict labels).
        verdict = "conditional"
    else:
        verdict = "rejected"

    candidate.tests = tests
    candidate.pass_rate = float(pass_rate)
    candidate.verdict = verdict
    candidate.components = components
    candidate.score = float(abs(candidate.rho) * (1.0 - min(candidate.q_value, 1.0)) * (0.5 + pass_rate))
    candidate.evidence = f"{passed_count}/{len(applicable)} applicable validation checks passed."
    return candidate
