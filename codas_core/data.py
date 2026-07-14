"""Dataset loading, profiling, and feature-matrix construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from .models import DatasetProfile


class InsufficientDataError(ValueError):
    """Raised when a dataset is too sparse/degenerate for a rigorous run.

    This is an expected guardrail outcome (not a bug): the agent refuses to
    fabricate associations on underpowered, constant, or feature-less data.
    Callers should convert it into a clear user-facing boundary message.
    """




@dataclass
class AnalysisFrame:
    frame: pd.DataFrame
    target_column: str
    participant_id_column: str | None
    time_column: str | None
    feature_columns: list[str]
    confounder_columns: list[str]
    excluded_columns: list[str]
    feature_components: dict[str, list[str]]
    audit_log: list[str]


def _detect_delimiter(path: Path) -> str:
    """Detect the field delimiter from the header line (most frequent of , ; tab |). Delimiter
    chars are ASCII, so this is robust to the file's text encoding. Handles European `;` exports,
    TSV, and pipe-delimited files without guessing per-row."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            line = fh.readline()
    except Exception:
        return ","
    counts = {d: line.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _robust_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV the way real users actually upload it: detect the delimiter (`;` European/TSV/
    pipe, not just comma), try common encodings (Excel often exports cp1252/latin-1), handle the
    European comma-decimal convention (`80,5`), and tolerate ragged rows / unbalanced quotes (skip
    bad lines), so a real-world file degrades gracefully instead of crashing. Raises a clear
    InsufficientDataError only when nothing parses at all."""
    explicit_sep = ("sep" in kwargs) or ("delimiter" in kwargs)
    sep = None if explicit_sep else _detect_delimiter(path)

    def _read(enc: str, **extra) -> pd.DataFrame:
        opts = dict(kwargs)
        if sep is not None:
            opts["sep"] = sep
        opts.update(extra)
        return pd.read_csv(path, encoding=enc, **opts)

    def _numeric_cols(df: pd.DataFrame) -> int:
        n = 0
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().mean() > 0.8:
                n += 1
        return n

    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            base = _read(enc)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue  # wrong encoding -> try the next one
        except pd.errors.ParserError as exc:
            last_err = exc
            py_kwargs = {k: v for k, v in kwargs.items() if k != "low_memory"}
            for extra in ({"on_bad_lines": "skip"}, {"on_bad_lines": "skip", "quoting": csv.QUOTE_NONE}):
                try:
                    return pd.read_csv(path, encoding=enc, sep=(sep or ","), engine="python", **py_kwargs, **extra)
                except (pd.errors.ParserError, UnicodeDecodeError) as exc2:
                    last_err = exc2
                    continue
            continue
        # European comma-decimal: for `;`-delimited files, prefer decimal="," if it yields MORE
        # numeric columns (so "80,5" parses to 80.5 instead of staying a string).
        if sep == ";":
            try:
                euro = _read(enc, decimal=",", thousands=".")
                if _numeric_cols(euro) > _numeric_cols(base):
                    return euro
            except Exception:
                pass
        return base
    raise InsufficientDataError(
        f"Could not parse this CSV ({type(last_err).__name__ if last_err else 'unknown error'}: "
        f"{str(last_err)[:140] if last_err else ''}). Check the delimiter, quoting, and encoding "
        "(UTF-8 or Latin-1) — or re-export the file from your tool — and upload again."
    )


# Memory guard: above this file size, read a BOUNDED row sample instead of the whole CSV, so a large
# data export (which on Cloud Run's in-memory tmpfs would otherwise load fully into RAM) cannot OOM
# the 8Gi instance. The cap is by CELLS (≈20M), so wide files read fewer rows. The discovery engine
# further subsamples for screening and discloses it, so analysis on a representative sample stays honest.
_LARGE_FILE_BYTES = 80 * 1024 * 1024
# Keep the in-memory footprint small even for object-heavy files (e.g. data exports with dozens of
# repeated string/demographic columns): the engine subsamples to ~40k rows for screening anyway, so a
# ~3M-cell sample is plenty and keeps peak RSS bounded well under the 8Gi instance (which also holds the
# raw file in tmpfs during processing).
_LARGE_FILE_CELL_CAP = 3_000_000
_LARGE_FILE_MIN_ROWS = 20_000
_LARGE_FILE_CHUNK_CELLS = 600_000     # per-chunk parse footprint cap (rows = this // ncols)


def _bounded_read_rows(path: Path) -> int | None:
    """Row cap for a large file (None = read all). Reads only the header to size the cap by columns."""
    try:
        if path.stat().st_size <= _LARGE_FILE_BYTES:
            return None
        ncols = max(1, _robust_read_csv(path, nrows=1).shape[1])
        return max(_LARGE_FILE_MIN_ROWS, _LARGE_FILE_CELL_CAP // ncols)
    except Exception:
        return None


def _count_data_rows(path: Path) -> int:
    """Fast streaming line count (O(1) memory) minus the header row."""
    with open(path, "rb") as f:
        n = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    return max(1, n - 1)


def _read_csv_systematic_sample(path: Path, cap_rows: int) -> pd.DataFrame:
    """Memory-bounded REPRESENTATIVE read of a large CSV. Streams the file LINE BY LINE (O(1) memory —
    pandas' chunked parser still buffers GBs for a multi-GB file), keeps the header + every stride-th
    data line into a small in-memory buffer, then parses only that. Spans the WHOLE file, so all
    participants are represented even when rows are grouped by subject; peak memory ≈ the sample, not
    the file."""
    import io
    total = _count_data_rows(path)
    stride = max(1, total // max(1, cap_rows))
    if stride <= 1:
        return _robust_read_csv(path, low_memory=False)
    buf = io.StringIO()
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header:
                return _robust_read_csv(path, nrows=cap_rows, low_memory=False)
            buf.write(header)
            for i, line in enumerate(f):       # i=0 is the FIRST data row
                if i % stride == 0:
                    buf.write(line)
    except Exception:
        return _robust_read_csv(path, nrows=cap_rows, low_memory=False)  # fall back to head sample
    buf.seek(0)
    try:
        # on_bad_lines="skip": real data exports are often ragged (a few rows with an extra field).
        return pd.read_csv(buf, low_memory=False, on_bad_lines="skip")
    except Exception:
        return _robust_read_csv(path, nrows=cap_rows, low_memory=False)


def _normalize_string_extension_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce pandas 'string'/Arrow-backed string columns (pandas 2.x ``StringDtype`` /
    ``ArrowStringArray``) to plain numpy ``object`` dtype.

    Why: sklearn's array indexing (``train_test_split``, and ``groups=`` for
    ``GroupKFold``/``StratifiedGroupKFold``) cannot integer-index an Arrow-backed array and raises
    ``"only integer scalar arrays can be converted to a scalar index"``. Any dataset with a string
    participant-id column (the common case) would otherwise crash the validation/CV pipeline. This
    is generic — it inspects dtypes only, makes no dataset-specific assumption, and leaves numeric,
    datetime, and already-object columns untouched."""
    for col in df.columns:
        dt = df[col].dtype
        try:
            is_str_ext = isinstance(dt, pd.StringDtype) or (
                pd.api.types.is_extension_array_dtype(dt) and pd.api.types.is_string_dtype(dt)
            )
        except Exception:
            is_str_ext = False
        if is_str_ext:
            df[col] = df[col].astype(object)
    return df


def read_csv_dataset(path: str | Path, nrows: int | None = None, *, coerce_numeric: bool = True) -> pd.DataFrame:
    """Read a CSV with light schema repair for exported biomedical tables.

    For files above ``_LARGE_FILE_BYTES`` and when the caller did not request an explicit ``nrows``, a
    memory-bounded REPRESENTATIVE (systematic, whole-file) row sample is read so a multi-GB upload
    cannot exhaust the instance while still covering all participants."""
    path = Path(path)
    if nrows is None:
        _cap = _bounded_read_rows(path)
        if _cap is not None:
            df = _read_csv_systematic_sample(path, _cap)
            df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed: 0$")]
            return _normalize_string_extension_dtypes(_coerce_numeric_like_columns(df) if coerce_numeric else df)
    df = _robust_read_csv(path, nrows=nrows, low_memory=False)
    unnamed_fraction = sum(str(column).startswith("Unnamed") for column in df.columns) / max(1, len(df.columns))
    if unnamed_fraction > 0.35:
        probe = _robust_read_csv(path, header=None, nrows=3)
        second_row = probe.iloc[1].dropna().astype(str).str.lower().tolist() if len(probe) > 1 else []
        # General "is the second row actually the header?" test: header cells are names (non-numeric),
        # whereas a real data row contains numbers. No column-name assumptions.
        header_like = bool(second_row) and not any(
            re.fullmatch(r"-?\d+(?:\.\d+)?", str(v).strip()) for v in second_row
        )
        if header_like:
            df = _robust_read_csv(path, header=1, nrows=nrows, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed: 0$")]
    if coerce_numeric:
        df = _coerce_numeric_like_columns(df)
    return _normalize_string_extension_dtypes(df)


def _coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Promote numeric-looking object columns without touching datetime-like text.

    User uploads often contain commas, percent signs, or currency-like prefixes that
    make pandas treat otherwise numeric sensor/lab columns as strings. CoDaS should
    still audit and model those columns, while preserving dates and free text.
    """
    work = df.copy()
    for column in work.columns:
        series = work[column]
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue
        if _looks_like_datetime(series):
            continue
        cleaned = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        observed = cleaned.notna() & (cleaned != "")
        if int(observed.sum()) < 20:
            continue
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if float(numeric.notna().sum() / max(1, observed.sum())) >= 0.8:
            work[column] = numeric
    return work


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False
    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8




def profile_dataframe(
    df: pd.DataFrame,
    target_column: str | None = None,
    participant_id_column: str | None = None,
    time_column: str | None = None,
) -> DatasetProfile:
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    datetime_columns = [column for column in df.columns if _looks_like_datetime(df[column])]
    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]
    # Structural profile only. The engine NEVER infers a target, participant, or time column from
    # column names: suggested_targets is just the numeric columns, and participant/time stay as the
    # caller passed them (or None). Semantic role choice belongs to the caller or the LLM agent.
    suggested_targets = list(numeric_columns)

    missing_fraction = {
        str(column): float(df[column].isna().mean())
        for column in df.columns
    }
    return DatasetProfile(
        rows=int(len(df)),
        columns=int(len(df.columns)),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        missing_fraction=missing_fraction,
        suggested_targets=suggested_targets,
        target_column=target_column,
        participant_id_column=participant_id_column,
        time_column=time_column,
    )


def _add_temporal_features(
    work: pd.DataFrame,
    numeric: list[str],
    participant_id_column: str,
    time_column: str,
    out_index: pd.Index,
) -> pd.DataFrame | None:
    """Per-participant temporal features that summary statistics cannot capture:

      * `<feat>_trendslope`          — OLS slope of the feature vs time (within-person trend; S16).
      * `<feat>_weekend_minus_weekday` — weekend-mean minus weekday-mean (social-jetlag style; S15).
      * `<feat>_night_over_day`      — night (00-06h) over day mean ratio (circadian; S14), only when
                                       the timestamp has sub-daily resolution.

    All are computed vectorised per feature (no per-group Python loop) and bounded by a budget so a
    very wide/long file degrades gracefully to summary stats only.
    """
    parsed = pd.to_datetime(work[time_column], errors="coerce")
    if parsed.notna().mean() < 0.6:
        return None
    n_participants = work[participant_id_column].nunique(dropna=True)
    # Budget: skip the enhancement on very large feature*participant grids (it is additive, not core).
    if not numeric or n_participants < 2 or (len(numeric) * max(1, n_participants)) > 6_000_000:
        return None

    tmp = work[[participant_id_column] + numeric].copy()
    # Continuous time in days-from-first (works for daily and hourly cadence alike).
    t0 = parsed.min()
    tmp["_t"] = (parsed - t0) / pd.Timedelta(days=1)
    tmp["_hour"] = parsed.dt.hour
    tmp["_dow"] = parsed.dt.dayofweek  # 0=Mon .. 6=Sun
    gid = tmp[participant_id_column]

    new_cols: dict[str, pd.Series] = {}

    # (1) Trend slope: cov(t,x)/var(t) within participant — vectorised via centered sums.
    t_centered = tmp["_t"] - gid.map(tmp.groupby(participant_id_column)["_t"].mean())
    t_var = (t_centered ** 2).groupby(gid).sum()
    valid_slope = t_var > 0
    for col in numeric:
        x_centered = tmp[col] - gid.map(tmp.groupby(participant_id_column)[col].mean())
        num = (t_centered * x_centered).groupby(gid).sum()
        slope = (num / t_var.where(valid_slope)).reindex(out_index)
        new_cols[f"{col}_trendslope"] = slope

    # (2) Weekday vs weekend contrast (needs both present across the dataset).
    has_weekend = bool((tmp["_dow"] >= 5).any() and (tmp["_dow"] < 5).any())
    if has_weekend:
        wk_mask = tmp["_dow"] < 5
        for col in numeric:
            wk = tmp.loc[wk_mask].groupby(participant_id_column)[col].mean()
            we = tmp.loc[~wk_mask].groupby(participant_id_column)[col].mean()
            new_cols[f"{col}_weekend_minus_weekday"] = (we - wk).reindex(out_index)

    # (3) Circadian night/day ratio (needs sub-daily resolution: >3 distinct hours).
    has_subdaily = int(tmp["_hour"].nunique(dropna=True)) > 3
    if has_subdaily:
        night_mask = tmp["_hour"] < 6
        for col in numeric:
            night = tmp.loc[night_mask].groupby(participant_id_column)[col].mean()
            day = tmp.loc[~night_mask].groupby(participant_id_column)[col].mean()
            new_cols[f"{col}_night_over_day"] = (night / day.replace(0, np.nan)).reindex(out_index)

    if not new_cols:
        return None
    out = pd.DataFrame(new_cols, index=out_index)
    # Drop columns that are entirely NaN or constant (no information / would just be noise).
    keep = [c for c in out.columns if out[c].notna().sum() >= 3 and out[c].nunique(dropna=True) > 1]
    return out[keep] if keep else None


def _aggregate_longitudinal(
    df: pd.DataFrame,
    target_column: str,
    participant_id_column: str,
    time_column: str | None,
    excluded_columns: set[str],
    confounder_columns: list[str] | None = None,
) -> pd.DataFrame:
    reserved = {target_column, participant_id_column}
    if time_column:
        reserved.add(time_column)
    confounders = [column for column in (confounder_columns or []) if column in df.columns]
    reserved.update(confounders)
    work = df.copy()
    for column in list(work.columns):
        if column in reserved or column in excluded_columns:
            continue
        clock_feature = _clock_hour_feature(work[column], str(column))
        if clock_feature is not None:
            work[f"{column}_clock_hour"] = clock_feature

    numeric = [
        column
        for column in work.columns
        if column not in reserved
        and column not in excluded_columns
        and pd.api.types.is_numeric_dtype(work[column])
    ]
    if not numeric:
        raise ValueError("No numeric sensor columns are available for longitudinal aggregation.")

    aggregations = {column: ["mean", "std", "min", "max", "median"] for column in numeric}
    feature_df = work.groupby(participant_id_column).agg(aggregations)
    feature_df.columns = [f"{name}_{agg}" for name, agg in feature_df.columns]
    # Build all coefficient-of-variation columns in ONE concat. Inserting them one-by-one
    # (feature_df[cv]=...) fragments the already-wide aggregate frame — O(cols^2) and a flood of
    # pandas PerformanceWarnings on a 100+-column data file, which measurably slows feature
    # engineering. Values are identical; only the assembly is vectorised.
    cv_data = {
        f"{column}_cv": feature_df[f"{column}_std"] / feature_df[f"{column}_mean"].replace(0, np.nan).abs()
        for column in numeric
    }
    cv_columns = list(cv_data.keys())
    if cv_data:
        feature_df = pd.concat([feature_df, pd.DataFrame(cv_data, index=feature_df.index)], axis=1)

    # --- Temporal feature engineering (S14/S15/S16) ---
    # Summary stats (mean/std/min/max/median/cv) alone miss predictors whose signal lives in the
    # TIME STRUCTURE: a within-person trend/slope, a weekday-vs-weekend contrast, or a circadian
    # (time-of-day) pattern. When a parseable time column exists, engineer per-participant temporal
    # features so these signals enter screening instead of being invisible. Bounded by a budget and
    # wrapped in try/except so it can never break a discovery (degrades to summary stats only).
    if time_column and time_column in work.columns:
        try:
            # Guard: skip temporal feature engineering for columns that look like outcome
            # proxies or label derivatives. Without this guard, a near-perfect outcome proxy
            # (e.g. daily_mood_selfreport, ρ≈0.999 with the target) would be blocked by the
            # construct-validity gate in raw form — but its temporal derivative (trendslope,
            # ρ≈0.26) would slip through the same gate and get validated, which is circular.
            # Strategy: also exclude columns that are strongly correlated with the target
            # in the aggregate frame (|rho| >= 0.95 at the participant level).
            _temporal_numeric = [col for col in numeric if col not in excluded_columns]
            if _temporal_numeric and target_column in df.columns:
                try:
                    # Block temporal feature engineering for construct-circular raw features.
                    # Compute participant-level aggregate correlation with the target for each
                    # candidate raw feature and skip those with |rho| >= 0.95 (near-identical
                    # to the target → any temporal derivative is equally circular).
                    _target_agg = df.groupby(participant_id_column)[target_column].median()
                    from .statistics import safe_spearman as _ss
                    _construct_blocked: set[str] = set()
                    for _col in list(_temporal_numeric):
                        if _col not in df.columns:
                            continue
                        _col_agg = pd.to_numeric(df[_col], errors="coerce")
                        if _col_agg.notna().sum() < 10:
                            continue
                        _col_part = df.groupby(participant_id_column)[_col].mean()
                        _r, _, _ = _ss(_col_part.reindex(_target_agg.index), _target_agg)
                        import numpy as _np2
                        if _np2.isfinite(_r) and abs(_r) >= 0.95:
                            _construct_blocked.add(_col)
                    if _construct_blocked:
                        _temporal_numeric = [col for col in _temporal_numeric if col not in _construct_blocked]
                except Exception:
                    pass  # guard is best-effort; never fail the core aggregation
            _temporal = _add_temporal_features(
                work, _temporal_numeric, participant_id_column, time_column, feature_df.index,
            )
            if _temporal is not None and not _temporal.empty:
                feature_df = pd.concat([feature_df, _temporal], axis=1)
        except Exception:
            pass  # temporal features are an enhancement; never fail the core aggregation

    target_values = df[target_column].dropna()
    if target_values.nunique(dropna=True) == 2:
        target_df = df.groupby(participant_id_column)[target_column].max().rename(target_column)
        audit_target_note = "max/event aggregation for binary target"
    else:
        target_df = df.groupby(participant_id_column)[target_column].median().rename(target_column)
        audit_target_note = "median aggregation for continuous target"
    confounder_df = None
    if confounders:
        confounder_parts = []
        grouped = work.groupby(participant_id_column)
        for column in confounders:
            if pd.api.types.is_numeric_dtype(df[column]):
                confounder_parts.append(grouped[column].median().rename(column))
            else:
                confounder_parts.append(grouped[column].agg(lambda values: values.dropna().iloc[0] if values.dropna().size else np.nan).rename(column))
        confounder_df = pd.concat(confounder_parts, axis=1) if confounder_parts else None
    merged = feature_df.join(target_df)
    if confounder_df is not None:
        merged = merged.join(confounder_df)
    merged = merged.reset_index()
    merged.attrs["codas_target_aggregation_note"] = audit_target_note
    return merged


def _clock_hour_feature(series: pd.Series, column_name: str) -> pd.Series | None:
    """Extract a time-of-day hour feature from sleep/wake timestamp columns."""
    name = column_name.lower()
    if not any(token in name for token in ("sleep", "bed", "wake", "start", "end", "left", "entered")):
        return None
    if not any(token in name for token in ("time", "datetime", "timestamp")):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return None
    sample = series.dropna().astype(str).head(40)
    if sample.empty:
        return None
    # Suppress pandas' per-call "Could not infer format" UserWarning. On ULTRA-WIDE files
    # (GLOBEM: 5,529 cols, many object "..._sleep_..._starttime_..." columns pass the name gate)
    # this function is called per column, and emitting+formatting thousands of identical warnings to
    # stderr is itself an I/O bottleneck that stalled discovery (>9 min). Parsing stays correct.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() < 0.65:
            return None
        parsed_all = pd.to_datetime(series.astype("string"), errors="coerce")
    return parsed_all.dt.hour + parsed_all.dt.minute / 60.0 + parsed_all.dt.second / 3600.0


def _add_ratio_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_components: dict[str, list[str]],
    max_ratio_features: int,
) -> tuple[list[str], list[str]]:
    added: list[str] = []
    if max_ratio_features <= 0:
        return feature_columns, added

    # Generic, domain-agnostic ratio enumeration: pairwise ratios over the strongest usable numeric
    # features. There are NO hardcoded or named ratios of any kind.
    usable = []
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 20:
            continue
        if (finite.abs() > 1e-9).mean() < 0.95:
            continue
        usable.append(column)

    usable = usable[:12]
    for i, numerator in enumerate(usable):
        for denominator in usable[i + 1 :]:
            if len(added) >= max_ratio_features:
                return feature_columns + added, added
            ratio_name = f"{numerator}_over_{denominator}"
            reverse_name = f"{denominator}_over_{numerator}"
            if ratio_name in frame.columns or reverse_name in frame.columns:
                continue
            den = pd.to_numeric(frame[denominator], errors="coerce").replace(0, np.nan)
            rev_den = pd.to_numeric(frame[numerator], errors="coerce").replace(0, np.nan)
            frame[ratio_name] = pd.to_numeric(frame[numerator], errors="coerce") / den
            frame[reverse_name] = pd.to_numeric(frame[denominator], errors="coerce") / rev_den
            feature_components[ratio_name] = [numerator, denominator]
            feature_components[reverse_name] = [denominator, numerator]
            added.extend([ratio_name, reverse_name])
            if len(added) >= max_ratio_features:
                return feature_columns + added, added
    return feature_columns + added, added


def _add_interaction_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_components: dict[str, list[str]],
    max_interaction_features: int = 12,
) -> list[str]:
    """Multiplicative INTERACTION features z(a)*z(b) for the leading numeric pairs (S5).

    Ratios capture one two-feature relationship; interactions capture another — an effect present only
    when two features are jointly high/low, with neither main effect alone. Standardized/centered so the
    product is a genuine interaction term (both-high & both-low ⇒ positive), not scale-dominated. Has its
    OWN budget so it is never crowded out by the ratio budget. Detects BETWEEN-unit interactions; pure
    within-person/within-row interactions still need a mixed-effects interaction term (documented).
    """
    if max_interaction_features <= 0:
        return []
    # Prioritise the PRIMARY signal carrier of each distinct base feature for pairing: the central-
    # tendency (`_mean`) aggregate, or the raw feature for cross-sectional data. Pairing the means of
    # DISTINCT base features (activity_index_mean × sleep_index_mean) yields the interpretable, likely
    # interactions; pairing a feature with its own std/min/max is redundant. This makes the meaningful
    # interaction reachable within the small budget instead of being crowded out by same-family products.
    def _base_of(col: str) -> str:
        for suf in ("_mean", "_std", "_min", "_max", "_median", "_cv", "_trendslope",
                    "_weekend_minus_weekday", "_night_over_day"):
            if col.endswith(suf):
                return col[: -len(suf)]
        return col
    # Derive interaction candidates from the FULL frame, not the (already univariate-pruned)
    # feature_columns list: an interaction-only predictor has null MAIN effects, so its components
    # are exactly the ones the univariate screen drops — they must still be reachable here.
    frame_cols = [str(c) for c in frame.columns]
    mean_cols = [c for c in frame_cols if c.endswith("_mean") and "_over_" not in c and "_x_" not in c]
    primary = mean_cols if mean_cols else [c for c in feature_columns if "_over_" not in c and "_x_" not in c]
    usable, seen_base = [], set()
    for column in primary:
        if "_over_" in column or "_x_" in column:  # skip already-derived ratio/interaction features
            continue
        base = _base_of(column)
        if base in seen_base:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 20 or finite.nunique() <= 2:
            continue
        usable.append(column)
        seen_base.add(base)
    usable = usable[:8]  # cap pair explosion: 8 distinct base features -> up to 28 pairs, budget-capped
    added: list[str] = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if len(added) >= max_interaction_features:
                return added
            inter_name = f"{a}_x_{b}"
            if inter_name in frame.columns:
                continue
            av = pd.to_numeric(frame[a], errors="coerce")
            bv = pd.to_numeric(frame[b], errors="coerce")
            sa = av.std(ddof=0)
            sb = bv.std(ddof=0)
            if not (sa and sb and np.isfinite(sa) and np.isfinite(sb)):
                continue
            prod = ((av - av.mean()) / sa) * ((bv - bv.mean()) / sb)
            if prod.notna().sum() < 20 or prod.nunique(dropna=True) <= 2:
                continue
            frame[inter_name] = prod
            feature_components[inter_name] = [a, b]
            added.append(inter_name)
    return added


def _validate_and_encode_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, str | None]:
    """Validate the target and encode a binary categorical one to 0/1. Raises InsufficientDataError
    for an absent / constant / high-cardinality-text / multi-class-text target. Returns the
    (possibly target-encoded) frame and an optional encoding note. Extracted from build_analysis_frame."""
    if target_column not in df.columns:
        raise InsufficientDataError(f"Target column '{target_column}' is not present in the dataset.")
    # Constant target = nothing to associate/predict. (Too-few-observations is left
    # to the downstream screening gate, which fires AFTER participant aggregation so
    # legitimate small longitudinal cohorts are not blocked pre-aggregation.)
    _target_numeric = pd.to_numeric(df[target_column], errors="coerce").dropna()
    if len(_target_numeric) > 0 and float(_target_numeric.std(ddof=0) or 0.0) == 0.0:
        raise InsufficientDataError(
            f"Target '{target_column}' has no variance (all observed values are identical); there is nothing to associate or predict."
        )
    # Reject a target that is mostly non-numeric AND high-cardinality (a date,
    # identifier, or free-text column mistakenly chosen as the outcome). A binary
    # or low-cardinality categorical target is allowed and handled downstream.
    # NOTE: compute the numeric fraction over OBSERVED (non-null) values so a
    # heavily-missing but genuinely numeric target is not mistaken for text.
    _observed_target = df[target_column].dropna()
    _binary_target_note: str | None = None
    if len(_observed_target) > 0:
        _numeric_frac = float(pd.to_numeric(_observed_target, errors="coerce").notna().mean())
        _target_unique = int(_observed_target.nunique())
        if _numeric_frac < 0.5:
            # Non-numeric (text/categorical) target. A 2-class target is encoded to 0/1 so the
            # numeric screening/modeling can proceed (classification-style). High-cardinality looks
            # like a date/id/free-text column, and 3-10 unordered text categories (e.g. a multi-level
            # "sex" field) are not a usable single outcome — both are rejected with a clear message.
            # Without this, a low-cardinality text target slipped through and crashed the numeric
            # screening downstream (safe_spearman: "could not convert string to float: 'Male'").
            if _target_unique > 10:
                raise InsufficientDataError(
                    f"Target '{target_column}' is not a usable outcome: its observed values are mostly non-numeric with "
                    f"{_target_unique} distinct values, which looks like a date, identifier, or free-text column rather than a "
                    "non-numeric (categorical) endpoint. Choose a numeric or binary target."
                )
            if _target_unique == 2:
                _cats = list(pd.Categorical(_observed_target).categories)
                df = df.copy()
                _codes = pd.Series(pd.Categorical(df[target_column], categories=_cats).codes, index=df.index)
                df[target_column] = _codes.where(_codes >= 0)
                _binary_target_note = (
                    f"Encoded binary categorical target '{target_column}' as 0/1 ({_cats[0]}=0, {_cats[1]}=1)."
                )
            else:
                raise InsufficientDataError(
                    f"Target '{target_column}' is a non-numeric categorical column with {_target_unique} categories; "
                    "association discovery requires a numeric or binary outcome. Choose a numeric endpoint, or a binary "
                    "target for classification."
                )
    return df, _binary_target_note


def _drop_identifier_like_columns(work: pd.DataFrame, feature_columns: list[str], audit_log: list[str]) -> list[str]:
    """Drop identifier / row-index columns that survived as numeric features — a structural
    all-unique-consecutive-integer detector plus an id/index name match with near-uniqueness.
    Prevents a sequential id on a target-sorted export from being screened as a predictor.
    Appends to audit_log and returns the filtered feature list. Extracted from build_analysis_frame."""
    # Drop trivial identifier / row-index columns that survived as numeric features. A column whose
    # NAME looks like an id/index AND whose values are ~unique per row is a record id, not a
    # predictor. Without this, a sequential id that happens to track the target (sorted exports —
    # the A15 trivial-leakage trap) could be screened and ratio-engineered into a spurious
    # "predictor". Requiring BOTH the name pattern AND near-uniqueness keeps real measurements
    # (e.g. a low-cardinality "device_id" grouping, or any genuinely continuous feature).
    _ID_NAMES = {
        "id", "index", "rowid", "row_id", "row_index", "rowindex", "record_id", "recordid",
        "record_number", "uuid", "guid", "seq", "sequence", "sample_id", "sampleid", "row",
        "row_number", "rownum", "serial", "serial_number",
    }
    _AGG_SUFFIXES = ("_mean", "_median", "_min", "_max", "_std", "_sum", "_count", "_first", "_last", "_iqr", "_range")

    def _is_identifier_like(col: str) -> bool:
        # STRUCTURAL row-index/ID detector (name-independent): a column whose values are all-unique
        # CONSECUTIVE integers (an arange-like 0..n-1 / 1..n in any order) is a row index or record id,
        # never a continuous predictor. Caught regardless of column name — fixes the row-order leakage
        # where a sequential id on a target-SORTED export was screened+validated as a "predictor"
        # (found via injection probe on thyroid_recurrence / obesity_level, which are sorted by target).
        _series0 = work[col].dropna()
        _n0 = len(_series0)
        if _n0 >= 20 and _series0.nunique() == _n0:
            _vals = pd.to_numeric(_series0, errors="coerce").dropna()
            if len(_vals) == _n0 and np.all(np.isfinite(_vals)) and np.allclose(np.mod(_vals, 1.0), 0.0):
                _span = float(_vals.max() - _vals.min())
                if abs(_span - (_n0 - 1)) <= max(1.0, 0.001 * _n0):  # consecutive integer sequence
                    return True
        name = str(col).strip().lower()
        # Longitudinal aggregation renames "record_id" -> "record_id_mean"/"_min"/... so match the
        # base name (suffix stripped) as well as the raw name.
        base = name
        for suffix in _AGG_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        name_hit = any(
            candidate in _ID_NAMES
            or candidate.endswith("_id") or candidate.endswith("_index")
            or candidate.endswith("_uuid") or candidate.endswith("_guid")
            or candidate.startswith("unnamed:")
            for candidate in (name, base)
        )
        if not name_hit:
            return False
        series = work[col].dropna()
        if len(series) < 20 or series.nunique() < 0.95 * len(series):
            return False
        # A record id / row index is INTEGER-valued and near-unique. A continuous physiological "index"
        # (e.g. a fitness or activity index that is a ratio of measurements) is also near-unique but NOT
        # integer-valued, and must not be mistaken for an identifier just because its name ends in
        # "index". Require integer-valued data before dropping on the name match.
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return len(vals) == len(series) and bool(np.all(np.mod(vals.to_numpy(dtype=float), 1.0) == 0.0))

    identifier_like = [column for column in feature_columns if _is_identifier_like(column)]
    if identifier_like:
        feature_columns = [column for column in feature_columns if column not in identifier_like]
        audit_log.append(
            f"Excluded identifier/index-like columns from predictor features (near-unique per row): "
            f"{', '.join(map(str, identifier_like))}."
        )
    return feature_columns


def build_analysis_frame(
    df: pd.DataFrame,
    target_column: str,
    participant_id_column: str | None = None,
    time_column: str | None = None,
    excluded_columns: list[str] | None = None,
    confounder_columns: list[str] | None = None,
    max_ratio_features: int = 24,
) -> AnalysisFrame:
    df, _binary_target_note = _validate_and_encode_target(df, target_column)
    excluded = set(excluded_columns or [])
    confounders = [column for column in (confounder_columns or []) if column in df.columns]
    audit_log: list[str] = []
    if _binary_target_note:
        audit_log.append(_binary_target_note)

    # Outcome/label columns OTHER than the target are construct-circular as predictor candidates
    # (predicting one outcome from another). Block them up front and record the reason — this is the
    # NOTE: no NAME-based outcome/proxy blocking. Construct circularity and label leakage are caught
    # STATISTICALLY by the validation battery (construct-validity / near-determinism hard gates on the
    # actual target's values). A caller who already knows certain columns are alternate outcomes or
    # proxies can pass them in request.excluded_columns.

    work = df.copy()
    # Aggregate to one row per participant when a time axis is present OR when a
    # participant appears in multiple rows (repeated measures). The latter guards
    # against pseudo-replication: treating non-independent rows from the same
    # person as independent observations would inflate effective n and significance.
    _has_duplicate_participants = (
        participant_id_column
        and participant_id_column in work.columns
        and int(work[participant_id_column].nunique(dropna=True)) < len(work)
    )
    # Aggregation is correct when each participant contributes repeated measures of a
    # participant-LEVEL outcome (one value per person, a constant label). But when
    # the TARGET itself varies WITHIN a participant (a stress label that is 0 at baseline
    # and 1 under stress; a per-visit severity score) and there is no time axis to model
    # it longitudinally, collapsing to one row per participant destroys the very signal
    # we want — the target would aggregate to a near-constant. In that case keep the
    # row-level data; within-participant correlation is handled by the mixed-effects model.
    _target_varies_within_participant = False
    if _has_duplicate_participants:
        _target_for_variation = pd.to_numeric(work[target_column], errors="coerce")
        _unique_per_participant = (
            work.assign(_codas_target_var=_target_for_variation)
            .groupby(participant_id_column)["_codas_target_var"]
            .nunique(dropna=True)
        )
        if len(_unique_per_participant):
            _target_varies_within_participant = float((_unique_per_participant >= 2).mean()) >= 0.20
    _should_aggregate = bool(
        participant_id_column
        and participant_id_column in work.columns
        and (time_column or _has_duplicate_participants)
    )
    if _should_aggregate and not time_column and _target_varies_within_participant:
        _should_aggregate = False
        audit_log.append(
            f"Kept row-level repeated measures: target '{target_column}' varies within "
            f"'{participant_id_column}' for a substantial fraction of participants, so participant "
            "aggregation would erase the within-person signal. Within-participant correlation is "
            "addressed by the mixed-effects model instead."
        )
    if _should_aggregate:
        work = _aggregate_longitudinal(work, target_column, participant_id_column, time_column, excluded, confounders)
        target_note = work.attrs.get("codas_target_aggregation_note", "target aggregation")
        if time_column:
            audit_log.append(
                f"Aggregated longitudinal rows by '{participant_id_column}' using mean/std/min/max/median features and {target_note}."
            )
        else:
            audit_log.append(
                f"Aggregated repeated-measures rows (no time column detected) by '{participant_id_column}' to avoid "
                f"pseudo-replication, using mean/std/min/max/median features and {target_note}."
            )
    elif participant_id_column and participant_id_column not in work.columns:
        audit_log.append(f"Participant ID column '{participant_id_column}' was not found and was ignored.")
        participant_id_column = None

    reserved = {target_column}
    if participant_id_column:
        reserved.add(participant_id_column)
    if time_column:
        reserved.add(time_column)
    reserved.update(excluded)
    reserved.update(confounders)

    feature_columns = [
        column
        for column in work.columns
        if column not in reserved
        and pd.api.types.is_numeric_dtype(work[column])
    ]
    feature_columns = [
        column
        for column in feature_columns
        if work[column].notna().sum() >= 20 and work[column].nunique(dropna=True) > 1
    ]

    feature_columns = _drop_identifier_like_columns(work, feature_columns, audit_log)
    feature_components = {column: [column] for column in feature_columns}
    feature_columns, added_ratios = _add_ratio_features(
        work,
        feature_columns,
        feature_components,
        max_ratio_features=max_ratio_features,
    )
    if added_ratios:
        audit_log.append(f"Generated {len(added_ratios)} ratio features before label-aware screening.")

    # Interaction (product) features get their OWN budget so they are never crowded out by ratios.
    try:
        added_interactions = _add_interaction_features(work, feature_columns, feature_components)
        if added_interactions:
            feature_columns = feature_columns + added_interactions
            audit_log.append(f"Generated {len(added_interactions)} interaction (product) features before screening.")
    except Exception:
        pass  # interactions are additive; never fail the frame build

    keep_columns = list(dict.fromkeys(
        [column for column in [participant_id_column, target_column] if column]
        + confounders
        + feature_columns
        + [column for column in excluded if column in work.columns]
    ))
    work = work[keep_columns].replace([np.inf, -np.inf], np.nan)
    return AnalysisFrame(
        frame=work,
        target_column=target_column,
        participant_id_column=participant_id_column,
        time_column=time_column,
        feature_columns=feature_columns,
        confounder_columns=confounders,
        excluded_columns=[column for column in excluded if column in work.columns],
        feature_components=feature_components,
        audit_log=audit_log,
    )
