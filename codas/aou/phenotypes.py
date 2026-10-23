"""Intraday digital phenotypes: per-participant summaries of minute-level wearable series over the
baseline window, computed in BigQuery next to the daily summaries.

Daily means and SDs say how much someone moves or sleeps; the phenotypes here describe how the day is
organised, which daily totals cannot show. Each has a published, disease-agnostic definition:

* minute activity (a table with one row per minute and a count, e.g. steps):
  - ``rel_amplitude``: relative amplitude, (M10 - L5) / (M10 + L5), of the 24-h activity profile, where
    M10 and L5 are the most active 10 h and least active 5 h (non-parametric circadian rhythm analysis;
    Van Someren et al., Chronobiol Int 1999);
  - ``interdaily_stability``: how alike the days are, variance of the 24-h profile over the variance of
    all hourly values (Witting et al., Biol Psychiatry 1990);
  - ``intradaily_variability``: how fragmented the rhythm is, mean squared hour-to-hour change over the
    variance (same source; differences are taken within each day so gaps between days do not count);
  - ``cosinor_rel_amplitude`` and ``acrophase_hour``: amplitude (relative to the mesor) and clock time of
    the peak of a 24-h cosine fitted to the profile (Cornelissen, Theor Biol Med Model 2014);
  - ``l5_onset_hour``: clock hour at which the least active 5 h start, a rest-timing marker, in hours
    after noon (12-36) so night-time onsets either side of midnight stay close;
  - ``peak1_mean`` and ``peak30_mean``: the day's highest one-minute count and the mean of its 30 highest
    minutes (peak cadence; Tudor-Locke et al., Br J Sports Med 2018; del Pozo Cruz et al., JAMA Intern
    Med 2022), averaged over days;
  - ``fragmentation``: active-to-sedentary transition probability, bouts of consecutive active minutes
    per active minute (Di et al., J Gerontol A 2017).
* sleep stages (one row per stage segment: start, duration, stage label, the night it belongs to):
  - ``midpoint_hour``, ``midpoint_sd``, ``onset_sd``: mean and night-to-night SD of sleep midpoint and
    onset, in clock hours after noon so a night crossing midnight stays continuous (sleep regularity;
    Phillips et al., Sci Rep 2017; Lunsford-Avery et al., Sci Rep 2018);
  - ``social_jetlag``: mean midpoint on free nights (ending Saturday or Sunday) minus work nights
    (Wittmann et al., Chronobiol Int 2006);
  - ``wake_bouts`` and ``transitions_per_hour``: wake segments per night and stage changes per hour
    asleep (sleep fragmentation);
  - ``rem_latency``: minutes from sleep onset to the first REM segment.
* minute heart rate (one row per minute: time, beats per minute; plausible values 30-220 only):
  - ``resting_hr``: mean over days of the mean of the day's 30 lowest minutes, a resting heart-rate proxy
    (resting heart rate and mortality/cardiovascular risk: Zhang et al., CMAJ 2016; wearable resting
    heart rate: Quer et al., JAMA Netw Open 2020);
  - ``night_hr``, ``day_hr`` and ``night_dip``: mean heart rate 00:00-05:59 and 10:00-19:59, and
    1 - night/day (nocturnal heart-rate dipping);
  - ``night_masd``: mean absolute difference between consecutive night minutes, a short-term variability
    proxy (minute averages, so not heart-rate variability in the beat-to-beat sense);
  - ``rel_amplitude``: 24-h cosinor amplitude of the hourly profile relative to its mean;
  - ``per_step``: heart-rate rise per step/min over minutes with steps, the slope of heart rate on minute
    steps from a minute-activity source (``steps_from``; a fitness marker, cf. heart rate per step in
    wearable studies).

Only the days the primary wearable counts as valid enter the minute-activity phenotypes, so a day the
device was barely worn does not flatten the rhythm. ``<prefix>_days`` counts the days that contributed,
a wear-adherence column like the daily sources' (adjustment-only, never a candidate).

Column names differ between data releases, so each role lists acceptable names; ``resolve`` picks the
first one the table has. A source marked optional whose columns cannot all be found is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

MINUTE_ACTIVITY = {
    "rel_amplitude": "relative amplitude (M10 - L5)/(M10 + L5) of the 24-h activity profile",
    "interdaily_stability": "interdaily stability: how alike the daily activity profiles are (0-1)",
    "intradaily_variability": "intradaily variability: hour-to-hour fragmentation of the activity rhythm",
    "cosinor_rel_amplitude": "24-h cosinor amplitude relative to the mean (rhythm strength)",
    "acrophase_hour": "clock hour of the fitted 24-h activity peak",
    "l5_onset_hour": "clock hour at which the least active 5 hours begin, hours after noon (24 = midnight)",
    "peak1_mean": "mean over days of the highest one-minute count",
    "peak30_mean": "mean over days of the mean of the 30 highest minutes (peak-30 cadence)",
    "fragmentation": "active-to-sedentary transition probability (activity bouts per active minute)",
}
SLEEP_STAGES = {
    "midpoint_hour": "mean sleep midpoint, clock hours after noon (24 = midnight)",
    "midpoint_sd": "night-to-night SD of the sleep midpoint (hours)",
    "onset_sd": "night-to-night SD of sleep onset (hours)",
    "social_jetlag": "free-night minus work-night sleep midpoint (hours)",
    "wake_bouts": "wake segments per night",
    "transitions_per_hour": "sleep-stage segments per hour asleep (fragmentation)",
    "rem_latency": "minutes from sleep onset to the first REM segment",
}
MINUTE_HEART_RATE = {
    "resting_hr": "resting heart rate proxy: mean over days of the mean of the day's 30 lowest minutes (bpm)",
    "night_hr": "mean heart rate 00:00-05:59 (bpm)",
    "day_hr": "mean heart rate 10:00-19:59 (bpm)",
    "night_dip": "nocturnal heart-rate dip: 1 - night mean / day mean",
    "night_masd": "mean absolute difference of consecutive night minutes (bpm; a variability proxy, not HRV)",
    "rel_amplitude": "24-h cosinor amplitude of the hourly heart-rate profile relative to its mean",
    "per_step": "heart-rate rise per step/min in minutes with steps (slope of heart rate on minute steps)",
}
KINDS = {
    "minute_activity": {"roles": ("time", "value"), "optional_roles": (), "phenotypes": MINUTE_ACTIVITY},
    "sleep_stages": {"roles": ("date", "start", "level", "duration"), "optional_roles": ("main",),
                     "phenotypes": SLEEP_STAGES},
    "minute_heart_rate": {"roles": ("time", "value"), "optional_roles": (), "phenotypes": MINUTE_HEART_RATE},
}


@dataclass
class PhenotypeSource:
    """A minute- or segment-level wearable table and the phenotypes to derive from it."""

    kind: str
    table: Any  # a table name, or acceptable names (first one the CDR has wins)
    prefix: str
    # role -> acceptable column names, first match wins (or a single name).
    columns: dict[str, Any]
    optional: bool = True
    # Stage labels for sleep_stages, matched case-insensitively.
    wake_levels: list[str] = field(default_factory=lambda: ["wake", "awake"])
    rem_levels: list[str] = field(default_factory=lambda: ["rem"])
    # minute_heart_rate: prefix of a minute_activity source to pair minutes with (for per_step).
    steps_from: str = ""

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"phenotype kind must be one of {sorted(KINDS)}, not {self.kind!r}.")
        for value, what in [*((t, "table") for t in self.tables()), (self.prefix, "prefix")]:
            if not _IDENT.match(str(value)):
                raise ValueError(f"Invalid phenotype {what}: {value!r}")
        if self.steps_from and (self.kind != "minute_heart_rate" or not _IDENT.match(self.steps_from)):
            raise ValueError(f"steps_from applies to minute_heart_rate sources and names a phenotype prefix.")
        spec = KINDS[self.kind]
        for role in spec["roles"]:
            if not self.candidates(role):
                raise ValueError(f"phenotype source {self.table!r} needs a column for role {role!r}.")
        for role in self.columns:
            if role not in spec["roles"] + spec["optional_roles"]:
                raise ValueError(f"phenotype source {self.table!r}: unknown role {role!r}.")
            for name in self.candidates(role):
                if not _IDENT.match(name):
                    raise ValueError(f"Invalid column name {name!r} for role {role!r}.")

    def tables(self) -> list[str]:
        return [self.table] if isinstance(self.table, str) else [str(t) for t in self.table]

    def phenotype_names(self) -> list[str]:
        names = list(KINDS[self.kind]["phenotypes"])
        return [n for n in names if n != "per_step" or self.steps_from]

    def candidates(self, role: str) -> list[str]:
        names = self.columns.get(role) or []
        return [names] if isinstance(names, str) else [str(n) for n in names]

    def column(self, role: str) -> str | None:
        names = self.candidates(role)
        return names[0] if names else None

    def output_columns(self) -> list[str]:
        return [f"{self.prefix}_days"] + [f"{self.prefix}_{p}" for p in self.phenotype_names()]

    def needed(self) -> list[str]:
        return ["person_id"] + [c for c in (self.column(r) for r in self.columns) if c]


def describe(source: PhenotypeSource) -> list[tuple[str, str]]:
    """(column, one-line definition) for each phenotype the source yields."""
    defs = KINDS[source.kind]["phenotypes"]
    return [(f"{source.prefix}_{name}", defs[name]) for name in source.phenotype_names()]


def resolve(source: PhenotypeSource, present: set[tuple[str, str]]) -> tuple[PhenotypeSource | None, str | None]:
    """The source fixed to the first acceptable table the CDR has, and each role to the first acceptable
    column that table has; or (None, reason)."""
    spec = KINDS[source.kind]
    tables = [t for t in source.tables() if (t, "person_id") in present]
    if not tables:
        return None, f"{' / '.join(source.tables())}: table or person_id column not found"
    table = tables[0]
    fixed: dict[str, str] = {}
    for role in spec["roles"] + spec["optional_roles"]:
        found = next((c for c in source.candidates(role) if (table, c) in present), None)
        if found is None and role in spec["roles"]:
            return None, f"{table}: no column for {role} (looked for {', '.join(source.candidates(role))})"
        if found is not None:
            fixed[role] = found
    return PhenotypeSource(kind=source.kind, table=table, prefix=source.prefix, columns=fixed,
                           optional=source.optional, wake_levels=list(source.wake_levels),
                           rem_levels=list(source.rem_levels), steps_from=source.steps_from), None


def _window(alias: str, column: str) -> str:
    return (f"DATE({alias}.{column}) >= DATE_SUB(ie.index_date, INTERVAL @window_days DAY)\n"
            f"    AND DATE({alias}.{column}) < ie.index_date")


def minute_activity_ctes(i: int, source: PhenotypeSource, table: str) -> list[str]:
    """CTEs ending in ``q{i}`` (person_id plus the source's output columns). Needs ``valid_days``."""
    q, time, value = f"q{i}", source.column("time"), source.column("value")
    p = source.prefix
    two_pi = "(2 * ACOS(-1))"
    return [
        # Minutes with a value, on the primary wearable's valid days in the window.
        f"{q}_min AS (\n"
        f"  SELECT s.person_id, DATE(s.{time}) AS d, EXTRACT(HOUR FROM s.{time}) AS h,\n"
        f"    EXTRACT(HOUR FROM s.{time}) * 60 + EXTRACT(MINUTE FROM s.{time}) AS minute_of_day,\n"
        f"    CAST(s.{value} AS FLOAT64) AS v\n"
        f"  FROM {table} s\n"
        f"  JOIN valid_days vd ON vd.person_id = s.person_id AND vd.d = DATE(s.{time})\n"
        f"  WHERE s.{value} IS NOT NULL\n"
        ")",
        # Every hour of every contributing day; an hour with no recorded minute counts as zero.
        f"{q}_grid AS (\n"
        "  SELECT g.person_id, g.d, hr AS h, COALESCE(SUM(m.v), 0) AS x\n"
        f"  FROM (SELECT DISTINCT person_id, d FROM {q}_min) g\n"
        "  CROSS JOIN UNNEST(GENERATE_ARRAY(0, 23)) AS hr\n"
        f"  LEFT JOIN {q}_min m ON m.person_id = g.person_id AND m.d = g.d AND m.h = hr\n"
        "  GROUP BY g.person_id, g.d, hr\n"
        ")",
        f"{q}_prof AS (\n  SELECT person_id, h, AVG(x) AS mh FROM {q}_grid GROUP BY person_id, h\n)",
        f"{q}_all AS (\n"
        f"  SELECT person_id, COUNT(DISTINCT d) AS ndays, VAR_POP(x) AS vx FROM {q}_grid GROUP BY person_id\n"
        ")",
        f"{q}_diff AS (\n"
        "  SELECT person_id, AVG(dx * dx) AS msd FROM (\n"
        f"    SELECT person_id, x - LAG(x) OVER (PARTITION BY person_id, d ORDER BY h) AS dx FROM {q}_grid\n"
        "  ) WHERE dx IS NOT NULL GROUP BY person_id\n"
        ")",
        f"{q}_cos AS (\n"
        "  SELECT person_id, AVG(mh) AS mesor, VAR_POP(mh) AS vprof,\n"
        f"    2 * AVG(mh * COS({two_pi} * h / 24)) AS a, 2 * AVG(mh * SIN({two_pi} * h / 24)) AS b\n"
        f"  FROM {q}_prof GROUP BY person_id\n"
        ")",
        # Mean activity in every circular 10-h and 5-h window of the profile, by start hour.
        f"{q}_win AS (\n"
        "  SELECT p.person_id, st AS start_h,\n"
        "    AVG(IF(MOD(p.h - st + 24, 24) < 10, p.mh, NULL)) AS m10,\n"
        "    AVG(IF(MOD(p.h - st + 24, 24) < 5, p.mh, NULL)) AS l5\n"
        f"  FROM {q}_prof p CROSS JOIN UNNEST(GENERATE_ARRAY(0, 23)) AS st\n"
        "  GROUP BY p.person_id, st\n"
        ")",
        f"{q}_ml AS (\n  SELECT person_id, MAX(m10) AS m10, MIN(l5) AS l5 FROM {q}_win GROUP BY person_id\n)",
        f"{q}_l5 AS (\n"
        "  SELECT person_id, start_h AS l5_onset FROM (\n"
        "    SELECT person_id, start_h, ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY l5, start_h) AS rn\n"
        f"    FROM {q}_win\n"
        "  ) WHERE rn = 1\n"
        ")",
        f"{q}_peak AS (\n"
        "  SELECT person_id, AVG(peak1) AS peak1, AVG(peak30) AS peak30 FROM (\n"
        "    SELECT person_id, d, MAX(v) AS peak1, SUM(IF(rn <= 30, v, 0)) / 30 AS peak30 FROM (\n"
        "      SELECT person_id, d, v, ROW_NUMBER() OVER (PARTITION BY person_id, d ORDER BY v DESC) AS rn\n"
        f"      FROM {q}_min\n"
        "    ) GROUP BY person_id, d\n"
        "  ) GROUP BY person_id\n"
        ")",
        # A bout starts at an active minute whose previous active minute is not the minute before.
        f"{q}_bouts AS (\n"
        "  SELECT person_id, SUM(IF(prev IS NULL OR minute_of_day - prev > 1, 1, 0)) AS bouts, COUNT(*) AS active\n"
        "  FROM (\n"
        "    SELECT person_id, minute_of_day,\n"
        "      LAG(minute_of_day) OVER (PARTITION BY person_id, d ORDER BY minute_of_day) AS prev\n"
        f"    FROM {q}_min WHERE v > 0\n"
        "  ) GROUP BY person_id\n"
        ")",
        f"{q} AS (\n"
        f"  SELECT al.person_id, al.ndays AS {p}_days,\n"
        f"    SAFE_DIVIDE(ml.m10 - ml.l5, ml.m10 + ml.l5) AS {p}_rel_amplitude,\n"
        f"    SAFE_DIVIDE(c.vprof, al.vx) AS {p}_interdaily_stability,\n"
        f"    SAFE_DIVIDE(df.msd, al.vx) AS {p}_intradaily_variability,\n"
        f"    SAFE_DIVIDE(SQRT(c.a * c.a + c.b * c.b), c.mesor) AS {p}_cosinor_rel_amplitude,\n"
        f"    ATAN2(c.b, c.a) * 24 / {two_pi} - 24 * FLOOR(ATAN2(c.b, c.a) / {two_pi}) AS {p}_acrophase_hour,\n"
        f"    MOD(l5.l5_onset + 12, 24) + 12 AS {p}_l5_onset_hour,\n"
        f"    pk.peak1 AS {p}_peak1_mean, pk.peak30 AS {p}_peak30_mean,\n"
        f"    SAFE_DIVIDE(bt.bouts, bt.active) AS {p}_fragmentation\n"
        f"  FROM {q}_all al\n"
        f"  JOIN {q}_cos c ON c.person_id = al.person_id\n"
        f"  JOIN {q}_ml ml ON ml.person_id = al.person_id\n"
        f"  JOIN {q}_l5 l5 ON l5.person_id = al.person_id\n"
        f"  LEFT JOIN {q}_diff df ON df.person_id = al.person_id\n"
        f"  LEFT JOIN {q}_peak pk ON pk.person_id = al.person_id\n"
        f"  LEFT JOIN {q}_bouts bt ON bt.person_id = al.person_id\n"
        ")",
    ]


def _clock_after_noon(expr: str) -> str:
    """Clock time of a DATETIME as hours after noon (12 = noon, 24 = midnight, 30 = 6 am)."""
    return f"(MOD(EXTRACT(HOUR FROM {expr}) + 12, 24) + EXTRACT(MINUTE FROM {expr}) / 60 + 12)"


def sleep_stage_ctes(j: int, source: PhenotypeSource, table: str) -> tuple[list[str], list[tuple[str, str, Any]]]:
    """CTEs ending in ``z{j}``, and their parameters (the wake and REM stage labels)."""
    z, p = f"z{j}", source.prefix
    date, start, level, duration = (source.column(r) for r in ("date", "start", "level", "duration"))
    main = source.column("main")
    params = [(f"{z}_wake", "ARRAY<STRING>", [s.lower() for s in source.wake_levels]),
              (f"{z}_rem", "ARRAY<STRING>", [s.lower() for s in source.rem_levels])]
    only_main = f"\n  WHERE LOWER(CAST(s.{main} AS STRING)) = 'true'" if main else ""
    onset = _clock_after_noon("onset")
    midpoint = _clock_after_noon("DATETIME_ADD(onset, INTERVAL DIV(DATETIME_DIFF(wake_time, onset, SECOND), 2) SECOND)")
    # Free nights end on Saturday or Sunday; 1900-01-07 was a Sunday (a dialect-neutral weekday).
    free = "MOD(DATE_DIFF(d, DATE '1900-01-07', DAY), 7) IN (0, 6)"
    ctes = [
        f"{z}_seg AS (\n"
        f"  SELECT s.person_id, DATE(s.{date}) AS d, DATETIME(s.{start}) AS st,\n"
        f"    CAST(s.{duration} AS FLOAT64) AS dur, LOWER(CAST(s.{level} AS STRING)) AS lvl\n"
        f"  FROM {table} s\n"
        f"  JOIN index_events ie ON ie.person_id = s.person_id\n"
        f"    AND {_window('s', date)}{only_main}\n"
        ")",
        f"{z}_night AS (\n"
        "  SELECT person_id, d, MIN(st) AS onset,\n"
        "    MAX(DATETIME_ADD(st, INTERVAL CAST(ROUND(dur * 60) AS INT64) SECOND)) AS wake_time,\n"
        f"    COUNT(*) AS segments, SUM(IF(lvl IN UNNEST(@{z}_wake), 1, 0)) AS wake_bouts,\n"
        f"    SUM(IF(lvl IN UNNEST(@{z}_wake), 0, dur)) AS asleep_min,\n"
        f"    MIN(IF(lvl IN UNNEST(@{z}_rem), st, NULL)) AS first_rem\n"
        f"  FROM {z}_seg GROUP BY person_id, d\n"
        ")",
        f"{z}_clock AS (\n"
        f"  SELECT person_id, d, segments, wake_bouts, asleep_min, {onset} AS onset_h, {midpoint} AS mid_h,\n"
        "    DATETIME_DIFF(first_rem, onset, MINUTE) AS rem_latency\n"
        f"  FROM {z}_night\n"
        ")",
        f"{z} AS (\n"
        f"  SELECT person_id, COUNT(*) AS {p}_days,\n"
        f"    AVG(mid_h) AS {p}_midpoint_hour, STDDEV_SAMP(mid_h) AS {p}_midpoint_sd,\n"
        f"    STDDEV_SAMP(onset_h) AS {p}_onset_sd,\n"
        f"    AVG(IF({free}, mid_h, NULL)) - AVG(IF({free}, NULL, mid_h)) AS {p}_social_jetlag,\n"
        f"    AVG(wake_bouts) AS {p}_wake_bouts,\n"
        f"    AVG(SAFE_DIVIDE(segments, asleep_min / 60)) AS {p}_transitions_per_hour,\n"
        f"    AVG(rem_latency) AS {p}_rem_latency\n"
        f"  FROM {z}_clock GROUP BY person_id\n"
        ")",
    ]
    return ctes, params


def minute_heart_rate_ctes(i: int, source: PhenotypeSource, table: str, steps_cte: str | None = None) -> list[str]:
    """CTEs ending in ``h{i}`` (person_id plus the source's output columns). Needs ``valid_days``; with
    ``steps_cte`` (a minute-activity source's ``q<k>_min``: person_id, d, minute_of_day, v) adds per_step."""
    h, time, value, p = f"h{i}", source.column("time"), source.column("value"), source.prefix
    two_pi = "(2 * ACOS(-1))"
    ctes = [
        f"{h}_min AS (\n"
        f"  SELECT s.person_id, DATE(s.{time}) AS d, EXTRACT(HOUR FROM s.{time}) AS h,\n"
        f"    EXTRACT(HOUR FROM s.{time}) * 60 + EXTRACT(MINUTE FROM s.{time}) AS minute_of_day,\n"
        f"    CAST(s.{value} AS FLOAT64) AS hr\n"
        f"  FROM {table} s\n"
        f"  JOIN valid_days vd ON vd.person_id = s.person_id AND vd.d = DATE(s.{time})\n"
        f"  WHERE s.{value} BETWEEN 30 AND 220\n"
        ")",
        f"{h}_low AS (\n"
        "  SELECT person_id, d, AVG(hr) AS low30 FROM (\n"
        "    SELECT person_id, d, hr, ROW_NUMBER() OVER (PARTITION BY person_id, d ORDER BY hr) AS rn\n"
        f"    FROM {h}_min\n"
        "  ) WHERE rn <= 30 GROUP BY person_id, d\n"
        ")",
        f"{h}_part AS (\n"
        "  SELECT person_id, AVG(IF(h < 6, hr, NULL)) AS night, AVG(IF(h >= 10 AND h < 20, hr, NULL)) AS dayhr,\n"
        "    COUNT(DISTINCT d) AS ndays\n"
        f"  FROM {h}_min GROUP BY person_id\n"
        ")",
        f"{h}_masd AS (\n"
        "  SELECT person_id, AVG(ABS(hr - prev_hr)) AS masd FROM (\n"
        "    SELECT person_id, h, hr, minute_of_day,\n"
        "      LAG(hr) OVER (PARTITION BY person_id, d ORDER BY minute_of_day) AS prev_hr,\n"
        "      LAG(minute_of_day) OVER (PARTITION BY person_id, d ORDER BY minute_of_day) AS prev_minute\n"
        f"    FROM {h}_min\n"
        "  ) WHERE h < 6 AND minute_of_day - prev_minute = 1 GROUP BY person_id\n"
        ")",
        f"{h}_prof AS (\n  SELECT person_id, h, AVG(hr) AS mh FROM {h}_min GROUP BY person_id, h\n)",
        f"{h}_cos AS (\n"
        "  SELECT person_id, AVG(mh) AS mesor,\n"
        f"    2 * AVG(mh * COS({two_pi} * h / 24)) AS a, 2 * AVG(mh * SIN({two_pi} * h / 24)) AS b\n"
        f"  FROM {h}_prof GROUP BY person_id HAVING COUNT(*) >= 12\n"
        ")",
    ]
    step_col, step_join = "", ""
    if steps_cte and source.steps_from:
        ctes.append(
            f"{h}_step AS (\n"
            "  SELECT m.person_id, SAFE_DIVIDE(COVAR_POP(m.hr, s.v), VAR_POP(s.v)) AS per_step\n"
            f"  FROM {h}_min m JOIN {steps_cte} s\n"
            "    ON s.person_id = m.person_id AND s.d = m.d AND s.minute_of_day = m.minute_of_day\n"
            "  WHERE s.v > 0 GROUP BY m.person_id\n"
            ")")
        step_col = f",\n    st.per_step AS {p}_per_step"
        step_join = f"\n  LEFT JOIN {h}_step st ON st.person_id = pt.person_id"
    ctes.append(
        f"{h} AS (\n"
        f"  SELECT pt.person_id, pt.ndays AS {p}_days, lw.resting AS {p}_resting_hr,\n"
        f"    pt.night AS {p}_night_hr, pt.dayhr AS {p}_day_hr, 1 - SAFE_DIVIDE(pt.night, pt.dayhr) AS {p}_night_dip,\n"
        f"    md.masd AS {p}_night_masd,\n"
        f"    SAFE_DIVIDE(SQRT(c.a * c.a + c.b * c.b), c.mesor) AS {p}_rel_amplitude{step_col}\n"
        f"  FROM {h}_part pt\n"
        f"  JOIN (SELECT person_id, AVG(low30) AS resting FROM {h}_low GROUP BY person_id) lw ON lw.person_id = pt.person_id\n"
        f"  LEFT JOIN {h}_masd md ON md.person_id = pt.person_id\n"
        f"  LEFT JOIN {h}_cos c ON c.person_id = pt.person_id{step_join}\n"
        ")")
    return ctes
