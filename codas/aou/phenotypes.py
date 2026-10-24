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
  With ``night_minutes`` ([low, high] minutes asleep) only nights in that range count, e.g. main sleep of
  3-16 h as in the All of Us wearable studies.
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
    wearable studies);
  - ``sleep_hr`` and ``sleep_dip``: mean heart rate within the counted main-sleep nights of a sleep-stage
    source (``sleep_from``: from sleep onset to the end of the last segment), and 1 - sleep/day. Unlike
    ``night_hr`` (clock hours 00:00-05:59) this follows when each person actually slept;
  - ``sedentary_hr``: mean heart rate in awake, inactive minutes: outside every main-sleep period, with no
    steps recorded (needs both ``steps_from`` and ``sleep_from``), a daytime resting level that sleep does
    not dominate the way it dominates ``resting_hr``.
  With ``min_minutes_per_day`` only days with that many heart-rate minutes count (600 = 10 h of data, the
  usual wear-time rule): a day worn for an hour would otherwise give a "resting" value from that hour.

Only the days the primary wearable counts as valid enter the minute-activity phenotypes, so a day the
device was barely worn does not flatten the rhythm. ``<prefix>_days`` counts the days that contributed,
a wear-adherence column like the daily sources' (adjustment-only, never a candidate). With ``min_days`` a
person with fewer contributing days (nights) keeps that count but has the source's phenotypes missing: a
rhythm or variability estimated from a handful of days is mostly noise.

Column names differ between data releases, so each role lists acceptable names; ``resolve`` picks the
first one the table has. A source marked optional whose columns cannot all be found is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
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
    "sleep_hr": "mean heart rate during the main-sleep periods of the sleep segments (bpm)",
    "sleep_dip": "heart-rate dip during sleep: 1 - main-sleep mean / day mean (10:00-19:59)",
    "sedentary_hr": "mean heart rate in awake minutes without steps (outside main sleep, no steps recorded; bpm)",
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
    # minute_heart_rate: prefix of a sleep_stages source whose main-sleep periods give sleep_hr and sleep_dip.
    sleep_from: str = ""
    # minute_heart_rate: a day counts only with at least this many heart-rate minutes (600 = 10 h of data).
    min_minutes_per_day: int | None = None
    # sleep_stages: [low, high] minutes asleep for a night to count; either may be null.
    night_minutes: list[float | None] | None = None
    # Any kind: with fewer contributing days (nights) than this, the person's phenotypes are missing.
    min_days: int | None = None

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"phenotype kind must be one of {sorted(KINDS)}, not {self.kind!r}.")
        for value, what in [*((t, "table") for t in self.tables()), (self.prefix, "prefix")]:
            if not _IDENT.match(str(value)):
                raise ValueError(f"Invalid phenotype {what}: {value!r}")
        for link in (self.steps_from, self.sleep_from):
            if link and (self.kind != "minute_heart_rate" or not _IDENT.match(link)):
                raise ValueError("steps_from and sleep_from apply to minute_heart_rate sources and name a "
                                 "phenotype prefix.")
        if self.min_minutes_per_day is not None and (self.kind != "minute_heart_rate"
                                                     or not whole(self.min_minutes_per_day, 1, 1440)):
            raise ValueError("min_minutes_per_day applies to minute_heart_rate sources: whole minutes, 1-1440.")
        if self.night_minutes is not None and (self.kind != "sleep_stages" or not bounds(self.night_minutes)):
            raise ValueError("night_minutes applies to sleep_stages sources: [low, high] minutes asleep, "
                             "either may be null.")
        if self.min_days is not None and not whole(self.min_days, 1, 100_000):
            raise ValueError("min_days must be a whole number of days, at least 1.")
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
        needs = {"per_step": self.steps_from, "sleep_hr": self.sleep_from, "sleep_dip": self.sleep_from,
                 "sedentary_hr": self.steps_from and self.sleep_from}
        return [n for n in KINDS[self.kind]["phenotypes"] if needs.get(n, True)]

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


def whole(value: Any, low: int, high: int) -> bool:
    """``value`` is an int (not a bool) in [low, high]."""
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def bounds(value: Any) -> bool:
    """``value`` is [low, high] of numbers or nulls, with low <= high when both are given."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in value):
        return False
    return None in value or value[0] <= value[1]


def range_text(limits: Any, unit: str = "") -> str:
    """[low, high] in words: "180-960 min", "at least 180 min", "at most 960 min"."""
    low, high = limits
    unit = f" {unit}" if unit else ""
    if low is not None and high is not None:
        return f"{low:g}-{high:g}{unit}"
    return f"at least {low:g}{unit}" if low is not None else f"at most {high:g}{unit}"


def quality_rules(source: PhenotypeSource) -> list[str]:
    """The source's quality rules in words (for the goal text and the report)."""
    rules = []
    if source.min_minutes_per_day:
        rules.append(f"{source.prefix}: only days with at least {source.min_minutes_per_day} heart-rate minutes "
                     f"({source.min_minutes_per_day / 60:g} h of data) count")
    if source.night_minutes is not None:
        rules.append(f"{source.prefix}: only main-sleep nights with {range_text(source.night_minutes, 'min')} "
                     "asleep count")
    if source.min_days:
        rules.append(f"{source.prefix}: phenotypes need at least {source.min_days} contributing days; with fewer "
                     "they are missing")
    return rules


def select_columns(alias: str, source: PhenotypeSource) -> list[str]:
    """The source's output columns as SELECT items from its CTE ``alias``. With ``min_days`` a person with
    fewer contributing days keeps the count but has every phenotype missing."""
    days = f"{alias}.{source.prefix}_days"
    items = [days]
    for name in source.phenotype_names():
        column = f"{source.prefix}_{name}"
        items.append(f"IF({days} >= {int(source.min_days)}, {alias}.{column}, NULL) AS {column}"
                     if source.min_days else f"{alias}.{column}")
    return items


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
    # replace() carries every other setting over, so a new field can never be dropped here.
    return replace(source, table=table, columns=fixed), None


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
    having = []
    for bound, op, value in zip(("low", "high"), (">=", "<="), source.night_minutes or (None, None)):
        if value is not None:  # a night counts only with its minutes asleep in range
            having.append(f"SUM(IF(lvl IN UNNEST(@{z}_wake), 0, dur)) {op} @{z}_night_{bound}")
            params.append((f"{z}_night_{bound}", "FLOAT64", float(value)))
    night_rule = ("\n  HAVING " + " AND ".join(having)) if having else ""
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
        # Every main-sleep period, counted or not (heart rate needs to know when people were in bed).
        f"{z}_period AS (\n"
        "  SELECT person_id, d, MIN(st) AS onset,\n"
        "    MAX(DATETIME_ADD(st, INTERVAL CAST(ROUND(dur * 60) AS INT64) SECOND)) AS wake_time\n"
        f"  FROM {z}_seg GROUP BY person_id, d\n"
        ")",
        f"{z}_night AS (\n"
        "  SELECT person_id, d, MIN(st) AS onset,\n"
        "    MAX(DATETIME_ADD(st, INTERVAL CAST(ROUND(dur * 60) AS INT64) SECOND)) AS wake_time,\n"
        f"    COUNT(*) AS segments, SUM(IF(lvl IN UNNEST(@{z}_wake), 1, 0)) AS wake_bouts,\n"
        f"    SUM(IF(lvl IN UNNEST(@{z}_wake), 0, dur)) AS asleep_min,\n"
        f"    MIN(IF(lvl IN UNNEST(@{z}_rem), st, NULL)) AS first_rem\n"
        f"  FROM {z}_seg GROUP BY person_id, d{night_rule}\n"
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


def minute_heart_rate_ctes(i: int, source: PhenotypeSource, table: str, steps_cte: str | None = None,
                           sleep: str | None = None) -> list[str]:
    """CTEs ending in ``h{i}`` (person_id plus the source's output columns). Needs ``valid_days``; with
    ``steps_cte`` (a minute-activity source's ``q<k>_min``: person_id, d, minute_of_day, v) adds per_step;
    with ``sleep`` (a sleep-stage source's alias ``z<k>``: its ``_period`` CTE holds every main-sleep period,
    its ``_night`` CTE the counted nights) sleep_hr and sleep_dip; with both, sedentary_hr. With
    ``min_minutes_per_day`` only days with that many heart-rate minutes count."""
    h, time, value, p = f"h{i}", source.column("time"), source.column("value"), source.prefix
    two_pi = "(2 * ACOS(-1))"
    minutes = f"{h}_raw" if source.min_minutes_per_day else f"{h}_min"
    ctes = [
        f"{minutes} AS (\n"
        f"  SELECT s.person_id, DATE(s.{time}) AS d, EXTRACT(HOUR FROM s.{time}) AS h,\n"
        f"    EXTRACT(HOUR FROM s.{time}) * 60 + EXTRACT(MINUTE FROM s.{time}) AS minute_of_day,\n"
        f"    CAST(s.{value} AS FLOAT64) AS hr\n"
        f"  FROM {table} s\n"
        f"  JOIN valid_days vd ON vd.person_id = s.person_id AND vd.d = DATE(s.{time})\n"
        f"  WHERE s.{value} BETWEEN 30 AND 220\n"
        ")",
    ]
    if source.min_minutes_per_day:
        ctes += [
            f"{h}_worn AS (\n"
            f"  SELECT person_id, d FROM {minutes} GROUP BY person_id, d\n"
            f"  HAVING COUNT(DISTINCT minute_of_day) >= {int(source.min_minutes_per_day)}\n"
            ")",
            f"{h}_min AS (\n"
            f"  SELECT r.* FROM {minutes} r JOIN {h}_worn w ON w.person_id = r.person_id AND w.d = r.d\n"
            ")",
        ]
    ctes += [
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
    sleep_col, sleep_join = "", ""
    if sleep and source.sleep_from:
        # Each main-sleep period [onset, wake time) meets the minutes of every date it touches (usually two);
        # a period whose night is not counted (too short or too long) still marks its minutes as in bed.
        stamp = "DATETIME_ADD(DATETIME(m.d), INTERVAL CAST(m.minute_of_day AS INT64) MINUTE)"
        period = (f"  FROM {sleep}_period p LEFT JOIN {sleep}_night n ON n.person_id = p.person_id AND n.d = p.d\n")
        ctes += [
            f"{h}_win AS (\n"
            "  SELECT p.person_id, DATE(p.onset) AS d, p.onset, p.wake_time, n.person_id IS NOT NULL AS counted\n"
            + period +
            "  UNION ALL\n"
            "  SELECT p.person_id, DATE(p.wake_time) AS d, p.onset, p.wake_time, n.person_id IS NOT NULL AS counted\n"
            + period +
            "  WHERE DATE(p.wake_time) > DATE(p.onset)\n"
            ")",
            f"{h}_inbed AS (\n"
            "  SELECT m.person_id, m.d, m.minute_of_day, m.hr, w.counted\n"
            f"  FROM {h}_min m JOIN {h}_win w ON w.person_id = m.person_id AND w.d = m.d\n"
            f"  WHERE {stamp} >= w.onset AND {stamp} < w.wake_time\n"
            ")",
            f"{h}_sleep AS (\n  SELECT person_id, AVG(hr) AS sleep_hr FROM {h}_inbed WHERE counted GROUP BY person_id\n)",
        ]
        sleep_col = f",\n    sl.sleep_hr AS {p}_sleep_hr, 1 - SAFE_DIVIDE(sl.sleep_hr, pt.dayhr) AS {p}_sleep_dip"
        sleep_join = f"\n  LEFT JOIN {h}_sleep sl ON sl.person_id = pt.person_id"
        if steps_cte and source.steps_from:  # awake (outside every main-sleep period) and no steps that minute
            ctes.append(
                f"{h}_sed AS (\n"
                "  SELECT m.person_id, AVG(m.hr) AS sedentary_hr\n"
                f"  FROM {h}_min m\n"
                f"  LEFT JOIN (SELECT DISTINCT person_id, d, minute_of_day FROM {steps_cte} WHERE v > 0) s\n"
                "    ON s.person_id = m.person_id AND s.d = m.d AND s.minute_of_day = m.minute_of_day\n"
                f"  LEFT JOIN (SELECT DISTINCT person_id, d, minute_of_day FROM {h}_inbed) b\n"
                "    ON b.person_id = m.person_id AND b.d = m.d AND b.minute_of_day = m.minute_of_day\n"
                "  WHERE s.person_id IS NULL AND b.person_id IS NULL\n"
                "  GROUP BY m.person_id\n"
                ")")
            sleep_col += f",\n    sd.sedentary_hr AS {p}_sedentary_hr"
            sleep_join += f"\n  LEFT JOIN {h}_sed sd ON sd.person_id = pt.person_id"
    ctes.append(
        f"{h} AS (\n"
        f"  SELECT pt.person_id, pt.ndays AS {p}_days, lw.resting AS {p}_resting_hr,\n"
        f"    pt.night AS {p}_night_hr, pt.dayhr AS {p}_day_hr, 1 - SAFE_DIVIDE(pt.night, pt.dayhr) AS {p}_night_dip,\n"
        f"    md.masd AS {p}_night_masd,\n"
        f"    SAFE_DIVIDE(SQRT(c.a * c.a + c.b * c.b), c.mesor) AS {p}_rel_amplitude{step_col}{sleep_col}\n"
        f"  FROM {h}_part pt\n"
        f"  JOIN (SELECT person_id, AVG(low30) AS resting FROM {h}_low GROUP BY person_id) lw ON lw.person_id = pt.person_id\n"
        f"  LEFT JOIN {h}_masd md ON md.person_id = pt.person_id\n"
        f"  LEFT JOIN {h}_cos c ON c.person_id = pt.person_id{step_join}{sleep_join}\n"
        ")")
    return ctes
