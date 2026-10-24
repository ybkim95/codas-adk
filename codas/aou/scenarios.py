"""Synthetic scenarios with a planted truth, for tests and for the dress rehearsal that runs the real models
before any All of Us query. No row describes a real person.

S1, latent reactivity. Each person's resting heart rate rises the day after a short night by their own amount
(their reactivity, in beats per minute per hour of sleep lost), and the hazard of a condition rises with that
reactivity. A person's mean resting heart rate, mean sleep and mean steps do not depend on it, so no per-person
mean or SD can find it; the within-person coupling of resting heart rate on the previous night's sleep
(``codas.core.phenotype_dsl``, summary ``coupling`` with lag 1, ``S1_COUPLING``) can. The scenario is written
in the layout an extraction leaves (the discovery cohort table and its days beside it, the sealed table and its
days in the sealed folder), for ``runner.run_spec(csv=...)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from codas.aou import cohort, daystore
from codas.aou.cohort import CohortSpec, OutcomeSpec, WearableSource

S1_STEM = "scenario_reactivity"
S1_COUPLING = {"name": "rest_after_sleep", "signal": "hr_rest", "summary": "coupling",
               "couple": {"with": "sleep_asleep", "lag": 1}, "min_days": 14}
S1_LABEL = "a condition driven by heart-rate reactivity to short sleep (synthetic)"


def reactivity_spec(stem: str = S1_STEM, days: int = 60) -> CohortSpec:
    """The cohort spec the S1 tables follow: three daily sources, a time-to-event outcome, a sealed half."""
    spec = CohortSpec(
        name=stem, outcome=OutcomeSpec(kind="condition", concept_ids=[990002], name="condition"),
        wearables=[WearableSource(table="daily_activity", date_column="date", columns=["steps"], prefix="act"),
                   WearableSource(table="daily_sleep", date_column="date", columns=["asleep"], prefix="sleep"),
                   WearableSource(table="daily_heart_rate", date_column="date", columns=["rest"], prefix="hr")],
        window_days=days, min_wear_days=14, design="incident", lag_days=0, holdout_fraction=0.5,
        notes="synthetic scenario S1 (codas.aou.scenarios)")
    spec.validate()
    return spec


def reactivity(out: str | Path, *, n: int = 3000, days: int = 60, seed: int = 20251024, stem: str = S1_STEM,
               hazard_ratio: float = 1.8) -> Path:
    """Write S1 to ``out`` and return the discovery cohort table's path. ``hazard_ratio`` is per SD of
    reactivity; about one participant in eight has the event."""
    spec = reactivity_spec(stem, days)
    rng = np.random.default_rng(seed)
    age = np.round(rng.normal(55, 12, n)).clip(20, 90)
    female = rng.integers(0, 2, n)
    fitness = rng.normal(size=n)
    react = rng.gamma(4.0, 0.15, n)                 # bpm per hour of sleep lost, mean 0.6
    sleep_mean = rng.normal(420, 30, n)
    base = rng.normal(64, 5, n) - 2 * fitness
    worn = rng.random((n, days)) < rng.uniform(0.75, 1.0, n)[:, None]
    asleep = rng.normal(sleep_mean[:, None], 60, (n, days)).clip(120, 720)
    before = np.concatenate([sleep_mean[:, None], asleep[:, :-1]], axis=1)  # the night before each day
    rest = base[:, None] + react[:, None] * (sleep_mean[:, None] - before) / 60 + rng.normal(0, 1.5, (n, days))
    steps = rng.normal(7000 + 2000 * fitness[:, None], 1800, (n, days)).clip(0)
    z = (react - react.mean()) / react.std()
    hazard = 0.05 / 365 * np.exp(np.log(hazard_ratio) * z + 0.03 * (age - 55))
    t_event, t_censor = rng.exponential(1 / hazard), rng.uniform(365, 4 * 365, n)

    signals = {"act_steps": steps, "sleep_asleep": asleep, "hr_rest": rest}
    masked = {k: np.where(worn, v, np.nan) for k, v in signals.items()}
    table = pd.DataFrame({"condition": (t_event <= t_censor).astype(int),
                          cohort.FOLLOWUP_COLUMN: np.ceil(np.minimum(t_event, t_censor)), "age": age, "female": female})
    for prefix, column in (("act", "steps"), ("sleep", "asleep"), ("hr", "rest")):
        values = masked[f"{prefix}_{column}"]
        table[f"{prefix}_days"] = worn.sum(axis=1)
        table[f"{prefix}_{column}_mean"] = np.nanmean(values, axis=1)
        table[f"{prefix}_{column}_sd"] = np.nanstd(values, axis=1, ddof=1)
    table[cohort.ERA_COLUMN] = rng.choice([2019, 2020, 2021, 2022], n)
    sealed = rng.random(n) < spec.holdout_fraction
    day = np.arange(-days, 0)

    def day_table(people: np.ndarray) -> pd.DataFrame:
        r, d = np.nonzero(worn[people])
        frame = pd.DataFrame({"row": r, "day": day[d], "weekday": (day[d] % 7) + 1})
        for name, values in signals.items():
            frame[name] = np.round(values[people][r, d], 3)
        return frame

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{stem}.csv"
    discovery, holdout = np.flatnonzero(~sealed), np.flatnonzero(sealed)
    table.iloc[discovery][cohort.output_columns(spec)].to_csv(csv, index=False)
    cohort.holdout_path(csv).parent.mkdir(parents=True, exist_ok=True)
    table.iloc[holdout][cohort.holdout_columns(spec)].to_csv(cohort.holdout_path(csv), index=False)
    day_table(discovery).to_csv(daystore.days_path(csv), index=False, compression="gzip")
    day_table(holdout).to_csv(daystore.holdout_days_path(csv), index=False, compression="gzip")
    return csv
