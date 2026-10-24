"""Scenario S1 plants a marker that lives in how a person's days respond, not in their means: only a phenotype
defined on the day-by-day data finds it, on both shares."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.aou import cohort, daystore, scenarios
from codas.core import phenotype_dsl as dsl
from codas.core.survival import cox_ph


def _hr(frame: pd.DataFrame, x: np.ndarray) -> tuple[float, float]:
    z = (x - np.nanmean(x)) / np.nanstd(x)
    fit = cox_ph(frame[cohort.FOLLOWUP_COLUMN], frame["condition"], np.c_[z, frame["age"], frame["female"]])
    return float(np.exp(fit.coef[0])), float(fit.p[0])


def test_s5_reverse_causation_fails_the_lag_check(tmp_path):
    """S5: a coming diagnosis lowers the feature before it is made (a prodrome), and the feature has no effect
    of its own. The effect is carried by the events of the first year, so it replicates on the holdout but
    vanishes once they are left out: not robust. A real effect of the same size survives the lag."""
    from codas.aou.confirm import confirm, preregister

    rng = np.random.default_rng(55)
    n = 8000
    age = rng.normal(size=n)
    t_event = rng.exponential(1 / (0.25 / 365 * np.exp(0.3 * age)))
    t_censor = rng.uniform(2 * 365, 5 * 365, n)
    early = t_event < 365
    prodrome = rng.normal(size=n) - 1.2 * early                     # the coming diagnosis lowers it
    real = rng.normal(size=n)
    t_real = rng.exponential(1 / (0.25 / 365 * np.exp(0.3 * age - 0.25 * real)))
    frame = pd.DataFrame({"prodrome": prodrome, "real": real, "age": age,
                          "followup_days": np.ceil(np.minimum(t_event, t_censor)), "y": (t_event <= t_censor).astype(int)})
    other = frame.assign(followup_days=np.ceil(np.minimum(t_real, t_censor)), y=(t_real <= t_censor).astype(int))
    for name, table, feature in (("rc", frame, "prodrome"), ("ok", other, "real")):
        table.to_csv(tmp_path / f"{name}.holdout.csv", index=False)
        state = {"target_column": "y", "confounder_columns": ["age"], "excluded_columns": [],
                 "latest_report": {"candidates": [{"feature": feature, "rho": -0.05, "q_value": 0.001,
                                                   "verdict": "validated", "components": [feature]}]},
                 "proposed_features": []}
        prereg = preregister(state, tmp_path, name, survival_time="followup_days", sensitivity_lag_days=365)
        (row,) = confirm(prereg, tmp_path / f"{name}.holdout.csv", tmp_path)["hypotheses"]
        assert row["replicated"] and row["hr_per_sd"] < 1, name
        if feature == "prodrome":
            assert "lag" in row["falsification_failed"] and not row["robust"]
        else:
            assert "lag" in row["falsification_passed"] and row["robust"]


def test_s1_is_found_only_by_a_dynamic_phenotype(tmp_path):
    csv = scenarios.reactivity(tmp_path, n=2000)
    spec = scenarios.reactivity_spec()
    discovery, sealed = pd.read_csv(csv), pd.read_csv(cohort.holdout_path(csv))
    assert list(discovery.columns) == cohort.output_columns(spec) and list(sealed.columns) == cohort.holdout_columns(spec)
    assert cohort.holdout_path(csv).parent.name == "sealed" and 800 < len(discovery) < 1200
    for table, days in ((discovery, pd.read_csv(daystore.days_path(csv))),
                        (sealed, pd.read_csv(daystore.holdout_days_path(csv)))):
        assert set(days["row"]) <= set(range(len(table))) and {"day", "weekday", "hr_rest", "sleep_asleep"} <= set(days)
        hr, p = _hr(table, dsl.compute(days, scenarios.S1_COUPLING).to_numpy())
        assert hr < 0.8 and p < 1e-6                                  # the reactivity, by its coupling
        for column in ("hr_rest_mean", "sleep_asleep_mean", "act_steps_mean"):
            _, p_mean = _hr(table, table[column].to_numpy())
            assert p_mean > 0.001, column                             # no mean shows it
    # The same seed writes the same scenario.
    again = scenarios.reactivity(tmp_path / "again", n=2000)
    pd.testing.assert_frame_equal(pd.read_csv(again), discovery)
