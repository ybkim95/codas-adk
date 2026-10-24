"""A feature followed over time: the counting-process Cox fit, the time-varying exposure, the course before a
diagnosis, the month store's SQL against the synthetic CDR, and the holdout's use of them."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codas.core import trajectory
from codas.core.survival import cox_ph

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"


def test_a_split_follow_up_gives_the_unsplit_fit():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=(n, 2))
    t = np.ceil(rng.exponential(1 / np.exp(0.5 * x[:, 0] - 0.3 * x[:, 1])) * 100)
    c = np.ceil(rng.uniform(10, 300, n))
    time, event = np.minimum(t, c), (t <= c).astype(float)
    whole = cox_ph(time, event, x)
    cut = np.floor(time / 3)
    ok = cut > 0
    split = cox_ph(np.r_[np.where(ok, cut, time), time[ok]], np.r_[np.where(ok, 0, event), event[ok]],
                   np.r_[x, x[ok]], entry=np.r_[np.zeros(n), cut[ok]])
    assert np.allclose(whole.coef, split.coef, atol=1e-10) and np.allclose(whole.se, split.se, atol=1e-10)
    # Late entry leaves a row out of the risk sets before it enters.
    late = cox_ph(time, event, x, entry=np.where(np.arange(n) % 2, time / 2, 0))
    assert late.converged and not np.allclose(late.coef, whole.coef)


def _drifting(seed=1, n=3000, blocks=40, effect=0.6):
    """Each person's value drifts block to block; the hazard in a block follows the value six blocks before."""
    rng = np.random.default_rng(seed)
    level = rng.normal(size=n)
    walk = level[:, None] + np.cumsum(rng.normal(0, 0.35, (n, blocks)), axis=1)
    rows, event_at = [], np.full(n, np.nan)
    for i in range(n):
        for b in range(6, blocks):
            if rng.random() < 0.004 * np.exp(effect * walk[i, b - 6]):
                event_at[i] = b * 30 + 15
                break
    end = np.where(np.isfinite(event_at), event_at, blocks * 30.0)
    blocks_frame = pd.DataFrame({"row": np.repeat(np.arange(n), blocks), "block": np.tile(np.arange(blocks), n),
                                 "v": walk.ravel()})
    people = pd.DataFrame({"row": np.arange(n), "start": 180.0, "end": end, "event": np.isfinite(event_at).astype(int),
                           "baseline": walk[:, :6].mean(axis=1)})
    return people, blocks_frame


def test_the_time_varying_exposure_recovers_an_effect_that_follows_the_current_value():
    people, blocks = _drifting()
    iv = trajectory.intervals(people, blocks, "v", block_days=30, lag=6, window=1)
    assert (iv["stop"] > iv["entry"]).all() and iv.groupby("row")["event"].sum().max() == 1
    tv = cox_ph(iv["stop"], iv["event"], iv[["exposure"]].to_numpy(), entry=iv["entry"])
    assert 0.45 < tv.coef[0] < 0.75                 # the planted 0.6 per unit of the value six blocks before


def test_the_course_before_a_diagnosis_tells_a_prodrome_from_a_stable_difference():
    rng = np.random.default_rng(3)
    n, blocks = 4000, 60
    age, female = rng.normal(55, 10, n), rng.integers(0, 2, n)
    event = rng.random(n) < 0.15
    end = np.where(event, rng.integers(40, 60, n) * 30 + 10, blocks * 30.0)

    def frame(shift_fn):
        v = rng.normal(0, 1, (n, blocks))
        for i in np.flatnonzero(event):
            ref = int(end[i] // 30)
            v[i] += shift_fn(np.arange(blocks) - ref)
        return pd.DataFrame({"row": np.repeat(np.arange(n), blocks), "block": np.tile(np.arange(blocks), n),
                             "v": v.ravel()})

    people = pd.DataFrame({"row": np.arange(n), "event": event.astype(int), "end": end, "female": female, "age": age})
    windows = [(-36, -24), (-24, -12), (-12, 0)]
    prodrome = trajectory.matched_trajectory(people, frame(lambda r: np.where(r >= -12, -0.8, 0.0)), "v",
                                             block_days=30, windows=windows, sd=1.0, min_cases=20)
    assert prodrome["onset"] == "prodromal"
    assert prodrome["windows"][2]["difference_sd"] < -0.5 and abs(prodrome["windows"][0]["difference_sd"]) < 0.2
    stable = trajectory.matched_trajectory(people, frame(lambda r: np.full(r.shape, 0.5)), "v", block_days=30,
                                           windows=windows, sd=1.0, min_cases=20)
    assert stable["onset"] == "stable"
    none = trajectory.matched_trajectory(people, frame(lambda r: np.zeros(r.shape)), "v", block_days=30,
                                         windows=windows, sd=1.0, min_cases=20)
    assert none["onset"] == "none"
    few = trajectory.matched_trajectory(people.assign(event=0).assign(event=lambda d: (d.row < 15).astype(int)),
                                        frame(lambda r: np.zeros(r.shape)), "v", block_days=30, windows=windows,
                                        sd=1.0, min_cases=20)
    assert few["onset"] == "not estimated" and all(w.get("status") == "too few cases" for w in few["windows"])


def test_onset_classes():
    w = lambda lo, hi: {"ci95": [lo, hi]}  # noqa: E731
    assert trajectory.onset([w(-0.1, 0.1), w(0.1, 0.4), w(0.3, 0.6)]) == "prodromal"
    assert trajectory.onset([w(0.1, 0.3), w(0.1, 0.4), w(0.3, 0.6)]) == "stable"
    assert trajectory.onset([w(-0.1, 0.1), w(-0.2, 0.1)]) == "none"
    assert trajectory.onset([w(0.1, 0.3), w(-0.1, 0.1), w(-0.6, -0.3)]) == "mixed"
    assert trajectory.onset([]) == "not estimated"


def test_the_month_store_sql_matches_the_rows(monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    import duckdb

    from codas.aou import cohort, monthstore
    from codas.aou.cohort import CohortSpec
    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    spec = replace(CohortSpec.from_json(SPEC), design="incident", window_days=60, month_store={})
    present = duckdb_query(con)(None, *cohort.schema_check_sql(spec, SYNTHETIC_CDR))
    spec, _ = cohort.resolve_sources(spec, list(zip(present["table_name"], present["column_name"])))
    store = duckdb_query(con)(None, *monthstore.build_monthstore_sql(spec, SYNTHETIC_CDR))
    primary = spec.wearables[0]
    rows = con.execute(f"SELECT * FROM {primary.table}").df()
    lo, hi = spec.valid_day.get("min"), spec.valid_day.get("max")
    rows = rows[rows[spec.valid_day["column"]].between(lo, hi)]
    rows["date"] = pd.to_datetime(rows[primary.date_column])
    index = rows.groupby("person_id")["date"].min() + pd.Timedelta(days=spec.window_days)
    rows["block"] = np.floor((rows["date"] - rows["person_id"].map(index)).dt.days / 30).astype(int)
    column = primary.columns[0]
    expected = rows.groupby(["person_id", "block"])[column].mean()
    got = store.set_index(["person_id", "block"])[f"{primary.prefix}_{column}"]
    assert set(got.index) == set(expected.index)
    assert np.allclose(got.loc[expected.index].to_numpy(), expected.to_numpy())
    assert (store["block"] < 0).any()               # blocks before the index are kept, floored
    assert monthstore.column_of(f"{primary.prefix}_{column}_mean", monthstore.block_columns(spec)) == \
        f"{primary.prefix}_{column}"
    assert monthstore.column_of(f"{primary.prefix}_{column}_sd", monthstore.block_columns(spec)) is None


def test_the_holdout_follows_a_registered_feature_over_time(tmp_path):
    """A pre-registered daily mean is followed on the sealed share: as a time-varying exposure and before each
    diagnosis; both reach the markdown and the scorecard."""
    from codas.aou import scorecard
    from codas.aou.confirm import confirm, confirmation_markdown, preregister

    people, blocks = _drifting(seed=5, n=2500)
    people2, blocks2 = _drifting(seed=6, n=2500)
    rng = np.random.default_rng(7)

    def table(p):
        return pd.DataFrame({"act_v_mean": p["baseline"], "y": p["event"], "followup_days": p["end"] - 180,
                             "age": rng.normal(55, 10, len(p)), "female": rng.integers(0, 2, len(p))})

    table(people).to_csv(tmp_path / "s.csv", index=False)
    table(people2).to_csv(tmp_path / "s.holdout.csv", index=False)
    blocks2.rename(columns={"v": "act_v"}).assign(days=20).to_csv(tmp_path / "s.months.holdout.csv.gz", index=False)
    state = {"target_column": "y", "confounder_columns": ["age"], "excluded_columns": [],
             "latest_report": {"candidates": [{"feature": "act_v_mean", "rho": 0.2, "q_value": 1e-9,
                                               "verdict": "validated", "components": ["act_v_mean"]}]}}
    prereg = preregister(state, tmp_path, "s", survival_time="followup_days", discovery_csv=tmp_path / "s.csv",
                         followed={"block_days": 30, "origin_days": 180})
    assert prereg["followed"]["lag_blocks"] == 6 and prereg["followed"]["windows"][0] == [-36, -24]
    result = confirm(prereg, tmp_path / "s.holdout.csv", tmp_path, months_csv=tmp_path / "s.months.holdout.csv.gz")
    (row,) = result["hypotheses"]
    tv = row["time_varying"]
    assert tv["hr_per_sd"] > 1 and tv["ci95"][0] > 1 and tv["same_direction"] and tv["intervals"] > len(people2)
    assert row["trajectory"]["windows"] and "onset" in row["trajectory"]
    assert "Followed over time" in confirmation_markdown(result)
    cells = scorecard.coverage(result, True)[0]["cells"]
    assert cells["time_varying"][0] == "yes" and cells["trajectory"][0] in ("yes", "no")
    # Without the month store nothing is followed, and the scorecard says so.
    plain = confirm(prereg, tmp_path / "s.holdout.csv", tmp_path)
    assert "time_varying" not in plain["hypotheses"][0]
    assert scorecard.coverage(plain, True)[0]["cells"]["time_varying"][0] == "no"
