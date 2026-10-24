"""The data atlas: coverage over calendar time, who has which data, wear against the health record, repeated
laboratory values during wear and the survey questions answered, from the discovery share only and checked
against the synthetic CDR's rows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codas.aou import cohort, data_atlas, phenome
from codas.aou import feasibility as feas
from codas.aou.cohort import CohortSpec, holdout_bucket

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"
CATALOG = ROOT / "scripts" / "aou" / "outcomes_catalog.json"


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    import duckdb

    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    query = duckdb_query(con)
    monkeypatch.setattr(cohort, "_query", query)
    base = replace(CohortSpec.from_json(SPEC), design="incident", window_days=60, lag_days=7, min_wear_days=20,
                   holdout_fraction=0.5)
    base, _ = feas.verify_covariates(base, SYNTHETIC_CDR, None)
    present = query(None, *cohort.schema_check_sql(base, SYNTHETIC_CDR))
    base, _ = cohort.resolve_sources(base, list(zip(present["table_name"], present["column_name"])))
    outcomes = feas.resolve_codes(feas.load_catalog(CATALOG), SYNTHETIC_CDR, None)
    problems = feas.verify_catalog(outcomes, SYNTHETIC_CDR, None)
    labs = [o for o in outcomes if problems.get(o.name) is None and o.kind == "measurement"]
    return {"con": con, "cdr": SYNTHETIC_CDR, "base": base, "labs": labs, "out": tmp_path}


def _discovery_ids(con, spec) -> set[int]:
    people = [int(p) for (p,) in con.execute("SELECT DISTINCT person_id FROM activity_summary").fetchall()]
    return {p for p in people if holdout_bucket(p, spec.split_salt) >= round(spec.holdout_fraction * 1000)}


def test_the_atlas_reads_only_the_discovery_share_and_matches_the_rows(synthetic):
    con, cdr, base, out = synthetic["con"], synthetic["cdr"], synthetic["base"], synthetic["out"]
    people = phenome.person_table(base, cdr, None, out)
    atlas = data_atlas.build(base, synthetic["labs"], cdr, None, out, people, log=lambda *_: None)
    discovery = _discovery_ids(con, base)
    # Coverage of the primary source by month equals a count over the discovery participants' rows.
    rows = con.execute("SELECT person_id, date FROM activity_summary").df()
    rows = rows[rows.person_id.isin(discovery)]
    expected = rows.assign(month=pd.to_datetime(rows.date).dt.strftime("%Y-%m")).groupby("month").person_id.nunique()
    primary = {m["month"]: m["participants"] for m in atlas["coverage"] if m["source"] == base.wearables[0].prefix}
    assert primary == expected.to_dict()
    assert all(m["mean_days"] is None or m["mean_days"] >= 1 for m in atlas["coverage"])
    # Who has which data: every source of the person table is compared, and a group of 1-20 hides both sizes.
    sources = {r["source"] for r in atlas["selection"]}
    assert set(data_atlas.sources_of(people, base)) == sources
    for r in atlas["selection"]:
        sizes = [r["n_with"], r["n_without"]]
        assert None in sizes or all(v == 0 or v > 20 for v in sizes)
        assert (r["n_with"] or 0) + (r["n_without"] or 0) in (len(people), r["n_with"] or 0, r["n_without"] or 0)
    # Repeated lab values during wear: hand-counted for one catalog lab.
    lab = next(o for o in synthetic["labs"] if o.name == "hba1c")
    span = rows.assign(date=pd.to_datetime(rows.date)).groupby("person_id").date.agg(["min", "max"])
    m = con.execute("SELECT person_id, measurement_concept_id, measurement_date, value_as_number FROM measurement").df()
    m = m[m.measurement_concept_id.isin(lab.concept_ids) & m.value_as_number.between(lab.value_min, lab.value_max)]
    m = m.join(span, on="person_id", how="inner")
    m = m[(pd.to_datetime(m.measurement_date) >= m["min"]) & (pd.to_datetime(m.measurement_date) <= m["max"])]
    counts = m.groupby("person_id").size()
    found = next(r for r in atlas["repeat_labs"] if r["outcome"] == "hba1c")
    assert found["participants"] == int((counts >= 2).sum())
    # Survey questions: the synthetic survey's question is among those answered, by discovery participants only.
    survey = {q["question"]: q["participants"] for q in atlas["survey"]}
    answered = con.execute("SELECT DISTINCT person_id FROM observation WHERE observation_source_concept_id = 1585857"
                           ).df().person_id
    assert survey.get(1585857) == int(answered.isin(discovery).sum()) > 0
    # The copy beside the reports is masked; the Markdown says what it shows.
    shown = json.loads((out / data_atlas.FILE).read_text())
    assert not [v for q in shown["survey"] + shown["coverage"] for k, v in q.items()
                if k == "participants" and isinstance(v, int) and 1 <= v <= 20]
    text = data_atlas.markdown(atlas)
    assert "Coverage over calendar time" in text and "Who has which data" in text


def test_selection_flags_a_source_held_by_different_people():
    rng = np.random.default_rng(1)
    n = 2000
    age = rng.normal(55, 12, n)
    people = pd.DataFrame({"age": age, "female": rng.integers(0, 2, n), "act_steps_mean": rng.normal(size=n),
                           "hr_rest_mean": np.where(age > 55, rng.normal(size=n), np.nan)})
    spec = CohortSpec.from_json(SPEC)
    spec = replace(spec, wearables=[replace(spec.wearables[0], prefix="act"),
                                    replace(spec.wearables[0], prefix="hr")], phenotypes=[], covariates=[])
    rows = {r["source"]: r for r in data_atlas.selection(people, spec)}
    assert rows["hr"]["notable"] == ["age"] and rows["hr"]["traits"]["age"] > 1
    assert rows["act"]["n_without"] == 0 and rows["act"]["traits"] == {}


def test_wear_and_visits_by_fifth():
    rng = np.random.default_rng(2)
    n = 1000
    days = rng.integers(20, 180, n)
    people = pd.DataFrame({"act_days": days, "ehr_visits": rng.poisson(days / 30)})
    spec = CohortSpec.from_json(SPEC)
    spec = replace(spec, wearables=[replace(spec.wearables[0], prefix="act")])
    out = data_atlas.wear_and_visits(people, spec)
    assert out["rho"] > 0.5 and out["mean_visits_by_fifth_of_wear"] == sorted(out["mean_visits_by_fifth_of_wear"])
