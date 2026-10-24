"""The phenome-wide scan: its tests agree with the exact fits, each outcome's cohort is the extraction's, only the
discovery share is read, planted pairs are found, and outcomes are selected by their hits."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codas.aou import cohort, phenome
from codas.aou import feasibility as feas
from codas.aou.cohort import CohortSpec, build_cohort_sql, holdout_bucket
from codas.core.statistics import partial_spearman
from codas.core.survival import breslow_fit, cox_ph, score_tests

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"
CATALOG = ROOT / "scripts" / "aou" / "outcomes_catalog.json"


def _survival(rng, n, rate, effect, missing_old=0.0):
    """Age and sex drive the hazard; x tracks age and adds ``effect`` per SD; x is missing more often in the old."""
    age, sex = rng.normal(size=n), rng.integers(0, 2, n).astype(float)
    x = 0.4 * age + rng.normal(size=n)
    x[rng.random(n) < 0.15 + missing_old * (age > 0)] = np.nan
    lin = 0.5 * age + 0.2 * sex + effect * np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x))
    t_event, t_censor = rng.exponential(1 / (rate * np.exp(lin))), rng.uniform(0.5, 3, n)
    time = np.ceil(np.minimum(t_event, t_censor) * 365)
    order = np.argsort(-time, kind="mergesort")
    return time[order], (t_event <= t_censor)[order], np.c_[age, sex][order], x[order]


@pytest.mark.parametrize("rate,effect", [(0.005, 0.0), (0.02, 0.15), (0.08, -0.1), (0.08, 0.3)])
def test_score_tests_match_the_cox_fit(rate, effect):
    """One base fit on every row, then a score test per column: its z and one-step log hazard ratio match the
    exact Cox fit on the rows where the column is observed, although those rows are older on average."""
    time, event, z, x = _survival(np.random.default_rng(1), 20000, rate, effect, missing_old=0.2)
    (test,) = score_tests(time, event, z, x, breslow_fit(time, event, z))
    seen = np.isfinite(x)
    exact = cox_ph(time[seen], event[seen], np.c_[(x[seen] - x[seen].mean()) / x[seen].std(), z[seen]])
    wald = exact.coef[0] / exact.se[0]
    assert abs(test["log_hr"] - exact.coef[0]) < 0.005
    assert abs(test["z"] - wald) < 0.02 + 0.005 * abs(wald)  # a score and a Wald statistic part only at large effects
    assert test["n"] == int(seen.sum()) and test["events"] == int(event[seen].sum())


def test_the_base_fit_and_the_test_behave():
    time, event, z, x = _survival(np.random.default_rng(2), 20000, 0.02, 0.2)
    beta = breslow_fit(time, event, z)
    assert np.allclose(beta, cox_ph(time, event, z).coef, atol=2e-3)  # few ties: Breslow and Efron agree
    tests = score_tests(time, event, z, np.c_[x, 3 * x - 2, np.ones_like(x)], beta)
    assert abs(tests[0]["z"] - tests[1]["z"]) < 1e-9          # a rescaled column is the same test
    assert math.isnan(tests[2]["z"])                            # a constant column cannot be tested
    # Under the null, about 5% of columns reach p < 0.05.
    rng = np.random.default_rng(3)
    null = score_tests(time, event, z, rng.normal(size=(len(time), 400)), beta)
    assert 0.02 < np.mean([t["p"] < 0.05 for t in null]) < 0.09


def test_partial_spearman_for_many_columns_is_the_same_estimator():
    rng = np.random.default_rng(4)
    n = 2000
    cov = np.c_[rng.normal(size=n), rng.integers(0, 2, n)]
    y = cov[:, 0] + rng.normal(size=n)
    x = np.c_[0.3 * y + rng.normal(size=n), rng.normal(size=n), np.round(rng.normal(size=n))]
    x[rng.random(n) < 0.3, 0] = np.nan
    for j, (rho, p, m) in enumerate(phenome.partial_spearman_many(x, y, cov)):
        one = partial_spearman(x[:, j], y, cov)
        assert np.allclose([rho, p], one[:2], atol=1e-12) and m == one[2]


def test_a_hit_needs_every_condition():
    pair = {"z": 4.0, "q": 0.001, "magnitude": "small", "half_z": [2.5, 3.0], "t1_z": 3.0, "t1_p": 0.003}
    assert phenome.is_hit(pair)
    for change in ({"q": 0.2}, {"magnitude": "negligible"}, {"half_z": [2.5, 1.0]}, {"half_z": [2.5, -3.0]},
                   {"t1_z": -3.0}, {"t1_p": 0.2}, {"half_z": [2.5]}, {"z": float("nan")}):
        assert not phenome.is_hit({**pair, **change}), change


def test_specificity_separates_general_markers_from_specific_hits():
    """A feature that is a hit for many outcomes is a broad marker; one that singles out few is specific; with few
    outcomes screened it is not assessed. Among outcomes with the same new hits, one whose hits are specific
    ranks first."""
    def pairs(feature, outcomes, hit_for):
        return [{"feature": feature, "outcome": f"o{i}", "t1_z": 4.0 if i < hit_for else 0.5, "hit": i < hit_for}
                for i in range(outcomes)]

    profile = phenome.specificity(pairs("broad", 40, 12) + pairs("specific", 40, 2) + pairs("shared", 40, 6)
                                  + pairs("few", 5, 1))
    assert {f: v["specificity"] for f, v in profile.items()} == {
        "broad": "broad", "specific": "specific", "shared": "shared", "few": "not assessed"}
    assert profile["broad"]["breadth"] == 0.3 and profile["specific"]["median_abs_t1_z"] == 0.5
    rows = [{"name": "a", "kind": "condition", "tested": True, "hits": 3, "novel_hits": 0, "specific_hits": 0,
             "best": {"t1_z": 9.0}},
            {"name": "b", "kind": "condition", "tested": True, "hits": 1, "novel_hits": 0, "specific_hits": 1,
             "best": {"t1_z": 4.0}},
            {"name": "c", "kind": "condition", "tested": True, "hits": 1, "novel_hits": 1, "specific_hits": 0,
             "best": {"t1_z": 3.0}}]
    assert [r["name"] for r in phenome.select({"outcomes": rows}, set(), top_k=3)] == ["c", "b", "a"]


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    import duckdb

    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    query = duckdb_query(con)
    calls: list[str] = []
    monkeypatch.setattr(cohort, "_query", lambda client, sql, params: calls.append(sql) or query(client, sql, params))
    base = replace(CohortSpec.from_json(SPEC), design="incident", window_days=60, lag_days=7, min_wear_days=20,
                   holdout_fraction=0.5)
    base, _ = feas.verify_covariates(base, SYNTHETIC_CDR, None)
    present = query(None, *cohort.schema_check_sql(base, SYNTHETIC_CDR))
    base, _ = cohort.resolve_sources(base, list(zip(present["table_name"], present["column_name"])))
    outcomes = feas.resolve_codes(feas.load_catalog(CATALOG), SYNTHETIC_CDR, None)
    problems = feas.verify_catalog(outcomes, SYNTHETIC_CDR, None)
    verified = [o for o in outcomes if problems.get(o.name) is None]
    return {"cdr": SYNTHETIC_CDR, "query": query, "base": base, "verified": verified, "out": tmp_path, "calls": calls,
            "problems": problems}


def _discovery(frame):
    return frame[~frame["_holdout"].astype(bool)].set_index("person_id")


@pytest.mark.parametrize("horizon", [None, 120])
def test_each_outcomes_cohort_is_the_extractions(synthetic, horizon):
    """Who is in each outcome's cohort, who has the event and when, or the value: the same as the extraction's
    discovery share, derived from one person table and one diagnosis and one laboratory matrix."""
    cdr, query = synthetic["cdr"], synthetic["query"]
    base = replace(synthetic["base"], horizon_days=horizon)
    conditions = [o for o in synthetic["verified"] if o.kind == "condition"]
    labs = [o for o in synthetic["verified"] if o.kind == "measurement"]
    assert {"t2d", "depression"} <= {o.name for o in conditions} and {"hba1c", "hdl"} <= {o.name for o in labs}
    people = phenome.person_table(base, cdr, None, synthetic["out"])
    dx = query(None, *phenome.condition_matrix_sql(conditions, base, cdr))
    values = query(None, *phenome.lab_matrix_sql(labs, base, cdr))
    for i, o in enumerate(conditions):
        spec = feas.spec_for(o, base)
        extracted = _discovery(query(None, *build_cohort_sql(spec, cdr, phenotypes=False)))
        eligible, event, time = phenome.condition_cohort(people, dx, i, base.lag_days, o.min_occurrences)
        mine = pd.DataFrame({"event": event[eligible].astype(int), "time": time[eligible]},
                            index=people["person_id"][eligible])
        assert set(mine.index) == set(extracted.index), o.name
        both = mine.join(extracted)
        assert (both["event"] == both[o.name]).all() and np.allclose(both["time"], both["followup_days"]), o.name
    for i, o in enumerate(labs):
        extracted = _discovery(query(None, *build_cohort_sql(feas.spec_for(o, base), cdr, phenotypes=False)))
        mine = pd.Series(phenome.lab_values(people, values, i), index=people["person_id"]).dropna()
        assert set(mine.index) == set(extracted.index), o.name
        assert np.allclose(mine.sort_index(), extracted[o.name].sort_index()), o.name


def test_the_scan_reads_only_the_discovery_share(synthetic):
    """The sealed share is filtered out in the queries themselves: nothing downloaded holds a sealed person."""
    cdr, query, base = synthetic["cdr"], synthetic["query"], synthetic["base"]
    sealed = lambda ids: [int(p) for p in ids if holdout_bucket(int(p), base.split_salt) < 500]  # noqa: E731
    people = phenome.person_table(base, cdr, None, synthetic["out"])
    conditions = [o for o in synthetic["verified"] if o.kind == "condition"]
    labs = [o for o in synthetic["verified"] if o.kind == "measurement"]
    assert len(people) and not sealed(people["person_id"])
    assert not sealed(query(None, *phenome.condition_matrix_sql(conditions, base, cdr))["person_id"])
    assert not sealed(query(None, *phenome.lab_matrix_sql(labs, base, cdr))["person_id"])
    assert "_holdout" not in people.columns
    everyone = query(None, *build_cohort_sql(replace(base, holdout_fraction=0.0), cdr, phenotypes=False,
                                             person_level=True))
    assert len(sealed(everyone["person_id"])) > 0  # without a holdout the same query keeps everyone
    with pytest.raises(ValueError, match="incident design"):
        build_cohort_sql(replace(base, design="window"), cdr, person_level=True)


def test_the_scan_finds_planted_pairs_and_selects_by_signal(synthetic, europe_pmc, monkeypatch):
    """The synthetic CDR plants activity -> HbA1c and HDL, night-to-night sleep variability -> depression, and
    body size -> calories and BMI. The scan finds them across the whole matrix, consistently in both halves and
    after adjustment; the literature (recorded answers: step count established, anything else unexplored) ranks
    the outcomes; already tested outcomes are skipped; displayed counts of 1-20 are masked."""
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "1")
    paper = {"pmid": "123", "title": "Daily steps and incident disease", "citedByCount": "50", "pubYear": "2020",
             "abstractText": "Background. We followed adults. Each additional 1000 daily steps was associated with "
                             "lower risk (HR 0.88, 95% CI 0.84-0.92). Conclusions follow."}
    asked = europe_pmc({"step count": (120, 4, [paper])})
    cdr, base, out = synthetic["cdr"], synthetic["base"], synthetic["out"]
    result = phenome.scan(synthetic["verified"], base, cdr, None, out, min_events=5, min_n=15, log=lambda *_: None)
    hits = {(p["outcome"], p["feature"]): p for p in result["pairs"] if p["hit"]}
    assert hits[("hba1c", "act_steps_mean")]["estimate"] < -0.3 and hits[("hdl", "act_steps_mean")]["estimate"] > 0.3
    assert any(o == "depression" and "sleep" in f and p["estimate"] > 1 for (o, f), p in hits.items())
    assert result["tested_pairs"] == result["features"] * sum(r["tested"] for r in result["outcomes"])
    assert all(0 <= p["q"] <= 1 and p["q"] >= p["p"] for p in result["pairs"] if p["p"] is not None)
    # Literature: only catalog terms leave the machine; activity hits are established, others count as new.
    assert asked and not any(re.search(r"\d{5,}", u) for u in asked)
    steps = hits[("hba1c", "act_steps_mean")]
    assert steps.get("novelty") in (None, "established")
    by_name = {r["name"]: r for r in result["outcomes"]}
    assert by_name["depression"]["novel_hits"] >= 1
    chosen = [r["name"] for r in phenome.select(result, set(), top_k=1)]
    assert chosen and set(chosen) <= {r["name"] for r in result["outcomes"] if r["hits"]}
    assert "depression" in chosen  # its sleep hits are unexplored in the recorded literature
    assert "depression" not in [r["name"] for r in phenome.select(result, {"depression"}, top_k=1)]
    table = phenome.scan_table(result, synthetic["problems"], phenome.select(result, set(), top_k=1), {"hdl"})
    assert "<=20" in table and "already tested on this holdout" in table and "concept" in table
    assert "| 17 |" not in table and "| 20 |" not in table
    assert "act_steps_mean" in phenome.hits_table(result)
    # The replication atlas: the literature (not this screen) names the pairs; each established one is re-tested
    # with q across the atlas and the effect its most cited paper states, quoted from the abstract.
    atlas = result["atlas"]
    assert atlas["status"] == "available" and atlas["pairs"]
    pairs = {(r["outcome"], r["construct"]): r for r in atlas["pairs"]}
    steps_hba1c = next(r for (o, c), r in pairs.items() if o == "hba1c" and "step" in c)
    assert steps_hba1c["q"] < 0.05 and steps_hba1c["t1_estimate"] < 0 and steps_hba1c["papers"] == 120
    assert steps_hba1c["published"][0]["statement"].startswith("Each additional 1000 daily steps")
    assert steps_hba1c["published"][0]["key"] == "PMID:123"
    # Only constructs the recorded literature establishes ("step count" in their terms) are listed, each with one
    # representative feature chosen without the outcome.
    established = {"daily step count", "day-to-day variability of physical activity"}
    assert {c for _, c in pairs} <= established
    assert len({r["feature"] for r in atlas["pairs"]}) == len({c for _, c in pairs})
    assert "Replication atlas" in phenome.atlas_table(result) and "HR 0.88" in phenome.atlas_table(result)
    # Every hit carries its feature's breadth; the profile covers every feature.
    assert set(result["specificity"]) == {p["feature"] for p in result["pairs"]}
    assert all(p["specificity"] in ("specific", "shared", "broad", "not assessed") and 0 < p["breadth"] <= 1
               for p in hits.values())
    short = phenome.scan_table(result, synthetic["problems"], phenome.select(result, set(), top_k=1), set(), limit=2)
    assert short.count("\n| ") == 3 and "more outcomes" in short  # the header, two outcomes; the rest counted
    # The second call is served from the cache: no query runs.
    calls = len(synthetic["calls"])
    again = phenome.scan(synthetic["verified"], base, cdr, None, out, min_events=5, min_n=15, log=lambda *_: None)
    assert again == result and len(synthetic["calls"]) == calls
    # The copy beside the reports has counts of 1-20 masked; the raw one stays in the sealed folder.
    from codas.aou.cohort import sealed_dir
    from codas.aou.privacy import mask_small_counts

    shown = json.loads((out / phenome.SCAN_FILE).read_text())
    assert shown == mask_small_counts(result) and shown != result
    assert not [v for o in shown["outcomes"] + shown["pairs"] for k, v in o.items()
                if k in ("n", "events") and isinstance(v, int) and 1 <= v <= 20]
    assert list((sealed_dir(out) / "phenome_cache").glob("phenome_scan_*.json"))
    assert not list(out.glob("phenome_scan_*.json"))


def test_explore_selects_outcomes_by_the_scan(synthetic, monkeypatch):
    from codas.aou import runner

    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "0")
    base, out = synthetic["base"], synthetic["out"]
    result = runner.explore(CATALOG, base, synthetic["cdr"], client=object(), top_k=1, min_measurement_n=30,
                            min_cases=10, out_dir=out, screen_only=True, log=lambda *_: None)
    assert result["selection"] == "scan" and result["scan"]["hits"] > 0 and result["chosen"]
    summary = Path(result["summary_path"]).read_text()
    assert "## Phenome scan" in summary and "### Strongest hits" in summary
    assert {r["name"] for r in result["rows"]} == {o.name for o in synthetic["verified"]}
    from codas.aou import notebook_views as views

    assert views.phenome_hits_table(result) is not None
    assert views.phenome_figure(result)._repr_png_().startswith(b"\x89PNG")
    by_size = runner.explore(CATALOG, base, synthetic["cdr"], client=object(), top_k=1, min_measurement_n=30,
                             min_cases=10, out_dir=out, screen_only=True, log=lambda *_: None, selection="size")
    assert by_size["selection"] == "size" and by_size["scan"] is None
    assert "## Feasibility screen" in Path(by_size["summary_path"]).read_text()
