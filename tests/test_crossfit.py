"""Two-way confirmation and the full-cohort scan: share B seals exactly the participants share A lets the agents
see, each direction keeps its own ledger, the two directions are summarised side by side, and the full-cohort
scan is pre-registered and opens what it reads on both halves."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from codas.aou import cohort, crossfit, ledger, phenome
from codas.aou import feasibility as feas
from codas.aou.cohort import CohortSpec

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
    return {"cdr": SYNTHETIC_CDR, "query": query, "base": base, "out": tmp_path,
            "verified": [o for o in outcomes if problems.get(o.name) is None]}


def test_share_b_seals_what_share_a_shows(synthetic):
    cdr, base, query, out = synthetic["cdr"], synthetic["base"], synthetic["query"], synthetic["out"]
    a, b = base, replace(base, discovery_share="B")
    seen_a = set(phenome.person_table(a, cdr, None, out / "a")["person_id"])
    seen_b = set(phenome.person_table(b, cdr, None, out / "b")["person_id"])
    everyone = set(phenome.person_table(a, cdr, None, out / "all", everyone=True)["person_id"])
    assert seen_a and seen_b and not seen_a & seen_b and seen_a | seen_b == everyone
    assert all(cohort.is_sealed(p, b) for p in seen_a) and all(cohort.is_sealed(p, a) for p in seen_b)
    # The cohort table's flag follows the same rule.
    frame = query(None, *cohort.build_cohort_sql(b, cdr, person_id=True))
    assert all(bool(f) == cohort.is_sealed(p, b) for p, f in zip(frame["person_id"], frame[cohort.HOLDOUT_COLUMN]))
    with pytest.raises(ValueError, match="discovery_share"):
        replace(base, discovery_share="C").validate()


def test_each_direction_has_its_own_folder_and_ledger(tmp_path):
    from codas.aou import runner

    spec = CohortSpec.from_json(SPEC)
    key_a, key_b = runner.holdout_key(spec, "p.d"), runner.holdout_key(replace(spec, discovery_share="B"), "p.d")
    assert key_a == "p.d|salt=0|permille=500" and key_b == key_a + "|share=B"
    assert crossfit.run_folder(tmp_path / "aou", "B") == tmp_path / "aou_share_b"
    assert crossfit.run_folder(tmp_path / "aou_share_b", "A") == tmp_path / "aou"
    assert crossfit.run_folder(tmp_path / "aou_share_b", "B") == tmp_path / "aou_share_b"


def _confirmation(folder: Path, stem: str, rows: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_confirmation.json").write_text(json.dumps({"hypotheses": rows}))


def test_the_two_directions_side_by_side(tmp_path):
    out = tmp_path / "aou"
    row = lambda f, rep, d=1: {"feature": f, "direction": d, "same_direction": True, "hr_per_sd": 1.2,  # noqa: E731
                               "ci95": [1.1, 1.3], "replicated": rep, "robust": rep}
    _confirmation(out, "explore_t2d", [row("x", True), row("y", False), row("z", True)])
    _confirmation(crossfit.run_folder(out, "B"), "explore_t2d", [row("x", True), row("w", True), row("z", True, -1)])
    _confirmation(out, "explore_hdl", [row("x", True)])
    result = crossfit.summary(out)
    (t2d,) = result["outcomes"]
    assert t2d["confirmed_both_ways"] == ["x"]          # z flipped direction; y failed one way; w ran one way only
    assert result["only_a"] == ["explore_hdl"] and result["only_b"] == []
    path = crossfit.write(out)
    assert path == out / crossfit.FILE and "| explore_t2d | x |" in path.read_text() and "| yes |" in path.read_text()
    assert crossfit.write(tmp_path / "empty") is None


def test_the_full_cohort_scan_is_preregistered_and_opens_both_halves(synthetic, monkeypatch):
    from codas.aou import runner

    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "0")
    base, out = synthetic["base"], synthetic["out"] / "aou"
    result = runner.explore(CATALOG, base, synthetic["cdr"], client=object(), min_measurement_n=30, min_cases=10,
                            out_dir=out, log=lambda *_: None, full_cohort=True)
    prereg = json.loads((out / phenome.FULL_PREREG).read_text())
    assert prereg["settings"]["participants"] == "every" and prereg["features"] and prereg["sha256"]
    scan = json.loads((out / phenome.FULL_SCAN_FILE).read_text())
    discovery = phenome.person_table(base, synthetic["cdr"], None, out)
    assert scan["participants"] > len(discovery)        # the sealed share is included
    assert "Full-cohort scan" in Path(result["summary_path"]).read_text()
    screened = {r["name"] for r in scan["outcomes"] if r.get("tested")}
    for share in ("A", "B"):
        key = runner.holdout_key(replace(base, discovery_share=share), synthetic["cdr"])
        folder = crossfit.run_folder(out, share)
        summary = ledger.summary(folder, key)
        assert set(summary["opened_without_a_test"]) == screened and summary["hypotheses"] == 0
        by_name = {o.name: o for o in synthetic["verified"]}
        opened = runner._opened_outcomes(folder, [key], by_name, replace(base, discovery_share=share), True)
        assert screened <= opened
