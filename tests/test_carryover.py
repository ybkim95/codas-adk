"""Carrying the holdout ledger to another workspace: exported after an exploration, adopted once in the new
workspace under its own key, its outcomes opened there, with a fingerprint of who the split seals."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from codas.aou import carryover, cohort, ledger
from codas.aou.cohort import CohortSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"


def test_the_ledger_travels_and_opens_its_outcomes_in_the_new_workspace(monkeypatch, tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("sqlglot")
    import duckdb

    from codas.aou import runner
    from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query

    con = duckdb.connect()
    build_synthetic_cdr(con)
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    spec = replace(CohortSpec.from_json(SPEC), design="incident", holdout_fraction=0.5)
    old_out, new_out = tmp_path / "old" / "aou", tmp_path / "new" / "aou"
    old_key = runner.holdout_key(spec, SYNTHETIC_CDR)
    old_out.mkdir(parents=True)
    ledger.append(old_out, old_key, [{"stem": "explore_t2d", "outcome_key": "k1", "feature": "x", "role": "novel",
                                      "tested": True, "rejected": True, "prereg": "abc", "p": 0.001}])
    exported = carryover.export(old_out, spec, SYNTHETIC_CDR, None, tmp_path / "old" / "reports")
    record = json.loads(exported.read_text())
    assert record["ledger"][0]["feature"] == "x" and record["fingerprint"]["sealed_participants"] > 20
    assert "person_id" not in exported.read_text()
    # The new workspace sees the file in its reports folder; its CDR has another name.
    record["cdr"] = "elsewhere.C2025"
    reports = tmp_path / "new" / "reports"
    reports.mkdir(parents=True)
    (reports / "carry_over.json").write_text(json.dumps(record))
    new_key = old_key.replace(SYNTHETIC_CDR, SYNTHETIC_CDR)  # same split settings; the CDR is what differs in use
    got = carryover.adopt(new_out, new_key, spec, SYNTHETIC_CDR, None, reports, log=lambda *_: None)
    assert got["entries"] == 1 and got["same_people_sealed"]
    stems, keys = ledger.opened(new_out, new_key)
    assert "explore_t2d" in stems and "k1" in keys
    entry = ledger.load(new_out, new_key)[0]
    assert entry["inherited_from"] == old_key
    # Adopted once: running again adds nothing.
    assert carryover.adopt(new_out, new_key, spec, SYNTHETIC_CDR, None, reports, log=lambda *_: None) is None
    assert len(ledger.load(new_out, new_key)) == 1
    # Entries for other split settings stay behind.
    other = replace(spec, split_salt=7)
    other_key = runner.holdout_key(other, SYNTHETIC_CDR)
    (reports / "carry_over_2.json").write_text(json.dumps({**record, "cdr": "third.C"}))
    assert carryover.adopt(tmp_path / "third", other_key, other, SYNTHETIC_CDR, None, reports,
                           log=lambda *_: None)["entries"] == 0
