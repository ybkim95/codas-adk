"""The data inventory: every table of the CDR, masked participant coverage, and minute heart-rate
tables found by their shape rather than by name."""

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from codas.aou import cohort, inventory  # noqa: E402
from codas.aou.privacy import MASK  # noqa: E402
from codas.aou.synthetic import SYNTHETIC_CDR, build_synthetic_cdr, duckdb_query  # noqa: E402


def test_inventory_lists_tables_with_masked_coverage(monkeypatch):
    con = duckdb.connect()
    build_synthetic_cdr(con)
    con.execute("CREATE TABLE rare AS SELECT person_id FROM person LIMIT 5")  # a table only 5 people are in
    monkeypatch.setattr(cohort, "_query", duckdb_query(con))
    inv = inventory.inventory(SYNTHETIC_CDR, None)
    tables = inv["tables"].set_index("table_name")
    assert {"measurement", "observation", "steps_intraday", "heart_rate_minute_level"} <= set(tables.index)
    assert tables.loc["person", "participants"] == 600
    assert tables.loc["rare", "participants"] == MASK and tables.loc["rare", "wearable_participants"] == MASK
    assert inv["minute_heart_rate"] == [{"table": "heart_rate_minute_level", "time": "datetime",
                                         "value": "heart_rate_value"}]


def test_minute_heart_rate_tables_are_found_by_shape_not_name():
    cols = pd.DataFrame([("hr_feed", "person_id", "INT64"), ("hr_feed", "ts", "DATETIME"), ("hr_feed", "bpm", "FLOAT64"),
                         ("heart_rate_summary", "person_id", "INT64"), ("heart_rate_summary", "date", "DATE"),
                         ("heart_rate_summary", "min_heart_rate", "INT64")],
                        columns=["table_name", "column_name", "data_type"])
    assert inventory.minute_heart_rate_tables(cols) == [{"table": "hr_feed", "time": "ts", "value": "bpm"}]
