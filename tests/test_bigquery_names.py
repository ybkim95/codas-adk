"""SQL that DuckDB and BigQuery read differently is refused by the test harness: a CTE named like one of its
own columns runs in DuckDB (the bare name is the column) but fails on the CDR (the bare name is the row)."""

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("sqlglot")

from codas.aou import carryover, cohort  # noqa: E402
from codas.aou.synthetic import bigquery_name_clashes, duckdb_query  # noqa: E402

CLASH = ("WITH first_wear AS (SELECT person_id, MIN(d) AS first_wear FROM t GROUP BY person_id)\n"
         "SELECT person_id, DATE_ADD(first_wear, INTERVAL 180 DAY) AS index_date FROM first_wear")


def test_a_cte_named_like_its_column_is_refused():
    assert bigquery_name_clashes(CLASH) == ["first_wear"]
    assert bigquery_name_clashes(CLASH.replace("AS first_wear FROM", "AS first_day FROM")
                                 .replace("DATE_ADD(first_wear", "DATE_ADD(first_day")) == []
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS person_id, DATE '2020-01-01' AS d")
    with pytest.raises(ValueError, match="whole row"):
        duckdb_query(con)(None, CLASH, [])


def test_the_shared_first_wear_cte_is_unambiguous():
    spec = cohort.CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    assert bigquery_name_clashes(carryover.split_sql(spec, "p.d")[0]) == []
