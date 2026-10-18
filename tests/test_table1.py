"""Cohort characteristics for the discovery and sealed shares, masked for counts of 1-20."""

import pandas as pd

from codas.aou import cohort
from codas.aou.cohort import CohortSpec
from codas.aou.table1 import cohort_table, cohort_table_markdown

SPEC = {
    "name": "t", "design": "incident", "window_days": 30, "min_wear_days": 5, "holdout_fraction": 0.5,
    "outcome": {"kind": "condition", "name": "dx", "concept_ids": [500]},
    "covariates": [{"name": "cov_m", "kind": "measurement", "concept_ids": [1]}],
    "wearables": [{"table": "activity_summary", "date_column": "date", "prefix": "act", "columns": ["steps"]}],
}


def test_counts_and_percents_are_masked_where_they_would_reveal_a_small_count(tmp_path):
    spec = CohortSpec.from_dict(SPEC)
    n = 200
    disc = pd.DataFrame({"dx": [1] * 50 + [0] * 150, "followup_days": [100.0] * n, "age": [40.0 + i % 30 for i in range(n)],
                         "female": [1] * 190 + [0] * 10, "cov_m": [25.0] * 185 + [None] * 15, "act_days": [30] * n,
                         "act_steps_mean": [1.0] * n, "act_steps_sd": [1.0] * n})
    path = tmp_path / "t.csv"
    disc.to_csv(path, index=False)
    disc.head(15).to_csv(cohort.holdout_path(path), index=False)
    table = cohort_table(spec, path)
    rows = {r[0]: r[1:] for r in table["rows"]}
    assert table["columns"] == ["characteristic", "discovery", "holdout"]
    assert rows["participants"] == ["200", "<=20"]
    assert rows["dx"][0] == "50 (25.0%)" and rows["dx"][1] == "<=20"
    assert rows["female"][0] == "<=20"  # 10 men: the complement is a small count
    assert rows["cov_m"][0] == "25 (0); missing <=20"
    assert "act_steps_mean" not in rows  # wearable features are the candidates, not cohort characteristics
    assert "| participants | 200 | <=20 |" in cohort_table_markdown(table)
