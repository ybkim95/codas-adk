"""Cohort characteristics of the discovery and sealed shares ("Table 1"), masked for publication.

Describes the outcome, follow-up, demographics, covariates and days of wearable data: a 0/1 column as
count (percent), any other as mean (SD), with the share of participants missing a value. Which
columns these are comes from the spec. A count of 1-20, or one that the total minus it would reveal,
is shown as ``<=20`` with no percent, as the Dissemination Policy requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from codas.aou.cohort import FOLLOWUP_COLUMN, CohortSpec, holdout_path, is_survival
from codas.aou.privacy import MASK, SMALL_CELL_MAX


def described_columns(spec: CohortSpec) -> list[str]:
    cols = [spec.outcome.name] + ([FOLLOWUP_COLUMN] if is_survival(spec) else [])
    cols += ["age", "female"] if spec.include_demographics else []
    cols += [c.name for c in spec.covariates] + [f"{s.prefix}_days" for s in spec.wearables]
    return cols


def _small(k: int) -> bool:
    return 1 <= k <= SMALL_CELL_MAX


def _describe(values: pd.Series) -> str:
    x = pd.to_numeric(values, errors="coerce")
    n, present = len(x), int(x.notna().sum())
    if present == 0:
        return "-"
    if _small(present):
        return MASK
    if set(x.dropna().unique()) <= {0, 1}:
        k = int(x.sum())
        text = MASK if _small(k) or _small(present - k) else f"{k:,} ({k / present:.1%})"
    else:
        text = f"{x.mean():.4g} ({x.std():.3g})"
    missing = n - present
    if missing:
        text += f"; missing {MASK if _small(missing) or _small(present) else f'{missing / n:.0%}'}"
    return text


def cohort_table(spec: CohortSpec, discovery_csv: str | Path) -> dict[str, Any]:
    """Masked characteristics of the discovery table and, when one exists, its sealed holdout."""
    shares = {"discovery": Path(discovery_csv)}
    if spec.holdout_fraction > 0 and holdout_path(discovery_csv).exists():
        shares["holdout"] = holdout_path(discovery_csv)
    frames = {name: pd.read_csv(path) for name, path in shares.items()}
    rows = [["participants", *[MASK if _small(len(f)) else f"{len(f):,}" for f in frames.values()]]]
    for column in described_columns(spec):
        if all(column in f.columns for f in frames.values()):
            rows.append([column, *[_describe(f[column]) for f in frames.values()]])
    return {"columns": ["characteristic", *frames], "rows": rows,
            "note": "0/1 columns: count (percent); others: mean (SD). Counts of 1-20 are masked."}


def cohort_table_markdown(table: dict[str, Any]) -> str:
    lines = ["| " + " | ".join(table["columns"]) + " |", "|" + "---|" * len(table["columns"])]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table["rows"]]
    return "\n".join(lines) + f"\n\n{table['note']}\n"
