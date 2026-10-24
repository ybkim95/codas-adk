"""A column name never carries two definitions: the sealed holdout's ledger is cumulative, so a feature that
changed meaning under the same name would read as the same feature across runs. Every source of the
example spec has a recorded definition (its settings and the SQL that computes it); changing one fails this
test until the source gets a new prefix, or, for a change that leaves every value as it was, until the
record is updated with a note saying why."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from codas.aou import cohort
from codas.aou.cohort import CohortSpec, build_cohort_sql

LOCK = cohort.ROOT / "scripts" / "aou" / "feature_definitions.json"
SPEC = cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json"


def _ctes(sql: str, alias: str) -> str:
    """The text of every CTE named ``alias`` or ``alias_*``, with the alias written as ``SRC``."""
    blocks = re.findall(rf"(?ms)^(?:WITH )?({alias}(?:_\w+)? AS \(\n.*?\n\))", sql)
    return re.sub(rf"\b{alias}(?=b_|_|\b)", "SRC", "\n".join(blocks))


def definitions() -> dict[str, str]:
    """Each source prefix of the example spec -> a hash of what its columns mean."""
    spec = CohortSpec.from_json(SPEC)
    sql, _params = build_cohort_sql(spec, "p.d")
    select = sql[sql.rindex("\nSELECT\n"):sql.index("\nFROM index_events ie")]
    design = {k: getattr(spec, k) for k in ("design", "window_days", "valid_day")}
    out = {}
    sources = [(f"w{i}", s) for i, s in enumerate(spec.wearables)]
    sources += [({"minute_activity": "q", "sleep_stages": "z", "minute_heart_rate": "h"}[p.kind] + str(k), p)
                for k, p in enumerate(spec.phenotypes)]
    for alias, source in sources:
        settings = {k: v for k, v in asdict(source).items() if k != "optional"}
        columns = [re.sub(rf"\b{alias}\.", "SRC.", line.strip()) for line in select.splitlines()
                   if re.search(rf"\b{alias}\.", line)]
        text = json.dumps([design, settings, _ctes(sql, alias), columns], sort_keys=True, default=str)
        out[source.prefix] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return out


def test_a_changed_definition_needs_a_new_column_prefix():
    recorded = json.loads(LOCK.read_text())["sources"]
    current = definitions()
    changed = sorted(p for p in current if p in recorded and recorded[p]["definition"] != current[p])
    unrecorded = sorted(set(current) - set(recorded))
    assert not changed, (f"The definition of {changed} changed. Give the source a new prefix so no column name carries "
                         f"two meanings across runs, then record it in {LOCK.name}; only if every value stays as it "
                         "was may the recorded hash be updated, with a note saying why.")
    assert not unrecorded, f"Record the definitions of {unrecorded} in {LOCK.name}."


def test_retired_prefixes_are_never_reused():
    lock = json.loads(LOCK.read_text())
    assert not set(lock["retired"]) & set(lock["sources"])
    assert not set(lock["retired"]) & set(definitions())
