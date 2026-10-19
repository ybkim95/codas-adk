"""The notebook itself and the tables its checks cell shows."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from codas.aou import notebook_views as views
from codas.aou.cohort import CohortSpec

ROOT = Path(__file__).resolve().parents[1]


def test_every_notebook_code_cell_compiles():
    import nbformat

    nb = nbformat.read(ROOT / "notebooks" / "aou_codas_quickstart.ipynb", as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell {i}", "exec")
    assert "__NOTEBOOK_VERSION__" not in "".join(c.source for c in nb.cells)  # the version is stamped


def test_schema_and_concept_tables():
    spec = CohortSpec.from_json(ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    present = pd.DataFrame({"table_name": ["activity_summary"] * 3, "column_name": ["date", "steps", "calories_out"]})
    table = views.schema_table(spec, present)
    act = table.set_index("table").loc["activity_summary"]
    assert act["found"] == 3 and "very_active_minutes" in act["missing"]
    assert table.set_index("table").loc["sleep_daily_summary", "found"] == 0
    items = [("outcome", SimpleNamespace(name="a", concept_ids=[1, 2])), ("covariate", SimpleNamespace(name="b", concept_ids=[3]))]
    rows = [{"concept_id": 1, "concept_name": "One", "domain_id": "Measurement"}]
    concepts = views.concept_table(items, rows, {"a": "concept 2 not in the CDR", "b": None})
    assert list(concepts["name in this CDR"]) == ["One", "(not found)", "(not found)"]
    assert list(concepts["check"]) == ["concept 2 not in the CDR"] * 2 + ["ok"]


def test_environment_table_lists_settings_and_machine():
    env = {"commit": "abc", "analysis_code": "def", "python": "3.12", "cpus": 4, "memory_total_gb": 15,
           "memory_available_gb": 12, "disk_free_gb": 80, "packages": {"pandas": "2"}, "settings": {"CODAS_X": "1"}}
    table = views.environment_table(env, project="p")
    assert table.iloc[0].tolist() == ["project", "p"] and "package pandas" in set(table["setting"])


def test_confirmation_table_with_nothing_registered(tmp_path):
    import json

    (tmp_path / "x_confirmation.json").write_text(json.dumps({"hypotheses": []}))
    table = views.confirmation_table({"summary_path": str(tmp_path / "s.md")}, "x")
    assert table.empty and "failed checks" in table.columns
