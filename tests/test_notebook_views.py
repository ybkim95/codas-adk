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


def test_charts_scale_to_the_page():
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 2))
    fig.subplots().plot([0, 1], [0, 1])
    html = views.Chart(fig)._repr_html_()
    assert html.startswith("<img") and "data:image/png;base64," in html and "max-width:100%" in html


def test_tables_wrap_long_identifiers_at_underscores_only():
    table = pd.DataFrame({"feature": ["act_days_over_act_steps_mean"], "q": [0.0417], "95% CI": ["-0.04 to 0.01"]})
    html = views.fit(table)._repr_html_()
    assert "act_<wbr>days_<wbr>over_<wbr>act_<wbr>steps_<wbr>mean" in html
    assert "0.0417" in html and "-0.04 to 0.01" in html


def test_report_tables_wrap_but_code_and_prose_stay_as_written():
    text = "Text with `a_b` and c_d.\n\n| feature | q |\n|---|---|\n| act_days_sd | 0.1 |\n| `x_y` z_w | 2 |"
    out = views.wrap_tables(text)
    assert out.splitlines()[0] == "Text with `a_b` and c_d."
    assert "| act_<wbr>days_<wbr>sd | 0.1 |" in out and "| `x_y` z_<wbr>w | 2 |" in out


def test_forest_hazard_axis_is_labelled_across_the_data(tmp_path):
    import json

    hyps = [{"feature": f"f{i}", "test": "cox", "hr_per_sd": hr, "ci95": [hr - 0.1, hr + 0.1], "replicated": False}
            for i, hr in enumerate((0.85, 1.0, 1.25))]
    (tmp_path / "explore_x_confirmation.json").write_text(json.dumps({"hypotheses": hyps}))
    chart = views.forest_figure({"summary_path": str(tmp_path / "s.md"), "runs": [{"cohort": "explore_x"}]})
    ax = chart.figure.axes[0]
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(chart.figure).draw()
    labels = [t.get_text() for t in ax.get_xticklabels(minor=False) + ax.get_xticklabels(minor=True) if t.get_text()]
    values = [float(t) for t in labels]
    assert min(values) < 1 < max(values) and len(values) >= 3, labels
    assert [t.get_text() for t in ax.get_yticklabels()] == ["x: f0", "x: f1", "x: f2"]
