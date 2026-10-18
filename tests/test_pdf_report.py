"""The compiled PDF renders with or without a Unicode font installed (Workbench images differ)."""

import pytest

pytest.importorskip("fpdf")
pytest.importorskip("markdown")

from codas.aou import pdf_report  # noqa: E402

PHASES = {
    "target_column": "y",
    "report": "Effect ($\\rho = -0.29$, $R^2 = 0.12$) — counts ≤20 are masked.\n\n| a | b |\n|---|---|\n| x | 1 |",
    "hypotheses": "Δ ≈ −0.3", "rounds": [{"round": 0, "validated_count": 2, "ml_metric_name": "r2", "ml_metric_value": 0.1,
                                          "passing_candidates": [{"feature": "f1", "rho": -0.3}, {"feature": "f2", "rho": 0.2}]}],
    "exclusion_log": [{"after_round": 0, "columns": ["z"], "reason": "derived from the outcome"}],
    "fact_sheet": {"rows": 100},
}
EVENTS = [{"round": 1, "kind": "text", "author": "critic_agent", "text": "**attack** $\\rho$"}]


@pytest.mark.parametrize("unicode_font", [True, False])
def test_pdf_builds_with_and_without_a_unicode_font(tmp_path, monkeypatch, unicode_font):
    if not unicode_font:
        monkeypatch.setattr(pdf_report, "_find_fonts", lambda: None)
    elif pdf_report._find_fonts() is None:
        pytest.skip("no Unicode TrueType font installed here")
    path = pdf_report.build_pdf(tmp_path, "run", PHASES, EVENTS, "| a | b |\n|---|---|\n| 1 | 2 |", {"rows": 100})
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF")


def test_tex_math_is_rendered_as_symbols():
    assert pdf_report._clean_tex(r"$\rho = -0.29$, $R^2$ and \le 20") == "ρ = -0.29, R² and ≤ 20"


def test_cohort_table_and_holdout_forest_plots_are_rendered(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_report, "_find_fonts", lambda: None)  # core fonts keep the text searchable
    init = pdf_report._Doc.__init__

    def uncompressed(self, title):
        init(self, title)
        self.pdf.set_compression(False)

    monkeypatch.setattr(pdf_report._Doc, "__init__", uncompressed)
    confirmation = {
        "replicated": 1, "tested": 2, "robust": 1,
        "hypotheses": [{"feature": "steps_mean", "test": "cox", "hr_per_sd": 0.8, "ci95": [0.7, 0.9], "replicated": True},
                       {"feature": "sleep_sd", "test": "cox", "hr_per_sd": 1.05, "ci95": [0.95, 1.16], "replicated": False}],
        "positive_controls": [{"control": "age", "test": "cox", "hr_per_sd": 1.6, "ci95": [1.4, 1.8], "detected": True}],
    }
    table1 = {"columns": ["characteristic", "discovery", "holdout"], "rows": [["participants", "1,000", "<=20"]],
              "note": "Counts of 1-20 are masked."}
    phases = {**PHASES, "confirmation": confirmation, "cohort_table": table1}
    data = pdf_report.build_pdf(tmp_path, "run", phases, EVENTS, "", {"rows": 100}).read_bytes()
    # PDF strings escape parentheses.
    for text in (b"Cohort characteristics", b"hazard ratio per SD", b"positive control: age", b"0.8 \\(0.7-0.9\\)",
                 b"replicated on the sealed holdout"):
        assert text in data, text
    cox, rank = pdf_report._forest_items(confirmation)
    assert [c[0] for c in cox] == ["steps_mean", "sleep_sd", "positive control: age"] and rank == []
    assert [c[4] for c in cox] == [True, False, True]
