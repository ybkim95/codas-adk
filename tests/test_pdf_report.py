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


def _matplotlib_fonts(monkeypatch):
    import os

    import matplotlib

    monkeypatch.setattr(pdf_report, "_FONT_DIRS", (os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"),))
    assert pdf_report._find_fonts() is not None


def test_bold_code_spans_with_a_unicode_font(tmp_path, monkeypatch):
    """The Workbench failure: **`feature`** asks for bold monospace; with a TrueType font only the regular
    monospace face was registered, fpdf2's HTML writer raised mid-way and left no page open."""
    _matplotlib_fonts(monkeypatch)
    report = "1. **`act_days_over_act_sedentary_minutes_mean`** ($\\rho = -0.16$) and *`x`*, ***`y`***"
    path = pdf_report.build_pdf(tmp_path, "run", {**PHASES, "report": report}, EVENTS, "", {"rows": 100})
    assert path.read_bytes().startswith(b"%PDF") and not (tmp_path / "run_pdf_error.txt").exists()
    assert {"mono", "monoB", "monoI", "monoBI"} <= set(pdf_report._Doc("t").pdf.fonts)  # every style registered


def test_a_failing_section_keeps_the_rest_of_the_report(tmp_path, monkeypatch):
    def broken(self, *args, **kwargs):
        raise ValueError("drawing failed")

    monkeypatch.setattr(pdf_report._Doc, "bars", broken)
    path = pdf_report.build_pdf(tmp_path, "run", PHASES, EVENTS, "", {"rows": 100})
    assert path.read_bytes().startswith(b"%PDF")
    error = (tmp_path / "run_pdf_error.txt").read_text()
    assert "section appendix" in error and "drawing failed" in error and "Traceback" in error


def test_registered_hypotheses_and_agent_analyses_are_rendered(tmp_path):
    phases = {**PHASES,
              "a_priori_hypotheses": [{"feature": "stepsday_rel_amplitude", "direction": -1, "knownness": "emerging",
                                       "construct": "rhythm strength", "citation": "doi:x"}],
              "analyses": [{"tool": "subgroup_analysis", "args": {"feature": "f", "by": "female"},
                            "result": {"strata": [{"stratum": "female=0", "rho": 0.1, "ci95": [0.0, 0.2]}],
                                       "heterogeneity_p": 0.3, "i2": 0.1}},
                           {"tool": "dose_response", "args": {"feature": "f"}, "result": {"error": "no rows"}}]}
    path = pdf_report.build_pdf(tmp_path, "run", phases, EVENTS, "", {"rows": 100})
    assert path.read_bytes().startswith(b"%PDF") and not (tmp_path / "run_pdf_error.txt").exists()


def test_methods_debates_lessons_and_analysis_figures_are_rendered(tmp_path):
    phases = {**PHASES,
              "analyses": [
                  {"tool": "method:subgroup_stratified", "round": 0, "args": {"by": "female"},
                   "result": {"method": "subgroup_stratified", "pool": "stat", "by": "female", "results": [
                       {"feature": "f", "by": "female", "heterogeneity_p": 0.01, "i2": 0.8,
                        "strata": [{"stratum": "female=0", "rho": 0.1, "ci95": [0.05, 0.15]},
                                   {"stratum": "female=1", "rho": 0.3, "ci95": [0.25, 0.35]}]}]}},
                  {"tool": "method:random_forest", "round": 1, "args": {},
                   "result": {"method": "random_forest", "pool": "ml", "metric": "r2", "value": 0.1,
                              "ci95": [0.05, 0.15], "null_p": 0.1,
                              "importance": [{"feature": "f", "importance": 0.2}, {"feature": "g", "importance": 0.05}]}},
                  {"tool": "dose_response", "round": 1, "args": {"feature": "f"},
                   "result": {"feature": "f", "model": "cox", "measure": "hazard ratio vs the median",
                              "nonlinearity_p": 0.4, "curve": [
                                  {"percentile": p, "value": float(p), "effect": 1 + (p - 50) / 200,
                                   "ci95": [0.9 + (p - 50) / 200, 1.1 + (p - 50) / 200]} for p in (10, 25, 50, 75, 90)]}}],
              "gap_analysis": [{"round": 0, "gaps": ["sex difference?"], "next_methods": [{"method": "subgroup_stratified",
                                                                                         "by": "female"}]}],
              "candidate_debates": [{"feature": "f", "verdict": "caveat", "reason": "smaller adjusted", "evidence": "x"}],
              "lessons": [{"lesson": "a column measures wear", "about": ["w"], "evidence": "critic"}],
              "report_review": "1. A number not in the facts."}
    from codas.aou.pdf_report import _analysis_figures

    titles = [t for t, _items, _log in _analysis_figures(phases["analyses"])]
    assert any(t.startswith("Subgroups") for t in titles) and any(t.startswith("Dose-response") for t in titles)
    assert any(t.startswith("Feature importance") for t in titles)
    path = pdf_report.build_pdf(tmp_path, "run", phases, EVENTS, "", {"rows": 100})
    assert path.read_bytes().startswith(b"%PDF") and not (tmp_path / "run_pdf_error.txt").exists()
