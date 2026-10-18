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
