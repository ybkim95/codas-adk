"""Unit tests for the Section 2.6 numeric verification (count correction) pass."""

from __future__ import annotations

from codas_agents.numeric_audit import verify_and_correct

_FS = {"rows": 7497, "candidate_features_screened": 228, "internal_battery_passing_variants": 12}


def test_corrects_sample_size_near_miss():
    text = "We analysed 7,500 participants across three cohorts."
    fixed, corr = verify_and_correct(text, _FS)
    assert "7,497 participants" in fixed
    assert corr == [{"key": "rows", "from": 7500, "to": 7497}]


def test_corrects_n_equals_label():
    fixed, corr = verify_and_correct("The DWB cohort (N = 7480) was used.", _FS)
    assert "N = 7,497" in fixed and corr[0]["to"] == 7497


def test_leaves_exact_values_untouched():
    text = "We analysed 7,497 participants; 228 features were screened; 12 validated candidates remained."
    fixed, corr = verify_and_correct(text, _FS)
    assert fixed == text and corr == []


def test_does_not_rewrite_large_discrepancy():
    # 9,000 is >5% from 7,497 -> a possible fabrication left for the grounding audit, not a typo.
    fixed, corr = verify_and_correct("A surprising 9,000 participants were included.", _FS)
    assert fixed == "A surprising 9,000 participants were included." and corr == []


def test_corrects_feature_and_candidate_counts():
    fixed, corr = verify_and_correct("230 features were screened and 13 validated biomarkers survived.", _FS)
    assert "228 features were screened" in fixed
    assert "228" in fixed and "13 validated biomarkers" not in fixed  # 13 -> 12
    keys = {c["key"] for c in corr}
    assert keys == {"candidate_features_screened", "internal_battery_passing_variants"}


def test_missing_fact_sheet_is_safe():
    assert verify_and_correct("7,500 participants", None) == ("7,500 participants", [])


class _FakeContext:
    def __init__(self, state):
        self.state = state


def test_report_callback_corrects_and_writes_audit_file(tmp_path, monkeypatch):
    from codas_agents.callbacks import report_grounding_audit

    monkeypatch.setenv("CODAS_AUDIT_DIR", str(tmp_path))
    ctx = _FakeContext({
        "report": "The cohort had 7,510 participants; sleep variability rho=0.252.",
        "fact_sheet": {"rows": 7497, "candidate_features_screened": 228},
        "latest_report": {"candidates": [{"rho": 0.252}]},
        "rounds": [],
    })
    report_grounding_audit(ctx)  # never raises
    assert "7,497 participants" in ctx.state["report"]  # count corrected toward ground truth
    audits = list(tmp_path.glob("numeric_audit_*.json"))
    assert len(audits) == 1  # per-run audit file written
