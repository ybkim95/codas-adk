import pytest


@pytest.fixture(autouse=True)
def _reports_in_tmp(tmp_path, monkeypatch):
    """Runs mirror their outputs to codas_reports/; keep tests from writing into the checkout."""
    monkeypatch.setenv("CODAS_REPORTS_DIR", str(tmp_path / "codas_reports"))
