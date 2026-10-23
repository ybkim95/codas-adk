import os

import pytest

# The AoU runner raises the loop cap to the paper's five rounds; the scripted tests keep three.
os.environ.setdefault("CODAS_MAX_DISCOVERY_ROUNDS", "3")


@pytest.fixture(autouse=True)
def _reports_in_tmp(tmp_path, monkeypatch):
    """Runs mirror their outputs to codas_reports/; keep tests from writing into the checkout."""
    monkeypatch.setenv("CODAS_REPORTS_DIR", str(tmp_path / "codas_reports"))


@pytest.fixture(autouse=True)
def _fresh_literature_cache():
    """The literature search caches identical queries for a run; each test starts with none."""
    try:
        from codas.agents import tools
    except ImportError:  # the agent layer is optional; engine-only installs have no cache to clear
        return
    tools.LITERATURE_CACHE.clear()
