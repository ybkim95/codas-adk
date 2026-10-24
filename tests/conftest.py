import os

import pytest

# The AoU runner raises the loop cap to the paper's five rounds; the scripted tests keep three.
os.environ.setdefault("CODAS_MAX_DISCOVERY_ROUNDS", "3")


@pytest.fixture(autouse=True)
def _reports_in_tmp(tmp_path, monkeypatch):
    """Runs mirror their outputs to codas_reports/; keep tests from writing into the checkout."""
    monkeypatch.setenv("CODAS_REPORTS_DIR", str(tmp_path / "codas_reports"))


@pytest.fixture(autouse=True)
def _literature_offline(monkeypatch):
    """No test reaches a bibliographic service: every request fails as an unreachable network would. A test
    that needs answers installs its own recorded responses in place of ``_urlopen``."""
    import urllib.error

    from codas.core import literature

    def offline(url: str, timeout: float):
        raise urllib.error.URLError("the network is off in tests")

    monkeypatch.setattr(literature, "_urlopen", offline)
    monkeypatch.setattr(literature, "_sleep", lambda _s: None)
    for key in literature.EGRESS:
        literature.EGRESS[key] = 0


@pytest.fixture
def europe_pmc(monkeypatch):
    """Recorded Europe PMC answers: ``install({term: (papers, reviews, [paper, ...])})`` makes a search whose
    query names ``term`` find that many papers and reviews, the listed papers first (by citations, or the
    oldest first when sorted by date); any other search finds nothing. Returns the URLs asked for."""
    import json
    import urllib.parse

    from codas.core import literature

    def install(answers: dict) -> list:
        asked: list[str] = []

        def opener(url: str, timeout: float):
            asked.append(url)
            params = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}
            query = params.get("query", "").lower()
            hit = next((v for term, v in answers.items() if term.lower() in query), (0, 0, []))
            total, reviews, papers = hit
            if "pub_type:" in query:
                total, papers = reviews, papers[:1]
            elif params.get("sort", "").startswith("P_PDATE_D"):
                papers = sorted(papers, key=lambda p: p.get("pubYear", 0))[:1]
            results = papers[: int(params.get("pageSize", "10"))]
            return 200, json.dumps({"hitCount": total, "resultList": {"result": results}}).encode()

        monkeypatch.setattr(literature, "_urlopen", opener)
        return asked

    return install


@pytest.fixture(autouse=True)
def _fresh_tool_usage():
    """Tools record the model calls they make on their own for the execution report; each test starts
    with none recorded."""
    try:
        from codas.agents import tools
    except ImportError:  # the agent layer is optional; engine-only installs record nothing
        return
    tools.MODEL_USAGE.clear()
