"""Which literature sources answer from the workspace, and what leaves the machine when they are asked.

Offline: the HTTP opener and the grounded model are fakes."""

from __future__ import annotations

import pytest

from codas.core import literature
from codas.core.gemini import GroundedResponse


def _opener(calls):
    def opener(url, timeout):
        calls.append(url)
        if "openalex" in url:
            raise OSError("Connection refused")
        if "semanticscholar" in url:
            return 429, b'{"message": "Too Many Requests"}'
        return 200, b'{"hitCount": 1}'
    return opener


def test_the_probe_reports_every_source_and_sends_only_a_fixed_word(monkeypatch):
    monkeypatch.delenv("SPRINGER_NATURE_API_KEY", raising=False)
    calls: list[str] = []
    before = dict(literature.EGRESS)
    grounded = GroundedResponse(text="Evidence.", model="gemini-3.8-flash", configured=True,
                                sources=[{"title": "a", "url": "https://example.org"}], queries=["q"])
    rows = {r["source"]: r for r in literature.probe(opener=_opener(calls), grounding=lambda q: grounded)}

    assert rows["Europe PMC"]["reachable"] and rows["Europe PMC"]["status"] == 200
    assert rows["Crossref (Nature Portfolio DOIs, 10.1038)"]["reachable"]
    assert not rows["OpenAlex"]["reachable"] and "Connection refused" in rows["OpenAlex"]["note"]
    assert not rows["Semantic Scholar"]["reachable"] and rows["Semantic Scholar"]["status"] == 429
    # A keyed source is not tried without its key, and says how Nature papers are found instead.
    springer = rows["Springer Nature Metadata API"]
    assert springer["reachable"] is False and "SPRINGER_NATURE_API_KEY" in springer["note"]
    assert not any("springernature" in u for u in calls)
    assert rows["Google Search grounding (Gemini, Vertex)"]["reachable"] and "caching" in \
        rows["Google Search grounding (Gemini, Vertex)"]["note"]
    assert rows["Google Scholar"]["reachable"] is None and "terms" in rows["Google Scholar"]["note"]

    assert rows["Europe PMC, fielded query with a review filter"]["reachable"]
    assert rows["Crossref, lookup by DOI"]["reachable"] and rows["Europe PMC, lookup by DOI"]["reachable"]
    # Only the fixed probe word or the known DOI went out, and every request was counted.
    assert calls and all(literature.PROBE_QUERY in u or literature.KNOWN_DOI.split("/")[1] in u for u in calls)
    assert literature.EGRESS["requests"] - before["requests"] == len(calls)
    assert literature.EGRESS["sent_bytes"] > before["sent_bytes"]


def test_literature_requests_stop_at_the_cap(monkeypatch):
    monkeypatch.setattr(literature, "MAX_REQUESTS", literature.EGRESS["requests"])
    calls: list[str] = []
    rows = literature.probe(opener=_opener(calls))
    assert calls == [] and all(not r["reachable"] for r in rows if r["reachable"] is not None)
    with pytest.raises(literature.EgressLimit):
        literature.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=x", opener=_opener(calls))
