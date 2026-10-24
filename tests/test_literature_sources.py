"""Papers a citation can point to: Europe PMC and Crossref answers parsed from recorded responses (no network),
novelty classed from counts, the circuit breaker, retries, the cache, query sanitising and citation tokens."""

from __future__ import annotations

import json
import urllib.error

import pytest

from codas.core import literature

STEPS = {"pmid": "30000001", "doi": "10.1000/steps.1", "title": "Daily <i>step</i> count and glycaemia",
         "journalTitle": "A Journal", "pubYear": "2019", "citedByCount": 210,
         "pubTypeList": {"pubType": ["review", "Journal Article"]}, "abstractText": "<h4>Background</h4>Steps."}
OLDEST = {"pmid": "10000001", "title": "An early step count study", "journalTitle": "B Journal", "pubYear": "1998",
          "citedByCount": 3, "pubType": "research-article"}


def test_europe_pmc_records_become_citable_papers(europe_pmc):
    europe_pmc({"step count": (2, 1, [STEPS])})
    hits, (paper,) = literature.europe_pmc('TITLE:"step count"', page_size=5)
    assert hits == 2 and paper.key == "PMID:30000001" and paper.doi == "10.1000/steps.1"
    assert paper.title == "Daily step count and glycaemia" and paper.abstract == "BackgroundSteps."
    assert paper.year == 2019 and paper.cited_by == 210 and "review" in paper.pub_types
    assert literature.Paper(source="x", title="t", doi="10.1/a").key == "DOI:10.1/a"


def test_novelty_is_counted_not_judged(europe_pmc):
    asked = europe_pmc({"step count": (40, 0, [STEPS, OLDEST]), "peak cadence": (3, 1, [STEPS]),
                        "sleep regularity": (4, 0, [OLDEST])})
    hba1c = ["HbA1c", "glycated hemoglobin"]
    many = literature.evidence("daily step count", ["step count"], hba1c)
    assert many["novelty"] == "established" and many["papers"] == 40 and many["earliest_year"] == 1998
    assert [p["key"] for p in many["top_cited"]] == ["PMID:30000001", "PMID:10000001"]
    assert literature.evidence("peak cadence", [], hba1c)["novelty"] == "established"     # one review suffices
    assert literature.evidence("sleep regularity", [], hba1c)["novelty"] == "emerging"
    before = len(asked)
    none = literature.evidence("REM latency", ["REM sleep latency"], hba1c)
    assert none["novelty"] == "unexplored" and none["papers"] == 0 and len(asked) == before + 1  # one query only
    assert 'TITLE:"REM sleep latency" OR ABSTRACT:"REM sleep latency"' in none["query"]


def test_an_unreachable_service_means_not_assessed_and_is_asked_once():
    calls = []

    def down(url, timeout):
        calls.append(url)
        raise urllib.error.URLError("no route")

    terms = {"constructs": {"a": {"synonyms": ["x"]}, "b": {"synonyms": ["y"]}, "c": {"synonyms": ["z"]}}}
    literature._urlopen, saved = down, literature._urlopen
    try:
        matrix = literature.evidence_matrix(["a", "b", "c"], ["outcome"], terms)
    finally:
        literature._urlopen = saved
    assert [m["novelty"] for m in matrix.values()] == ["not assessed"] * 3 and len(calls) == 1
    assert "no route" in matrix["c"]["error"]


def test_rate_limits_are_retried_and_missing_dois_are_none(monkeypatch):
    answers = iter([(429, b""), (200, json.dumps({"hitCount": 1, "resultList": {"result": [STEPS]}}).encode())])
    monkeypatch.setattr(literature, "_urlopen", lambda url, timeout: next(answers))
    assert literature.europe_pmc("q")[0] == 1
    monkeypatch.setattr(literature, "_urlopen", lambda url, timeout: (404, b"{}"))
    assert literature.lookup("DOI:10.9999/none") is None
    monkeypatch.setattr(literature, "_urlopen", lambda url, timeout: (500, b""))
    with pytest.raises(literature.Unreachable):
        literature.europe_pmc("q")


def test_a_doi_and_a_pmid_are_looked_up(monkeypatch):
    def opener(url, timeout):
        if "crossref" in url:
            return 200, json.dumps({"message": {"DOI": "10.1038/x", "title": ["A Nature paper"],
                                                "container-title": ["Nature Medicine"],
                                                "published": {"date-parts": [[2022, 10]]}, "type": "journal-article",
                                                "is-referenced-by-count": 99}}).encode()
        return 200, json.dumps({"hitCount": 1, "resultList": {"result": [STEPS]}}).encode()

    monkeypatch.setattr(literature, "_urlopen", opener)
    doi = literature.lookup("DOI:10.1038/x")
    assert (doi.title, doi.journal, doi.year, doi.cited_by) == ("A Nature paper", "Nature Medicine", 2022, 99)
    assert literature.lookup("PMID:30000001").key == "PMID:30000001"


def test_the_cache_answers_a_repeated_query_without_asking_again(tmp_path, europe_pmc):
    asked = europe_pmc({"step count": (5, 0, [STEPS])})
    cache = literature.Cache(tmp_path / "cache.jsonl")
    literature.europe_pmc('"step count"', cache=cache)
    literature.europe_pmc('"step count"', cache=literature.Cache(tmp_path / "cache.jsonl"))  # a new process
    assert len(asked) == 1


def test_only_sanitised_query_text_leaves(europe_pmc):
    asked = europe_pmc({})
    text = literature.sanitize("steps and HbA1c in 12 participants, person 123456789012 " + "x" * 400)
    assert "123456789012" not in text and "[1-20] participants" in text and len(text) == 300
    literature.evidence("daily step count", ["steps 1234567"], ["HbA1c"])
    assert "1234567" not in asked[0]


def test_citation_tokens_and_the_reference_list():
    text = "As [PMID:30000001] showed, and [DOI:10.1038/s41591-022-02012-w]; again [PMID:30000001]. [PMID:x]"
    keys = literature.tokens(text)
    assert keys == ["PMID:30000001", "DOI:10.1038/s41591-022-02012-w", "PMID:x"]  # a malformed key is still one
    store = {"PMID:30000001": {"key": "PMID:30000001", "pmid": "30000001", "title": "Steps", "journal": "J",
                               "year": 2019}}
    refs = literature.references_markdown(keys, store)
    assert refs == "1. Steps. J, 2019. [PMID:30000001](https://europepmc.org/article/MED/30000001)"


def test_columns_map_to_constructs_by_phenotype_kind_or_name_pattern():
    terms = literature.load_terms(literature.__file__.rsplit("codas", 1)[0] + "scripts/aou/literature_terms.json")
    kinds = {"hr10h": "minute_heart_rate", "stepsday": "minute_activity", "sleepseg": "sleep_stages"}
    found = literature.constructs_of(["act_steps_mean", "act_steps_sd", "act_days", "hr10h_rel_amplitude",
                                      "stepsday_rel_amplitude", "hr10h_sedentary_hr", "sleepseg_midpoint_sd",
                                      "mainsleep_minute_asleep_mean"], terms, kinds)
    assert found == {"act_steps_mean": "daily step count",
                     "act_steps_sd": "day-to-day variability of physical activity",
                     "hr10h_rel_amplitude": "circadian heart rate rhythm",       # same name, different kind
                     "stepsday_rel_amplitude": "rest-activity rhythm",
                     "hr10h_sedentary_hr": "resting heart rate", "sleepseg_midpoint_sd": "sleep regularity",
                     "mainsleep_minute_asleep_mean": "sleep duration"}           # wear counts name no construct
    assert set(terms["outcomes"]) >= {"hba1c", "t2d", "depression"}


def test_a_bibliography_is_checked_doi_by_doi(monkeypatch):
    from codas.aou import notebook_views as views

    text = ("| 1 | Master H. Association of step counts over time with the risk of chronic disease. Nat Med 2022. "
            "[doi](https://doi.org/10.1038/s41591-022-02012-w) |\n| 2 | Someone. A paper. 10.9999/missing. |\n"
            "| 3 | Other. Unrelated title words. https://doi.org/10.1000/other |")
    items = literature.references_in(text)
    assert [i["doi"] for i in items] == ["10.1038/s41591-022-02012-w", "10.9999/missing", "10.1000/other"]

    def opener(url, timeout):
        if "missing" in url:
            return 404, b"{}"
        title = ("Association of step counts over time with the risk of chronic disease in the All of Us Research "
                 "Program" if "s41591" in url else "Completely different subject matter")
        return 200, json.dumps({"message": {"DOI": "x", "title": [title], "container-title": ["J"],
                                            "published": {"date-parts": [[2022]]}}}).encode()

    monkeypatch.setattr(literature, "_urlopen", opener)
    rows = literature.verify_references(items)
    assert [r["status"] for r in rows] == ["found", "not found", "found"]
    assert rows[0]["title_in_citation"] > 0.6 and rows[2]["title_in_citation"] == 0
    assert views.references_summary(rows).set_index("status")["DOIs"].to_dict() == {"found": 2, "not found": 1}
    assert list(views.references_doubtful(rows)["doi"]) == ["10.9999/missing", "10.1000/other"]


def test_an_unreachable_index_leaves_the_rest_unchecked():
    rows = literature.verify_references([{"doi": "10.1000/a", "context": "a"}, {"doi": "10.1000/b", "context": "b"}])
    assert [r["status"] for r in rows] == ["not checked", "not checked"] and "network is off" in rows[0]["note"]


def test_the_research_basis_cites_more_than_fifty_papers_by_doi():
    from pathlib import Path

    folder = Path(literature.__file__).resolve().parents[2] / "docs" / "research"
    items = literature.references_in("\n".join(p.read_text() for p in sorted(folder.glob("*.md"))))
    assert len(items) >= 50 and (folder / "README.md").exists()


def test_effect_statements_are_quoted_as_written():
    from codas.core import literature

    abstract = ("Background. Cohort of adults. Higher sleep regularity was associated with lower mortality "
                "(HR 0.70, 95% CI 0.62-0.79). The odds ratio for diabetes was 1.31 (1.10 to 1.56); "
                "results held. No interval here, HR 1.2.")
    found = literature.effect_statements(abstract, ["sleep regularity"])
    assert found[0] == "Higher sleep regularity was associated with lower mortality (HR 0.70, 95% CI 0.62-0.79)."
    assert all(s in abstract for s in found) and not any("No interval" in s for s in found)
    assert literature.effect_statements("", ["x"]) == []
