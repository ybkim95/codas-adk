"""The citation contract: hypotheses cite papers retrieved in this run by key, their knownness comes from the
counted evidence, the paper search fills the run's store, and the report's references are checked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from codas.agents import tools
from codas.aou import cohort, runner
from codas.aou.cohort import CohortSpec

STEPS = {"key": "PMID:30000001", "pmid": "30000001", "title": "Daily step count and glycaemia", "journal": "J",
         "year": 2019}


def _context(tmp_path, **state):
    frame = pd.DataFrame({"hba1c": np.linspace(5, 7, 30), "act_steps_mean": np.linspace(1, 2, 30),
                          "sleep_minute_asleep_mean": np.linspace(3, 4, 30)})
    frame.to_csv(tmp_path / "c.csv", index=False)
    return SimpleNamespace(state={"csv_path": str(tmp_path / "c.csv"), "target_column": "hba1c", **state})


def _register(ctx, *hypotheses):
    return tools.register_hypotheses(json.dumps(list(hypotheses)), ctx)


def _hypothesis(feature="act_steps_mean", citation="", knownness="emerging"):
    return {"feature": feature, "direction": "negative", "construct": "c", "mechanism": "m", "citation": citation,
            "knownness": knownness}


def test_with_retrieved_papers_a_hypothesis_must_cite_one_of_them(tmp_path):
    evidence = {"daily step count": {"novelty": "established", "papers": 40, "reviews": 2, "earliest_year": 1998}}
    ctx = _context(tmp_path, literature_store={"PMID:30000001": STEPS}, literature_evidence=evidence,
                   construct_of={"act_steps_mean": "daily step count"})
    out = _register(ctx, _hypothesis(), _hypothesis("sleep_minute_asleep_mean", "[PMID:99999999]"))
    assert [r["feature"] for r in out["rejected"]] == ["act_steps_mean", "sleep_minute_asleep_mean"]
    assert "cite at least one retrieved paper" in out["rejected"][0]["reason"]
    assert "PMID:99999999 was not retrieved" in out["rejected"][1]["reason"]
    out = _register(ctx, _hypothesis(citation="as shown [PMID:30000001]", knownness="unexplored"),
                    _hypothesis("sleep_minute_asleep_mean", "[PMID:30000001]"))
    steps, sleep = out["registered"]
    # The counted evidence decides; the agent's own label is kept beside it.
    assert steps["knownness"] == "established" and steps["declared_knownness"] == "unexplored"
    assert steps["evidence"]["papers"] == 40 and steps["citations"] == ["PMID:30000001"]
    assert sleep["knownness"] == "not assessed" and sleep["evidence"] is None   # its construct was not counted


def test_without_retrieved_papers_nothing_can_be_cited_and_novelty_is_not_assessed(tmp_path):
    ctx = _context(tmp_path)
    out = _register(ctx, _hypothesis(), _hypothesis("sleep_minute_asleep_mean", "[DOI:10.1/made.up]"))
    assert [h["feature"] for h in out["registered"]] == ["act_steps_mean"]
    assert out["registered"][0]["knownness"] == "not assessed"
    assert "was not retrieved" in out["rejected"][0]["reason"]


def test_the_paper_search_fills_the_runs_store(tmp_path, europe_pmc, monkeypatch):
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "1")
    europe_pmc({"step count": (7, 1, [{"pmid": "30000001", "title": "Steps", "journalTitle": "J", "pubYear": "2019",
                                       "abstractText": "a" * 900}])})
    ctx = _context(tmp_path, literature_cache_path=str(tmp_path / "cache.jsonl"))
    found = tools.search_papers("step count and HbA1c in 12 participants", ctx)
    assert found["hits"] == 7 and [p["key"] for p in found["papers"]] == ["PMID:30000001"]
    assert "[1-20] participants" in found["query"] and len(found["papers"][0]["abstract"]) == 400
    assert "PMID:30000001" in ctx.state["literature_store"]
    assert _register(ctx, _hypothesis(citation="[PMID:30000001]"))["registered"]
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "0")
    assert tools.search_papers("anything", ctx)["papers"] == []


def test_an_unreachable_paper_search_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "1")
    out = tools.search_papers("step count", _context(tmp_path))
    assert out["papers"] == [] and "did not answer" in out["note"]


def test_the_runner_counts_the_evidence_before_the_agents(tmp_path, europe_pmc, monkeypatch):
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "1")
    asked = europe_pmc({"step count": (40, 2, [{"pmid": "30000001", "title": "Steps", "pubYear": "2019"}])})
    spec = CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    state = runner.literature_evidence(spec, "HbA1c", tmp_path)
    assert state["literature_status"] == "available" and "PMID:30000001" in state["literature_store"]
    assert state["construct_of"]["act_steps_mean"] == "daily step count"
    assert state["construct_of"]["hr10h_sleep_dip"] == "heart rate during sleep"
    assert "act_days" not in state["construct_of"] and "bmi_cov" not in state["construct_of"]
    assert state["literature_evidence"]["daily step count"]["novelty"] == "established"
    assert "daily step count (act_steps_mean): established" in state["literature_digest"]
    assert all("HbA1c" in __import__("urllib.parse").parse.unquote(u) for u in asked)  # the catalog's outcome terms
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "0")
    assert runner.literature_evidence(spec, "HbA1c", tmp_path)["literature_status"] == "off"


def test_an_offline_run_says_the_evidence_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("CODAS_AOU_LITERATURE_SEARCH", "1")
    spec = CohortSpec.from_json(cohort.ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    state = runner.literature_evidence(spec, "HbA1c", tmp_path)
    assert state["literature_status"].startswith("unavailable") and not state["literature_store"]
    assert all(e["novelty"] == "not assessed" for e in state["literature_evidence"].values())


def test_the_report_lists_the_retrieved_papers_it_cites_and_flags_the_rest():
    state = {"literature_store": {"PMID:30000001": STEPS}, "report": "Steps matter [PMID:30000001].",
             "mechanisms": "See [DOI:10.1/made.up].", "a_priori_hypotheses": [{"citation": "[PMID:30000001]"}]}
    runner._with_references(state)
    assert state["citations"] == {"used": ["PMID:30000001", "DOI:10.1/made.up"], "verified": ["PMID:30000001"],
                                  "unverified": ["DOI:10.1/made.up"], "retrieved": 1}
    assert "## References\n\n1. Daily step count and glycaemia. J, 2019." in state["report"]
    assert "Not citations: DOI:10.1/made.up" in state["report"]
    quiet = {"report": "No citations here."}
    runner._with_references(quiet)
    assert quiet["report"] == "No citations here." and quiet["citations"]["used"] == []
