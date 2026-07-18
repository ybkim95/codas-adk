"""Engine smoke + determinism tests (no LLM, no network)."""

from pathlib import Path

from codas.core.discovery import DiscoveryRequest, run_discovery_from_csv

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_dataset.csv"


def _run():
    return run_discovery_from_csv(
        SAMPLE,
        DiscoveryRequest(target_column="depression_score", top_k=5, validation_resamples=150, random_state=17),
    )


def test_discovery_runs_and_reports_candidates():
    report = _run().to_dict()
    assert report["fact_sheet"]["target_column"] == "depression_score"
    assert report["fact_sheet"]["rows"] > 0
    assert len(report["candidates"]) > 0
    # The sample is designed to carry a recoverable signal: at least one candidate should validate.
    assert any(c["verdict"] == "validated" for c in report["candidates"])


def test_engine_is_deterministic():
    a = _run().to_dict()["candidates"]
    b = _run().to_dict()["candidates"]
    assert [c["feature"] for c in a] == [c["feature"] for c in b]
    assert [round(c["rho"], 10) for c in a] == [round(c["rho"], 10) for c in b]
