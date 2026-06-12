"""Service-layer tests: auth and the deterministic endpoints (explicit target, no inference)."""

import os

os.environ["CODAS_AGENT_API_KEYS"] = "test-key"  # set before importing the app

from pathlib import Path

from fastapi.testclient import TestClient

from codas_service.app import app

client = TestClient(app)
CSV = (Path(__file__).resolve().parents[1] / "examples" / "sample_dataset.csv").read_text()
HEADERS = {"X-CoDaS-Agent-Key": "test-key"}


def test_healthz_is_open():
    assert client.get("/healthz").json()["status"] == "ok"


def test_auth_required_with_keys_configured():
    assert client.get("/v1/health").status_code == 401
    assert client.get("/v1/health", headers={"X-CoDaS-Agent-Key": "wrong"}).status_code == 401
    assert client.get("/v1/health", headers=HEADERS).status_code == 200


def test_discover_requires_explicit_target():
    r = client.post(
        "/v1/discover",
        headers=HEADERS,
        json={"csv": CSV, "target_column": "depression_score", "top_k": 5, "validation_resamples": 120},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["report"]["fact_sheet"]["target_column"] == "depression_score"
    assert len(body["report"]["candidates"]) > 0


def test_discover_rejects_unknown_target():
    r = client.post("/v1/discover", headers=HEADERS, json={"csv": CSV, "target_column": "not_a_column"})
    assert r.status_code == 400


def test_profile_is_structural_only():
    r = client.post("/v1/profile", headers=HEADERS, json={"csv": CSV})
    assert r.status_code == 200, r.text
    profile = r.json()["profile"]
    # suggested_targets is just the numeric columns (no name-based ranking).
    assert "depression_score" in profile["suggested_targets"]
    assert profile["rows"] == 420
