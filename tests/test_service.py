"""Service-layer tests: auth and the deterministic endpoints (explicit target, no inference)."""

import os

os.environ["CODAS_AGENT_API_KEYS"] = "test-key"  # set before importing the app

from pathlib import Path

from fastapi.testclient import TestClient

from codas.service.app import app

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


# --- resilience: a transient model-backend error must not surface as a 500 ---

class _FakeResult:
    text = "ok"
    events: list = []


def _force_gemini_configured(monkeypatch):
    from codas.core import gemini
    monkeypatch.setattr(gemini, "configured", lambda: True)


def test_agent_transient_llm_error_retries_then_returns_503(monkeypatch):
    import codas.agents.runtime as rt
    _force_gemini_configured(monkeypatch)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("503 UNAVAILABLE: model is experiencing high demand")

    monkeypatch.setattr(rt, "run_adk_agent_text", boom)
    monkeypatch.setenv("CODAS_AGENT_RETRIES", "1")
    import codas.service.app as appmod
    monkeypatch.setattr(appmod, "_AGENT_RETRIES", 1)
    r = client.post("/v1/agent", headers=HEADERS, json={"csv": CSV, "query": "find predictors"})
    assert r.status_code == 503, r.text          # retryable, not a 500
    assert calls["n"] == 2                         # one try + one retry


def test_agent_retries_then_succeeds(monkeypatch):
    import codas.agents.runtime as rt
    _force_gemini_configured(monkeypatch)
    seq = [RuntimeError("503 high demand"), _FakeResult()]

    def flaky(*a, **k):
        nxt = seq.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(rt, "run_adk_agent_text", flaky)
    r = client.post("/v1/agent", headers=HEADERS, json={"csv": CSV, "query": "find predictors"})
    assert r.status_code == 200 and r.json()["status"] == "completed"


def test_agent_non_transient_error_is_not_swallowed(monkeypatch):
    import codas.agents.runtime as rt
    _force_gemini_configured(monkeypatch)

    def real_bug(*a, **k):
        raise ValueError("a genuine programming error")

    monkeypatch.setattr(rt, "run_adk_agent_text", real_bug)
    quiet = TestClient(app, raise_server_exceptions=False)
    r = quiet.post("/v1/agent", headers=HEADERS, json={"csv": CSV, "query": "x"})
    assert r.status_code == 500  # a real bug must NOT be masked as a friendly 503


def test_old_uploads_are_pruned(tmp_path, monkeypatch):
    """Disk stays bounded: upload CSVs past the TTL are removed, recent ones kept."""
    import os
    import time as _time

    import codas.service.app as appmod
    monkeypatch.setattr(appmod, "_AGENT_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(appmod, "_UPLOAD_TTL_SECONDS", 100)
    old = tmp_path / "old.csv"; old.write_text("x")
    recent = tmp_path / "recent.csv"; recent.write_text("x")
    past = _time.time() - 500
    os.utime(old, (past, past))
    appmod._prune_old_uploads()
    assert not old.exists() and recent.exists()
