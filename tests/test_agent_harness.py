"""Harness tests: the tool-calling sandbox guardrail, the ADK tools, and engine logging.

These verify the agent harness without a live Gemini call (deterministic, offline). A live
end-to-end agent trace is exercised separately when a GOOGLE_API_KEY is configured.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codas.agents.agent import preview_columns, profile_dataset
from codas.agents.agent import run_discovery as run_discovery_tool
from codas.agents.callbacks import before_tool_guardrail
from codas.core.discovery import DiscoveryRequest, run_discovery

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_dataset.csv"


class _Tool:
    name = "run_discovery"


# --- tool-calling sandbox (path guardrail) ---

def test_sandbox_blocks_paths_outside_allowed_roots():
    blocked = before_tool_guardrail(_Tool(), {"csv_path": "/etc/passwd"}, None)
    assert isinstance(blocked, dict) and "error" in blocked


def test_sandbox_allows_paths_inside_allowed_roots():
    assert before_tool_guardrail(_Tool(), {"csv_path": str(SAMPLE)}, None) is None


# --- the deterministic ADK tools ---

def test_tool_profile_dataset():
    out = profile_dataset(str(SAMPLE))
    assert out["rows"] == 420
    assert "depression_score" in out["suggested_targets"]


def test_tool_run_discovery():
    out = run_discovery_tool(str(SAMPLE), "depression_score", top_k=5)
    assert out["fact_sheet"]["target_column"] == "depression_score"
    assert out["candidates"]


def test_tool_preview_columns():
    out = preview_columns(str(SAMPLE), limit=10)
    assert out["columns"] and len(out["preview_rows"]) == 3


# --- engine logging / trace ---

def test_engine_emits_trace_logs(caplog):
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = df["x1"] + rng.normal(0, 1, n)
    with caplog.at_level(logging.INFO, logger="codas.engine"):
        run_discovery(df, DiscoveryRequest(target_column="y", top_k=4, validation_resamples=80))
    messages = [r.message for r in caplog.records if r.name == "codas.engine"]
    assert any("discovery start" in m for m in messages)
    assert any("discovery done" in m for m in messages)
