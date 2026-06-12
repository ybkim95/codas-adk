"""CoDaS service: a thin HTTP surface over the deterministic engine and the ADK agent.

Two layers are exposed:

* ``/v1/discover`` | ``/v1/profile`` — the deterministic engine (``codas_core``). Stateless, fast,
  reproducible; identical numbers for identical input. No LLM required. The caller specifies the
  target column explicitly: the engine makes NO assumption about column names or problem domain.
* ``/v1/agent`` — the google-adk SequentialAgent (``codas_agents``) orchestrating the same
  deterministic tools over Gemini. Here the LLM chooses the target/roles from the schema and the
  task description. Requires a Gemini API key; degrades to 503 without one.

Auth is server-to-server API key (see ``codas_service.auth``). There is no browser/Firebase auth and
no local dataset registry: callers hand the data over inline, keeping the service stateless.
"""

from __future__ import annotations

import base64
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from codas_core import gemini
from codas_core.data import InsufficientDataError, profile_dataframe, read_csv_dataset
from codas_core.discovery import DiscoveryRequest, run_discovery
from codas_service.auth import require_api_key

_MAX_INLINE_CSV_BYTES = int(os.getenv("CODAS_AGENT_MAX_INLINE_CSV_MB", "50")) * 1024 * 1024
_MIN_RESAMPLES = int(os.getenv("CODAS_AGENT_MIN_RESAMPLES", "100"))
# ADK tools are guardrailed to read only under these roots; write inline CSV here for /v1/agent.
_AGENT_UPLOAD_DIR = Path(__file__).resolve().parents[1] / ".codas_runs" / "agent_uploads"

app = FastAPI(
    title="CoDaS",
    version="1.0.0",
    description="AI Co-Data-Scientist: a deterministic, domain-agnostic association-discovery "
    "engine with a google-adk + Gemini agent layer.",
)

# CORS: never wildcard-with-credentials. Allowed origins are explicit (comma-separated env), and
# credentials are off, so the browser CORS contract is respected.
_cors_origins = [o.strip() for o in os.getenv("CODAS_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/v1")


class _InlineData(BaseModel):
    csv: str | None = Field(default=None, description="Inline CSV text (utf-8).")
    csv_base64: str | None = Field(default=None, description="Base64-encoded CSV bytes (binary-safe).")


class ProfilePayload(_InlineData):
    pass


class DiscoverPayload(_InlineData):
    # The caller names the target explicitly. The engine never infers a target from column names.
    target_column: str = Field(description="Name of the outcome column to discover predictors of.")
    # Optional explicit roles. Anything left unset is simply not used (no name-based guessing):
    # without a participant id, rows are treated as independent; without confounders, none are adjusted.
    participant_id_column: str | None = None
    time_column: str | None = None
    excluded_columns: list[str] = Field(default_factory=list)
    confounder_columns: list[str] = Field(default_factory=list)
    top_k: int = 15
    validation_resamples: int = 300


class AgentPayload(_InlineData):
    query: str = Field(description="Natural-language task for the ADK agent pipeline.")
    model: str | None = Field(default=None, description="Gemini model id; defaults to CODAS_GEMINI_MODEL.")


def _decode_csv_text(data: _InlineData) -> str:
    if data.csv_base64:
        try:
            raw = base64.b64decode(data.csv_base64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="csv_base64 is not valid base64.") from exc
        if len(raw) > _MAX_INLINE_CSV_BYTES:
            raise HTTPException(status_code=413, detail=f"Inline CSV exceeds {_MAX_INLINE_CSV_BYTES} bytes.")
        return raw.decode("utf-8", errors="replace")
    if data.csv and data.csv.strip():
        if len(data.csv.encode("utf-8")) > _MAX_INLINE_CSV_BYTES:
            raise HTTPException(status_code=413, detail=f"Inline CSV exceeds {_MAX_INLINE_CSV_BYTES} bytes.")
        return data.csv
    raise HTTPException(status_code=400, detail="Provide a non-empty `csv` or `csv_base64`.")


def _load_inline_df(data: _InlineData):
    """Parse inline CSV with the canonical loader, then discard the scratch file."""
    text = _decode_csv_text(data)
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    scratch = Path(handle.name)
    try:
        handle.write(text)
        handle.close()
        return read_csv_dataset(scratch)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=f"Dataset is unusable: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc
    finally:
        scratch.unlink(missing_ok=True)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Unauthenticated liveness probe."""
    return {"status": "ok"}


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Authenticated readiness probe; also reports whether the Gemini agent layer is configured."""
    principal = require_api_key(request)
    return {"status": "ok", "principal": principal, "gemini_configured": gemini.configured()}


@router.post("/profile")
def profile(payload: ProfilePayload, request: Request) -> dict[str, Any]:
    """Structural summary of an inline CSV (dtypes, missingness, numeric columns). No inference."""
    require_api_key(request)
    return {"profile": profile_dataframe(_load_inline_df(payload)).to_dict()}


@router.post("/discover")
def discover(payload: DiscoverPayload, request: Request) -> dict[str, Any]:
    """Run the full deterministic discovery for an explicit target and return the report."""
    require_api_key(request)
    df = _load_inline_df(payload)
    if payload.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"target_column '{payload.target_column}' is not in the CSV.")
    request_obj = DiscoveryRequest(
        target_column=payload.target_column,
        participant_id_column=payload.participant_id_column,
        time_column=payload.time_column,
        excluded_columns=payload.excluded_columns,
        confounder_columns=payload.confounder_columns,
        top_k=payload.top_k,
        validation_resamples=max(payload.validation_resamples, _MIN_RESAMPLES),
    )
    try:
        report = run_discovery(df, request_obj).to_dict()
    except InsufficientDataError as exc:
        return {"status": "data_insufficient", "reason": str(exc), "target_column": payload.target_column}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "completed", "report": report}


@router.post("/agent")
def agent(payload: AgentPayload, request: Request) -> dict[str, Any]:
    """Run the google-adk SequentialAgent over the dataset using Gemini.

    The agent plans, profiles, chooses the target/roles, runs the deterministic discovery tool, and
    narrates the result. All numbers still come from the deterministic engine; the LLM only
    orchestrates and explains. Requires a Gemini API key; returns 503 if unconfigured.
    """
    require_api_key(request)
    if not gemini.configured():
        raise HTTPException(status_code=503, detail="Gemini is not configured; set GOOGLE_API_KEY to use /v1/agent.")

    # Lazy import so the engine endpoints work even if google-adk is not installed.
    from codas_agents.agent import root_agent
    from codas_agents.runtime import run_adk_agent_text

    text = _decode_csv_text(payload)
    _AGENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _AGENT_UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
    csv_path.write_text(text, encoding="utf-8")
    try:
        prompt = f"Dataset CSV path: {csv_path}\n\nTask: {payload.query}"
        result = run_adk_agent_text(
            root_agent,
            prompt,
            app_name="codas",
            user_id="service",
            session_id=uuid.uuid4().hex,
        )
    finally:
        csv_path.unlink(missing_ok=True)
    return {"status": "completed", "text": result.text, "events": result.events}


app.include_router(router)
