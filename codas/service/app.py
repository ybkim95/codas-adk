"""CoDaS service: a thin HTTP surface over the deterministic engine and the ADK agent.

Two layers are exposed:

* ``/v1/discover`` | ``/v1/profile`` — the deterministic engine (``codas.core``). Stateless, fast,
  reproducible; identical numbers for identical input. No LLM required. The caller specifies the
  target column explicitly: the engine makes NO assumption about column names or problem domain.
* ``/v1/agent`` (+ ``/v1/agent/feedback``) — the google-adk Orchestrator (``codas.agents``) running
  the six-phase pipeline over the same deterministic tools with Gemini. Here the LLM chooses
  the target/roles from the schema and the task, iterates a deepening search loop, and writes the
  report; ``/v1/agent/feedback`` resumes the SAME session so a domain expert can steer an optional
  next iteration. Requires a Gemini API key; degrades to 503 without one.

Auth is server-to-server API key (see ``codas.service.auth``). There is no browser/Firebase auth and
no local dataset registry: callers hand the data over inline, keeping the service stateless.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from codas.core import gemini
from codas.core.data import InsufficientDataError, profile_dataframe, read_csv_dataset
from codas.core.discovery import DiscoveryRequest, run_discovery
from codas.service.auth import require_api_key

_MAX_INLINE_CSV_BYTES = int(os.getenv("CODAS_AGENT_MAX_INLINE_CSV_MB", "50")) * 1024 * 1024
_MIN_RESAMPLES = int(os.getenv("CODAS_AGENT_MIN_RESAMPLES", "100"))
# ADK tools are guardrailed to read only under these roots; write inline CSV here for /v1/agent.
# The file is keyed by session id and kept for the session's lifetime so a feedback turn can re-read
# it. (In multi-instance production, back this with object storage and a TTL instead of local disk.)
_AGENT_UPLOAD_DIR = Path(__file__).resolve().parents[2] / ".codas_runs" / "agent_uploads"
_AGENT_APP_NAME = "codas"
# Bound disk: agent upload CSVs are kept for feedback re-entry, then pruned past this TTL so a
# long-running instance does not accumulate them without limit.
_UPLOAD_TTL_SECONDS = int(os.getenv("CODAS_AGENT_UPLOAD_TTL_HOURS", "24")) * 3600


def _prune_old_uploads() -> None:
    """Best-effort removal of agent upload CSVs older than the TTL (keeps disk bounded)."""
    if not _AGENT_UPLOAD_DIR.exists():
        return
    cutoff = time.time() - _UPLOAD_TTL_SECONDS
    for path in _AGENT_UPLOAD_DIR.glob("*.csv"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            pass

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


class AgentFeedbackPayload(BaseModel):
    session_id: str = Field(description="The session_id returned by /v1/agent, to resume in place.")
    feedback: str = Field(description="Domain-expert feedback to steer the next discovery iteration.")


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
    return {"status": "ok", "principal": principal, "gemini": gemini.status()}


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


# Process-persistent session store, so a /v1/agent/feedback turn can resume the SAME session (with
# all prior rounds and memory) instead of restarting the discovery from scratch.
_AGENT_SESSIONS = None


def _agent_sessions():
    global _AGENT_SESSIONS
    if _AGENT_SESSIONS is None:
        from codas.agents.runtime import new_session_service

        _AGENT_SESSIONS = new_session_service()
    return _AGENT_SESSIONS


# One internal retry for a transient model-backend error (a 503/429/500 spike), so a temporary Gemini
# overload does not surface to the caller as a 500. Configurable; backoff is linear and short.
_AGENT_RETRIES = int(os.getenv("CODAS_AGENT_RETRIES", "1"))


def _is_transient_llm_error(exc: Exception) -> bool:
    """True for a retryable model-backend hiccup (overload / rate limit / internal), not a real bug."""
    blob = f"{type(exc).__name__} {exc}"
    return any(s in blob for s in ("ServerError", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                                   "429", "500 INTERNAL", "high demand"))


@router.post("/agent")
def agent(payload: AgentPayload, request: Request) -> dict[str, Any]:
    """Run the google-adk Orchestrator (six-phase pipeline) over the dataset using Gemini.

    The agents plan, profile, choose the target/roles, iterate a deepening discovery loop with
    parallel statistical/ML interpretation and adversarial validation, then write a report. All
    numbers still come from the deterministic engine; the LLMs orchestrate, interpret, and explain.
    The dataset path is seeded into shared memory and the session is persisted, so the returned
    ``session_id`` can be handed to ``/v1/agent/feedback`` to steer an optional next iteration.
    Requires a Gemini API key; returns 503 if unconfigured.
    """
    require_api_key(request)
    if not gemini.configured():
        raise HTTPException(status_code=503, detail="Gemini is not configured; set GOOGLE_API_KEY to use /v1/agent.")

    from codas.agents.agent import root_agent  # lazy: engine endpoints work without google-adk installed
    from codas.agents.runtime import run_adk_agent_text

    text = _decode_csv_text(payload)
    _AGENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _prune_old_uploads()
    # Each attempt is a fully fresh session + upload, so an internal retry never resumes partial state.
    last_exc: Exception | None = None
    for attempt in range(_AGENT_RETRIES + 1):
        session_id = uuid.uuid4().hex
        csv_path = _AGENT_UPLOAD_DIR / f"{session_id}.csv"
        csv_path.write_text(text, encoding="utf-8")
        try:
            result = run_adk_agent_text(
                root_agent,
                f"Task: {payload.query}",  # the dataset is in shared memory (state['csv_path'])
                app_name=_AGENT_APP_NAME,
                user_id="service",
                session_id=session_id,
                state={"csv_path": str(csv_path)},
                session_service=_agent_sessions(),
            )
            return {"status": "completed", "session_id": session_id, "text": result.text, "events": result.events}
        except TimeoutError as exc:
            csv_path.unlink(missing_ok=True)
            raise HTTPException(status_code=504, detail="Discovery exceeded its time budget; try a smaller dataset or fewer rounds.") from exc
        except Exception as exc:  # noqa: BLE001
            csv_path.unlink(missing_ok=True)
            if not _is_transient_llm_error(exc):
                raise
            last_exc = exc
            if attempt < _AGENT_RETRIES:
                time.sleep(2 * (attempt + 1))
    raise HTTPException(status_code=503, detail=f"Model backend temporarily unavailable; retry shortly. ({last_exc})")


@router.post("/agent/feedback")
def agent_feedback(payload: AgentFeedbackPayload, request: Request) -> dict[str, Any]:
    """Resume a finished discovery with domain-expert feedback (the optional human-in-the-loop).

    Re-enters the SAME session — its target/roles, prior rounds, and Fact Sheet are intact in shared
    memory — so the orchestrator incorporates the feedback and runs further deepening rounds (the loop
    continues from the existing round count) rather than starting over. Returns the updated report.
    """
    require_api_key(request)
    if not gemini.configured():
        raise HTTPException(status_code=503, detail="Gemini is not configured; set GOOGLE_API_KEY to use /v1/agent.")
    if not (_AGENT_UPLOAD_DIR / f"{payload.session_id}.csv").exists():
        raise HTTPException(status_code=404, detail="Unknown or expired session_id; start a new /v1/agent run.")

    from codas.agents.agent import root_agent
    from codas.agents.runtime import run_adk_agent_text

    prompt = (
        "Domain-expert feedback on the discovery so far. Incorporate it, and if it calls for more evidence "
        f"run further discovery rounds before revising the report.\n\nFeedback: {payload.feedback}"
    )
    try:
        result = run_adk_agent_text(
            root_agent,
            prompt,
            app_name=_AGENT_APP_NAME,
            user_id="service",
            session_id=payload.session_id,  # resume in place: prior memory carries over
            session_service=_agent_sessions(),
        )
    except Exception as exc:  # noqa: BLE001
        # No auto-retry here (the session is mid-mutation); surface a transient backend hiccup as a
        # retryable 503 rather than a 500 so the caller can resend the same feedback.
        if _is_transient_llm_error(exc):
            raise HTTPException(status_code=503, detail=f"Model backend temporarily unavailable; retry shortly. ({exc})") from exc
        raise
    return {"status": "completed", "session_id": payload.session_id, "text": result.text, "events": result.events}


app.include_router(router)
