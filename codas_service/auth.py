"""API-key authentication for the CoDaS service layer.

Server-to-server only. There is no human/browser auth here by design: the service is meant to be
called by another machine (e.g. an AI co-scientist) or run locally for development.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient", "testserver"}


def agent_api_keys() -> set[str]:
    """Accepted API keys, comma-separated in CODAS_AGENT_API_KEYS (supports zero-downtime rotation)."""
    raw = os.getenv("CODAS_AGENT_API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def _is_local(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    return host in _LOCAL_HOSTS


def require_api_key(request: Request) -> str:
    """Authorize a request by API key; return the caller principal or raise 401.

    Fail-closed: if no keys are configured the request is accepted only from localhost, so a
    deployed instance is never open to the internet without an explicit key.
    """
    presented = request.headers.get("x-codas-agent-key", "").strip()
    if not presented:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            presented = header[len("bearer "):].strip()

    keys = agent_api_keys()
    if keys:
        if presented and any(hmac.compare_digest(presented, key) for key in keys):
            return "agent"
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    if _is_local(request):
        return "local-dev"
    raise HTTPException(
        status_code=401,
        detail="API is disabled. Set CODAS_AGENT_API_KEYS to allow remote calls.",
    )
