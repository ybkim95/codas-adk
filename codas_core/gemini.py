"""Gemini client boundary for CoDaS chat responses."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

from codas_core.settings import load_local_env


load_local_env()


DEFAULT_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
]

MODEL_ALIASES = {
    # Keep visible choices aligned with the live models/list endpoint and a
    # generateContent probe in this environment. Deprecated preview names still
    # degrade to a nearby live model instead of surfacing a 404 to users.
    "models/gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "models/gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "models/gemini-3-pro-preview": "gemini-3-pro-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    "models/gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "models/gemini-3.1-pro-preview-customtools": "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-pro-preview-customtools": "gemini-3.1-pro-preview-customtools",
    "models/gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "models/gemini-flash-latest": "gemini-flash-latest",
    "gemini-flash-latest": "gemini-flash-latest",
    "models/gemini-pro-latest": "gemini-pro-latest",
    "gemini-pro-latest": "gemini-pro-latest",
    "models/gemini-flash-lite-latest": "gemini-flash-lite-latest",
    "gemini-flash-lite-latest": "gemini-flash-lite-latest",
    "models/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "models/gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "models/gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    # Historical/deprecated selections.
    "models/gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    "gemini-2.5-flash": "gemini-3.5-flash",
    "models/gemini-2.5-flash": "gemini-3.5-flash",
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
    "models/gemini-2.5-pro": "gemini-3.1-pro-preview",
}


def _normalize_model_id(value: str | None) -> str:
    model = (value or "").strip()
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    return MODEL_ALIASES.get(model, model)


@dataclass
class GroundedResponse:
    text: str | None
    model: str
    configured: bool
    sources: list[dict[str, str]]
    queries: list[str]
    error: str | None = None


def _grounding_from_response(response: Any) -> tuple[list[dict[str, str]], list[str]]:
    try:
        candidate = response.candidates[0]
    except Exception:
        return [], []
    metadata = getattr(candidate, "grounding_metadata", None)
    if not metadata:
        return [], []
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in getattr(metadata, "grounding_chunks", []) or []:
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None) if web else None
        if not uri or uri in seen:
            continue
        seen.add(uri)
        sources.append({
            "title": getattr(web, "title", None) or "source",
            "url": uri,
        })
    queries = [str(item) for item in (getattr(metadata, "web_search_queries", None) or [])]
    return sources, queries


def configured() -> bool:
    return bool(
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
    )


def configured_source() -> str:
    if os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY"
    if os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY"
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE":
        return "vertex_ai"
    return "missing"


def configured_models() -> list[str]:
    raw = os.getenv("CODAS_GEMINI_MODELS", "")
    raw_models = [item.strip() for item in raw.split(",") if item.strip()] or list(DEFAULT_MODELS)
    models: list[str] = []
    seen: set[str] = set()
    for item in raw_models:
        normalized = _normalize_model_id(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            models.append(normalized)
    default = _normalize_model_id(os.getenv("CODAS_GEMINI_MODEL", "").strip())
    if default and default not in models:
        models.insert(0, default)
    return models or list(DEFAULT_MODELS)


def default_model() -> str:
    configured_default = os.getenv("CODAS_GEMINI_MODEL", "").strip()
    if configured_default:
        normalized = _normalize_model_id(configured_default)
        if normalized in configured_models():
            return normalized
    return configured_models()[0]


def normalize_model(model: str | None) -> str:
    requested = _normalize_model_id(model)
    models = configured_models()
    if requested and requested in models:
        return requested
    return default_model()


def _candidate_models(model: str | None) -> list[str]:
    selected = normalize_model(model)
    ordered = [selected] + [candidate for candidate in configured_models() if candidate != selected]
    # Bound the fallback cascade: trying all configured models on repeated errors could
    # add ~20s each and balloon synthesis latency. The first few cover the realistic
    # available models; if those all fail the request should surface the error fast.
    max_fallbacks = int(os.getenv("CODAS_MAX_MODEL_FALLBACKS", "3"))
    return ordered[: max(1, max_fallbacks)]


def _model_unavailable_error(error: str | None) -> bool:
    if not error:
        return False
    lower = error.lower()
    return any(token in lower for token in ("404", "not_found", "no longer available", "model is not found"))


def status() -> dict[str, Any]:
    return {
        "configured": configured(),
        "source": configured_source(),
        "default_model": default_model(),
        "models": configured_models(),
    }


def _client():
    from google import genai

    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE":
        return genai.Client()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def generate_grounded_reply(
    message: str,
    *,
    model: str | None = None,
    context: str = "",
    max_output_tokens: int = 1400,
) -> GroundedResponse:
    selected_model = normalize_model(model)
    if not configured():
        return GroundedResponse(
            text=None,
            model=selected_model,
            configured=False,
            sources=[],
            queries=[],
            error="Gemini API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env.",
        )
    last_error: str | None = None
    for candidate_model in _candidate_models(model):
        try:
            from google.genai import types

            client = _client()
            prompt = f"{context.strip()}\n\nTask:\n{message}".strip()
            response = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=max_output_tokens,
                ),
            )
            sources, queries = _grounding_from_response(response)
            text = (response.text or "").strip()
            if not text:
                last_error = f"{candidate_model} returned an empty grounded text response."
                continue
            return GroundedResponse(
                text=text,
                model=candidate_model,
                configured=True,
                sources=sources,
                queries=queries,
            )
        except Exception as exc:
            last_error = str(exc)
            if _model_unavailable_error(last_error):
                continue
            return GroundedResponse(text=None, model=candidate_model, configured=True, sources=[], queries=[], error=last_error)
    return GroundedResponse(text=None, model=selected_model, configured=True, sources=[], queries=[], error=last_error or "No configured Gemini model is available.")
