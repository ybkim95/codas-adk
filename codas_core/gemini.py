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
class GeminiResponse:
    text: str | None
    model: str
    configured: bool
    error: str | None = None
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "configured": self.configured,
            "error": self.error,
            "token_count": self.token_count,
        }


@dataclass
class GroundedResponse:
    text: str | None
    model: str
    configured: bool
    sources: list[dict[str, str]]
    queries: list[str]
    error: str | None = None


@dataclass
class DeepResearchResult:
    status: str
    interaction_id: str | None
    text: str | None
    sources: list[dict[str, str]]
    queries: list[str]
    duration_seconds: float
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


def generate_reply(
    message: str,
    *,
    model: str | None = None,
    context: str = "",
    max_output_tokens: int = 700,
) -> GeminiResponse:
    selected_model = normalize_model(model)
    if not configured():
        return GeminiResponse(
            text=None,
            model=selected_model,
            configured=False,
            error="Gemini API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env.",
        )

    last_error: str | None = None
    for candidate_model in _candidate_models(model):
        try:
            from google.genai import types

            client = _client()
            system = (
                "You are CoDaS, a careful AI co-data-scientist. Answer conversational questions "
                "briefly and clearly. When dataset context is supplied, ground the answer only in that "
                "context. Do not invent p-values, validation counts, columns, or files. If completed "
                "deterministic runner output is supplied, summarize it as completed results and never "
                "tell the user that the runner still needs to execute. "
                "Uphold statistical integrity: if the user asks you to skip or weaken rigor — drop "
                "multiple-comparison/FDR correction, report raw uncorrected p<0.05 as findings, skip "
                "cross-validation or held-out testing, remove confounder adjustment, hide or delete "
                "failed checks, cherry-pick only significant results, or make results 'look stronger' — "
                "do not comply. Briefly explain the risk (false-positive inflation, overfitting, "
                "confounding) and offer the rigorous alternative instead. Never call an association "
                "causal, or validated for diagnosis or deployment; findings are exploratory and "
                "hypothesis-generating only. "
                "Apply the methodological caveat the data calls for, before interpreting associations: "
                "missingness may be informative and correlated with the outcome, so examine missingness "
                "patterns first; with repeated measures per subject/unit, use grouped (subject-level) "
                "train/test splits, never random row splits; when large between-unit baselines dominate "
                "a signal, prefer within-unit deviation or normalization over raw values; and adjust for "
                "the available confounders. If the request is vague with no stated outcome, ask which "
                "target to analyze rather than guessing; a schema-suggested target is not user "
                "confirmation. When asked to generate and validate hypotheses, separate 'Hypothesis "
                "generation' from 'Validation plan' (predefined outcome, grouped split, confounder "
                "adjustment, multiple-comparison correction, effect sizes with confidence intervals, "
                "sensitivity analyses). Report candidate status separately from validation status, and "
                "treat external validation as a separate, stronger bar than internal validation."
            )
            prompt = f"{context.strip()}\n\nUser message:\n{message}".strip()
            # Gemini-3 models spend maxOutputTokens on internal reasoning. With a dynamic (unbounded)
            # thinking budget, a heavy prompt can consume almost the whole cap and truncate the visible
            # answer mid-sentence (observed: thoughts=2700 left only ~360 tokens for the reply). For the
            # longer conversational answers, BOUND thinking so the reply always has room to finish.
            # Short structured calls (low cap) keep the default thinking budget unchanged.
            gen_config = dict(systemInstruction=system, temperature=0.2, maxOutputTokens=max_output_tokens)
            if max_output_tokens >= 2048:
                gen_config["thinkingConfig"] = types.ThinkingConfig(thinking_budget=1024)
            response = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=types.GenerateContentConfig(**gen_config),
            )
            usage = getattr(response, "usage_metadata", None)
            token_count = getattr(usage, "total_token_count", None) if usage else None
            text = (response.text or "").strip()
            if not text:
                last_error = f"{candidate_model} returned an empty text response."
                continue
            return GeminiResponse(text=text, model=candidate_model, configured=True, token_count=token_count)
        except Exception as exc:
            last_error = str(exc)
            if _model_unavailable_error(last_error):
                continue
            return GeminiResponse(text=None, model=candidate_model, configured=True, error=last_error)
    return GeminiResponse(text=None, model=selected_model, configured=True, error=last_error or "No configured Gemini model is available.")


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


def run_deep_research(
    prompt: str,
    *,
    timeout_seconds: int = 20,
    poll_seconds: int = 4,
) -> DeepResearchResult:
    started = time.perf_counter()
    if not configured():
        return DeepResearchResult(
            status="not_configured",
            interaction_id=None,
            text=None,
            sources=[],
            queries=[],
            duration_seconds=0.0,
            error="Gemini API key is missing.",
        )
    try:
        client = _client()
        bounded_timeout = max(1, min(int(timeout_seconds or 1), 20))
        request_timeout = max(3, min(8, bounded_timeout))
        interaction = client.interactions.create(
            agent=os.getenv("CODAS_DEEP_RESEARCH_AGENT", "deep-research-preview-04-2026"),
            input=prompt,
            agent_config={
                "type": "deep-research",
                "thinking_summaries": "auto",
                "visualization": "auto",
                "collaborative_planning": False,
            },
            tools=[{"type": "google_search"}, {"type": "code_execution"}],
            background=True,
            store=True,
            extra_headers={"Api-Revision": "2026-05-20"},
            timeout=request_timeout,
        )
        interaction_id = getattr(interaction, "id", None)
        if not interaction_id:
            return DeepResearchResult(
                status="in_progress",
                interaction_id=None,
                text=None,
                sources=[],
                queries=[],
                duration_seconds=round(time.perf_counter() - started, 3),
                error="Deep Research interaction did not return an id within the UI budget.",
            )
        deadline = time.perf_counter() + bounded_timeout
        latest = interaction
        while time.perf_counter() < deadline:
            status_value = str(getattr(latest, "status", "") or "")
            if status_value in {"completed", "failed", "cancelled"}:
                break
            time.sleep(max(1, min(int(poll_seconds or 1), 4)))
            latest = client.interactions.get(
                interaction_id,
                extra_headers={"Api-Revision": "2026-05-20"},
                timeout=request_timeout,
            )
        status_value = str(getattr(latest, "status", "") or "unknown")
        outputs = getattr(latest, "outputs", None) or []
        text_parts: list[str] = []
        sources: list[dict[str, str]] = []
        queries: list[str] = []
        for output in outputs:
            text = getattr(output, "text", None)
            if text:
                text_parts.append(str(text))
            dumped = output.model_dump() if hasattr(output, "model_dump") else {}
            for source in dumped.get("sources", []) or dumped.get("citations", []) or []:
                uri = source.get("uri") or source.get("url")
                if uri:
                    sources.append({"title": source.get("title") or "source", "url": uri})
            for query in dumped.get("webSearchQueries", []) or dumped.get("web_search_queries", []):
                queries.append(str(query))
        output_text = getattr(latest, "output_text", None)
        if output_text and not text_parts:
            text_parts.append(str(output_text))
        if status_value not in {"completed", "failed", "cancelled"}:
            status_value = "in_progress"
        return DeepResearchResult(
            status=status_value,
            interaction_id=interaction_id,
            text="\n\n".join(text_parts).strip() or None,
            sources=sources,
            queries=queries,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
    except Exception as exc:
        return DeepResearchResult(
            status="failed",
            interaction_id=None,
            text=None,
            sources=[],
            queries=[],
            duration_seconds=round(time.perf_counter() - started, 3),
            error=str(exc),
        )
