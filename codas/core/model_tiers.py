"""The Gemini models CoDaS runs on: three tiers, how each falls back, how hard each thinks, and list prices.

One table for every caller (the agent graph, the literature search, the notebook's model probe and the
execution report), so retiring or adding a model is a change in one place.

* ``reasoning``: design, adversarial debate, mechanism, novelty and reporting.
* ``fast``: the repeated discovery-loop tasks (the search step, the statistical and ML interpreters).
* ``lite``: short bookkeeping turns (the convergence check, artifacts, lessons).

Each tier's candidates are listed newest first; the notebook's probe takes the first one that answers in
the workspace project and sets the tier's environment variable. Without a probe the first candidate is used.
CODAS_GEMINI_MODEL forces one model for every tier. Model IDs and prices are from the Vertex AI model and
pricing pages (Oct 2026); ``gemini-3.1-pro-preview`` and older are superseded and are not candidates.
"""

from __future__ import annotations

import json
import logging
import os

TIERS = ("reasoning", "fast", "lite")
CANDIDATES: dict[str, tuple[str, ...]] = {
    "reasoning": ("gemini-4-argon", "gemini-3.8-flash"),
    "fast": ("gemini-3.8-flash", "gemini-3.5-flash"),
    "lite": ("gemini-3.5-flash-lite", "gemini-3.8-flash"),
}
ENV = {"reasoning": "CODAS_GEMINI_REASONING_MODEL", "fast": "CODAS_GEMINI_FAST_MODEL",
       "lite": "CODAS_GEMINI_LITE_MODEL"}
# Superseded model names and their successors: a setting that still names one runs on the successor.
SUPERSEDED = {
    "gemini-3.1-pro-preview": "gemini-4-argon", "gemini-3.1-pro-preview-customtools": "gemini-4-argon",
    "gemini-3-pro-preview": "gemini-4-argon", "gemini-2.5-pro": "gemini-4-argon",
    "gemini-3-flash-preview": "gemini-3.8-flash", "gemini-2.5-flash": "gemini-3.8-flash",
    "gemini-2.0-flash": "gemini-3.8-flash", "gemini-3.1-flash-lite": "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite-preview": "gemini-3.5-flash-lite", "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
}
# Gemini thinking_level per tier; an agent may ask for a different level. Thinking tokens bill as output.
THINKING = {"reasoning": "MEDIUM", "fast": "LOW", "lite": "LOW"}
# When a tier's model is out of capacity or not served, the call goes to these tiers' models, in order.
_FALLBACK = {"reasoning": ("fast", "lite"), "fast": ("reasoning", "lite"), "lite": ("fast", "reasoning")}
# USD per 1M tokens (input, output), standard pay-as-you-go on the global endpoint. Introductory prices:
# gemini-4-argon 2/10 until 2026-11-30, then 4/20; gemini-3.8-flash 0.75/3.75 until 2026-12-31, then 1.50/7.50.
# CODAS_MODEL_PRICES_JSON ({"model": [input, output], ...}) overrides or adds entries.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-4-argon": (2.0, 10.0),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.5, 9.0),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


LOGGER = logging.getLogger("codas.models")


def current(model_id: str) -> str:
    """``model_id``, or its successor when it is superseded."""
    name = model_id.strip()
    successor = SUPERSEDED.get(name.split("/")[-1])
    if successor:
        LOGGER.warning("model %s is superseded; using %s", name, successor)
        return successor
    return name


def model(tier: str) -> str:
    """The model id serving ``tier``."""
    return current(os.getenv("CODAS_GEMINI_MODEL") or os.getenv(ENV[tier]) or CANDIDATES[tier][0])


def fallback_order(tier: str) -> list[str]:
    """The tier's model, then the other tiers' models, without repeats."""
    return list(dict.fromkeys(model(t) for t in (tier, *_FALLBACK[tier])))


def prices() -> dict[str, tuple[float, float]]:
    table = dict(PRICES)
    raw = os.getenv("CODAS_MODEL_PRICES_JSON", "").strip()
    if raw:
        try:
            table.update({str(k): (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()})
        except (ValueError, TypeError, IndexError, KeyError, AttributeError):
            pass  # a malformed override leaves the listed prices in place
    return table


def price(model_id: str | None) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens for a served model id (a dated version such as
    ``gemini-3.8-flash-001`` takes its base model's price), or None when it is not listed."""
    if not model_id:
        return None
    table = prices()
    name = model_id.split("/")[-1]
    return table.get(name) or next((v for k, v in sorted(table.items(), key=lambda kv: -len(kv[0]))
                                    if name.startswith(k)), None)


def cost(model_id: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD at list price; ``output_tokens`` should include thinking tokens."""
    p = price(model_id)
    return None if p is None else (input_tokens * p[0] + output_tokens * p[1]) / 1e6
