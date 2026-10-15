"""Privacy guard for running CoDaS on All of Us data.

Two All of Us rules shape what may leave the deterministic engine:

* The Data User Code of Conduct forbids giving workspace access to any AI tool that could result in
  the export of participant-level data. The LLM therefore sees only aggregates: no tool response may
  carry raw rows, and the model must be served from the workspace's own Google Cloud project
  (Vertex AI), not a consumer API key.
* The Data and Statistics Dissemination Policy forbids publishing any participant count from 1 to 20.
  Counts in that range are masked before they reach the model, the shared memory it reads, or the
  final report, so nothing the agents write can carry one out of the Workbench.

The guard is switched on with ``CODAS_AOU_MODE=1``. With it off, every function here is a no-op for
callers, so the public CoDaS behaviour is unchanged.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Dissemination policy: counts 1..20 (inclusive) may not be shown. Zero is allowed.
SMALL_CELL_MAX = 20
MASK = f"<={SMALL_CELL_MAX}"

# Tool-response keys that hold row-level records. They are dropped outright in AoU mode.
ROW_LEVEL_KEYS = frozenset({"preview_rows", "rows_sample", "records", "sample_rows", "head"})

# Keys whose integer value is a count of participants or rows.
_COUNT_KEYS = frozenset({
    "n", "rows", "count", "cases", "controls", "positives", "negatives", "support",
    "n_participants", "participants", "n_rows", "n_cases", "n_controls",
})
_COUNT_SUFFIXES = ("_count", "_n", "_rows", "_participants", "_cases", "_controls")
_COUNT_PREFIXES = ("n_", "num_", "count_")
# Counts of things that are not people (features, checks, rounds) are structural and stay visible.
_STRUCTURAL_HINTS = ("feature", "column", "candidate", "check", "test", "round", "variant",
                     "warning", "budget", "top_k", "resample", "iteration", "correction",
                     "validated", "evaluated", "passing", "cluster", "fold")

_TRUE = {"1", "true", "yes", "on"}


def aou_mode() -> bool:
    return os.getenv("CODAS_AOU_MODE", "").strip().lower() in _TRUE


def literature_search_enabled() -> bool:
    """Literature grounding sends an LLM-written query to Google Search. It never contains rows
    (the model never sees any), but a deployment can turn it off with CODAS_AOU_LITERATURE_SEARCH=0."""
    return os.getenv("CODAS_AOU_LITERATURE_SEARCH", "1").strip().lower() in _TRUE


def _is_count_key(key: str) -> bool:
    k = key.strip().lower()
    if any(hint in k for hint in _STRUCTURAL_HINTS):
        return False
    return k in _COUNT_KEYS or k.endswith(_COUNT_SUFFIXES) or k.startswith(_COUNT_PREFIXES)


def _is_small_count(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 1 <= value <= SMALL_CELL_MAX
    if isinstance(value, float) and value.is_integer():
        return 1 <= value <= SMALL_CELL_MAX
    return False


def mask_small_counts(obj: Any) -> Any:
    """Return a copy of ``obj`` with every participant/row count in 1..20 replaced by ``"<=20"``."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if _is_count_key(str(key)) and _is_small_count(value):
                out[key] = MASK
            else:
                out[key] = mask_small_counts(value)
        return out
    if isinstance(obj, list):
        return [mask_small_counts(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(mask_small_counts(item) for item in obj)
    return obj


def sanitize_tool_response(response: Any) -> Any:
    """Drop row-level payloads and mask small counts in a tool response before the model sees it."""
    if isinstance(response, dict):
        cleaned = {k: sanitize_tool_response(v) for k, v in response.items() if k not in ROW_LEVEL_KEYS}
        return mask_small_counts(cleaned)
    if isinstance(response, list):
        return [sanitize_tool_response(item) for item in response]
    return response


# "n = 12", "n: 12", "N=12" and "12 participants/people/cases/..." in free text.
_N_EQUALS = re.compile(r"(?<![A-Za-z0-9_])([nN]\s*[=:]\s*)(\d{1,3}(?:,\d{3})*)(?![\d.])")
_N_NOUN = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:,\d{3})*)(\s+(?:participants?|people|persons?|individuals?|patients?|"
    r"subjects?|cases?|controls?|respondents?|users?|women|men|adults?)\b)",
    re.I,
)


def _mask_number(text: str) -> str | None:
    value = int(text.replace(",", ""))
    return MASK if 1 <= value <= SMALL_CELL_MAX else None


def mask_small_counts_in_text(text: str) -> str:
    """Mask participant counts in 1..20 written in prose (``n = 12``, ``12 participants``)."""
    if not text:
        return text

    def _eq(m: re.Match[str]) -> str:
        masked = _mask_number(m.group(2))
        return f"{m.group(1)}{masked}" if masked else m.group(0)

    def _noun(m: re.Match[str]) -> str:
        masked = _mask_number(m.group(1))
        return f"{masked}{m.group(2)}" if masked else m.group(0)

    return _N_NOUN.sub(_noun, _N_EQUALS.sub(_eq, text))


def sanitize_shared_state(state: Any) -> None:
    """Mask small counts in the memory keys that prompts interpolate ({fact_sheet?}, {rounds?})."""
    for key in ("fact_sheet", "rounds"):
        try:
            value = state.get(key)
        except Exception:
            return
        if value:
            state[key] = mask_small_counts(value)


class AouBackendError(RuntimeError):
    """Raised when AoU mode is on but the model backend would send data outside the workspace."""


def check_llm_backend() -> None:
    """In AoU mode, require Vertex AI in the workspace's project and refuse consumer API keys.

    A ``GOOGLE_API_KEY``/``GEMINI_API_KEY`` routes calls to the public Gemini API under a personal
    key, outside the workspace's Google Cloud project; Vertex AI keeps them inside it.
    """
    if not aou_mode():
        return
    problems = []
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().upper() != "TRUE":
        problems.append("set GOOGLE_GENAI_USE_VERTEXAI=TRUE")
    if not os.getenv("GOOGLE_CLOUD_PROJECT", "").strip():
        problems.append("set GOOGLE_CLOUD_PROJECT to the workspace's Google Cloud project")
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.getenv(key, "").strip():
            problems.append(f"unset {key} (AoU mode serves models only through Vertex AI)")
    if problems:
        raise AouBackendError("CODAS_AOU_MODE is on but the model backend is not workspace-local: "
                              + "; ".join(problems) + ".")
