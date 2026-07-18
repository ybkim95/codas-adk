"""Minimal .env loader for local development.

Loads ``KEY=VALUE`` pairs from a ``.env`` file at the repository root into
``os.environ`` without overwriting variables already present in the environment.
Deliberately dependency-free (no python-dotenv) so the engine stays lightweight
and import-safe in restricted runtimes.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(env_path: str | Path | None = None) -> None:
    path = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        # Blank inherited values count as unset, so a shell snippet like
        # `GOOGLE_API_KEY= uvicorn ...` does not mask the real value in .env.
        if not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")
