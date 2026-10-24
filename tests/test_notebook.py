"""The notebook users run is exactly what its generator writes, so it can always be regenerated and
reviewed, and it names no model itself: the probe reads each tier's candidates from the code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "aou_codas_quickstart.ipynb"


def _sources(path: Path) -> list[str]:
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def test_the_committed_notebook_is_what_the_generator_writes(tmp_path):
    pytest.importorskip("nbformat")
    fresh = tmp_path / "notebook.ipynb"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "aou" / "make_notebook.py"), str(fresh)], check=True,
                   capture_output=True, text=True)
    assert _sources(fresh) == _sources(NOTEBOOK), "run python scripts/aou/make_notebook.py and commit the notebook"


def test_the_notebook_takes_its_models_from_the_tier_table():
    from codas.core import model_tiers

    text = "\n".join(_sources(NOTEBOOK))
    assert "model_tiers.CANDIDATES" in text
    named = {m for tier in model_tiers.CANDIDATES.values() for m in tier} | set(model_tiers.SUPERSEDED)
    assert not [m for m in named if f'"{m}"' in text]
