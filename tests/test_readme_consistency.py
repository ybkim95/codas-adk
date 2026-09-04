"""The README must describe the repository as it actually is.

A reviewer of this code reported that the README's engine snippet named a target column
`outcome` that does not exist in the bundled sample. Documentation drifts away from the code
silently, because nothing runs it. These tests run it.
"""
from __future__ import annotations

import csv
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SAMPLE = ROOT / "examples" / "sample_dataset.csv"


def _sample_columns() -> set[str]:
    with SAMPLE.open(encoding="utf-8", newline="") as fh:
        return set(next(csv.reader(fh)))


def test_documented_target_columns_exist_in_the_sample() -> None:
    """Every target_column= the README names must be a real column in the shipped sample."""
    named = re.findall(r'target_column\s*=\s*["\']([^"\']+)["\']', README)
    assert named, "the README no longer shows a target_column example; update this test"
    missing = sorted(c for c in named if c not in _sample_columns())
    assert not missing, (
        f"README names target column(s) {missing} that are absent from {SAMPLE.name}. "
        f"Available: {sorted(_sample_columns())}"
    )


def test_readme_csv_paths_exist() -> None:
    """A path the README tells the reader to run must be present."""
    for path in re.findall(r'read_csv_dataset\(\s*["\']([^"\']+)["\']', README):
        if path.startswith(("/", "your_", "path/")) or "<" in path:
            continue  # a placeholder the reader substitutes, not a shipped file
        assert (ROOT / path).exists(), f"README reads {path}, which is not in the repository"


def test_readme_python_version_matches_pyproject() -> None:
    """The stated Python version must be the one the package actually requires."""
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = re.search(r"(\d+\.\d+)", requires["project"]["requires-python"]).group(1)
    assert re.search(rf"Python\s+{re.escape(floor)}\b", README), (
        f"pyproject requires Python >= {floor}; the README does not say so"
    )


def test_data_manifest_lists_every_data_file() -> None:
    """The manifest claims to be exhaustive. Hold it to that."""
    manifest = (ROOT / "examples" / "DATA_MANIFEST.md").read_text(encoding="utf-8")
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    data = [f for f in tracked if f.lower().endswith(
        (".csv", ".tsv", ".parquet", ".json", ".xlsx", ".pkl", ".npy"))]
    missing = [f for f in data if f not in manifest]
    assert not missing, f"data files absent from DATA_MANIFEST.md: {missing}"
