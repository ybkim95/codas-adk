"""The incident register (docs/INCIDENTS.md) is only worth keeping if every row's guard is a test that exists:
a renamed or deleted test would otherwise leave a past failure unguarded without anyone noticing."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_REF = re.compile(r"(test_\w+\.py)?::(test_\w+)(\[.*\])?")


def test_every_incident_names_guard_tests_that_exist():
    rows = [line for line in (ROOT / "docs" / "INCIDENTS.md").read_text(encoding="utf-8").splitlines()
            if re.match(r"\|\s*\d+\s*\|", line)]
    assert len(rows) >= 50
    numbers = [int(row.split("|")[1]) for row in rows]
    assert numbers == list(range(1, len(rows) + 1))  # numbered in order, none dropped
    problems = []
    for row in rows:
        guard = row.rstrip().rstrip("|").rsplit("|", 1)[-1]
        current, found = None, 0
        for ref in re.findall(r"`([^`]+)`", guard):
            match = _REF.fullmatch(ref.strip())
            if not match:
                continue
            current = match.group(1) or current  # "::test_x" refers to the file named before it
            path = ROOT / "tests" / (current or "")
            if current and path.exists() and re.search(rf"^def {match.group(2)}\(", path.read_text(encoding="utf-8"), re.M):
                found += 1
            else:
                problems.append(f"row {row.split('|')[1].strip()}: {current}::{match.group(2)} does not exist")
        if not found:
            problems.append(f"row {row.split('|')[1].strip()}: names no guard test")
    assert not problems, problems
