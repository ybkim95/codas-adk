"""Shared data models for CoDaS deterministic runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    missing_fraction: dict[str, float]
    suggested_targets: list[str]
    target_column: str | None = None
    participant_id_column: str | None = None
    time_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class ValidationTestResult:
    name: str
    dimension: str
    passed: bool
    applicable: bool = True
    hard_gate: bool = False
    metric: float | None = None
    p_value: float | None = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class Candidate:
    feature: str
    rho: float
    p_value: float
    q_value: float
    n: int
    direction: str
    score: float
    verdict: str = "untested"
    pass_rate: float = 0.0
    components: list[str] = field(default_factory=list)
    tests: list[ValidationTestResult] = field(default_factory=list)
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = _jsonable(asdict(self))
        data["tests"] = [test.to_dict() for test in self.tests]
        return data


@dataclass
class DiscoveryReport:
    profile: DatasetProfile
    candidates: list[Candidate]
    fact_sheet: dict[str, Any]
    audit_log: list[str]
    warnings: list[str]
    markdown_report: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "fact_sheet": self.fact_sheet,
            "audit_log": self.audit_log,
            "warnings": self.warnings,
            "markdown_report": self.markdown_report,
        }
