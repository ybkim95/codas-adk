"""CoDaS quickstart: deterministic association discovery on the sample dataset.

Engine-only, no Gemini key required:

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from a checkout without installing: put the repo root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codas_core.discovery import DiscoveryRequest, run_discovery_from_csv

SAMPLE = Path(__file__).resolve().parent / "sample_dataset.csv"


def main() -> None:
    report = run_discovery_from_csv(
        SAMPLE,
        DiscoveryRequest(target_column="depression_score", top_k=5, validation_resamples=200),
    )
    fs = report.fact_sheet
    print(f"target : {fs.get('target_column')}   rows: {fs.get('rows')}   features screened: {fs.get('candidate_features_screened')}")
    metric_value = fs.get("ml_metric_value")
    metric_str = f"{metric_value:.3f}" if isinstance(metric_value, (int, float)) else str(metric_value)
    print(f"model  : {fs.get('ml_metric_name')}={metric_str}   above_chance={fs.get('ml_above_chance')}")
    print(f"passed : {fs.get('reported_battery_passing_variants')} validated of {len(report.candidates)} reported")
    print("-" * 72)
    for c in report.candidates:
        print(f"  {c.verdict:20s} {c.feature:38s} rho={c.rho:+.3f}  q={c.q_value:.2e}")
    if report.warnings:
        print("\nmethodological warnings (first 3):")
        for w in report.warnings[:3]:
            print(f"  - {w[:110]}")


if __name__ == "__main__":
    main()
