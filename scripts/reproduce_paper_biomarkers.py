#!/usr/bin/env python3
"""Compare selected reference biomarker effects with this deterministic engine.

This runs the deterministic engine (``codas.core``) on each cohort table and checks that a set of
named candidate biomarkers with configured reference Spearman effect sizes. The governed inputs and
their full preprocessing lineage are not included, so this script is not stand-alone evidence that the
reported statistics reproduce.

Reference comparisons configured for matching participant-level analysis tables:
  * DWB (target PHQ-8):   main sleep-duration variability            ρ ≈ +0.252
  * WEAR-ME (HOMA-IR):    C-reactive protein                         ρ ≈ +0.393
  * WEAR-ME (HOMA-IR):    HDL cholesterol                            ρ ≈ −0.412

Honest scope. This is the deterministic ``codas.core`` path — it screens univariate features and a
bounded family of engineered ratio features, then runs the internal validation battery. The exact
set of *validated* candidates and the agent-constructed composites (e.g. the night-to-day social
media ratio, or a bespoke steps/RHR fitness index) depend on the full agent loop, which proposes
transformations for the engine to evaluate (see ``codas.agents``). Reference deltas can be evaluated
only with matching analysis tables; validated-candidate *counts* are configuration-sensitive and are
reported as such — never as a fixed number.

Cohort roles are supplied by the caller as data (a JSON config or CLI flags), never inferred from
column names: the engine applies no name-based rules. The participant-level cohort tables are real
clinical data and are NOT shipped in this repository; pass their paths explicitly.

    python scripts/reproduce_paper_biomarkers.py --config scripts/paper_cohorts.example.json

Each cohort entry declares: ``csv`` (path), ``target`` (outcome column), optional ``participant`` /
``time`` role columns, ``exclude`` (columns the caller declares are outcome proxies / other labels,
supplied by the domain expert, not hard-coded), and an optional ``expect`` map of
{feature substring: reference ρ} used only to print a replication delta. Nothing in ``expect``
influences the engine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codas.core.data import read_csv_dataset
from codas.core.discovery import DiscoveryRequest, run_discovery
from codas.core.statistics import safe_spearman


def _run_cohort(name: str, cfg: dict) -> None:
    if not Path(cfg["csv"]).exists():
        raise SystemExit(f"[{name}] CSV not found: {cfg['csv']}. Edit the config to point at the "
                         f"participant-level cohort table (real clinical data is not shipped in this repo).")
    df = read_csv_dataset(cfg["csv"])
    target = cfg["target"]
    if target not in df.columns:
        raise SystemExit(f"[{name}] target {target!r} not in {cfg['csv']} "
                         f"(columns include: {list(df.columns)[:12]} ...)")
    exclude = [c for c in cfg.get("exclude", []) if c in df.columns and c != target]
    report = run_discovery(df, DiscoveryRequest(
        target_column=target,
        participant_id_column=cfg.get("participant") or None,
        time_column=cfg.get("time") or None,
        excluded_columns=exclude,
        top_k=cfg.get("top_k", 40),
        validation_resamples=cfg.get("resamples", 1000),  # 1,000 permutation/bootstrap resamples
    ))
    validated = [c for c in report.candidates if c.verdict == "validated"]
    print(f"\n{'=' * 88}\n{name}  (N={len(df):,}, target={target!r})\n{'=' * 88}")
    print(f"  screened {report.fact_sheet.get('candidate_features_screened')} features; "
          f"{len(validated)} validated by the internal battery (count is configuration-sensitive).")
    for c in validated[:10]:
        print(f"    validated  rho={c.rho:+.3f}  q={c.q_value:.1e}  {c.feature}")

    # Comparison against the reference effects (documentation only — the engine never sees
    # `expect`). We recompute rho directly on the analysis frame so a feature that was screened but
    # not in the top-k still gets a faithful comparison.
    expect = cfg.get("expect", {})
    if expect:
        print("  reference effect reproduction:")
        for needle, ref_rho in expect.items():
            col = next((c for c in df.columns if needle.lower() in c.lower()), None)
            if col is None:
                print(f"    {needle:38s} reference rho={ref_rho:+.3f}  |  column not present")
                continue
            rho, _, n = safe_spearman(df[col], df[target])
            delta = abs(rho - ref_rho)
            flag = "OK" if delta <= 0.02 else ("~" if delta <= 0.05 else "DIFF")
            print(f"    {col:38.38s} reference rho={ref_rho:+.3f}  engine rho={rho:+.3f}  "
                  f"d={delta:.3f}  [{flag}]  (n={n})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="JSON file mapping cohort name -> {csv, target, ...}.")
    args = ap.parse_args()

    cohorts = json.loads(Path(args.config).read_text())
    print("Evaluating selected biomarker effects with the deterministic engine.")
    for name, cfg in cohorts.items():
        if name.startswith("__"):  # documentation keys (e.g. "__doc__") are not cohorts
            continue
        _run_cohort(name, cfg)
    print(f"\n{'=' * 88}\nConfigured reference deltas are printed when matching analysis columns and expectations are\n"
          f"provided. Validated-candidate counts are configuration-sensitive (feature family, declared\n"
          f"exclusions, thresholds) and are reported as computed, not as a fixed number.\n{'=' * 88}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
