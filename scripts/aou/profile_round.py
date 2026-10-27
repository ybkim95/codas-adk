"""Where one discovery round spends its time on a cohort table, without reading or printing any data.

Runs a single round of the engine as the All of Us runner configures it (adjusted screen, the follow-up time for a
diagnosis, multiply imputed covariates, the first round's candidate budget), with the spec's covariates, age, sex and
the wear-day counts as confounders. Every ``--every`` seconds it records which CoDaS function the round is in (names
and line numbers only); at the end, or after ``--minutes``, it prints how the samples split across those functions
and the stack seen most often. Usage, in the Workbench::

    python scripts/aou/profile_round.py .codas_runs/aou/explore_acute_kidney_injury.csv --minutes 15
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--every", type=float, default=10.0)
    args = parser.parse_args(argv)

    import pandas as pd

    from codas.agents import tools
    from codas.aou import cohort, runner
    from codas.core.discovery import DiscoveryRequest, run_discovery_from_csv

    spec = cohort.CohortSpec.from_json(ROOT / "scripts" / "aou" / "hba1c_fitbit.example.json")
    header = list(pd.read_csv(args.csv, nrows=0).columns)
    target = header[0]
    survival = cohort.FOLLOWUP_COLUMN in header
    confounders = [c for c in ["age", "female", *(c.name for c in spec.covariates)] if c in header]
    confounders += [c for c in header if c.endswith("_days") and c not in confounders]
    request = DiscoveryRequest(
        target_column=target, excluded_columns=[cohort.FOLLOWUP_COLUMN] if survival else [],
        confounder_columns=confounders, context_columns=[],
        top_k=tools._TOPK_BASE, max_ratio_features=tools._RATIO_FEATURES_BASE,
        validation_resamples=tools._ROUND_RESAMPLES, covariate_imputations=runner.COVARIATE_IMPUTATIONS,
        screen_adjust=True, survival_time_column=cohort.FOLLOWUP_COLUMN if survival else None)
    print(f"columns {len(header)}, confounders {len(confounders)}, survival {survival}, imputations "
          f"{runner.COVARIATE_IMPUTATIONS}, resamples {tools._ROUND_RESAMPLES}", flush=True)

    done: dict[str, object] = {}

    def work() -> None:
        try:
            run_discovery_from_csv(args.csv, request)
            done["ok"] = True
        except Exception as exc:  # noqa: BLE001 - reported, the profile still prints
            done["error"] = f"{type(exc).__name__}: {exc}"[:200]

    worker = threading.Thread(target=work, daemon=True)
    started = time.monotonic()
    worker.start()
    where: collections.Counter[str] = collections.Counter()
    stacks: collections.Counter[str] = collections.Counter()
    while worker.is_alive() and time.monotonic() - started < args.minutes * 60:
        worker.join(args.every)
        frame = sys._current_frames().get(worker.ident)
        if frame is None:
            break
        ours = [f for f in traceback.extract_stack(frame) if "/codas/" in f.filename]
        if ours:
            inner = ours[-1]
            where[f"{Path(inner.filename).name}:{inner.name}"] += 1
            stacks[" > ".join(f"{Path(f.filename).stem}.{f.name}:{f.lineno}" for f in ours[-6:])] += 1
        print(f"{time.monotonic() - started:7.0f} s  {' > '.join(f.name for f in ours[-4:])}", flush=True)
    elapsed = time.monotonic() - started
    state = "finished" if done.get("ok") else done.get("error") or "still running (stopped here)"
    print(f"\nround {state} after {elapsed:.0f} s\n\nshare of samples by CoDaS function:")
    total = sum(where.values()) or 1
    for name, count in where.most_common(12):
        print(f"  {count / total:5.0%}  {name}")
    if stacks:
        print("\nstack seen most often:\n  " + stacks.most_common(1)[0][0].replace(" > ", "\n  > "))
    sys.stdout.flush()
    os._exit(0)   # the round may still be running in its thread


if __name__ == "__main__":
    raise SystemExit(main())
