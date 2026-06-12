"""Robustness benchmark: run the deterministic engine on REAL public datasets across domains.

Proves the engine handles real, messy, varied data with no special-casing. Downloads are cached in
``.cache/`` (gitignored); the script skips any dataset it cannot fetch, so it is safe offline (the
sklearn-bundled clinical datasets always run). Exit code is nonzero if any dataset CRASHES with an
unexpected exception — that is the robustness contract.

    python scripts/benchmark_datasets.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from codas_core.data import InsufficientDataError, read_csv_dataset
from codas_core.discovery import DiscoveryRequest, run_discovery

CACHE = Path(__file__).resolve().parents[1] / ".cache"
CACHE.mkdir(exist_ok=True)


def _load_url(name: str, url: str) -> pd.DataFrame:
    dest = CACHE / f"{name}.csv"
    if not dest.exists():
        # curl uses the system trust store (portable + avoids Python's missing-CA issue on macOS).
        subprocess.run(["curl", "-fsSL", "-m", "40", "-o", str(dest), url], check=True)
    # Route real files through the engine loader (delimiter/encoding detection on `;`-files etc.).
    return read_csv_dataset(dest)


def _load_sklearn(loader: str) -> pd.DataFrame:
    from sklearn import datasets

    return getattr(datasets, loader)(as_frame=True).frame  # target column is named "target"


# (name, kind, source, target, participant_id, domain)
DATASETS = [
    ("breast_cancer", "sklearn", "load_breast_cancer", "target", None, "clinical/cell-nucleus biomarkers"),
    ("diabetes_progression", "sklearn", "load_diabetes", "target", None, "clinical biomarkers"),
    ("pima_diabetes", "url", "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv", "Outcome", None, "clinical (glucose/insulin/BMI)"),
    ("exercise_pulse", "url", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv", "pulse", "id", "physiological (repeated measures)"),
    ("penguins", "url", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv", "body_mass_g", None, "biology (has NaN + categoricals)"),
    ("auto_mpg", "url", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv", "mpg", None, "engineering"),
    ("wine_quality_red", "url", "https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/winequality-red.csv", "quality", None, "chemistry (`;`-delimited)"),
]


def main() -> int:
    rows = []
    crashes = 0
    for name, kind, source, target, pid, domain in DATASETS:
        try:
            df = _load_sklearn(source) if kind == "sklearn" else _load_url(name, source)
        except (subprocess.CalledProcessError, OSError, InsufficientDataError) as exc:
            rows.append((name, domain, "SKIP (download failed)", ""))
            print(f"  skip   {name}: {exc}")
            continue

        target_col = target if target in df.columns else None
        if target_col is None:
            rows.append((name, domain, f"SKIP (no target '{target}')", ""))
            continue

        request = DiscoveryRequest(
            target_column=target_col,
            participant_id_column=(pid if pid and pid in df.columns else None),
            top_k=8,
            validation_resamples=300,
        )
        try:
            report = run_discovery(df, request).to_dict()
        except InsufficientDataError as exc:
            rows.append((name, domain, "boundary", str(exc)[:50]))
            continue
        except Exception as exc:  # an unexpected exception is a robustness failure
            crashes += 1
            rows.append((name, domain, f"CRASH: {type(exc).__name__}", str(exc)[:60]))
            continue

        fs = report["fact_sheet"]
        verdicts: dict[str, int] = {}
        for c in report["candidates"]:
            verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
        metric = ""
        if fs.get("ml_metric_name"):
            metric = f"{fs['ml_metric_name']}={fs['ml_metric_value']:.3f}"
        detail = (f"n={fs.get('rows')} feat_screened={fs.get('candidate_features_screened')} "
                  f"cands={len(report['candidates'])} {verdicts} {metric}".strip())
        rows.append((name, domain, "ok", detail))

    width = max(len(r[0]) for r in rows)
    print("\n=== Real-dataset robustness benchmark ===")
    for name, domain, status, detail in rows:
        print(f"  {name:<{width}}  {status:<10}  {domain}\n  {'':<{width}}  -> {detail}")
    ran = sum(1 for r in rows if r[2] == "ok")
    print(f"\n{ran} ran, {crashes} crashed, {len(rows)} total.")
    return 1 if crashes else 0


if __name__ == "__main__":
    raise SystemExit(main())
