"""Real-data tests on sklearn-bundled clinical datasets (offline, no network).

These are genuine clinical-biomarker datasets — the engine's home turf. The broader cross-domain
network benchmark (penguins, wine, mpg, pima, ...) lives in scripts/benchmark_datasets.py.
"""

import pytest

from codas.core.discovery import DiscoveryRequest, run_discovery

sklearn_datasets = pytest.importorskip("sklearn.datasets")


def _frame(loader: str):
    return getattr(sklearn_datasets, loader)(as_frame=True).frame  # target column is "target"


def test_breast_cancer_real_clinical():
    df = _frame("load_breast_cancer")  # 569 x 31, binary target (cell-nucleus measurements)
    report = run_discovery(
        df, DiscoveryRequest(target_column="target", top_k=8, validation_resamples=200)
    ).to_dict()
    assert report["fact_sheet"]["rows"] == 569
    assert report["fact_sheet"]["ml_metric_name"] == "auc"
    assert any(c["verdict"] == "validated" for c in report["candidates"])


def test_diabetes_progression_real_clinical():
    df = _frame("load_diabetes")  # 442 x 11, continuous target (disease progression)
    report = run_discovery(
        df, DiscoveryRequest(target_column="target", top_k=8, validation_resamples=200)
    ).to_dict()
    assert report["fact_sheet"]["rows"] == 442
    assert report["fact_sheet"]["ml_metric_name"] == "r2"
    assert any(c["verdict"] in {"validated", "conditional"} for c in report["candidates"])
