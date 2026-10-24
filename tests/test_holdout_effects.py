"""Effect sizes on the sealed holdout: frozen at registration, read in units a reader can weigh, with the
discrimination a feature adds and a magnitude class that says when a significant effect is negligible."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.aou.confirm import confirm, confirmation_markdown, preregister, reconciliation_markdown


def _survival(seed, n=6000):
    rng = np.random.default_rng(seed)
    h = pd.DataFrame({"x": rng.normal(100, 20, n), "inc": rng.normal(size=n), "age": rng.normal(size=n)})
    t = rng.exponential(1 / (0.15 * np.exp(0.4 * (h.x - 100) / 20 + 0.4 * h.inc + 0.3 * h.age)))
    censor = rng.uniform(2, 6, n)
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    return h


def _registered(feature, incumbent="none"):
    return {"feature": feature, "direction": 1, "knownness": "emerging", "construct": "c", "citation": "",
            "incumbent": incumbent, "shape": "monotone", "mechanism_steps": ["one", "two"],
            "magnitude": {"measure": "hr_per_sd", "low": 1.1, "high": 2.0, "source": "analogy"},
            "predictions": ["attenuation_below_half_after_incumbent"]}


def test_a_time_to_event_effect_is_read_per_natural_unit_by_quarter_and_in_absolute_risk(tmp_path):
    _survival(1).to_csv(tmp_path / "s.csv", index=False)
    _survival(2).to_csv(tmp_path / "s.holdout.csv", index=False)
    state = {"target_column": "y", "confounder_columns": ["age"], "excluded_columns": [],
             "latest_report": {"candidates": []}, "a_priori_hypotheses": [_registered("x", "inc")]}
    prereg = preregister(state, tmp_path, "s", survival_time="followup_days", discovery_csv=tmp_path / "s.csv")
    (h,) = prereg["hypotheses"]
    # Frozen before the holdout is read: the discovery SD, quarters and natural unit, and the prediction models.
    assert h["effect_scale"]["unit"] == 30.0 and len(h["effect_scale"]["cutpoints"]) == 3   # IQR of 20 * 1.35
    assert set(h["frozen_model"]["models"]) == {"base", "feature", "incumbent", "both"}
    assert 0 < prereg["absolute_risk_horizon_days"] <= 3 * 365
    result = confirm(prereg, tmp_path / "s.holdout.csv", tmp_path)
    (row,) = result["hypotheses"]
    e = row["effect_sizes"]
    # 0.4 log-hazard per SD of 20 units: about exp(0.4 * 30 / 20) = 1.82 per natural unit of 30.
    assert 1.6 < e["hr_per_unit"] < 2.05 and e["unit"] == 30.0
    assert e["quarter_hr"][0] == 1.0 and e["quarter_hr"][3] > e["quarter_hr"][1] > 1 and e["p_trend"] < 1e-10
    ci = e["cumulative_incidence"]
    assert ci == sorted(ci) and e["risk_difference"] > 0.1 and e["risk_difference_ci95"][0] > 0
    p = e["prediction"]
    assert p["metric"] == "C" and p["gain"] > 0.01 and p["gain_ci95"][0] > 0 and p["gain_beyond_incumbent"] > 0
    assert row["magnitude"] in {"moderate", "large"}
    md = confirmation_markdown(result)
    assert "Effect sizes on the holdout" in md and "per 30" in md and "lowest to highest quarter" in md
    assert "beyond inc" in md


def test_a_significant_but_negligible_effect_is_called_so(tmp_path):
    rng = np.random.default_rng(3)
    n = 30000

    def share():
        f = pd.DataFrame({"x": rng.normal(size=n), "age": rng.normal(size=n)})
        f["y"] = 0.06 * f.x + 0.3 * f.age + rng.normal(size=n)
        return f

    share().to_csv(tmp_path / "v.csv", index=False)
    share().to_csv(tmp_path / "v.holdout.csv", index=False)
    state = {"target_column": "y", "confounder_columns": ["age"], "excluded_columns": [],
             "latest_report": {"candidates": [{"feature": "x", "rho": 0.06, "q_value": 1e-9, "verdict": "validated",
                                               "components": ["x"]}]}}
    prereg = preregister(state, tmp_path, "v", discovery_csv=tmp_path / "v.csv")
    result = confirm(prereg, tmp_path / "v.holdout.csv", tmp_path)
    (row,) = result["hypotheses"]
    assert row["replicated"] and row["magnitude"] == "negligible"
    e = row["effect_sizes"]
    assert 0.04 < e["per_sd"] < 0.08 and e["partial_delta_r2"] < 0.01 and e["prediction"]["metric"] == "R2"
    assert "negligible, though significant" in reconciliation_markdown(result, {})
    assert "Significant but negligible in size" in confirmation_markdown(result)
