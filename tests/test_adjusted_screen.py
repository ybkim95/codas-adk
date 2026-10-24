"""The opt-in screens rank candidates the way the holdout tests them: adjusted for the declared confounders,
and for a time to event with a Cox model on the follow-up time. Off, the engine is unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codas.core.discovery import DiscoveryRequest, _rank_candidates, run_discovery


def _proxy_data(rng, n=3000):
    """``proxy`` only stands in for age; ``real`` moves the outcome on its own."""
    age = rng.normal(size=n)
    frame = pd.DataFrame({"age": age, "proxy": age + 0.4 * rng.normal(size=n), "real": rng.normal(size=n),
                          "noise": rng.normal(size=n)})
    frame["y"] = 0.8 * frame.age + 0.25 * frame.real + rng.normal(size=n)
    return frame


def test_a_feature_that_only_stands_in_for_a_confounder_does_not_lead_an_adjusted_screen():
    frame = _proxy_data(np.random.default_rng(0))
    plain = {c.feature: c for c in _rank_candidates(frame, "y", ["proxy", "real", "noise"])}
    adjust = frame[["age"]].to_numpy(dtype=float)
    adjusted = {c.feature: c for c in _rank_candidates(frame, "y", ["proxy", "real", "noise"], adjust=adjust,
                                                       adjust_names=["age"])}
    assert plain["proxy"].q_value < 1e-50 and plain["proxy"].score > plain["real"].score   # the proxy leads
    assert adjusted["proxy"].q_value > 0.05 and adjusted["real"].q_value < 1e-10           # adjusted, it does not
    assert adjusted["real"].screen["measure"] == "partial Spearman rho" and adjusted["real"].screen["adjusted_for"] == ["age"]
    assert abs(adjusted["real"].screen["effect"] - 0.25 / np.sqrt(1 + 0.0625)) < 0.06
    # rho and the direction stay the plain Spearman ones, which the validation battery uses.
    assert adjusted["proxy"].rho == plain["proxy"].rho and adjusted["proxy"].direction == plain["proxy"].direction


def _followup_confounded(rng, n=6000):
    """``x`` lowers the hazard but lengthens follow-up, so the event flag rises with it."""
    h = pd.DataFrame({"x": rng.normal(size=n), "age": rng.normal(size=n)})
    t = rng.exponential(1 / (0.4 * np.exp(-0.3 * h.x + 0.3 * h.age)))
    censor = np.minimum(rng.uniform(0.2, 1.0, n) * np.exp(1.2 * h.x), 6)
    h["followup_days"] = np.ceil(np.minimum(t, censor) * 365)
    h["y"] = (t <= censor).astype(int)
    return h


def test_a_time_to_event_screen_reads_the_hazard_not_the_event_flag():
    frame = _followup_confounded(np.random.default_rng(1))
    (plain,) = _rank_candidates(frame, "y", ["x"])
    (cox,) = _rank_candidates(frame, "y", ["x"], adjust=frame[["age"]].to_numpy(dtype=float), adjust_names=["age"],
                              survival_time=frame["followup_days"].to_numpy(dtype=float))
    assert plain.rho > 0.1                                   # the flag rises with x ...
    assert cox.screen["measure"] == "hazard ratio per SD (Cox)" and cox.screen["effect"] < 0.85   # ... the hazard falls
    assert cox.q_value < 1e-10 and cox.p_value == cox.screen["p"]


def test_the_engine_uses_the_screens_only_when_asked():
    frame = _proxy_data(np.random.default_rng(2))
    base = dict(target_column="y", confounder_columns=["age"], validation_resamples=40, top_k=5)
    off = run_discovery(frame, DiscoveryRequest(**base))
    assert all("screen" not in c.to_dict() for c in off.candidates) and "screen_measure" not in off.fact_sheet
    assert any("with Spearman correlation" in line for line in off.audit_log)
    on = run_discovery(frame, DiscoveryRequest(**base, screen_adjust=True))
    assert on.fact_sheet["screen_measure"] == "partial Spearman rho" and on.fact_sheet["screen_adjusted_for"] == ["age"]
    assert any("partial Spearman rho adjusted for age" in line for line in on.audit_log)
    assert "proxy" not in {c.feature for c in on.candidates if c.verdict in ("validated", "conditional")}
    timed = _followup_confounded(np.random.default_rng(3))
    report = run_discovery(timed, DiscoveryRequest(target_column="y", confounder_columns=["age"],
                                                   context_columns=["followup_days"], validation_resamples=40,
                                                   top_k=5, screen_adjust=True, survival_time_column="followup_days"))
    assert report.fact_sheet["screen_measure"].startswith("hazard ratio per SD (Cox")
    (x,) = [c for c in report.candidates if c.feature == "x"]
    assert x.screen["effect"] < 1 and x.screen["adjusted_for"] == ["age"]
