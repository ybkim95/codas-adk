#!/usr/bin/env python3
"""Load + soak harness for the deterministic endpoints (no API key).

The deterministic engine (`/v1/discover`, `/v1/profile`) is the high-QPS surface — the agent path is
LLM-bound and low-QPS by nature. This measures concurrent throughput and latency percentiles, and
soaks the engine path to check for memory growth (a leak would show as steadily rising Python heap).

    python scripts/loadtest.py [N_REQUESTS] [CONCURRENCY]
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CODAS_AGENT_API_KEYS", "loadtest-key")

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from codas.service.app import app

client = TestClient(app)
H = {"X-CoDaS-Agent-Key": "loadtest-key"}

_rng = np.random.default_rng(0)
_n = 1500
_x = _rng.normal(size=_n)
CSV = pd.DataFrame({
    "x1": _x, "x2": _rng.normal(size=_n), "x3": _rng.normal(size=_n),
    **{f"f{i}": _rng.normal(size=_n) for i in range(8)},
    "y": 0.5 * _x + _rng.normal(size=_n) * 0.8,
}).to_csv(index=False)
PAYLOAD = {"csv": CSV, "target_column": "y", "validation_resamples": 80, "top_k": 6}


def _one() -> tuple[float, int]:
    t0 = time.perf_counter()
    r = client.post("/v1/discover", headers=H, json=PAYLOAD)
    return time.perf_counter() - t0, r.status_code


def _pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(round(p / 100 * len(xs))))]


def load_test(n: int, c: int) -> None:
    _one()  # warm up
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        results = list(ex.map(lambda _: _one(), range(n)))
    wall = time.perf_counter() - t0
    lat = [s * 1000 for s, _ in results]  # ms
    ok = sum(code == 200 for _, code in results)
    print("=" * 78)
    print(f"LOAD TEST — /v1/discover  ({n} requests, concurrency {c}, {_n}x{CSV.count(chr(10))} CSV)")
    print("=" * 78)
    print(f"  success      : {ok}/{n} ({100*ok/n:.1f}%)   errors: {n - ok}")
    print(f"  throughput   : {n / wall:.1f} req/s   (wall {wall:.1f}s)")
    print(f"  latency (ms) : p50={_pct(lat,50):.0f}  p90={_pct(lat,90):.0f}  p99={_pct(lat,99):.0f}  "
          f"max={max(lat):.0f}  mean={statistics.mean(lat):.0f}")
    print(f"  VERDICT: {'PASS' if ok == n else 'FAIL — non-200 responses under load'}")


def soak_test(m: int) -> None:
    for _ in range(20):
        _one()  # warm up allocator
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    samples = []
    for i in range(m):
        _one()
        if (i + 1) % max(1, m // 6) == 0:
            cur = tracemalloc.get_traced_memory()[0]
            samples.append((i + 1, (cur - base) / 1024))
    growth = samples[-1][1] if samples else 0.0
    tracemalloc.stop()
    print("\n" + "=" * 78)
    print(f"SOAK TEST — {m} sequential /v1/discover (engine-path memory leak check)")
    print("=" * 78)
    for n_done, kb in samples:
        print(f"  after {n_done:4d} requests: heap +{kb:8.1f} KB vs baseline")
    print(f"  net growth over {m} requests: {growth:.1f} KB "
          f"({growth / max(1, m):.3f} KB/request)")
    print(f"  VERDICT: {'PASS — engine path is memory-stable' if growth < 5000 else 'FAIL — possible leak'}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    load_test(N, C)
    soak_test(180)
