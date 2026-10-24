"""The dress rehearsal: before any All of Us query, the real models run the whole pipeline once on a synthetic
scenario with a planted truth (``codas.aou.scenarios``, S1: heart-rate reactivity to short sleep), on synthetic
rows only, for one discovery round.

It answers whether this code and these models can do the job before hours of compute on real data: whether the
run finishes, hypotheses get registered, the report and the execution records are written (a failure there
stops the exploration before any query: ``RehearsalError``), and, as warnings, whether the agents defined a
phenotype from the day-by-day data, whether it was a dynamic one, whether they found the planted signal on the
sealed half, and whether the literature could be searched and cited. The result is kept per analysis-code
version and models, so a later Run All does not repeat it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from codas.aou import scenarios

FOLDER = "rehearsal"
LATEST = "latest.json"
DYNAMIC = ("coupling", "slope_per_30d", "mean_abs_change", "weekend_minus_weekday", "sd", "cv", "fraction")


class RehearsalError(RuntimeError):
    """The rehearsal found something that would also break the real run."""


def _models() -> list[str]:
    from codas.core import model_tiers

    return [model_tiers.model(tier) for tier in model_tiers.ENV]


def key() -> str:
    """The analysis code and the models a rehearsal result holds for."""
    from codas.aou.runner import _analysis_version

    return hashlib.sha256(json.dumps([_analysis_version(), _models()]).encode()).hexdigest()[:16]


def score(summary: dict[str, Any], state: dict[str, Any], out: Path) -> dict[str, Any]:
    """Each check with whether it passed; ``critical`` ones would break a real run too."""
    confirmation = state.get("confirmation") or {}
    defined = state.get("defined_phenotypes") or []
    columns = {d["column"]: d for d in defined}
    found = [r["feature"] for r in confirmation.get("hypotheses") or []
             if r.get("feature") in columns and (r.get("replicated") or r.get("reproduced"))
             and ((columns[r["feature"]].get("definition") or {}).get("summary") == "coupling")]
    citations = state.get("citations") or {}
    used, verified = len(citations.get("used") or []), len(citations.get("verified") or [])
    execution = summary.get("execution") or {}
    report = Path(str(summary.get("report_path") or ""))
    checks = [
        ("the pipeline ran to the end", True, True),
        ("hypotheses were registered before any statistic", bool(state.get("a_priori_hypotheses")), True),
        ("the sealed half was tested", bool(confirmation.get("hypotheses")), True),
        ("the report was written", report.is_file() and report.stat().st_size > 200, True),
        ("every model call's tokens were recorded", int(execution.get("tokens") or 0) > 0, True),
        ("a phenotype was defined on the day-by-day data", bool(defined), False),
        ("a dynamic phenotype was defined (how days vary, drift or respond)",
         any((d.get("definition") or {}).get("summary") in DYNAMIC for d in defined), False),
        ("the planted reactivity was found and replicated on the sealed half", bool(found), False),
        ("the literature could be searched", str(state.get("literature_status") or "").startswith("available"), False),
        ("citations were among the papers retrieved", (verified / used >= 0.5) if used else True, False),
    ]
    return {"checks": [{"check": c, "passed": bool(ok), "critical": crit} for c, ok, crit in checks],
            "found": found, "defined": [{"column": d["column"], "description": d.get("description")} for d in defined],
            "critical_failures": [c for c, ok, crit in checks if crit and not ok],
            "warnings": [c for c, ok, crit in checks if not crit and not ok]}


def rehearse(out_dir: str | Path, *, rounds: int = 1, n: int = 3000, log: Any = print,
             on_tool_call: Any = None, on_agent: Any = None) -> dict[str, Any]:
    """Run the pipeline once on S1 in ``out_dir/rehearsal`` (its own holdout ledger) and score it."""
    from codas.aou import runner

    folder = Path(out_dir) / FOLDER
    shutil.rmtree(folder, ignore_errors=True)  # a fresh scenario and ledger: an earlier test would make this one a repeat
    started = time.time()
    csv = scenarios.reactivity(folder, n=n)
    spec = scenarios.reactivity_spec()
    try:
        summary = runner.run_spec(spec, "synthetic.scenario", label=scenarios.S1_LABEL, out_dir=folder, csv=csv,
                                  check_schema=False, on_tool_call=on_tool_call, on_agent=on_agent,
                                  extra_state={"round_limit": rounds})
    except Exception as exc:  # noqa: BLE001 - the rehearsal's job is to catch exactly this before a real run
        result = {"key": key(), "minutes": round((time.time() - started) / 60, 1), "checks": [
            {"check": "the pipeline ran to the end", "passed": False, "critical": True}],
            "critical_failures": [f"the pipeline ran to the end ({type(exc).__name__}: {str(exc)[:300]})"],
            "warnings": [], "found": [], "defined": []}
    else:
        state = json.loads(Path(summary["files"]["state"]).read_text(encoding="utf-8"))
        result = {"key": key(), "minutes": round((time.time() - started) / 60, 1), **score(summary, state, folder),
                  "summary_path": summary.get("report_path")}
    (folder / f"rehearsal_{result['key']}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (folder / LATEST).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for c in result["checks"]:
        log(f"  rehearsal: {'ok  ' if c['passed'] else ('FAIL' if c['critical'] else 'warn')} {c['check']}")
    return result


def run_once(out_dir: str | Path, *, log: Any = print, **kwargs: Any) -> dict[str, Any]:
    """The rehearsal for this code and these models: run it unless a result is kept; raise on a critical
    failure, so the exploration stops before any query."""
    folder = Path(out_dir) / FOLDER
    kept = folder / f"rehearsal_{key()}.json"
    if kept.exists():
        result = json.loads(kept.read_text(encoding="utf-8"))
        log("rehearsal: passed earlier with this code and these models; not repeated")
    else:
        log("rehearsal: the real models run the pipeline once on synthetic data before any All of Us query")
        result = rehearse(out_dir, log=log, **kwargs)
    (folder / LATEST).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    if result["critical_failures"]:
        kept.unlink(missing_ok=True)  # a failed rehearsal runs again next time
        raise RehearsalError("the rehearsal on synthetic data failed, so no All of Us query was run: "
                             + "; ".join(result["critical_failures"]))
    return result
