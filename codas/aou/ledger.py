"""The sealed holdout's ledger: every hypothesis tested on it, in order, with the verdict it got.

Each look at a holdout is a chance at a false replication, so the ledger decides which looks may count:

* Identity. An entry is keyed by its outcome (the outcome's name and a hash of its concept set) and by its
  feature definition (a hash of the rebuilt feature's recipe).
* Repeat. A pair already tested on this holdout is reported but not tested again: re-running an analysis
  after a code change cannot buy a second chance.
* Opened outcome. Once any hypothesis about an outcome has been tested here, its holdout results are known
  and can shape later designs, so later registrations about that outcome are reported as non-confirmatory.
* Families.
  - ``novel``: emerging or unexplored a-priori hypotheses and exploratory candidates. They are tested under
    online false-discovery-rate control (LORD++, Javanmard & Montanari 2018; Ramdas et al. 2017) in
    ledger order. Each test gets a level that depends only on earlier tests, so a verdict is final when made
    and never changes as the ledger grows.
  - ``reference``: associations the literature already establishes. Each is tested at ``ALPHA`` on its own,
    like a positive control, and is never counted as a discovery.
* Direction. Hypotheses are directional (the pre-registered sign), so the test uses the one-sided p in that
  direction.
* Entries written before this scheme carry no family. They mark their outcomes as opened and are listed
  separately; they take no part in the online procedure.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

LEDGER_NAME = "holdout_ledger.jsonl"
ALPHA = 0.05
INITIAL_WEALTH = ALPHA / 2  # LORD++'s W0; must not exceed ALPHA
NOVEL, REFERENCE, NON_CONFIRMATORY = "novel", "reference", "non_confirmatory"
OPENED = "opened"  # an outcome whose holdout was read by something other than a registration (a full-cohort scan)
_REPEAT = "repeat of an earlier test of this outcome and feature on this holdout"
_OPENED = "this outcome was already tested on this holdout, so its holdout results were known when this was registered"


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def outcome_key(outcome: dict[str, Any] | None, stem: str) -> str:
    """The outcome's identity: its concept set and coding when known, else its name."""
    if not outcome:
        return _hash({"stem": stem})
    return _hash({k: (sorted(v) if isinstance(v, list) else v) for k, v in outcome.items()
                  if k in ("kind", "concept_ids", "codes", "value_min", "value_max", "min_occurrences")})


def feature_key(hypothesis: dict[str, Any]) -> str:
    """The feature's identity: its name together with the recipe that rebuilds it."""
    return _hash({"feature": hypothesis.get("feature"), "definition": hypothesis.get("definition") or {}})


def gamma(j: int) -> float:
    """The LORD++ spending sequence (Javanmard & Montanari 2018): positive, non-increasing, sums to about 1."""
    return 0.0722 * math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(j)))) if j >= 1 else 0.0


def lord_level(t: int, rejections: list[int], alpha: float = ALPHA, w0: float = INITIAL_WEALTH) -> float:
    """The level for the t-th novel test (1-based), given the indices of earlier rejections."""
    earlier = [tau for tau in rejections if tau < t]
    level = gamma(t) * w0
    if earlier:
        level += (alpha - w0) * gamma(t - earlier[0])
        level += alpha * sum(gamma(t - tau) for tau in earlier[1:])
    return level


def one_sided_p(p: float, same_direction: bool) -> float:
    """The p-value for an effect in the pre-registered direction, from a two-sided p of a symmetric test."""
    if not isinstance(p, (int, float)) or not math.isfinite(p):
        return float("nan")
    return p / 2 if same_direction else 1 - p / 2


def load(out_dir: Path, key: str) -> list[dict[str, Any]]:
    path = Path(out_dir) / LEDGER_NAME
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("key") == key:
            entries.append(entry)
    return entries


def opened(out_dir: Path, key: str) -> tuple[set[str], set[str]]:
    """The run names and outcome keys already tested on the holdout ``key``."""
    entries = load(out_dir, key)
    return {str(e.get("stem")) for e in entries}, {str(e["outcome_key"]) for e in entries if e.get("outcome_key")}


def _novel_stream(entries: list[dict[str, Any]]) -> tuple[int, list[int]]:
    """How many novel tests the ledger holds and at which of them the null was rejected."""
    count, rejections = 0, []
    for e in entries:
        if e.get("role") == NOVEL and e.get("tested"):
            count += 1
            if e.get("rejected"):
                rejections.append(count)
    return count, rejections


def decide(prior: list[dict[str, Any]], prereg: dict[str, Any], rows: list[dict[str, Any]],
           outcome: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Classify and judge this registration's hypotheses against everything tested before it.

    ``rows`` are the confirmation rows in pre-registered order (each with ``feature``, ``family``,
    ``knownness``, ``p`` and ``same_direction``); their hypotheses in ``prereg["hypotheses"]`` carry the
    feature definitions. Returns one ledger entry per row (not yet written)."""
    stem, sha = prereg["stem"], prereg["sha256"]
    okey = outcome_key(outcome, stem)
    earlier = [e for e in prior if e.get("prereg") != sha]
    opened = {e.get("stem") for e in earlier} | {e.get("outcome_key") for e in earlier if e.get("outcome_key")}
    tested_pairs = {(e.get("outcome_key"), e.get("feature_key")) for e in earlier if e.get("feature_key")}
    definitions = {h["feature"]: h for h in prereg.get("hypotheses") or []}
    count, rejections = _novel_stream(earlier)
    entries = []
    for row in rows:
        fkey = feature_key(definitions.get(row["feature"], {"feature": row["feature"]}))
        p_one = one_sided_p(row.get("p", float("nan")), bool(row.get("same_direction")))
        entry = {"key": None, "prereg": sha, "stem": stem, "outcome_key": okey, "feature": row["feature"],
                 "feature_key": fkey, "p": row.get("p"), "p_one_sided": p_one,
                 "same_direction": bool(row.get("same_direction")), "family": row.get("family")}
        if (okey, fkey) in tested_pairs:
            entry.update(role=NON_CONFIRMATORY, reason=_REPEAT, tested=False, rejected=False)
        elif stem in opened or okey in opened:
            entry.update(role=NON_CONFIRMATORY, reason=_OPENED, tested=False, rejected=False)
        elif row.get("family") == "a_priori" and row.get("knownness") == "established":
            testable = math.isfinite(p_one)
            entry.update(role=REFERENCE, level=ALPHA, tested=testable,
                         rejected=bool(testable and p_one <= ALPHA))
        else:
            if math.isfinite(p_one):
                count += 1
                level = lord_level(count, rejections)
                rejected = p_one <= level
                if rejected:
                    rejections.append(count)
                entry.update(role=NOVEL, level=level, tested=True, rejected=bool(rejected), order=count)
            else:  # the test could not be computed: it spends nothing and decides nothing
                entry.update(role=NOVEL, tested=False, rejected=False)
        entries.append(entry)
    return entries


def append(out_dir: Path, key: str, entries: list[dict[str, Any]]) -> None:
    path = Path(out_dir) / LEDGER_NAME
    with path.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps({**e, "key": key}, default=str) + "\n")


def mark_opened(out_dir: Path, key: str, outcomes: list[tuple[str, dict[str, Any] | None]], reason: str) -> None:
    """Record outcomes as opened on the holdout ``key`` without a test: later registrations about them are
    non-confirmatory (``decide``). ``outcomes`` are (name, identity) pairs."""
    append(out_dir, key, [{"stem": f"opened: {name}", "outcome_key": outcome_key(identity, name), "role": OPENED,
                           "reason": reason} for name, identity in outcomes])


def summary(out_dir: Path, key: str) -> dict[str, Any]:
    """What has been tested on the holdout ``key``: totals by family, the online procedure's state, and the
    frozen verdicts (novel replications and reproduced reference associations)."""
    every = load(out_dir, key)
    marks = [e for e in every if e.get("role") == OPENED]
    entries = [e for e in every if e.get("role") != OPENED]
    legacy = [e for e in entries if "role" not in e]
    current = [e for e in entries if "role" in e]
    novel = [e for e in current if e["role"] == NOVEL and e.get("tested")]
    reference = [e for e in current if e["role"] == REFERENCE and e.get("tested")]
    count, rejections = _novel_stream(current)
    return {
        "key": key, "hypotheses": len(entries), "registrations": len({e.get("prereg") for e in entries}),
        "legacy": len(legacy), "legacy_outcomes": sorted({str(e.get("stem")) for e in legacy}),
        "opened_outcomes": sorted({str(e.get("stem")) for e in entries}),
        "opened_without_a_test": sorted({str(e.get("stem"))[len("opened: "):] for e in marks}),
        "novel_tested": len(novel), "novel_replicated": [f"{e['stem']}: {e['feature']}" for e in novel if e.get("rejected")],
        "reference_tested": len(reference),
        "reference_reproduced": [f"{e['stem']}: {e['feature']}" for e in reference if e.get("rejected")],
        "non_confirmatory": sum(1 for e in current if e["role"] == NON_CONFIRMATORY),
        "next_level": lord_level(count + 1, rejections),
    }
