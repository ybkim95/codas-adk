"""Moving to another workspace without losing what the sealed half has already been used for.

A workspace's run folder holds the holdout ledger: every hypothesis tested on the sealed half, and every outcome
opened there. A new workspace (for example one on the Controlled Tier, which a Registered Tier workspace cannot
add for its network perimeter) starts with an empty folder, so an outcome already tested would look fresh, and a
second test of it would be counted as a confirmation. This module carries the ledger across:

* ``export`` writes ``carry_over.json`` beside the reports after every exploration: the ledger entries (feature
  names, p-values, verdicts and hashes; no participant data and no counts) and a fingerprint of the split (a
  SHA-256 of the sorted person ids on each side, and the two sizes, which are large), so the receiving workspace
  can tell whether it seals the same people;
* ``adopt`` runs at the start of an exploration: a ``carry_over*.json`` placed in the reports folder (uploaded
  there in the new workspace) is read once, its entries for the same split settings are appended to the ledger
  under the new holdout key and marked as inherited, and the fingerprint is compared. Importing is the cautious
  direction either way: an inherited outcome counts as opened, so it is never confirmed twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from codas.aou import cohort, ledger
from codas.aou.cohort import CohortSpec, QueryParam, _first_wear_cte, _valid_rule, resolve_cdr

FILE = "carry_over.json"
ADOPTED = "carry_over_adopted.json"


def split_sql(spec: CohortSpec, cdr: str) -> tuple[str, list[QueryParam]]:
    """Every participant with a valid day of the primary source, and whether the split seals them."""
    cdr = resolve_cdr(cdr)
    t = lambda name: f"`{cdr}.{name}`"  # noqa: E731 - table reference helper
    params: list[QueryParam] = []
    valid = _valid_rule(spec, params)
    params += [("holdout_permille", "INT64", int(round(spec.holdout_fraction * 1000))),
               ("split_salt", "INT64", int(spec.split_salt))]
    return ("WITH " + _first_wear_cte(spec.wearables[0], t, valid) + "\n"
            f"SELECT person_id, {cohort.sealed_condition(spec)} AS sealed FROM first_wear"), params


def fingerprint(spec: CohortSpec, cdr: str, client: Any) -> dict[str, Any]:
    """Hashes of the sorted person ids on each side of the split, with the sides' sizes."""
    frame = cohort._query(client, *split_sql(spec, cdr))
    out: dict[str, Any] = {}
    for side, flag in (("sealed", True), ("discovery", False)):
        ids = sorted(str(int(p)) for p in frame.loc[frame["sealed"].astype(bool) == flag, "person_id"])
        out[f"{side}_sha256"] = hashlib.sha256(",".join(ids).encode()).hexdigest()
        out[f"{side}_participants"] = len(ids)
    return out


def _settings(key: str) -> str:
    """The split settings part of a holdout key (salt, share fraction and side), without the CDR."""
    return key.split("|", 1)[1] if "|" in key else key


def export(out: Path, spec: CohortSpec, cdr: str, client: Any, reports: Path) -> Path:
    """Write ``carry_over.json`` (module doc) to ``reports`` and return its path."""
    path = Path(out) / ledger.LEDGER_NAME
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []
    try:
        mark = fingerprint(spec, cdr, client)
    except Exception as exc:  # noqa: BLE001 - the ledger still travels; the fingerprint is a check
        mark = {"error": f"{type(exc).__name__}"}
    record = {"cdr": resolve_cdr(cdr), "split": {"salt": spec.split_salt, "fraction": spec.holdout_fraction,
                                                 "share": spec.discovery_share},
              "fingerprint": mark, "ledger": entries}
    reports.mkdir(parents=True, exist_ok=True)
    target = reports / FILE
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return target


def adopt(out: Path, key: str, spec: CohortSpec, cdr: str, client: Any, reports_root: Path,
          log: Any = print) -> dict[str, Any] | None:
    """Import a carried-over ledger found under ``reports_root`` (module doc), once; None when there is none."""
    out = Path(out)
    done = out / ADOPTED
    seen = json.loads(done.read_text(encoding="utf-8")) if done.exists() else {}
    candidates = sorted(Path(reports_root).rglob("carry_over*.json")) if Path(reports_root).exists() else []
    result = None
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen or record.get("cdr") == resolve_cdr(cdr):
            continue  # already imported, or this workspace's own export
        wanted = _settings(key)
        entries = [{**e, "inherited_from": e.get("key")} for e in record.get("ledger") or []
                   if _settings(str(e.get("key", ""))) == wanted]
        if entries:
            out.mkdir(parents=True, exist_ok=True)
            ledger.append(out, key, [{k: v for k, v in e.items() if k != "key"} for e in entries])
        mine = {}
        try:
            mine = fingerprint(spec, cdr, client)
        except Exception as exc:  # noqa: BLE001 - reported, the import stands
            mine = {"error": type(exc).__name__}
        theirs = record.get("fingerprint") or {}
        same = bool(mine.get("sealed_sha256") and mine.get("sealed_sha256") == theirs.get("sealed_sha256"))
        result = {"file": path.name, "from_cdr": record.get("cdr"), "entries": len(entries), "same_people_sealed": same}
        seen[digest] = result
        log(f"carried over {len(entries)} ledger entries from {record.get('cdr')}: the sealed half here "
            + ("holds exactly the same people" if same else
               "does not hold exactly the same people (inherited outcomes still count as opened)"))
    if result is not None:
        out.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(seen, indent=1), encoding="utf-8")
    return result
