"""Genetic instruments for an exposure: variants published as associated with it, their effect alleles and weights.

* ``from_table``: instruments from a table (CSV, TSV or JSON records) with ``construct``, ``variant`` (an rsID),
  ``effect_allele``, ``other_allele``, ``beta`` and optionally ``p`` and ``source`` (a PMID or DOI).
* ``gwas_catalog``: instruments from the NHGRI-EBI GWAS Catalog for a trait name (genome-wide significant
  associations with a per-allele effect), through its public REST interface (v2); only the trait name leaves the
  machine. The interface's shape is read defensively; when it cannot be read, no instrument is returned and the
  reason is given, never a guess.
* ``align``: each instrument matched to a genotype file's variants (by rsID), its weight turned onto the file's
  counted allele (a flipped allele pair flips the sign), strand-ambiguous pairs (A/T, C/G) and mismatched alleles
  left out with the reason.
* ``prune``: one instrument per window of ``window_kb`` on a chromosome (the strongest), a stand-in for
  clumping on linkage disequilibrium when no reference panel is at hand.

Nothing here names a dataset, an exposure or an outcome.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

GENOME_WIDE = 5e-8
_GWAS = "https://www.ebi.ac.uk/gwas/rest/api/v2"   # v1 (the HAL interface) was retired in 2026
_AMBIGUOUS = ({"A", "T"}, {"C", "G"})


@dataclass(frozen=True)
class Instrument:
    construct: str
    variant: str
    effect_allele: str
    other_allele: str
    beta: float
    p: float | None = None
    source: str | None = None


def from_table(path: str | Path) -> list[Instrument]:
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(rows.get("instruments", rows) if isinstance(rows, dict) else rows)
    else:
        frame = pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",")
    out = []
    for r in frame.to_dict("records"):
        try:
            out.append(Instrument(str(r["construct"]), str(r["variant"]), str(r["effect_allele"]).upper(),
                                  str(r["other_allele"]).upper(), float(r["beta"]),
                                  None if pd.isna(r.get("p")) else float(r["p"]),
                                  None if pd.isna(r.get("source")) else str(r["source"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _p(assoc: dict[str, Any]) -> float | None:
    for key in ("p_value", "pvalue"):
        if assoc.get(key) is not None:
            try:
                return float(assoc[key])
            except (TypeError, ValueError):
                return None
    m = assoc.get("pvalue_mantissa", assoc.get("pvalueMantissa"))
    e = assoc.get("pvalue_exponent", assoc.get("pvalueExponent"))
    try:
        return float(m) * 10.0 ** int(e)
    except (TypeError, ValueError):
        return None


_BETA = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b.*\b(increase|decrease)\b", re.I)


def _effect(assoc: dict[str, Any]) -> float | None:
    """The per-allele effect: a beta signed by its direction (``"0.02 unit decrease"``, or ``betaNum`` with
    ``betaDirection``), else the log of a per-copy odds ratio; None when the record states neither."""
    text = assoc.get("beta")
    if isinstance(text, str) and (match := _BETA.match(text)):
        return float(match.group(1)) * (-1.0 if match.group(2).lower() == "decrease" else 1.0)
    if assoc.get("betaNum") is not None:
        return float(assoc["betaNum"]) * (-1.0 if str(assoc.get("betaDirection") or "").lower().startswith("de") else 1.0)
    for key in ("or_per_copy_number", "or_value", "orPerCopyNum"):
        try:
            value = float(assoc.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return math.log(value)
    return None


def _risk_allele(assoc: dict[str, Any]) -> tuple[str, str] | None:
    """The single variant and its effect allele (``rs123``, ``A``): from ``snp_allele``, ``snp_effect_allele``
    or the loci's strongest risk alleles; None for a haplotype, an interaction or an unknown allele."""
    if assoc.get("snp_allele"):
        pairs = [(str(a.get("rs_id") or ""), str(a.get("effect_allele") or "")) for a in assoc["snp_allele"]]
    else:
        names = assoc.get("snp_effect_allele") or [a.get("riskAlleleName") for locus in assoc.get("loci") or []
                                                   for a in locus.get("strongestRiskAlleles") or []]
        pairs = [tuple(str(n).rsplit("-", 1)) if n and "-" in str(n) else ("", "") for n in names]
    if len(pairs) != 1:
        return None
    variant, allele = pairs[0]
    if not variant.lower().startswith("rs") or allele.upper() not in ("A", "C", "G", "T"):
        return None
    return variant.lower(), allele.upper()


def parse_associations(construct: str, payload: dict[str, Any]) -> list[Instrument]:
    """Genome-wide significant associations with a per-allele effect from a GWAS Catalog association page
    (the REST interface's v2 records; the retired v1 shape is read too): the single variant's effect allele,
    the effect (a beta signed by its direction, else the log of a per-copy odds ratio) and p. An association
    without an effect allele or a per-allele effect is left out. The other allele is not part of the record; it
    is read from the genotype file when aligning (``align`` with ``other_allele`` "?")."""
    rows = (payload.get("_embedded") or {}).get("associations") or payload.get("associations") or []
    out = []
    for assoc in rows:
        p = _p(assoc)
        if p is None or p > GENOME_WIDE:
            continue
        risk = _risk_allele(assoc)
        beta = _effect(assoc)
        if risk is None or beta is None:
            continue
        pmid = assoc.get("pubmed_id") or ((assoc.get("study") or {}).get("publicationInfo") or {}).get("pubmedId")
        out.append(Instrument(construct, risk[0], risk[1], "?", beta, p, f"PMID:{pmid}" if pmid else None))
    return out


PAGES = 30          # association pages read per trait (100 records each, most significant first)
PAGE_TIMEOUT_S = 90  # the interface takes about 0.1 s per record


def _associations(query: dict[str, str], cache: Any, opener: Any) -> tuple[list[dict[str, Any]], int]:
    """The trait's associations, most significant first, until they stop being genome-wide significant."""
    from codas.core import literature

    url = f"{_GWAS}/associations?" + urllib.parse.urlencode({**query, "size": 100, "sort": "p_value", "direction": "asc"})
    rows, total = [], 0
    for _ in range(PAGES):
        page = literature._get_json(url, cache, opener, timeout=PAGE_TIMEOUT_S)
        got = (page.get("_embedded") or {}).get("associations") or []
        total = int((page.get("page") or {}).get("totalElements") or len(got))
        rows += got
        nxt = ((page.get("_links") or {}).get("next") or {}).get("href")
        if not got or not nxt or (_p(got[-1]) or 1.0) > GENOME_WIDE:
            break
        url = nxt
    return rows, total


def gwas_catalog(construct: str, trait: str, *, opener: Any = None, cache: Any = None) -> tuple[list[Instrument], str]:
    """Instruments for ``trait`` from the GWAS Catalog (REST interface v2), and a status line (module doc): the
    associations annotated with the trait by name, else with the ontology terms the trait name resolves to."""
    from codas.core import literature

    try:
        rows, total = _associations({"efo_trait": trait}, cache, opener)
        if not total:
            found = literature._get_json(f"{_GWAS}/efo-traits?" + urllib.parse.urlencode({"efo_trait": trait, "size": 3}),
                                         cache, opener, timeout=PAGE_TIMEOUT_S)
            terms = [t.get("efo_id") for t in (found.get("_embedded") or {}).get("efo_traits") or [] if t.get("efo_id")]
            if not terms:
                return [], f"no GWAS Catalog trait named {trait!r}"
            for term in terms[:3]:
                more, n = _associations({"efo_id": term}, cache, opener)
                rows, total = rows + more, total + n
        out = parse_associations(construct, {"associations": rows})
    except literature.Unreachable as exc:
        return [], f"GWAS Catalog not reachable ({exc})"
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return [], f"GWAS Catalog answer not readable ({type(exc).__name__})"
    best: dict[str, Instrument] = {}
    for i in out:  # one record per variant: the most significant
        if i.variant not in best or (i.p or 1) < (best[i.variant].p or 1):
            best[i.variant] = i
    return list(best.values()), f"{len(best)} genome-wide significant variants for {trait!r}"


def align(instruments: list[Instrument], variants: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Each instrument on the genotype file's counted allele ``a1``: (variant, weight on the a1 dosage, chrom,
    pos), and the reason each left-out instrument was dropped."""
    index = variants.drop_duplicates("id").set_index("id")
    kept, dropped = [], {}
    for i in instruments:
        if i.variant not in index.index:
            dropped[i.variant] = "not in the genotype file"
            continue
        row = index.loc[i.variant]
        a1, a2 = str(row["a1"]).upper(), str(row["a2"]).upper()
        if {a1, a2} in _AMBIGUOUS:
            dropped[i.variant] = "strand-ambiguous allele pair"
            continue
        other = a2 if i.effect_allele == a1 else a1 if i.effect_allele == a2 else None
        if other is None or (i.other_allele not in ("?", other)):
            dropped[i.variant] = "alleles do not match the genotype file"
            continue
        kept.append({"variant": i.variant, "weight": i.beta if i.effect_allele == a1 else -i.beta,
                     "chrom": str(row["chrom"]), "pos": int(row["pos"]), "p": i.p, "source": i.source})
    return kept, dropped


def prune(aligned: list[dict[str, Any]], window_kb: int = 500) -> list[dict[str, Any]]:
    """The strongest instrument in each ``window_kb`` window of a chromosome (module doc)."""
    chosen: list[dict[str, Any]] = []
    for a in sorted(aligned, key=lambda a: (a["p"] if a["p"] is not None else 1.0, -abs(a["weight"]))):
        if all(a["chrom"] != c["chrom"] or abs(a["pos"] - c["pos"]) > window_kb * 1000 for c in chosen):
            chosen.append(a)
    return chosen


def as_records(instruments: list[Instrument]) -> list[dict[str, Any]]:
    return [asdict(i) for i in instruments]
