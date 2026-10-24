"""Genotypes for genetic triangulation, read inside the Workbench from the Controlled Tier's genomic files.

Only the few variants that instrument a registered exposure are read: PLINK 1 binary files store each variant as a
fixed-length run of bytes (four genotypes per byte), so a variant is fetched by a byte-range read of the ``.bed``
file without downloading the rest (``read_variants``). The ``.bim`` (variants) and ``.fam`` (samples) files say
where each variant and person is; the ancestry file gives each person's genetic principal components, the
covariates that keep ancestry from confounding a genetic score.

Where these files are is not assumed: ``locate`` searches the CDR's storage folder (the ``CDR_STORAGE_PATH``
environment variable, or a path given) for array PLINK files and the ancestry predictions by their names, a few
levels deep, and reports what it found. Nothing here leaves the Workbench: genotypes become per-person scores and
dosages in the sealed folder, keyed like every other sealed table, and only aggregates are reported.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from codas.core.mendelian import allele_score

# A reader fetches bytes [start, end) of a URI (end None: to the end); local paths and gs:// URIs are served.
Reader = Callable[[str, int, int | None], bytes]
_BED_MAGIC = b"\x6c\x1b\x01"
_KEYWORDS = ("microarray", "array", "plink", "aux", "ancestry", "genomic", "wgs", "snpindel", "short_read")


def local_reader(uri: str, start: int = 0, end: int | None = None) -> bytes:
    with open(uri, "rb") as fh:
        fh.seek(start)
        return fh.read() if end is None else fh.read(end - start)


def gcs_reader(project: str | None = None) -> Reader:
    """Byte-range reads from Cloud Storage, billed to ``project`` (the CDR bucket is requester-pays)."""
    from google.cloud import storage

    client = storage.Client(project=project)

    def read(uri: str, start: int = 0, end: int | None = None) -> bytes:
        bucket, _, name = uri[len("gs://"):].partition("/")
        blob = client.bucket(bucket, user_project=project or client.project).blob(name)
        return blob.download_as_bytes(start=start, end=None if end is None else end - 1)

    return read


def reader_for(uri: str) -> Reader:
    return gcs_reader(os.getenv("GOOGLE_CLOUD_PROJECT")) if uri.startswith("gs://") else local_reader


@dataclass
class Plink:
    """A PLINK 1 binary fileset: the ``.bed`` URI, its variants (``.bim``) and samples (``.fam``)."""

    bed: str
    variants: pd.DataFrame   # chrom, id, cm, pos, a1, a2 (a1 is the counted allele)
    samples: pd.Series       # the person id of each sample, in file order

    @classmethod
    def open(cls, bed: str, read: Reader | None = None) -> "Plink":
        read = read or reader_for(bed)
        stem = bed[: -len(".bed")]
        bim = pd.read_csv(io.BytesIO(read(stem + ".bim", 0, None)), sep=r"\s+", header=None,
                          names=["chrom", "id", "cm", "pos", "a1", "a2"], dtype={"chrom": str, "id": str})
        fam = pd.read_csv(io.BytesIO(read(stem + ".fam", 0, None)), sep=r"\s+", header=None,
                          names=["fid", "iid", "father", "mother", "sex", "pheno"], dtype={"iid": str})
        if read(bed, 0, 3) != _BED_MAGIC:
            raise ValueError(f"{bed} is not a variant-major PLINK 1 .bed file.")
        return cls(bed, bim, fam["iid"].astype(str))

    def read_variants(self, ids: list[str], read: Reader | None = None) -> pd.DataFrame:
        """Allele-1 dosages (0, 1, 2; NaN missing) of the variants ``ids`` that the fileset holds, one column each,
        one row per sample (index: person id)."""
        read = read or reader_for(self.bed)
        n = len(self.samples)
        width = (n + 3) // 4
        where = {v: i for i, v in enumerate(self.variants["id"])}
        codes = np.array([2.0, np.nan, 1.0, 0.0])  # PLINK: 00 hom a1, 01 missing, 10 het, 11 hom a2
        out = {}
        for vid in ids:
            i = where.get(vid)
            if i is None:
                continue
            raw = np.frombuffer(read(self.bed, 3 + i * width, 3 + (i + 1) * width), dtype=np.uint8)
            bits = np.stack([(raw >> s) & 3 for s in (0, 2, 4, 6)], axis=1).ravel()[:n]
            out[vid] = codes[bits]
        return pd.DataFrame(out, index=self.samples.to_numpy())


def ancestry(uri: str, read: Reader | None = None, components: int = 16) -> pd.DataFrame:
    """Each person's genetic principal components (``pc1``...) and predicted ancestry group from the ancestry
    predictions file (a person id column and a ``pca_features`` list per row); index: person id."""
    read = read or reader_for(uri)
    frame = pd.read_csv(io.BytesIO(read(uri, 0, None)), sep="\t", dtype=str)
    person = next(c for c in frame.columns if c.lower() in ("research_id", "person_id", "s", "sample_id"))
    features = next(c for c in frame.columns if "pca" in c.lower())
    pcs = np.array([json.loads(v) if v.strip().startswith("[") else [float(x) for x in re.split(r"[,\s]+", v.strip())
                                                                       if x] for v in frame[features]], dtype=float)
    out = pd.DataFrame(pcs[:, :components], columns=[f"pc{i + 1}" for i in range(min(components, pcs.shape[1]))],
                       index=frame[person].astype(str).to_numpy())
    group = next((c for c in frame.columns if c.lower() == "ancestry_pred"), None)
    if group:
        out["ancestry_group"] = frame[group].to_numpy()
    return out


def locate(root: str | None = None, list_dir: Callable[[str], tuple[list[str], list[str]]] | None = None,
           depth: int = 5, limit: int = 5000) -> dict[str, Any]:
    """Find array PLINK filesets and the ancestry predictions under ``root`` (default: ``CDR_STORAGE_PATH``),
    descending only into folders whose names suggest genomic data. ``list_dir(prefix)`` returns (sub-folders,
    files) of a prefix; Cloud Storage is used for gs:// roots and the file system otherwise."""
    root = root or os.getenv("CDR_STORAGE_PATH") or ""
    if not root:
        return {"root": None, "status": "no CDR storage path (CDR_STORAGE_PATH is not set)"}
    list_dir = list_dir or (_gcs_lister() if root.startswith("gs://") else _local_lister)
    beds, ancestry_files, seen, frontier = [], [], 0, [(root.rstrip("/") + "/", 0)]
    while frontier and seen < limit:
        prefix, level = frontier.pop(0)
        folders, files = list_dir(prefix)
        seen += len(files) + len(folders)
        beds += [f for f in files if f.endswith(".bed")]
        ancestry_files += [f for f in files if f.endswith("ancestry_preds.tsv")]
        if level < depth:
            frontier += [(f, level + 1) for f in folders
                         if level == 0 or any(k in f.rstrip("/").rsplit("/", 1)[-1].lower() for k in _KEYWORDS)]
    arrays = [b for b in beds if "array" in b.lower()]
    return {"root": root, "status": "found" if (arrays or beds) and ancestry_files else "incomplete",
            "array_beds": sorted(arrays), "other_beds": sorted(set(beds) - set(arrays)),
            "ancestry": sorted(ancestry_files)}


def _local_lister(prefix: str) -> tuple[list[str], list[str]]:
    p = Path(prefix)
    if not p.is_dir():
        return [], []
    return ([str(c) + "/" for c in sorted(p.iterdir()) if c.is_dir()],
            [str(c) for c in sorted(p.iterdir()) if c.is_file()])


def _gcs_lister() -> Callable[[str], tuple[list[str], list[str]]]:
    from google.cloud import storage

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    client = storage.Client(project=project)

    def list_dir(prefix: str) -> tuple[list[str], list[str]]:
        bucket, _, path = prefix[len("gs://"):].partition("/")
        blobs = client.list_blobs(client.bucket(bucket, user_project=project), prefix=path, delimiter="/",
                                  max_results=2000)
        files = [f"gs://{bucket}/{b.name}" for b in blobs]
        return [f"gs://{bucket}/{p}" for p in blobs.prefixes], files

    return list_dir


def slug(construct: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", construct.lower()).strip("_")


def instrument_sets(spec_genetics: dict[str, Any], out: Path, log: Any = print) -> dict[str, Any]:
    """Each construct's instruments: from the spec's table, or from the GWAS Catalog by the traits the
    literature catalog names for the construct (``gwas_traits``), cached in the sealed folder."""
    from codas.aou import cohort, runner
    from codas.core import instruments as inst
    from codas.core import literature

    source = spec_genetics.get("instruments") or "gwas_catalog"
    sets: dict[str, Any] = {}
    if source != "gwas_catalog":
        for i in inst.from_table(Path(source) if Path(source).is_absolute() else cohort.ROOT / source):
            sets.setdefault(i.construct, {"instruments": [], "status": f"from {source}"})["instruments"].append(i)
        return sets
    terms = literature.load_terms(runner.LITERATURE_TERMS) if runner.LITERATURE_TERMS.exists() else {}
    cache = literature.Cache(str(Path(out) / "sealed" / "genetics" / "gwas_catalog_cache.jsonl"))
    for construct, entry in (terms.get("constructs") or {}).items():
        found, notes = [], []
        for trait in entry.get("gwas_traits") or []:
            got, status = inst.gwas_catalog(construct, trait, cache=cache)
            found += got
            notes.append(status)
        if entry.get("gwas_traits"):
            sets[construct] = {"instruments": found, "status": "; ".join(notes)}
            log(f"genetics: {construct}: {'; '.join(notes)}")
    return sets


def person_table(spec_genetics: dict[str, Any], people: pd.Series, out: Path, log: Any = print,
                 read: Reader | None = None, located: dict[str, Any] | None = None) -> tuple[pd.DataFrame | None,
                                                                                            dict[str, Any]]:
    """Every listed person's genetic principal components, allele scores per construct (``score_<construct>``)
    and instrument dosages (``g_<variant>``), indexed by person id, with the manifest of instruments used; None
    and the reason when the genomic files or instruments are missing. Cached in the sealed folder."""
    from codas.core import instruments as inst

    located = located or locate(spec_genetics.get("storage"))
    manifest: dict[str, Any] = {"located": located}
    if not located.get("array_beds") or not located.get("ancestry"):
        manifest["status"] = f"genomic files not found ({located.get('status')})"
        return None, manifest
    plink = Plink.open(located["array_beds"][0], read)
    sets = instrument_sets(spec_genetics, out, log)
    used: dict[str, list[dict[str, Any]]] = {}
    for construct, entry in sets.items():
        aligned, dropped = inst.align(entry["instruments"], plink.variants)
        chosen = inst.prune(aligned, int(spec_genetics.get("window_kb") or 500))
        if len(chosen) >= 2:
            used[construct] = chosen
        manifest.setdefault("constructs", {})[construct] = {"status": entry["status"], "kept": len(chosen),
                                                            "dropped": dropped}
    if not used:
        manifest["status"] = "no construct with two or more usable instruments"
        return None, manifest
    key = hashlib.sha256(json.dumps([located, used], default=str, sort_keys=True).encode()).hexdigest()[:16]
    cached = Path(out) / "sealed" / "genetics" / f"people_{key}.csv.gz"
    if cached.exists():
        table = pd.read_csv(cached, index_col=0)
        table.index = table.index.astype(str)
    else:
        variants = sorted({a["variant"] for chosen in used.values() for a in chosen})
        dosages = plink.read_variants(variants, read).add_prefix("g_")
        pcs = ancestry(located["ancestry"][0], read, int(spec_genetics.get("components") or 16))
        table = pcs.drop(columns=["ancestry_group"], errors="ignore").join(dosages, how="inner")
        for construct, chosen in used.items():
            cols = [f"g_{a['variant']}" for a in chosen]
            table[f"score_{slug(construct)}"] = allele_score(table[cols].to_numpy(), [a["weight"] for a in chosen])
        cached.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cached, compression="gzip")
    manifest.update(status="available", key=key,
                    instruments={c: [{k: a[k] for k in ("variant", "weight", "source")} for a in chosen]
                                 for c, chosen in used.items()})
    wanted = people.astype(str)
    return table.reindex(wanted.to_numpy()), manifest


def genetics_path(discovery_csv: str | Path) -> Path:
    """The sealed share's genetic table (keyed by the sealed table's rows), in the sealed folder."""
    from codas.aou import cohort

    p = Path(discovery_csv)
    return cohort.sealed_dir(p.parent) / (p.stem + ".genetics.holdout.csv.gz")


def write_sealed(spec_genetics: dict[str, Any], sealed_people: pd.Series, discovery_csv: str | Path, out: Path,
                 log: Any = print) -> dict[str, Any]:
    """The sealed share's genetic table and its manifest (``<stem>.instruments.json`` in the sealed folder)."""
    table, manifest = person_table(spec_genetics, sealed_people, out, log)
    path = genetics_path(discovery_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if table is not None:
        table.reset_index(drop=True).rename_axis("row").reset_index().to_csv(path, index=False, compression="gzip")
    path.with_name(Path(discovery_csv).stem + ".instruments.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    log(f"genetics: {manifest.get('status')}")
    return manifest
