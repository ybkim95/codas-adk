"""Genetic triangulation: PLINK genotypes read by byte range, ancestry components, the files located by name,
instruments parsed, aligned and pruned, Mendelian randomization recovering a planted causal effect where the
observational estimate is confounded, and the holdout's use of it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from codas.aou import genomics
from codas.core import instruments as inst
from codas.core import mendelian


def write_plink(stem: Path, dosages: np.ndarray, ids: list[str], people: list[str], a1: list[str], a2: list[str],
                chrom: list[str] | None = None, pos: list[int] | None = None) -> Path:
    """A variant-major PLINK 1 fileset: dosages are counts of a1 (NaN missing), one row per person."""
    n, k = dosages.shape
    code = {2.0: 0, 1.0: 2, 0.0: 3}
    out = bytearray(b"\x6c\x1b\x01")
    for j in range(k):
        bits = [1 if not np.isfinite(v) else code[float(v)] for v in dosages[:, j]]
        bits += [0] * (-n % 4)
        for b in range(0, len(bits), 4):
            out.append(bits[b] | bits[b + 1] << 2 | bits[b + 2] << 4 | bits[b + 3] << 6)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".bed").write_bytes(bytes(out))
    chrom = chrom or ["1"] * k
    pos = pos or [1_000_000 * (j + 1) for j in range(k)]
    stem.with_suffix(".bim").write_text("".join(f"{c}\t{i}\t0\t{p}\t{x}\t{y}\n"
                                                for c, i, p, x, y in zip(chrom, ids, pos, a1, a2)))
    stem.with_suffix(".fam").write_text("".join(f"{p}\t{p}\t0\t0\t0\t-9\n" for p in people))
    return stem.with_suffix(".bed")


def test_genotypes_are_read_by_byte_range(tmp_path):
    rng = np.random.default_rng(0)
    d = rng.integers(0, 3, (7, 5)).astype(float)
    d[2, 1] = np.nan
    bed = write_plink(tmp_path / "arrays", d, [f"rs{j}" for j in range(5)], [f"p{i}" for i in range(7)],
                      ["A"] * 5, ["G"] * 5)
    plink = genomics.Plink.open(str(bed))
    got = plink.read_variants(["rs3", "rs1", "rs9"])
    assert list(got.columns) == ["rs3", "rs1"] and list(got.index) == [f"p{i}" for i in range(7)]
    assert np.array_equal(got["rs3"].to_numpy(), d[:, 3])
    assert np.isnan(got.loc["p2", "rs1"]) and np.array_equal(np.delete(got["rs1"].to_numpy(), 2), np.delete(d[:, 1], 2))


def test_ancestry_components_and_locating_the_files(tmp_path):
    root = tmp_path / "cdr"
    write_plink(root / "microarray" / "plink" / "arrays", np.zeros((2, 1)), ["rs1"], ["1", "2"], ["A"], ["G"])
    anc = root / "wgs" / "aux" / "ancestry" / "ancestry_preds.tsv"
    anc.parent.mkdir(parents=True)
    anc.write_text("research_id\tancestry_pred\tpca_features\n1\teur\t[0.1, -0.2, 0.3]\n2\tafr\t[0.4, 0.5, -0.6]\n")
    (root / "cram" / "huge").mkdir(parents=True)            # never descended into: not a genomic-data name
    found = genomics.locate(str(root))
    assert found["status"] == "found" and found["array_beds"][0].endswith("microarray/plink/arrays.bed")
    assert found["ancestry"][0].endswith("ancestry_preds.tsv")
    pcs = genomics.ancestry(found["ancestry"][0], components=2)
    assert list(pcs.columns) == ["pc1", "pc2", "ancestry_group"] and pcs.loc["2", "pc2"] == 0.5
    assert genomics.locate("")["root"] is None


def test_instruments_are_parsed_aligned_and_pruned():
    payload = {"_embedded": {"associations": [   # the GWAS Catalog REST interface v2's records
        {"p_value": 3e-20, "beta": "0.4 unit increase", "snp_allele": [{"rs_id": "rs1", "effect_allele": "A"}],
         "pubmed_id": "111"},
        {"pvalue_mantissa": 2, "pvalue_exponent": -9, "beta": "0.2 unit decrease",
         "snp_allele": [{"rs_id": "rs2", "effect_allele": "C"}]},
        {"p_value": 1e-6, "beta": "0.9 unit increase",                                   # not genome-wide
         "snp_allele": [{"rs_id": "rs3", "effect_allele": "A"}]},
        {"p_value": 1e-30, "beta": "-", "or_per_copy_number": 1.5,
         "snp_allele": [{"rs_id": "rs4", "effect_allele": "T"}]},
        {"p_value": 1e-30, "beta": "0.3 unit increase",
         "snp_allele": [{"rs_id": "rs5", "effect_allele": "?"}]},                         # no allele
        {"p_value": 1e-30, "beta": "-", "snp_allele": [{"rs_id": "rs7", "effect_allele": "G"}]},   # no effect
        {"p_value": 1e-30, "beta": "0.1 unit increase",                                  # a haplotype
         "snp_allele": [{"rs_id": "rs8", "effect_allele": "G"}, {"rs_id": "rs9", "effect_allele": "A"}]}]}}
    found = inst.parse_associations("c", payload)
    by = {i.variant: i for i in found}
    assert set(by) == {"rs1", "rs2", "rs4"} and by["rs2"].beta == -0.2 and abs(by["rs4"].beta - np.log(1.5)) < 1e-12
    assert by["rs1"].source == "PMID:111" and abs(by["rs1"].p - 3e-20) < 1e-30
    legacy = {"_embedded": {"associations": [{"pvalueMantissa": 3, "pvalueExponent": -20, "betaNum": 0.4,
                                              "betaDirection": "decrease",
                                              "loci": [{"strongestRiskAlleles": [{"riskAlleleName": "rs1-A"}]}]}]}}
    assert [(i.variant, i.beta) for i in inst.parse_associations("c", legacy)] == [("rs1", -0.4)]
    bim = pd.DataFrame({"chrom": ["1", "1", "2", "3"], "id": ["rs1", "rs2", "rs4", "rs6"], "cm": 0,
                        "pos": [100_000, 300_000, 5_000, 1], "a1": ["G", "C", "A", "A"], "a2": ["A", "T", "T", "G"]})
    aligned, dropped = inst.align(found + [inst.Instrument("c", "rs6", "C", "T", 0.1)], bim)
    weights = {a["variant"]: a["weight"] for a in aligned}
    assert weights == {"rs1": -0.4, "rs2": -0.2}       # rs1's effect allele is the file's a2: the sign flips
    assert dropped == {"rs4": "strand-ambiguous allele pair", "rs6": "alleles do not match the genotype file"}
    assert [a["variant"] for a in inst.prune(aligned, window_kb=500)] == ["rs1"]   # rs2 is 200 kb away, weaker
    assert len(inst.prune(aligned, window_kb=100)) == 2


def test_the_gwas_catalog_is_read_page_by_page_through_its_v2_interface():
    """By trait name first, else by the ontology terms the name resolves to; most significant first, paging
    until the associations stop being genome-wide significant."""
    import urllib.parse

    def record(rs, p):
        return {"p_value": p, "beta": "0.1 unit increase", "snp_allele": [{"rs_id": rs, "effect_allele": "A"}]}

    asked = []

    def opener(url, timeout):
        asked.append(url)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        assert "/rest/api/v2/" in url and timeout >= 60
        if url.split("?")[0].endswith("/efo-traits"):
            body = {"_embedded": {"efo_traits": [{"efo_id": "OBA_1"}]}}
        elif query.get("efo_trait"):
            body = {"page": {"totalElements": 0}}
        elif query.get("page") == "1":
            body = {"_embedded": {"associations": [record("rs3", 1e-9), record("rs4", 1e-6)]},
                    "page": {"totalElements": 4}, "_links": {"next": {"href": url.replace("page=1", "page=2")}}}
        else:
            assert query["efo_id"] == "OBA_1" and query["sort"] == "p_value" and query["direction"] == "asc"
            body = {"_embedded": {"associations": [record("rs1", 1e-30), record("rs2", 1e-12)]},
                    "page": {"totalElements": 4}, "_links": {"next": {"href": url + "&page=1"}}}
        return 200, json.dumps(body).encode()

    got, status = inst.gwas_catalog("c", "some trait", opener=opener)
    assert sorted(i.variant for i in got) == ["rs1", "rs2", "rs3"] and status.startswith("3 genome-wide")
    assert not any("page=2" in u for u in asked)          # the page ending above 5e-8 was the last read

    gone = lambda url, timeout: (410, b"This legacy GWAS Catalog REST API is no longer available.")  # noqa: E731
    assert inst.gwas_catalog("c", "x", opener=gone) == ([], "GWAS Catalog not reachable (HTTP 410)")


def _genetic_cohort(seed=1, n=12000, k=10, causal=0.3):
    """Variants raise the exposure; a confounder raises both; the exposure raises the outcome hazard by
    ``causal`` per SD."""
    rng = np.random.default_rng(seed)
    g = rng.binomial(2, rng.uniform(0.2, 0.5, k), (n, k)).astype(float)
    gamma = rng.uniform(0.08, 0.15, k)
    u = rng.normal(size=n)
    x = g @ gamma + u + rng.normal(size=n)
    x = (x - x.mean()) / x.std()
    t = rng.exponential(1 / (0.05 * np.exp(causal * x + 0.8 * u)))
    c = rng.uniform(2, 8, n)
    return g, gamma, x, np.minimum(t, c) * 365, (t <= c).astype(float)


def test_mendelian_randomization_sees_through_a_confounder():
    g, gamma, x, time, event = _genetic_cohort()
    from codas.core.survival import cox_ph

    observed = cox_ph(time, event, x[:, None]).coef[0]
    whole = mendelian.one_sample(x, event, mendelian.allele_score(g, gamma), None, time)
    each = mendelian.per_variant(x, event, g, None, time)
    assert observed > 0.6                              # confounded
    assert whole["f_statistic"] > 50 and whole["ci95"][0] < 0.3 < whole["ci95"][1] and whole["ratio"] < 0.6
    assert each["variants"] == 10 and each["ivw"]["ci95"][0] < 0.3 < each["ivw"]["ci95"][1]
    assert "egger" in each and "weighted_median" in each


def test_the_holdout_triangulates_a_registered_feature(tmp_path):
    from codas.aou import scorecard
    from codas.aou.confirm import confirm, confirmation_markdown, preregister

    shares = []
    for seed in (2, 3):
        g, gamma, x, time, event = _genetic_cohort(seed)
        shares.append((g, gamma, pd.DataFrame({"hr_rest_mean": 60 + 8 * x, "y": event, "followup_days": time,
                                               "age": 55.0, "female": np.arange(len(x)) % 2})))
    shares[0][2].to_csv(tmp_path / "s.csv", index=False)
    shares[1][2].to_csv(tmp_path / "s.holdout.csv", index=False)
    g, gamma, _ = shares[1]
    genes = pd.DataFrame(g, columns=[f"g_rs{j}" for j in range(g.shape[1])])
    genes["score_resting_heart_rate"] = mendelian.allele_score(g, gamma)
    genes["pc1"] = 0.0
    genes.rename_axis("row").reset_index().to_csv(tmp_path / "genes.csv.gz", index=False)
    manifest = {"instruments": {"resting heart rate": [{"variant": f"rs{j}", "weight": float(w)}
                                                      for j, w in enumerate(gamma)]}}
    state = {"target_column": "y", "confounder_columns": ["age"], "excluded_columns": [],
             "latest_report": {"candidates": [{"feature": "hr_rest_mean", "rho": 0.2, "q_value": 1e-9,
                                               "verdict": "validated", "components": ["hr_rest_mean"]}]}}
    prereg = preregister(state, tmp_path, "s", survival_time="followup_days", discovery_csv=tmp_path / "s.csv")
    result = confirm(prereg, tmp_path / "s.holdout.csv", tmp_path,
                     genetics={"table": tmp_path / "genes.csv.gz", "manifest": manifest,
                               "constructs": {"hr_rest_mean": "resting heart rate"}})
    (row,) = result["hypotheses"]
    gen = row["genetic"]
    assert gen["measure"] == "hazard ratio per SD" and gen["variants"] == 10 and gen["same_direction"]
    assert gen["ci95"][0] < np.exp(0.3) < gen["ci95"][1] and row["hr_per_sd"] > gen["estimate"]
    assert "Genetic triangulation" in confirmation_markdown(result)
    cells = scorecard.coverage(result, True)[0]["cells"]
    assert cells["genetic"][0] == "yes" and "F" in cells["genetic"][1]


def test_without_genomic_files_the_extraction_says_why(tmp_path, monkeypatch):
    monkeypatch.delenv("CDR_STORAGE_PATH", raising=False)
    manifest = genomics.write_sealed({"instruments": "gwas_catalog"}, pd.Series(["1", "2"]), tmp_path / "s.csv",
                                     tmp_path, log=lambda *_: None)
    assert manifest["status"].startswith("genomic files not found")
    assert not genomics.genetics_path(tmp_path / "s.csv").exists()
    saved = json.loads((tmp_path / "sealed" / "s.instruments.json").read_text())
    assert saved["status"] == manifest["status"]
