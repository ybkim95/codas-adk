"""Where CoDaS can look up published work, and a probe of which sources answer from this machine.

Only query text ever leaves the machine: a short phrase about constructs and outcomes, never data. The
All of Us data use agreement forbids giving any tool access that could export participant-level data, and
the Workbench monitors outbound traffic, so every request is counted (``EGRESS``) and capped
(``CODAS_LITERATURE_MAX_REQUESTS`` per process).

Sources, and why each is or is not used:

* Europe PMC, Crossref, OpenAlex, NCBI E-utilities (PubMed), Semantic Scholar: open bibliographic APIs that
  return identifiers (PMID, DOI), titles, journals, years, publication types and citation counts, and allow
  caching. These are what a citation can point to.
* Nature and the other Springer Nature journals: their papers are indexed by DOI in Crossref, OpenAlex and
  Europe PMC (the probe checks the Nature Portfolio DOI prefix 10.1038 through Crossref). Springer Nature's
  own Metadata API needs a key; it is probed only when SPRINGER_NATURE_API_KEY is set (a workspace secret).
* Google Search grounding (Gemini on Vertex): web context for the agents' reasoning. Its terms forbid
  caching the results and harvesting the links, so its replies are never stored or reused, and its redirect
  links are not citations.
* Google Scholar: not used. It has no API, and its terms forbid automated queries.

What the agents cite comes from Europe PMC (search, with hit counts, publication types and citation counts)
and Crossref (looking a DOI up). Each retrieved paper has a key, ``PMID:<id>`` or ``DOI:<doi>``, that the
agents write as a citation token (``[PMID:...]``); a token that is not among the papers a run retrieved is
not a citation. How established a construct is for an outcome is counted (``evidence``), not judged.
Responses are cached on disk (these APIs allow it), so a repeated query is not sent again.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

# A fixed, generic probe query and a known published DOI: they say nothing about any analysis.
PROBE_QUERY = "wearable"
KNOWN_DOI = "10.1038/s41591-022-02012-w"
MAX_REQUESTS = int(os.getenv("CODAS_LITERATURE_MAX_REQUESTS", "3000"))
TIMEOUT_S = float(os.getenv("CODAS_LITERATURE_TIMEOUT_S", "15"))
_USER_AGENT = "CoDaS-literature/1.0"
# Bytes sent and received by every literature request in this process (request line and headers are
# estimated from the URL and the headers sent).
EGRESS = {"requests": 0, "sent_bytes": 0, "received_bytes": 0}


@dataclass(frozen=True)
class Provider:
    name: str
    probe_url: str
    key_env: str | None = None  # the provider needs this environment variable (an API key)


PROVIDERS = (
    Provider("Europe PMC", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&pageSize=1"),
    Provider("Crossref", "https://api.crossref.org/works?query={q}&rows=1"),
    Provider("Crossref (Nature Portfolio DOIs, 10.1038)",
             "https://api.crossref.org/works?query={q}&filter=prefix:10.1038&rows=1"),
    Provider("OpenAlex", "https://api.openalex.org/works?search={q}&per-page=1"),
    Provider("NCBI E-utilities (PubMed)",
             "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={q}&retmax=1&retmode=json"),
    Provider("Semantic Scholar", "https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields=title"),
    Provider("Springer Nature Metadata API", "https://api.springernature.com/meta/v2/json?q={q}&p=1&api_key={key}",
             key_env="SPRINGER_NATURE_API_KEY"),
    # The query forms the evidence search uses: title-or-abstract fields with a review filter, and looking a
    # published paper up by its DOI (the All of Us step-count paper, Master et al. 2022).
    Provider("Europe PMC, fielded query with a review filter",
             "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%28TITLE%3A%22{q}%22%20OR%20ABSTRACT%3A"
             "%22{q}%22%29%20AND%20PUB_TYPE%3A%22review%22&format=json&pageSize=1"),
    Provider("Europe PMC, lookup by DOI",
             "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI%3A%22" + KNOWN_DOI.replace("/", "%2F")
             + "%22&format=json&pageSize=1"),
    Provider("Crossref, lookup by DOI", "https://api.crossref.org/works/" + KNOWN_DOI),
)
NOT_USED = {"Google Scholar": "not used: no API, and its terms forbid automated queries"}


class EgressLimit(RuntimeError):
    """This process has made as many literature requests as it may."""


def _urlopen(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""


def get(url: str, *, opener: Callable[[str, float], tuple[int, bytes]] | None = None,
        timeout: float = TIMEOUT_S) -> tuple[int, bytes]:
    """One counted GET. Raises EgressLimit past MAX_REQUESTS; network errors propagate."""
    if EGRESS["requests"] >= MAX_REQUESTS:
        raise EgressLimit(f"{MAX_REQUESTS} literature requests already made in this run")
    EGRESS["requests"] += 1
    EGRESS["sent_bytes"] += len(url) + len(_USER_AGENT) + 64
    status, body = (opener or _urlopen)(url, timeout)
    EGRESS["received_bytes"] += len(body)
    return status, body


def probe(*, opener: Callable[[str, float], tuple[int, bytes]] | None = None, grounding: Any = None,
          timeout: float = 10.0) -> list[dict[str, Any]]:
    """Which literature sources answer from here: one tiny query per source.

    ``grounding`` is a callable returning a ``core.gemini.GroundedResponse`` for a query (the notebook
    passes ``gemini.generate_grounded_reply``); without it Google Search grounding is not probed. Returns
    one row per source: whether it answered, the HTTP status, the latency and the bytes received, or why
    it was not tried."""
    rows: list[dict[str, Any]] = []
    for p in PROVIDERS:
        row: dict[str, Any] = {"source": p.name, "reachable": False, "status": None, "latency_ms": None,
                               "received_bytes": None, "note": ""}
        key = os.getenv(p.key_env, "").strip() if p.key_env else ""
        if p.key_env and not key:
            row["note"] = f"needs {p.key_env} (not set); Nature papers are found by DOI through the open indexes"
            rows.append(row)
            continue
        started = time.monotonic()
        try:
            status, body = get(p.probe_url.format(q=PROBE_QUERY, key=key), opener=opener, timeout=timeout)
            row.update(status=status, received_bytes=len(body), reachable=200 <= status < 300,
                       note="" if 200 <= status < 300 else "answered with an error status")
        except EgressLimit as exc:
            row["note"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - a probe reports any failure, it never raises
            row["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        row["latency_ms"] = round((time.monotonic() - started) * 1000)
        rows.append(row)
    if grounding is not None:
        row = {"source": "Google Search grounding (Gemini, Vertex)", "reachable": False, "status": None,
               "latency_ms": None, "received_bytes": None, "note": ""}
        started = time.monotonic()
        try:
            reply = grounding("Cohort studies of wearable step counts and all-cause mortality")
            row["reachable"] = bool(reply.text and reply.sources)
            row["note"] = (f"{len(reply.sources)} web sources via {reply.model}; web context only (its terms "
                           "forbid caching or harvesting the links)" if reply.sources else
                           (reply.error or "answered without sources")[:160])
        except Exception as exc:  # noqa: BLE001
            row["note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        row["latency_ms"] = round((time.monotonic() - started) * 1000)
        rows.append(row)
    rows += [{"source": name, "reachable": None, "status": None, "latency_ms": None, "received_bytes": None,
              "note": why} for name, why in NOT_USED.items()]
    return rows

# --- Papers a citation can point to: Europe PMC search, Crossref lookup, evidence counts, citation tokens ----

# How the evidence on a construct and an outcome is classed, from counts alone.
ESTABLISHED_PAPERS = int(os.getenv("CODAS_NOVELTY_ESTABLISHED_PAPERS", "10"))
ESTABLISHED_REVIEWS = int(os.getenv("CODAS_NOVELTY_ESTABLISHED_REVIEWS", "1"))
NOT_ASSESSED = "not assessed"
_MAX_QUERY = 300
_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_CROSSREF = "https://api.crossref.org/works"
_REVIEW_TYPES = ('"review"', '"systematic review"', '"meta-analysis"')
_RETRY_STATUS = {429, 500, 502, 503, 504}
_PAUSE_S = float(os.getenv("CODAS_LITERATURE_PAUSE_S", "0.2"))  # between requests to one service
_sleep = time.sleep  # tests replace it
# A citation in agent text: [PMID:12345678] or [DOI:10.1038/s41591-022-02012-w]. Any bracketed PMID or DOI
# counts as one, well formed or not, so a made-up key is caught by the store check rather than ignored.
TOKEN = re.compile(r"\[((?:PMID|DOI):[^\]\s]+)\]")
_LONG_DIGITS = re.compile(r"\d{5,}")
_SMALL_COUNT = re.compile(r"\b([1-9]|1\d|20)(\s+)(people|persons|participants|patients|subjects|cases|individuals)\b",
                          re.IGNORECASE)


class Unreachable(RuntimeError):
    """A bibliographic service did not answer usefully (network, HTTP error, or the request cap)."""


@dataclass(frozen=True)
class Paper:
    """One retrieved paper, as a citation can point to it."""

    source: str
    title: str
    pmid: str | None = None
    doi: str | None = None
    journal: str | None = None
    year: int | None = None
    pub_types: tuple[str, ...] = ()
    cited_by: int | None = None
    abstract: str | None = None

    @property
    def key(self) -> str | None:
        """The citation token's content: ``PMID:<id>`` when there is a PubMed id, else ``DOI:<doi>``."""
        return f"PMID:{self.pmid}" if self.pmid else f"DOI:{self.doi}" if self.doi else None

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "source": self.source, "title": self.title, "pmid": self.pmid, "doi": self.doi,
                "journal": self.journal, "year": self.year, "pub_types": list(self.pub_types),
                "cited_by": self.cited_by, "abstract": self.abstract}


def sanitize(query: str) -> str:
    """A query as it may leave the machine: digit runs of five or more removed (an identifier or a count), a
    count of 1-20 next to a word for people masked, whitespace collapsed, at most 300 characters."""
    text = _SMALL_COUNT.sub(r"[1-20]\2\3", _LONG_DIGITS.sub(" ", str(query)))
    return " ".join(text.split())[:_MAX_QUERY]


class Cache:
    """Responses of the open bibliographic APIs on disk, so a repeated query is not sent again (these APIs
    allow it; Google Search grounding replies are never cached)."""

    def __init__(self, path: Any = None):
        from pathlib import Path

        self.path = Path(path) if path else None
        self._data: dict[str, Any] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    self._data[entry["url"]] = entry["body"]
                except (ValueError, KeyError, TypeError):
                    continue

    def get(self, url: str) -> Any:
        return self._data.get(url)

    def put(self, url: str, body: Any) -> None:
        self._data[url] = body
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"url": url, "body": body}) + "\n")


def _get_json(url: str, cache: Cache | None = None, opener: Any = None, retries: int = 2) -> Any:
    """A counted GET of a JSON body, from the cache when it holds the URL; retries on 429 and 5xx."""
    cached = cache.get(url) if cache else None
    if cached is not None:
        return cached
    for attempt in range(retries + 1):
        try:
            status, body = get(url, opener=opener)
        except EgressLimit as exc:
            raise Unreachable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - network errors: the service is unreachable from here
            raise Unreachable(f"{type(exc).__name__}: {str(exc)[:120]}") from exc
        if 200 <= status < 300:
            break
        if status not in _RETRY_STATUS or attempt == retries:
            raise Unreachable(f"HTTP {status}")
        _sleep(2.0 ** attempt)
    try:
        data = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise Unreachable("the answer was not JSON") from exc
    if cache:
        cache.put(url, data)
    _sleep(_PAUSE_S)
    return data


def _year(value: Any) -> int | None:
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


def _europe_pmc_paper(r: dict[str, Any]) -> Paper:
    types = (r.get("pubTypeList") or {}).get("pubType") if isinstance(r.get("pubTypeList"), dict) else None
    if isinstance(types, str):
        types = [types]
    if not types and r.get("pubType"):
        types = [t.strip() for t in str(r["pubType"]).split(";") if t.strip()]
    journal = r.get("journalTitle") or ((r.get("journalInfo") or {}).get("journal") or {}).get("title")
    abstract = r.get("abstractText")
    cited = str(r.get("citedByCount", ""))
    return Paper(source="Europe PMC", title=re.sub(r"<[^>]+>", "", str(r.get("title") or "")).strip(),
                 pmid=str(r["pmid"]) if r.get("pmid") else None, doi=r.get("doi") or None, journal=journal,
                 year=_year(r.get("pubYear")), pub_types=tuple(types or ()),
                 cited_by=int(cited) if cited.isdigit() else None,
                 abstract=re.sub(r"<[^>]+>", "", abstract)[:600] if abstract else None)


def europe_pmc(query: str, *, page_size: int = 10, sort: str | None = None, abstracts: bool = False,
               cache: Cache | None = None, opener: Any = None) -> tuple[int, list[Paper]]:
    """(hit count, papers) for a Europe PMC query in its own syntax (TITLE:, ABSTRACT:, PUB_TYPE:, DOI:)."""
    params = {"query": query, "format": "json", "pageSize": str(page_size),
              "resultType": "core" if abstracts else "lite"}
    if sort:
        params["sort"] = sort
    data = _get_json(f"{_EUROPE_PMC}?{urllib.parse.urlencode(params)}", cache, opener)
    results = (data.get("resultList") or {}).get("result") or []
    return int(data.get("hitCount") or 0), [_europe_pmc_paper(r) for r in results]


def _crossref_paper(item: dict[str, Any]) -> Paper:
    parts = ((item.get("published") or item.get("published-print") or item.get("published-online")
              or item.get("issued") or {}).get("date-parts") or [[None]])[0]
    return Paper(source="Crossref", title=" ".join(item.get("title") or []).strip(), doi=item.get("DOI"),
                 journal=" ".join(item.get("container-title") or []) or None, year=_year(parts[0] if parts else None),
                 pub_types=(item["type"],) if item.get("type") else (), cited_by=item.get("is-referenced-by-count"))


def crossref(query: str, *, rows: int = 5, cache: Cache | None = None, opener: Any = None) -> tuple[int, list[Paper]]:
    """(total, papers) for a free-text bibliographic query in Crossref (title, authors, journal, year)."""
    params = {"query.bibliographic": query, "rows": str(rows)}
    data = _get_json(f"{_CROSSREF}?{urllib.parse.urlencode(params)}", cache, opener)
    message = data.get("message") or {}
    return int(message.get("total-results") or 0), [_crossref_paper(i) for i in message.get("items") or []]


def lookup(key: str, *, cache: Cache | None = None, opener: Any = None) -> Paper | None:
    """The paper behind ``PMID:<id>`` (Europe PMC) or ``DOI:<doi>`` (Crossref), or None when it is not
    registered. Raises Unreachable when the service does not answer."""
    kind, _, value = key.partition(":")
    if kind == "PMID":
        _total, papers = europe_pmc(f"EXT_ID:{value} AND SRC:MED", page_size=1, cache=cache, opener=opener)
        return papers[0] if papers else None
    try:
        data = _get_json(f"{_CROSSREF}/{urllib.parse.quote(value.strip(), safe='/')}", cache, opener, retries=1)
    except Unreachable as exc:
        if str(exc) == "HTTP 404":
            return None
        raise
    return _crossref_paper(data.get("message") or {})


def _any_of(terms: list[str]) -> str:
    quoted = [t.replace('"', "").strip() for t in terms if t and t.strip()]
    return "(" + " OR ".join(f'TITLE:"{t}" OR ABSTRACT:"{t}"' for t in quoted) + ")"


def evidence(construct: str, synonyms: list[str], outcome_terms: list[str], *, cache: Cache | None = None,
             opener: Any = None) -> dict[str, Any]:
    """How much has been published on a construct and an outcome together, counted in Europe PMC: papers
    naming both in the title or abstract, how many are reviews or meta-analyses, the earliest year, and the
    most cited. Novelty is classed from these counts alone: ``established`` (at least ESTABLISHED_REVIEWS
    reviews or ESTABLISHED_PAPERS papers), ``emerging`` (fewer), ``unexplored`` (none); ``not assessed``
    when the service does not answer, never a guess. Only the terms leave the machine."""
    terms = list(dict.fromkeys(sanitize(t) for t in [construct, *synonyms] if t))
    outcome = list(dict.fromkeys(sanitize(t) for t in outcome_terms if t))
    query = f"{_any_of(terms)} AND {_any_of(outcome)}"
    out: dict[str, Any] = {"construct": construct, "terms": terms, "outcome_terms": outcome, "query": query}
    try:
        total, top = europe_pmc(query, page_size=3, sort="CITED desc", cache=cache, opener=opener)
        reviews = 0
        first: list[Paper] = []
        if total:
            reviews, _ = europe_pmc(f"{query} AND ({' OR '.join('PUB_TYPE:' + t for t in _REVIEW_TYPES)})",
                                    page_size=1, cache=cache, opener=opener)
            _, first = europe_pmc(query, page_size=1, sort="P_PDATE_D asc", cache=cache, opener=opener)
    except Unreachable as exc:
        return {**out, "novelty": NOT_ASSESSED, "error": str(exc)}
    novelty = ("established" if reviews >= ESTABLISHED_REVIEWS or total >= ESTABLISHED_PAPERS
               else "emerging" if total else "unexplored")
    return {**out, "papers": total, "reviews": reviews, "earliest_year": first[0].year if first else None,
            "top_cited": [p.as_dict() for p in top if p.key], "novelty": novelty}


def tokens(text: str) -> list[str]:
    """The citation tokens in ``text`` (``PMID:...`` and ``DOI:...``), in order, without repeats."""
    return list(dict.fromkeys(m.group(1).rstrip(".,;") for m in TOKEN.finditer(str(text or ""))))


def references_markdown(keys: list[str], store: dict[str, dict[str, Any]]) -> str:
    """A numbered reference list for ``keys`` from the run's literature store (unknown keys are skipped)."""
    lines = []
    for key in keys:
        paper = store.get(key)
        if not paper:
            continue
        where = ", ".join(str(x) for x in (paper.get("journal"), paper.get("year")) if x)
        link = f"https://doi.org/{paper['doi']}" if paper.get("doi") else f"https://europepmc.org/article/MED/{paper['pmid']}"
        lines.append(f"{len(lines) + 1}. {paper.get('title') or key}. {where}. [{key}]({link})")
    return "\n".join(lines)


def load_terms(path: Any) -> dict[str, Any]:
    """The construct-synonym catalog (``constructs``: name -> synonyms, column patterns, phenotype names by
    source kind; ``outcomes``: name -> search terms)."""
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def constructs_of(columns: list[str], terms: dict[str, Any], kinds: dict[str, str] | None = None) -> dict[str, str]:
    """Each column -> the first construct of the catalog that names it: an intraday phenotype
    ``<prefix>_<name>`` by its source kind and name (``kinds``: prefix -> kind), any column by a shell-style
    pattern of names. Columns no construct names are left out."""
    import fnmatch

    out: dict[str, str] = {}
    for column in columns:
        phenotype = next(((kinds[p], column[len(p) + 1:]) for p in sorted(kinds or {}, key=len, reverse=True)
                          if column.startswith(p + "_")), None)
        for construct, entry in (terms.get("constructs") or {}).items():
            by_kind = (entry.get("phenotypes") or {}).get(phenotype[0], []) if phenotype else []
            if (phenotype and phenotype[1] in by_kind) or any(
                    fnmatch.fnmatchcase(column, pattern) for pattern in entry.get("columns") or []):
                out[column] = construct
                break
    return out


def evidence_matrix(constructs: list[str], outcome_terms: list[str], terms: dict[str, Any], *,
                    cache: Cache | None = None, opener: Any = None) -> dict[str, dict[str, Any]]:
    """``evidence`` for each construct with the outcome. When the first construct cannot be assessed (the
    service is unreachable) the others are not tried: each is ``not assessed`` with the same reason."""
    out: dict[str, dict[str, Any]] = {}
    for construct in constructs:
        entry = (terms.get("constructs") or {}).get(construct) or {}
        result = evidence(construct, entry.get("synonyms") or [], outcome_terms, cache=cache, opener=opener)
        out[construct] = result
        if result["novelty"] == NOT_ASSESSED and len(out) == 1:
            for rest in constructs[1:]:
                out[rest] = {"construct": rest, "novelty": NOT_ASSESSED, "error": result.get("error")}
            break
    return out


def evidence_digest(matrix: dict[str, dict[str, Any]], columns_of: dict[str, list[str]], limit: int = 2) -> str:
    """The evidence matrix as text for the agents: per construct its columns, novelty class, counts and the
    most cited papers with their citation keys."""
    lines = []
    for construct, e in matrix.items():
        cols = ", ".join(columns_of.get(construct, [])[:6]) or "no column"
        if e.get("novelty") == NOT_ASSESSED:
            lines.append(f"- {construct} ({cols}): not assessed ({e.get('error') or 'no answer'})")
            continue
        top = "; ".join(f"[{p['key']}] {str(p.get('title') or '')[:100]} ({p.get('journal')}, {p.get('year')}; "
                        f"cited {p.get('cited_by')})" for p in (e.get("top_cited") or [])[:limit])
        lines.append(f"- {construct} ({cols}): {e['novelty']}: {e.get('papers')} papers with the outcome, "
                     f"{e.get('reviews')} reviews or meta-analyses, earliest {e.get('earliest_year')}"
                     + (f". Most cited: {top}" if top else ""))
    return "\n".join(lines)


# --- checking a bibliography written from abstracts --------------------------------------------------------

_DOI_IN_TEXT = re.compile(r"\b(10\.\d{4,9}/[^\s)\]|>\"'<]+)", re.IGNORECASE)


def references_in(text: str) -> list[dict[str, str]]:
    """Every distinct DOI in a Markdown text, with the line it appears on (what the citation says)."""
    found: dict[str, str] = {}
    for line in str(text).splitlines():
        for match in _DOI_IN_TEXT.finditer(line):
            found.setdefault(match.group(1).rstrip(".,;:").lower(), line.strip())
    return [{"doi": doi, "context": context} for doi, context in found.items()]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", re.sub(r"<[^>]+>", " ", str(text or "")).lower()))


def verify_references(items: list[dict[str, str]], *, cache: Cache | None = None,
                      opener: Any = None) -> list[dict[str, Any]]:
    """Look each DOI up in Crossref: whether it is registered, its title, journal and year, and how much of
    that title the citation's own line contains (a low share means the DOI may belong to another paper).
    After the first unanswered lookup the rest are reported as not checked."""
    rows: list[dict[str, Any]] = []
    reachable = True
    for item in items:
        row: dict[str, Any] = {"doi": item["doi"], "cited_as": item["context"][:160]}
        if not reachable:
            rows.append({**row, "status": "not checked"})
            continue
        try:
            paper = lookup(f"DOI:{item['doi']}", cache=cache, opener=opener)
        except Unreachable as exc:
            reachable = False
            rows.append({**row, "status": "not checked", "note": str(exc)})
            continue
        if paper is None:
            rows.append({**row, "status": "not found"})
            continue
        title = _words(paper.title)
        share = round(len(title & _words(item["context"])) / len(title), 2) if title else None
        rows.append({**row, "status": "found", "title": paper.title, "journal": paper.journal, "year": paper.year,
                     "title_in_citation": share})
    return rows
