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
"""

from __future__ import annotations

import os
import time
import urllib.error
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
