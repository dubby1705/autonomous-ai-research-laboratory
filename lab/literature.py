"""AARL Literature search — arXiv + Crossref (polite, rate-limited).

Uses only public, documented metadata APIs and stays well inside their
fair-use limits:

* arXiv API      (http://export.arxiv.org/api/query)   ~1 request / 3 s
* Crossref API   (https://api.crossref.org/works)      polite pool with mailto

Only metadata + abstract snippets are returned (no full text is copied);
each result carries a link so the GUI can send the reader to the legitimate
source.
"""
from __future__ import annotations

import re
import time
import threading
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from urllib.parse import quote

import requests

_LOCK = threading.Lock()
_LAST_ARXIV = 0.0
ARXIV_MIN_INTERVAL = 3.0          # arXiv asks for >= 3 s between calls
CROSSREF_MIN_INTERVAL = 1.0
_TIMEOUT = 15

_HEADERS = {"User-Agent": "AARL/1.0 (research exploration tool; mailto:aarl@example.org)"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _wait(target: float) -> None:
    global _LAST_ARXIV
    with _LOCK:
        now = time.time()
        delta = target - (now - _LAST_ARXIV)
        if delta > 0:
            time.sleep(delta)
        _LAST_ARXIV = time.time()


def _arxiv(query: str, limit: int) -> List[Dict[str, Any]]:
    _wait(ARXIV_MIN_INTERVAL)
    url = (
        "http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=%d"
        % (quote("all:" + query), min(max(limit, 1), 20))
    )
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.content)
    out: List[Dict[str, Any]] = []
    for entry in root.findall("a:entry", ns):
        title = _clean(entry.findtext("a:title", "", ns))
        if not title:
            continue
        authors = [_clean(a.findtext("a:name", "", ns)) for a in entry.findall("a:author", ns)]
        abstract = _clean(entry.findtext("a:summary", "", ns))[:900]
        link = _clean(entry.findtext("a:id", "", ns))
        published = _clean(entry.findtext("a:published", "", ns))[:10]
        out.append({
            "source": "arXiv",
            "title": title,
            "authors": [a for a in authors if a],
            "year": published[:4] if published else "",
            "abstract": abstract,
            "url": link,
            "identifier": link.rsplit("/abs/", 1)[-1],
            "license_note": "Metadata + abstract via the arXiv API; full text at the link.",
        })
    return out


def _crossref(query: str, limit: int) -> List[Dict[str, Any]]:
    _wait(CROSSREF_MIN_INTERVAL)
    url = (
        "https://api.crossref.org/works?query=%s&rows=%d&select=title,author,issued,abstract,DOI,URL,container-title"
        % (quote(query), min(max(limit, 1), 20))
    )
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    items = (resp.json() or {}).get("message", {}).get("items", [])
    out: List[Dict[str, Any]] = []
    for item in items:
        title = _clean(" ".join(item.get("title") or []))
        if not title:
            continue
        authors = [
            _clean(" ".join(x for x in (a.get("given", ""), a.get("family", "")) if x))
            for a in (item.get("author") or [])
        ]
        year = ""
        issued = (item.get("issued") or {}).get("date-parts") or []
        if issued and issued[0]:
            year = str(issued[0][0])
        abstract = _clean(re.sub(r"<[^>]+>", " ", item.get("abstract") or ""))[:900]
        doi = _clean(item.get("DOI", ""))
        out.append({
            "source": "Crossref",
            "title": title,
            "authors": [a for a in authors if a],
            "year": year,
            "abstract": abstract,
            "url": _clean(item.get("URL", "")) or ("https://doi.org/%s" % doi if doi else ""),
            "identifier": doi,
            "container": _clean(" ".join(item.get("container-title") or [])),
            "license_note": "Metadata via the Crossref REST API; access the publisher link for full text.",
        })
    return out


def search_literature(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Search both sources; failures of one source never block the other."""
    limit = max(1, min(int(limit or 8), 20))
    results: List[Dict[str, Any]] = []
    for fetch in (_arxiv, _crossref):
        try:
            results.extend(fetch(query, limit))
        except Exception:
            continue
    # De-duplicate on normalised title.
    seen = set()
    unique = []
    for r in results:
        key = re.sub(r"[^a-z0-9 ]", "", r.get("title", "").lower())[:120]
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique[: limit * 2]
