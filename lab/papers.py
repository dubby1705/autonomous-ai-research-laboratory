"""
AARL Lab — Researcher Paper Ingestion
=====================================
Parses a researcher-provided paper (PDF, plain text, or structured JSON data)
and integrates its useful information into the existing Knowledge JSON.

Integration rules
-----------------
* **Never overwrites** existing knowledge — new items are appended with the
  paper's provenance.
* If the paper *contradicts* an existing sourced claim, BOTH are preserved and
  an explicit contradiction relationship is recorded
  (``KnowledgeManager.record_contradiction``).
* Extracted claims are labelled ``claim_type='sourced'`` (from a paper) only
  when they come from the paper text itself; anything the extractor infers is
  labelled ``inferred``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from lab.knowledge import KnowledgeItem

log = logging.getLogger("aarl.lab.papers")

CLAIM_SOURCED = "sourced"
CLAIM_INFERRED = "inferred"

#: Provenance labels for the extraction method that ACTUALLY ran.
METHOD_LLM = "llm"
METHOD_DETERMINISTIC = "deterministic_section_parsing"
METHOD_STRUCTURED = "structured_passthrough"

# Common section headings in research papers (deterministic extraction).
_SECTIONS = {
    "title": None,  # handled separately
    "abstract": "abstract",
    "introduction": "research_problem",
    "background": "findings",
    "related work": "findings",
    "method": "methodology",
    "methodology": "methodology",
    "approach": "methodology",
    "materials and methods": "methodology",
    "experimental setup": "experimental_setup",
    "experiment": "results",
    "results": "results",
    "evaluation": "results",
    "discussion": "conclusions",
    "equations": "equations",
    "limitations": "limitations",
    "failure": "failures",
    "threats to validity": "limitations",
    "conclusion": "conclusions",
    "conclusions": "conclusions",
    "references": "references",
}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class PaperIngestor:
    def __init__(self, memory, knowledge, llm_service=None):
        self.memory = memory
        self.knowledge = knowledge
        self.llm = llm_service

    # ------------------------------------------------------------------ input
    def ingest(
        self,
        path: str = "",
        text: str = "",
        structured: Optional[Dict[str, Any]] = None,
        provided_by: str = "researcher",
        source_url: str = "",
    ) -> Dict[str, Any]:
        """Ingest one researcher paper. Exactly one of path/text/structured.

        ``source_url`` records where the material legitimately lives (e.g. the
        arXiv/Crossref landing page of a fetched record) so the reader can send the
        user to the source instead of reproducing copyrighted full text.
        """
        if structured:
            body, fmt = dict(structured), "structured"
        elif text:
            body, fmt = {"text": text}, "text"
        elif path:
            body, fmt = self._read_file(path), self._fmt(path)
        else:
            raise ValueError("Provide a path, text, or structured data")

        if fmt == "structured":
            extracted = self._normalise_structured(body)
            extraction_method = METHOD_STRUCTURED
        else:
            raw = str(body.get("text", "") or "")
            extracted, extraction_method = self._extract(raw)
            if not extracted.get("title"):
                extracted["title"] = os.path.splitext(os.path.basename(path or "untitled"))[0]

        return self._integrate(
            extracted, fmt, path, provided_by,
            extraction_method=extraction_method, source_url=source_url,
        )

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _fmt(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return "pdf"
        if ext in (".json",):
            return "structured"
        return "text"

    @staticmethod
    def _read_file(path: str) -> Dict[str, Any]:
        fmt = PaperIngestor._fmt(path)
        if fmt == "pdf":
            return {"text": PaperIngestor._read_pdf(path)}
        if fmt == "structured":
            with open(path, "r", encoding="utf-8") as handle:
                return {"structured": json.load(handle)}
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return {"text": handle.read()}

    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            from pypdf import PdfReader
        except Exception as exc:  # noqa: BLE001 - PDF support is optional
            raise RuntimeError(
                f"PDF parsing requires the 'pypdf' package ({exc}); "
                "provide the paper as plain text or structured JSON instead."
            )
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - skip unreadable pages
                pages.append("")
        return "\n".join(pages)

    # ------------------------------------------------------- extraction
    def _extract_deterministic(self, raw: str) -> Dict[str, Any]:
        """Section-aware, purely text-derived extraction (no invented content)."""
        text = raw.replace("\r\n", "\n")
        extracted: Dict[str, Any] = {
            "title": "", "authors": [], "year": None, "domain": "",
            "research_problem": "", "hypothesis": "", "methodology": [],
            "important_methods": [], "equations": [], "assumptions": [],
            "datasets": [], "experimental_setup": "", "results": [],
            "limitations": [], "failures": [], "conclusions": [],
            "supporting_evidence": [], "contradictory_evidence": [],
            "references": [], "relationships_to_other_research": [],
        }
        # ---- title / authors / year from the header block -----------------
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines:
            extracted["title"] = lines[0][:200]
        header = "\n".join(lines[:8])
        author_match = re.search(
            r"((?:[A-Z][A-Za-z.\-]+(?: [A-Z][A-Za-z.\-]+)+[,;] *){2,})", header
        )
        if author_match:
            extracted["authors"] = [a.strip(" ,;") for a in author_match.group(1).split(",")][:8]
        year_match = re.search(r"\b(19|20)\d{2}\b", header)
        if year_match:
            extracted["year"] = int(year_match.group(0))

        # ---- section-aware splitting --------------------------------------
        current, buckets = "header", {"header": []}
        for ln in lines[1:]:
            low = ln.lower().rstrip(":").strip()
            matched = None
            for heading, field in _SECTIONS.items():
                if heading and (low == heading or low.startswith(heading)):
                    matched = field
                    break
            if matched:
                current = matched
                buckets.setdefault(current, [])
            else:
                buckets.setdefault(current, []).append(ln)

        def paragraphs(field: str) -> List[str]:
            body = "\n".join(buckets.get(field, []))
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            return [p[:1200] for p in paras[:6]]

        extracted["abstract"] = (paragraphs("abstract") or [""])[0]
        extracted["research_problem"] = (paragraphs("research_problem") or [""])[0]
        extracted["methodology"] = paragraphs("methodology")
        extracted["experimental_setup"] = (paragraphs("experimental_setup") or [""])[0]
        extracted["results"] = paragraphs("results")
        extracted["limitations"] = paragraphs("limitations")
        extracted["failures"] = paragraphs("failures")
        extracted["conclusions"] = paragraphs("conclusions")
        refs = paragraphs("references")
        extracted["references"] = [
            r for p in refs for r in re.split(r"\[\d+\]|\n", p) if r.strip()
        ][:30]
        # equations: any line with an = sign inside methodology/results
        for field in ("methodology", "results"):
            for p in paragraphs(field):
                for ln in p.split("\n"):
                    if "=" in ln and any(c.isdigit() for c in ln):
                        extracted["equations"].append(ln.strip()[:200])
        extracted["equations"] = extracted["equations"][:10]
        return extracted

    # ------------------------------------------------------- extraction
    def _llm_usable(self) -> bool:
        """True only when an LLM service is present *and* configured.

        ``make_llm_service`` returns a service object even when no API key is
        configured (it exposes ``available=False``), so presence alone is not
        enough to conclude that an LLM call will succeed.
        """
        if self.llm is None:
            return False
        return bool(getattr(self.llm, "available", True))

    def _extract(self, raw: str) -> Tuple[Dict[str, Any], str]:
        """LLM-first extraction with a deterministic fallback (project pattern).

        Returns ``(extracted, extraction_method)`` where ``extraction_method``
        describes what *actually* ran, so provenance never claims an LLM
        extraction that did not happen.
        """
        if self._llm_usable():
            try:
                return self._extract_with_llm(raw), METHOD_LLM
            except Exception as exc:  # noqa: BLE001 - ingestion must never fail on this
                log.warning(
                    "LLM paper extraction unavailable (%s); falling back to "
                    "deterministic section parsing", exc,
                )
        return self._extract_deterministic(raw), METHOD_DETERMINISTIC

    def _extract_with_llm(self, raw: str) -> Dict[str, Any]:
        """LLM extraction of the paper's structured fields (text-verbatim where possible)."""
        system_prompt = (
            "You extract structured research knowledge from a paper. Use ONLY what the "
            "paper states — do not invent content. Respond ONLY with JSON: "
            '{"title": str, "authors": [str], "year": int, "domain": str, '
            '"research_problem": str, "hypothesis": str, "methodology": [str], '
            '"important_methods": [str], "equations": [str], "assumptions": [str], '
            '"datasets": [str], "experimental_setup": str, "results": [str], '
            '"limitations": [str], "failures": [str], "conclusions": [str], '
            '"supporting_evidence": [str], "contradictory_evidence": [str], '
            '"references": [str], "relationships_to_other_research": [str]}'
        )
        data = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=raw[:14000],
            temperature=0.1,
        )
        extracted = self._normalise_structured(data)
        return extracted

    @staticmethod
    def _normalise_structured(data: Dict[str, Any]) -> Dict[str, Any]:
        """Accept either a full extracted dict or raw structured research data."""
        data = data or {}
        #: ``abstract`` is the paper's own summary text, so it is preserved rather
        #: than being folded into ``structured_data``.
        text_keys = ("title", "abstract", "research_problem", "hypothesis", "experimental_setup", "domain")
        list_keys = (
            "authors", "methodology", "important_methods", "equations", "assumptions",
            "datasets", "results", "limitations", "failures", "conclusions",
            "supporting_evidence", "contradictory_evidence", "references",
            "relationships_to_other_research",
        )
        out: Dict[str, Any] = {}
        for key in text_keys:
            out[key] = str(data.get(key, "") or "")
        for key in list_keys:
            value = data.get(key) or []
            out[key] = [str(v) for v in value] if isinstance(value, list) else [str(value)]
        year = data.get("year")
        try:
            out["year"] = int(year) if year else None
        except (TypeError, ValueError):
            out["year"] = None
        # Structured research data may carry arbitrary metrics; keep them verbatim.
        extras = {
            k: v for k, v in data.items()
            if k not in set(text_keys) | set(list_keys) | {"year", "abstract"}
        }
        if extras:
            out["structured_data"] = extras
        return out

    # ------------------------------------------------------- integration
    def _integrate(
        self,
        extracted: Dict[str, Any],
        fmt: str,
        path: str,
        provided_by: str,
        extraction_method: str = "",
        source_url: str = "",
    ) -> Dict[str, Any]:
        paper_id = self.memory.next_id("papers", "PAPER")
        title = str(extracted.get("title", "") or "untitled")
        method = str(extraction_method or "").strip() or (
            METHOD_STRUCTURED if fmt == "structured" else METHOD_DETERMINISTIC
        )

        # 1. Store the paper record (full extraction, provenance intact).
        record = {
            "paper_id": paper_id,
            "title": title,
            "authors": list(extracted.get("authors") or []),
            "year": extracted.get("year"),
            "domain": str(extracted.get("domain", "") or ""),
            "provided_by": provided_by,
            "input_format": fmt,
            "source_file": path or "",
            # Where the material legitimately lives (link-out target for the reader).
            "source_url": str(source_url or ""),
            "extraction": extracted,
            "provenance": {
                "origin": "researcher_provided_paper",
                "source_url": str(source_url or ""),
                # The method that actually produced the extraction (never claims
                # an LLM extraction when the deterministic parser ran).
                "extraction_method": method,
                "llm_service_present": self.llm is not None,
                "llm_service_available": self._llm_usable(),
                "generated_by": "aarl.lab.papers.PaperIngestor",
            },
            "timestamp": _now(),
        }
        self.memory.append_record("papers", paper_id, record)
        self.memory.append_history({
            "event": "paper_ingested",
            "paper_id": paper_id,
            "title": title,
            "provided_by": provided_by,
        })
        # 2. Integrate extracted knowledge as SOURCED items (never overwrites).
        citation = citation_for(extracted)
        self._integrate_knowledge(record, citation)

        # 3. Detect contradictions with existing SOURCED knowledge — preserve
        #    BOTH claims and record the conflict explicitly.
        record["contradictions"] = self._detect_contradictions(record)

        # Revision-safe update: append_record versions rather than overwrites.
        self.memory.append_record("papers", paper_id, record, revision_of="integrated knowledge")
        return record

    def _detect_contradictions(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        from lab.knowledge import _jaccard, _polarity

        out: List[Dict[str, Any]] = []
        paper_id = record["paper_id"]
        for entry in record.get("integrated_items", []):
            if entry["section"] not in ("findings", "evidence"):
                continue
            new_item = self.knowledge.get(entry["item_id"])
            if new_item is None:
                continue
            for existing in self.knowledge.items(section=entry["section"], claim_type=CLAIM_SOURCED):
                if existing.item_id == new_item.item_id:
                    continue
                if existing.metadata.get("paper_id") == paper_id:
                    continue
                similarity = _jaccard(existing.statement, new_item.statement)
                if similarity < 0.25:
                    continue
                polarity_a = _polarity(existing.statement)
                if polarity_a != 0 and polarity_a == -_polarity(new_item.statement):
                    conflict = self.knowledge.record_contradiction(
                        existing.item_id,
                        new_item.item_id,
                        note=(
                            f"Paper claims conflict: '{existing.source_paper}' vs "
                            f"'{new_item.source_paper}' on a similar statement "
                            f"(similarity {similarity:.2f}, opposite polarity). "
                            "Both claims preserved."
                        ),
                    )
                    out.append(conflict)
        return out


    def _integrate_knowledge(self, record: Dict[str, Any], citation: str) -> None:
        extracted = record["extraction"]
        paper_id = record["paper_id"]

        def _add(section: str, statement: str, note: str) -> None:
            statement = str(statement or "").strip()
            if not statement:
                return
            item_id = self.knowledge.add(
                section=section,
                statement=statement,
                claim_type=CLAIM_SOURCED,
                provenance={
                    "paper_id": paper_id,
                    "citation": citation,
                    "extraction_method": record["provenance"]["extraction_method"],
                    "field": note,
                    "locator": record.get("source_file") or f"paper:{paper_id}:{note}",
                },
                source=f"paper:{paper_id}",
                source_paper=citation,
                confidence=0.7,  # sourced from a single paper
                metadata={"paper_id": paper_id, "field": note},
            )
            record.setdefault("integrated_items", []).append(
                {"section": section, "item_id": item_id, "field": note}
            )

        # The abstract is the paper's own summary text. Storing it verbatim (with
        # an explicit locator) means an abstract-only input — e.g. a metadata
        # search result — still contributes one provenance-exact knowledge item
        # instead of nothing at all.
        if extracted.get("abstract"):
            _add("findings", extracted["abstract"], "abstract")
        if extracted.get("research_problem"):
            _add("findings", extracted["research_problem"], "research_problem")
        if extracted.get("hypothesis"):
            _add("hypotheses", extracted["hypothesis"], "hypothesis")
        for m in extracted.get("methodology", []) + extracted.get("important_methods", []):
            _add("methods", m, "methodology")
        for eq in extracted.get("equations", []):
            _add("equations", eq, "equation")
        for a in extracted.get("assumptions", []):
            _add("assumptions", a, "assumption")
        for r in extracted.get("results", []) + extracted.get("supporting_evidence", []):
            _add("evidence", r, "result_or_evidence")
        for lim in extracted.get("limitations", []):
            _add("limitations", lim, "limitation")
        for f in extracted.get("failures", []):
            _add("limitations", f, "reported_failure")
        for c in extracted.get("conclusions", []):
            _add("findings", c, "conclusion")
        for ce in extracted.get("contradictory_evidence", []):
            _add("contradictions", ce, "contradictory_evidence")
        if extracted.get("structured_data"):
            _add(
                "evidence",
                "Structured data provided by the researcher: "
                + json.dumps(extracted["structured_data"], default=str)[:800],
                "structured_data",
            )


def citation_for(extracted: Dict[str, Any]) -> str:
    title = str(extracted.get("title", "") or "untitled")
    year = extracted.get("year")
    return f"{title} ({year})" if year else title

    # __PART6__




