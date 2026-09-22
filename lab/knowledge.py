"""
AARL Lab — Living Knowledge Base (Knowledge JSON)
=================================================
One append-only, provenance-first knowledge store shared by every stage of the
continuous research loop.

Conceptual shape::

    Knowledge JSON
    ├── papers          externally sourced documents (retrieval layer)
    ├── findings        extracted/observed findings
    ├── methods         techniques and procedures
    ├── equations       formulas with units and stated conditions
    ├── assumptions     working assumptions (explicitly NOT facts)
    ├── evidence        evidence items supporting/contradicting a claim
    ├── limitations     known limits of a claim, or of AARL's own simulation
    ├── contradictions  explicit A-contradicts-B records (both sides preserved)
    ├── relationships   typed links between knowledge items
    ├── hypotheses      AARL-generated hypotheses (mirrors HYP-### records)
    ├── experiments     simulation observations (mirrors EXP-### records)
    ── lessons         failure-derived lessons (mirrors LESSON-### records)

The store never treats model output as truth: every item carries a
``claim_type`` from :data:`CLAIM_TYPES` plus the provenance fields that type
requires, so *sourced knowledge*, *inference*, *generated hypotheses*,
*simulation observations* and *real-world observations* stay separable — for
ever, inside the JSON itself.

Nothing is deleted. A claim that is later contradicted keeps its place and gains
a ``contradicts`` relationship; a superseded claim gains ``superseded_by``.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .memory import ResearchMemory

log = logging.getLogger("aarl.lab.knowledge")

SCHEMA_VERSION = 2

#: The five evidence classes. AARL must never blur them.
CLAIM_TYPES = (
    "sourced",                 # extracted from a real document; never model-authored
    "inferred",                # model- or rule-derived reasoning
    "generated_hypothesis",    # AARL's own proposal
    "simulation_observation",  # produced by an executable simulator
    "real_world_observation",  # supplied by a human from a real experiment
)

CLAIM_TYPE_NOTES: Dict[str, str] = {
    "sourced": "Extracted from a supplied document. Never LLM-authored text.",
    "inferred": "Reasoning derived by a model or rule engine; NOT established fact.",
    "generated_hypothesis": "Proposed by AARL; untested until an experiment runs.",
    "simulation_observation": "Observed inside a simulator under stated parameters.",
    "real_world_observation": "Reported by a human researcher from a physical experiment.",
}

#: Knowledge sections.
SECTIONS = (
    "papers",
    "findings",
    "methods",
    "equations",
    "assumptions",
    "evidence",
    "limitations",
    "contradictions",
    "relationships",
    "hypotheses",
    "experiments",
    "lessons",
)

#: Provenance keys each claim type MUST provide (enforced by KnowledgeManager.add).
REQUIRED_PROVENANCE: Dict[str, Tuple[str, ...]] = {
    "sourced": ("source", "source_paper", "locator"),
    "inferred": ("derived_from",),
    "generated_hypothesis": ("hypothesis_id",),
    "simulation_observation": ("experiment_id",),
    "real_world_observation": ("real_world_experiment_id", "provided_by"),
}

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "that", "this",
    "as", "at", "from", "than", "then", "which", "we", "can", "may", "will",
}

_POSITIVE = {
    "increase", "increases", "increased", "higher", "improve", "improves",
    "improved", "improvement", "enhance", "enhances", "faster", "stronger",
    "better", "gain", "gains", "boost", "boosts", "supports", "support",
    "enables", "accelerates", "outperforms",
}
_NEGATIVE = {
    "decrease", "decreases", "decreased", "lower", "reduce", "reduces",
    "reduced", "worse", "slower", "degrade", "degrades", "degradation", "loss",
    "losses", "fails", "failed", "failure", "ineffective",
    "contradicts", "contradicted", "breaks", "broken", "exceeds", "violates",
}

#: Tokens that flip the polarity of a nearby positive/negative cue.
_NEGATORS = (
    "not", "no", "never", "cannot", "without", "unable", "fails", "failed",
    "ineffective", "lacks", "lacking",
)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9_]+", str(text).lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _jaccard(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _polarity(text: str) -> int:
    """Crude polarity signal, used only to *flag possible* contradictions.

    A positive cue inside a negation window (``not improve``) counts as negative,
    which is what makes "X improves Y" vs "X does not improve Y" screen correctly.
    """
    words = _tokens(text)
    pos = neg = 0
    for index, word in enumerate(words):
        window = words[max(0, index - 3): index]
        negated = any(negator in window for negator in _NEGATORS)
        if word in _POSITIVE:
            if negated:
                neg += 1
            else:
                pos += 1
        elif word in _NEGATIVE:
            neg += 1
        elif negated and word in _NEGATORS:
            neg += 1
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


@dataclass
class KnowledgeItem:
    """One provenance-carrying knowledge item."""

    item_id: str = ""
    section: str = ""
    statement: str = ""
    claim_type: str = "inferred"
    source: str = ""
    source_paper: str = ""
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    timestamp: str = ""
    relationship_type: str = ""
    derived_from: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "section": self.section,
            "statement": self.statement,
            "claim_type": self.claim_type,
            "source": self.source,
            "source_paper": self.source_paper,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "timestamp": self.timestamp,
            "relationship_type": self.relationship_type,
            "derived_from": list(self.derived_from),
            "relationships": [dict(r) for r in self.relationships],
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        data = data or {}
        return cls(
            item_id=str(data.get("item_id", "") or ""),
            section=str(data.get("section", "") or ""),
            statement=str(data.get("statement", "") or ""),
            claim_type=str(data.get("claim_type", "inferred") or "inferred"),
            source=str(data.get("source", "") or ""),
            source_paper=str(data.get("source_paper", "") or ""),
            confidence=float(data.get("confidence", 0.5) or 0.0),
            evidence=list(data.get("evidence") or []),
            timestamp=str(data.get("timestamp", "") or ""),
            relationship_type=str(data.get("relationship_type", "") or ""),
            derived_from=list(data.get("derived_from") or []),
            relationships=list(data.get("relationships") or []),
            provenance=dict(data.get("provenance") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    def content_key(self) -> str:
        """Stable dedupe key for (section, normalised statement)."""
        norm = " ".join(_tokens(self.statement))
        digest = hashlib.sha1(f"{self.section}|{norm}".encode("utf-8")).hexdigest()
        return digest[:16]

    def has_relationship(self, rel_type: str, target: str) -> bool:
        for rel in self.relationships:
            if rel.get("type") == rel_type and rel.get("target") == target:
                return True
        return False

    def add_relationship(
        self,
        rel_type: str,
        target: str,
        note: str = "",
        confidence: float = 0.5,
        needs_human_review: bool = False,
    ) -> None:
        if self.has_relationship(rel_type, target):
            return
        self.relationships.append(
            {
                "type": rel_type,
                "target": target,
                "note": note,
                "confidence": round(float(confidence), 3),
                "needs_human_review": bool(needs_human_review),
                "timestamp": _now(),
            }
        )
class KnowledgeManager:
    """The living knowledge base: append-only, provenance-first, contradiction-preserving."""

    def __init__(self, memory: ResearchMemory, research_question: str = "") -> None:
        self.memory = memory
        loaded = memory.load_knowledge()
        self.research_question = research_question or str(
            loaded.get("research_question", "") or ""
        )
        self.store: Dict[str, Any] = loaded or self._skeleton()

    # ------------------------------------------------------------------ store
    def _skeleton(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "research_question": self.research_question,
            "created_at": _now(),
            "last_updated": _now(),
            "claim_types": list(CLAIM_TYPES),
            "claim_type_notes": dict(CLAIM_TYPE_NOTES),
            "sections": list(SECTIONS),
            "items": [],
            "dedupe_index": {},
            "counts": {},
            "note": (
                "Append-only. Claims are never silently removed; contradictions are "
                "recorded as explicit relationships and both claims are preserved."
            ),
        }

    def reload(self) -> None:
        self.store = self.memory.load_knowledge() or self._skeleton()

    def save(self) -> str:
        self.store["research_question"] = self.research_question or str(
            self.store.get("research_question", "") or ""
        )
        self.store["schema_version"] = SCHEMA_VERSION
        self.store["last_updated"] = _now()
        self.store["counts"] = self.summary()["by_section"]
        return self.memory.save_knowledge(self.store)

    # ------------------------------------------------------------------ items
    def _raw_items(self) -> List[Dict[str, Any]]:
        items = self.store.get("items")
        if not isinstance(items, list):
            items = []
            self.store["items"] = items
        return items

    def _next_item_id(self) -> str:
        highest = 0
        for raw in self._raw_items():
            match = re.match(r"^K-(\d+)$", str(raw.get("item_id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"K-{highest + 1:04d}"

    def get(self, item_id: str) -> Optional[KnowledgeItem]:
        for raw in self._raw_items():
            if str(raw.get("item_id", "")) == item_id:
                return KnowledgeItem.from_dict(raw)
        return None

    def remove_by_paper(self, paper_id: str) -> List[str]:
        """Remove every knowledge item contributed by a paper.

        Used by the Research Library's "remove from AARL knowledge" action.
        Only items whose provenance/metadata names ``paper_id`` are removed —
        a paper cannot silently delete claims contributed elsewhere. Returns
        the removed item ids.
        """
        paper_id = str(paper_id or "")
        removed: List[str] = []
        kept: List[Dict[str, Any]] = []
        for raw in self._raw_items():
            meta = raw.get("metadata") or {}
            prov = raw.get("provenance") or {}
            if paper_id and paper_id in (str(meta.get("paper_id", "")),
                                         str(prov.get("paper_id", ""))):
                removed.append(str(raw.get("item_id", "")))
            else:
                kept.append(raw)
        if not removed:
            return []
        self.store["items"] = kept
        index = self.store.get("dedupe_index")
        if isinstance(index, dict):
            self.store["dedupe_index"] = {
                key: item_id for key, item_id in index.items()
                if item_id not in removed
            }
        self.save()
        return removed

    def items(
        self, section: Optional[str] = None, claim_type: Optional[str] = None
    ) -> List[KnowledgeItem]:
        out: List[KnowledgeItem] = []
        for raw in self._raw_items():
            item = KnowledgeItem.from_dict(raw)
            if section and item.section != section:
                continue
            if claim_type and item.claim_type != claim_type:
                continue
            out.append(item)
        return out

    def add(
        self,
        section: str,
        statement: str,
        claim_type: str,
        provenance: Optional[Dict[str, Any]] = None,
        source: str = "",
        source_paper: str = "",
        confidence: float = 0.5,
        evidence: Optional[List[str]] = None,
        relationship_type: str = "",
        derived_from: Optional[List[str]] = None,
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        autosave: bool = True,
    ) -> str:
        """Add one knowledge item and return its ``item_id``.

        Honesty gate: ``sourced`` items must carry ``source``, ``source_paper``
        and ``locator``; every other claim type must carry the provenance keys in
        :data:`REQUIRED_PROVENANCE`. Missing provenance raises ``ValueError``
        rather than storing an unverifiable claim.
        """
        if section not in SECTIONS:
            raise ValueError(f"unknown knowledge section '{section}'")
        if claim_type not in CLAIM_TYPES:
            raise ValueError(f"unknown claim_type '{claim_type}' (expected {CLAIM_TYPES})")
        statement = str(statement or "").strip()
        if not statement:
            raise ValueError("knowledge items require a non-empty statement")

        prov = dict(provenance or {})
        if source:
            prov.setdefault("source", source)
        if source_paper:
            prov.setdefault("source_paper", source_paper)
        missing = [key for key in REQUIRED_PROVENANCE[claim_type] if not prov.get(key)]
        if missing:
            raise ValueError(
                f"claim_type '{claim_type}' requires provenance keys {missing}; "
                "AARL refuses to store unverifiable claims"
            )

        item = KnowledgeItem(
            item_id=item_id or self._next_item_id(),
            section=section,
            statement=statement,
            claim_type=claim_type,
            source=str(prov.get("source", "") or ""),
            source_paper=str(prov.get("source_paper", "") or ""),
            confidence=float(confidence),
            evidence=list(evidence or []),
            timestamp=_now(),
            relationship_type=relationship_type,
            derived_from=list(derived_from or prov.get("derived_from") or []),
            provenance=prov,
            metadata=dict(metadata or {}),
        )

        key = item.content_key()
        index = self.store.setdefault("dedupe_index", {})
        existing_id = index.get(key)
        if existing_id and self.get(str(existing_id)) is not None:
            return str(existing_id)  # idempotent: the identical claim is already stored

        self._flag_potential_contradictions(item)
        self._raw_items().append(item.to_dict())
        index[key] = item.item_id
        if autosave:
            self.save()
        return item.item_id
    # ------------------------------------------------------------------ relationships
    def _mutate(self, item_id: str, mutate) -> Optional[KnowledgeItem]:
        """Apply an in-place change to a stored item (the items list stays ordered)."""
        for raw in self._raw_items():
            if str(raw.get("item_id", "")) == item_id:
                item = KnowledgeItem.from_dict(raw)
                mutate(item)
                raw.clear()
                raw.update(item.to_dict())
                self.save()
                return item
        return None

    def link(
        self,
        source_item_id: str,
        rel_type: str,
        target: str,
        note: str = "",
        confidence: float = 0.5,
        needs_human_review: bool = False,
    ) -> None:
        """Add a typed relationship from one knowledge item to a target record."""
        self._mutate(
            source_item_id,
            lambda item: item.add_relationship(
                rel_type, target, note, confidence, needs_human_review
            ),
        )

    def record_contradiction(
        self,
        item_a: str,
        item_b: str,
        note: str = "",
        needs_human_review: bool = True,
        confidence: float = 0.4,
    ) -> Dict[str, Any]:
        """Record that two claims disagree — **both claims are preserved**.

        Creates a ``contradictions`` section entry plus symmetric ``contradicts``
        relationships on both items, so the stored history can never show one
        source silently replacing another.
        """
        record_id = self.add(
            section="contradictions",
            statement=f"{item_a} contradicts {item_b}",
            claim_type="inferred",
            provenance={
                "derived_from": [item_a, item_b],
                "detector": "KnowledgeManager.record_contradiction",
                "detection_rule": "explicit report or polarity screen",
            },
            confidence=confidence,
            relationship_type="contradicts",
            source="AARL contradiction detector",
            metadata={"item_a": item_a, "item_b": item_b, "note": note},
        )
        self.link(item_a, "contradicts", item_b, note, confidence, needs_human_review)
        self.link(item_b, "contradicts", item_a, note, confidence, needs_human_review)
        self.link(item_a, "contradiction_record", record_id, note, confidence)
        self.link(item_b, "contradiction_record", record_id, note, confidence)
        self.memory.append_history(
            {
                "event": "contradiction_recorded",
                "item_a": item_a,
                "item_b": item_b,
                "record_id": record_id,
                "both_claims_preserved": True,
                "needs_human_review": needs_human_review,
            }
        )
        return {"record_id": record_id, "item_a": item_a, "item_b": item_b}

    def _flag_potential_contradictions(self, item: KnowledgeItem) -> None:
        """Flag (never delete) a claim that appears to disagree with stored knowledge.

        Screening heuristic only: pairs it finds are marked ``needs_human_review``
        with the detection rule written into provenance, because keyword polarity
        is not a scientific judgement.
        """
        if item.claim_type not in ("sourced", "inferred"):
            return
        if item.section in ("contradictions", "relationships", "evidence"):
            return
        polarity = _polarity(item.statement)
        if polarity == 0:
            return
        for raw in list(self._raw_items()):
            other = KnowledgeItem.from_dict(raw)
            if other.item_id == item.item_id or other.section != item.section:
                continue
            if other.claim_type not in ("sourced", "inferred"):
                continue
            if _jaccard(item.statement, other.statement) < 0.34:
                continue
            other_polarity = _polarity(other.statement)
            if other_polarity == 0 or other_polarity == polarity:
                continue
            self.record_contradiction(
                item.item_id,
                other.item_id,
                note=(
                    "Screened automatically: similar topic, opposing polarity. "
                    "Requires human review before either claim is treated as settled."
                ),
            )
            item.add_relationship(
                "contradicts",
                other.item_id,
                "auto-screened",
                0.35,
                needs_human_review=True,
            )
            return
# ------------------------------------------------------------------ retrieval
    def latest(self, section: str, n: int = 5) -> List[KnowledgeItem]:
        """Most recent items of a section (oldest first, truncated to ``n``)."""
        found = self.items(section=section)
        return found[-n:] if n > 0 else found

    def search(
        self,
        query: str,
        section: Optional[str] = None,
        claim_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[KnowledgeItem]:
        """Rank stored items by token overlap with ``query`` (deterministic)."""
        scored = []
        for item in self.items(section=section, claim_type=claim_type):
            score = _jaccard(query, item.statement)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].item_id))
        return [item for _, item in scored[: max(0, limit)]]

    def summary(self) -> Dict[str, Any]:
        """Counts per section and per claim type (used by reporting)."""
        by_section: Dict[str, int] = {section: 0 for section in SECTIONS}
        by_claim_type: Dict[str, int] = {claim: 0 for claim in CLAIM_TYPES}
        unresolved = 0
        for item in self.items():
            by_section[item.section] = by_section.get(item.section, 0) + 1
            by_claim_type[item.claim_type] = by_claim_type.get(item.claim_type, 0) + 1
            if item.section == "contradictions":
                unresolved += 1
        return {
            "total_items": len(self._raw_items()),
            "by_section": by_section,
            "by_claim_type": by_claim_type,
            "contradiction_records": unresolved,
            "schema_version": SCHEMA_VERSION,
            "path": self.memory.knowledge_path,
            "honesty_note": (
                "Only 'sourced' items come from supplied documents; 'inferred' items "
                "are model/rule reasoning and are NOT established facts."
            ),
        }