"""
AARL Discovery - Mechanism and the "Why this idea?" chain
===========================================================
Two reader-facing structures:

* **Mechanism** - the proposed causal chain of the direction, stage by stage,
  each stage tagged with whether stored evidence touches it. The mechanism is
  labelled ``PROPOSED MECHANISM`` and ``CONCEPTUAL - NOT IMPLEMENTED``; a
  described mechanism is never presented as a working one.

* **Why this idea?** - the reasoning chain that produced the direction:

      Research problem -> Observed limitation -> Cross-domain concept
      -> Mechanism -> Hypothesis

  Each stage carries the class it belongs to. The limitation stage is
  source-linked when a *sourced* limitation exists in the knowledge base and is
  otherwise explicitly marked as an AARL inference. This is what stops the panel
  from looking like a black box.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .text import content_token_set


def _overlap(left: str, right: str) -> float:
    a, b = content_token_set(left), content_token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def build_mechanism(
    concept,
    facet,
    subject: str,
    knowledge_items: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.16,
) -> Dict[str, Any]:
    """Stage-by-stage mechanism, with per-stage evidence state."""
    items = list(knowledge_items or [])
    steps: List[Dict[str, Any]] = []
    for index, template in enumerate(concept.mechanism_steps):
        text = str(template).format(subject=subject, facet=facet.label.lower())
        supporting: List[Dict[str, Any]] = []
        contradicting: List[Dict[str, Any]] = []
        for item in items:
            statement = str(item.get("statement") or "")
            if _overlap(text, statement) < threshold:
                continue
            row = {
                "item_id": item.get("item_id", ""),
                "claim_type": item.get("claim_type", ""),
                "statement": statement[:280],
                "source_paper": (item.get("provenance") or {}).get("citation")
                or item.get("source_paper") or item.get("source") or "",
                "section": item.get("section", ""),
            }
            if str(item.get("section")) == "contradictions":
                contradicting.append(row)
            else:
                supporting.append(row)
        if contradicting:
            state = "contradicted"
        elif supporting:
            state = "touched_by_stored_knowledge"
        else:
            state = "unsupported"
        steps.append({
            "index": index + 1,
            "text": text,
            "evidence_state": state,
            "evidence_state_label": {
                "contradicted": "A STORED RECORD CONTRADICTS THIS STAGE",
                "touched_by_stored_knowledge": "SOME STORED KNOWLEDGE IS RELEVANT",
                "unsupported": "NO STORED EVIDENCE ADDRESSES THIS STAGE",
            }[state],
            "supporting": supporting[:3],
            "contradicting": contradicting[:3],
        })
    return {
        "label": "PROPOSED MECHANISM",
        "implementation_status": "CONCEPTUAL - NOT IMPLEMENTED",
        "source_concept": concept.concept_id,
        "source_concept_name": concept.name,
        "source_domains": list(concept.source_domains),
        "steps": steps,
        "unsupported_stages": sum(1 for s in steps if s["evidence_state"] == "unsupported"),
        "note": (
            "This chain is the mechanism AARL is proposing, expressed conceptually. "
            "No implementation of it was built or measured, and an unsupported stage "
            "is a research target - not a hidden assumption."
        ),
    }

def find_sourced_limitation(
    facet,
    knowledge_items: List[Dict[str, Any]],
    threshold: float = 0.12,
) -> Optional[Dict[str, Any]]:
    """A *sourced* limitation for this facet, when the knowledge base has one.

    Only items whose ``claim_type`` is ``sourced`` or
    ``real_world_observation`` qualify - an AARL inference can never supply the
    literature backing for a limitation.
    """
    query = "%s %s %s" % (facet.label, facet.metric, facet.framing)
    best, best_score = None, 0.0
    for item in knowledge_items:
        if str(item.get("section")) != "limitations":
            continue
        if str(item.get("claim_type")) not in ("sourced", "real_world_observation"):
            continue
        score = _overlap(query, str(item.get("statement") or ""))
        if score > best_score:
            best, best_score = item, score
    if best is None or best_score < threshold:
        return None
    return {
        "item_id": best.get("item_id", ""),
        "statement": str(best.get("statement") or ""),
        "source_paper": (best.get("provenance") or {}).get("citation")
        or best.get("source_paper") or "",
        "locator": (best.get("provenance") or {}).get("locator", ""),
        "claim_type": best.get("claim_type", ""),
        "overlap": round(best_score, 3),
    }


def build_why_chain(
    problem: str,
    facet,
    concept,
    hypothesis_text: str,
    subject: str,
    knowledge_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The five-stage reasoning chain behind the direction."""
    sourced = find_sourced_limitation(facet, list(knowledge_items or []))
    if sourced:
        limitation_stage = {
            "stage": "Observed limitation",
            "value": sourced["statement"],
            "claim_class": "LITERATURE_EVIDENCE",
            "claim_class_label": "LITERATURE EVIDENCE",
            "source": sourced["source_paper"] or sourced["item_id"],
            "provenance": {"item_id": sourced["item_id"], "locator": sourced["locator"],
                           "overlap": sourced["overlap"]},
            "note": "Taken from a stored, source-linked limitation item.",
        }
    else:
        limitation_stage = {
            "stage": "Observed limitation",
            "value": facet.limitation,
            "claim_class": "INFERENCE",
            "claim_class_label": "INFERENCE (AARL - NOT SOURCED)",
            "source": "aarl.discovery.facets (facet assumption)",
            "provenance": {"facet_id": facet.facet_id},
            "note": (
                "AARL states this as an assumed limitation of the conventional "
                "approach. No stored source-linked limitation was found for this "
                "facet, so it must not be read as a literature fact."
            ),
        }
    stages = [
        {
            "stage": "Research problem",
            "value": problem,
            "claim_class": "STATED_BY_USER",
            "claim_class_label": "STATED BY RESEARCHER",
            "source": "user input",
            "provenance": {},
            "note": "The exact problem text AARL was given.",
        },
        limitation_stage,
        {
            "stage": "Cross-domain concept",
            "value": "%s - %s" % (concept.name, concept.description),
            "claim_class": "LIBRARY_ENTRY",
            "claim_class_label": "LIBRARY ENTRY",
            "source": ", ".join(concept.source_domains),
            "provenance": {"concept_id": concept.concept_id,
                           "library": "aarl.discovery.concepts"},
            "note": (
                "A curated cross-domain mechanism. Its transfer to %s is AARL's "
                "proposal, not an established result." % subject
            ),
        },
        {
            "stage": "Mechanism",
            "value": concept.description,
            "claim_class": "LIBRARY_ENTRY",
            "claim_class_label": "LIBRARY ENTRY",
            "source": "aarl.discovery.concepts:%s" % concept.concept_id,
            "provenance": {"steps": len(concept.mechanism_steps)},
            "note": "Expanded stage by stage in the mechanism view below.",
        },
        {
            "stage": "Hypothesis",
            "value": hypothesis_text or "(no hypothesis generated in this run)",
            "claim_class": "HYPOTHESIS_UNTESTED",
            "claim_class_label": "HYPOTHESIS (UNTESTED)",
            "source": "aarl.lab.hypotheses",
            "provenance": {},
            "note": "AARL-generated proposal. It is not a finding.",
        },
    ]
    return {
        "stages": stages,
        "cross_domains": list(concept.source_domains),
        "flow": " > ".join(s["stage"] for s in stages),
        "why_readable": (
            "AARL started from the stated problem, picked the '%s' dimension of it, "
            "took the '%s' mechanism from %s, and proposed applying it there."
            % (facet.label, concept.name, ", ".join(concept.source_domains))
        ),
        "honesty_note": (
            "Each stage is labelled with what it is. A limitation marked INFERENCE "
            "is an AARL assumption and is the first thing a reviewer should check."
        ),
    }