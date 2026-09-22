"""
AARL Discovery - Evidence Panel and Result Provenance
=====================================================
Two jobs:

1. **Evidence panel.** Groups the evidence that exists for a direction into
   *direct*, *indirect*, *contradicting* and *missing*, keeping the provenance of
   each item (source, paper, date, locator, what it supports, its limitations).
   Only provenance-linked knowledge items count as *direct* evidence; anything
   found by wording similarity is *indirect* and is labelled as such. Semantic
   similarity is never converted into scientific evidence.

2. **Result provenance.** States what a stored result actually *is*:
   ``ASSUMPTION`` / ``ASSUMPTION-BASED RESULT`` / ``SIMULATION RESULT`` /
   ``EXPERIMENTAL RESULT`` / ``LITERATURE EVIDENCE`` / ``INFERENCE`` /
   ``HYPOTHESIS (UNTESTED)``. A parametric model driven by assumed multipliers is
   never presented as experimental validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .text import content_token_set

#: Result classes AARL is allowed to print.
RESULT_CLASSES: Dict[str, Dict[str, str]] = {
    "ASSUMPTION": {
        "label": "ASSUMPTION", "css": "prov-assumption",
        "meaning": "A working assumption stated by AARL or by the hypothesis. Not evidence.",
    },
    "HYPOTHESIS_UNTESTED": {
        "label": "HYPOTHESIS (UNTESTED)", "css": "prov-hypothesis",
        "meaning": "AARL's own proposal. Nothing has tested it yet.",
    },
    "INFERENCE": {
        "label": "INFERENCE", "css": "prov-inference",
        "meaning": "Derived by a model or rule engine from other records. Not established fact.",
    },
    "LITERATURE_EVIDENCE": {
        "label": "LITERATURE EVIDENCE", "css": "prov-literature",
        "meaning": "Extracted from a supplied document, with its source recorded.",
    },
    "ASSUMPTION_BASED_RESULT": {
        "label": "ASSUMPTION-BASED RESULT", "css": "prov-assumption-result",
        "meaning": (
            "Produced by a parametric model whose changes are assumed multipliers "
            "rather than measured behaviour. It is NOT experimental validation."
        ),
    },
    "SIMULATION_RESULT": {
        "label": "SIMULATION RESULT", "css": "prov-simulation",
        "meaning": (
            "Produced by executing a seeded model. A model output under stated "
            "parameters - never real-world evidence."
        ),
    },
    "EXPERIMENTAL_RESULT": {
        "label": "EXPERIMENTAL RESULT", "css": "prov-experimental",
        "meaning": "Reported by a human researcher from a physical experiment.",
    },
    "NO_SIMULATION": {
        "label": "NO SIMULATION AVAILABLE", "css": "prov-none",
        "meaning": "No executable model supported this domain, so nothing was run.",
    },
}

#: Result classes that may never be described with validation wording.
NEVER_VALIDATED = ("ASSUMPTION_BASED_RESULT", "SIMULATION_RESULT", "ASSUMPTION",
                   "INFERENCE", "HYPOTHESIS_UNTESTED", "NO_SIMULATION")

def classify_simulation(simulation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Say exactly what a stored Simulation JSON record is."""
    sim = dict(simulation or {})
    if not sim:
        return {
            "result_class": "NO_SIMULATION",
            "label": RESULT_CLASSES["NO_SIMULATION"]["label"],
            "css": RESULT_CLASSES["NO_SIMULATION"]["css"],
            "meaning": RESULT_CLASSES["NO_SIMULATION"]["meaning"],
            "why": "No simulation record exists for this direction yet.",
            "is_reproducible": False,
            "experiment_id": "",
            "must_not_be_called": list(NEVER_VALIDATED),
            "assumptions": [],
        }

    status = str(sim.get("status", "") or "").lower()
    sim_type = str(sim.get("simulation_type", "") or "")
    observed = sim.get("observed_results") or {}
    modifiers = {}
    if isinstance(observed, dict):
        modifiers = observed.get("applied_modifications") or {}
    extraction = ""
    if isinstance(sim.get("parameters"), dict):
        extraction = str(sim["parameters"].get("extraction_source", "") or "")

    if status == "simulation_unavailable" or sim_type in ("", "none"):
        result_class = "NO_SIMULATION"
        why = (
            "No simulation engine supports this domain, so AARL recorded "
            "'simulation_unavailable' instead of inventing a result."
        )
    elif sim_type.startswith("domain_model:"):
        if modifiers:
            result_class = "ASSUMPTION_BASED_RESULT"
            why = (
                "The domain model is parametric: the hypothesis was translated into "
                "assumed multipliers ("
                + ", ".join("%s=%s" % (k, v) for k, v in list(modifiers.items())[:6])
                + "). The numbers follow directly from those assumptions"
                + ((" (extraction source: %s)." % extraction) if extraction else ".")
            )
        else:
            result_class = "SIMULATION_RESULT"
            why = (
                "The domain model was executed with no hypothesis-derived "
                "modifications, so baseline and hypothesis were identical."
            )
    else:
        result_class = "SIMULATION_RESULT"
        why = (
            "Executed by adapter '%s' with seed %s."
            % (sim_type or "unknown",
               (sim.get("reproducibility") or {}).get("seed", "n/a"))
        )

    reproducibility = sim.get("reproducibility") or {}
    return {
        "result_class": result_class,
        "label": RESULT_CLASSES[result_class]["label"],
        "css": RESULT_CLASSES[result_class]["css"],
        "meaning": RESULT_CLASSES[result_class]["meaning"],
        "why": why,
        "status": status,
        "simulation_type": sim_type,
        "experiment_id": sim.get("experiment_id", ""),
        "hypothesis_id": sim.get("hypothesis_id", ""),
        "is_reproducible": bool(reproducibility.get("deterministic")),
        "input_hash": reproducibility.get("input_hash", ""),
        "how_to_reproduce": reproducibility.get("how_to_reproduce", ""),
        "assumed_modifiers": dict(modifiers) if isinstance(modifiers, dict) else {},
        "constraints_violated": list(sim.get("constraints_violated") or []),
        "errors": list(sim.get("errors") or []),
        "notes": list(sim.get("notes") or []),
        "assumptions": assumptions_of(sim),
        "must_not_be_called": list(
            ["validated", "experimentally confirmed", "scientifically proven"]
            if result_class in NEVER_VALIDATED else []
        ),
    }

def assumptions_of(sim: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assumptions a reader needs in order to interpret the run, from the record."""
    out: List[Dict[str, Any]] = []
    params = sim.get("parameters") or {}
    if isinstance(params, dict):
        for key, value in params.items():
            if key == "domain_parameter_modifications":
                continue
            out.append({"name": str(key), "value": value, "source": "recorded parameter"})
        modifications = params.get("domain_parameter_modifications")
        if isinstance(modifications, dict):
            for key, value in modifications.items():
                out.append({
                    "name": str(key),
                    "value": value,
                    "source": "hypothesis-derived modification (assumed multiplier)",
                })
        if params.get("extraction_source"):
            out.append({
                "name": "parameter extraction",
                "value": params["extraction_source"],
                "source": "recorded provenance of the parameter values",
            })
    reproducibility = sim.get("reproducibility") or {}
    if reproducibility.get("constraints_used"):
        out.append({
            "name": "declared constraints",
            "value": reproducibility["constraints_used"],
            "source": "human-declared operating envelope",
        })
    return out


def _evidence_class(claim_type: str) -> str:
    if claim_type == "sourced":
        return "LITERATURE_EVIDENCE"
    if claim_type == "real_world_observation":
        return "EXPERIMENTAL_RESULT"
    if claim_type == "simulation_observation":
        return "SIMULATION_RESULT"
    if claim_type == "generated_hypothesis":
        return "HYPOTHESIS_UNTESTED"
    return "INFERENCE"


def _item_view(
    item: Dict[str, Any],
    kind: str,
    linked_hypothesis_ids: List[str],
    papers_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """One evidence row: provenance first, interpretation clearly separated."""
    prov = item.get("provenance") or {}
    claim_type = str(item.get("claim_type") or "inferred")
    paper_id = str(prov.get("paper_id") or (item.get("metadata") or {}).get("paper_id") or "")
    paper = papers_by_id.get(paper_id) or {}
    limitations = []
    if claim_type == "sourced":
        limitations.append(
            "Extracted from a single document; AARL has not re-run or reproduced it."
        )
    if claim_type == "inferred":
        limitations.append("Derived by rule/model reasoning, not observed directly.")
    if prov.get("needs_human_review"):
        limitations.append("Flagged for human review (automatic screening).")
    return {
        "evidence_id": item.get("item_id", ""),
        "kind": kind,
        "evidence_class": _evidence_class(claim_type),
        "evidence_class_label": RESULT_CLASSES[_evidence_class(claim_type)]["label"],
        "claim_type": claim_type,
        "claim_type_note": {
            "sourced": "Extracted from a supplied document; never model-authored text.",
            "inferred": "Rule/model reasoning; NOT established fact.",
            "generated_hypothesis": "AARL proposal; untested.",
            "simulation_observation": "Observed inside a simulator under stated parameters.",
            "real_world_observation": "Reported by a human researcher.",
        }.get(claim_type, ""),
        "statement": str(item.get("statement") or ""),
        "source": str(prov.get("source") or item.get("source") or ""),
        "source_paper": str(
            prov.get("citation") or item.get("source_paper") or paper.get("title") or ""
        ),
        "paper_id": paper_id,
        "date": paper.get("year") or prov.get("year") or "",
        "locator": str(prov.get("locator") or prov.get("source_locator") or ""),
        "supports_what": linked_hypothesis_ids,
        "section": item.get("section", ""),
        "confidence": item.get("confidence"),
        "limitations": limitations,
        "note": (
            "Direct evidence because the hypothesis record lists this item id in "
            "supporting_evidence / contradictory_evidence." if kind == "direct"
            else "Indirect: found by wording overlap with the direction. Not linked by "
                 "provenance, so it is not counted as direct evidence."
        ),
    }

def _jaccard(left: str, right: str) -> float:
    a, b = content_token_set(left), content_token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def evidence_panel(
    hypotheses: List[Dict[str, Any]],
    knowledge_items: List[Dict[str, Any]],
    papers_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    direction_text: str = "",
    mechanism_steps: Optional[List[str]] = None,
    uncertainties: Optional[List[str]] = None,
    overlap_threshold: float = 0.18,
    overlap_limit: int = 6,
) -> Dict[str, Any]:
    """Group evidence for a direction: direct / indirect / contradicting / missing.

    Direct evidence requires a *provenance link*: the hypothesis record must list
    the knowledge item id in ``supporting_evidence`` or ``contradictory_evidence``.
    Everything found by wording overlap is placed in ``indirect`` and labelled as
    such, so similarity is never silently promoted to evidence.
    """
    papers_by_id = papers_by_id or {}
    mechanism_steps = list(mechanism_steps or [])
    uncertainties = list(uncertainties or [])

    support_ids: List[str] = []
    contra_ids: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in knowledge_items:
        by_id[str(item.get("item_id", ""))] = item
    for hyp in hypotheses:
        hid = str(hyp.get("hypothesis_id", ""))
        for item_id in hyp.get("supporting_evidence") or []:
            if str(item_id) in by_id:
                support_ids.append(str(item_id))
        for item_id in hyp.get("contradictory_evidence") or []:
            if str(item_id) in by_id:
                contra_ids.append(str(item_id))

    support_set, contra_set = set(support_ids), set(contra_ids)
    hyp_by_item: Dict[str, List[str]] = {}
    for hyp in hypotheses:
        hid = str(hyp.get("hypothesis_id", ""))
        for item_id in (hyp.get("supporting_evidence") or []) + \
                (hyp.get("contradictory_evidence") or []):
            hyp_by_item.setdefault(str(item_id), [])
            if hid not in hyp_by_item[str(item_id)]:
                hyp_by_item[str(item_id)].append(hid)

    direct: List[Dict[str, Any]] = []
    contradicting: List[Dict[str, Any]] = []
    for item_id in sorted(support_set):
        direct.append(_item_view(by_id[item_id], "direct", hyp_by_item.get(item_id, []),
                                 papers_by_id))
    for item_id in sorted(contra_set):
        contradicting.append(_item_view(by_id[item_id], "contradicting",
                                        hyp_by_item.get(item_id, []), papers_by_id))

    excluded = support_set | contra_set
    indirect: List[Dict[str, Any]] = []
    text = " ".join([direction_text] + mechanism_steps)
    for item in knowledge_items:
        item_id = str(item.get("item_id", ""))
        if item_id in excluded:
            continue
        if str(item.get("section")) in ("relationships",):
            continue
        if str(item.get("claim_type")) == "generated_hypothesis":
            continue
        score = _jaccard(text, str(item.get("statement") or ""))
        if score < overlap_threshold:
            continue
        row = _item_view(item, "indirect", [], papers_by_id)
        row["similarity"] = round(score, 3)
        indirect.append(row)
    indirect.sort(key=lambda r: (-r["similarity"], r["evidence_id"]))
    indirect = indirect[:overlap_limit]

    missing = _missing_evidence(mechanism_steps, uncertainties, direct, indirect, text)
    return {
        "direct": direct,
        "indirect": indirect,
        "contradicting": contradicting,
        "missing": missing,
        "counts": {
            "direct": len(direct),
            "indirect": len(indirect),
            "contradicting": len(contradicting),
            "missing": len(missing),
        },
        "direct_requires": (
            "A provenance link in the hypothesis record. Wording similarity alone "
            "never produces direct evidence."
        ),
        "honesty_note": (
            "AARL never converts semantic similarity into scientific evidence. "
            "Indirect items are shown as leads, not as support."
        ),
    }

def _missing_evidence(
    mechanism_steps: List[str],
    uncertainties: List[str],
    direct: List[Dict[str, Any]],
    indirect: List[Dict[str, Any]],
    direction_text: str,
) -> List[Dict[str, Any]]:
    """What is absent, derived from the mechanism stages and the uncertainties.

    A stage counts as *covered* only when a direct evidence item overlaps it.
    Indirect items are reported as "lead only", which is deliberately weaker.
    """
    out: List[Dict[str, Any]] = []
    for index, step in enumerate(mechanism_steps):
        cover = max(
            [_jaccard(step, str(e.get("statement") or "")) for e in direct] or [0.0]
        )
        lead = max(
            [_jaccard(step, str(e.get("statement") or "")) for e in indirect] or [0.0]
        )
        if cover >= 0.2:
            continue
        out.append({
            "gap_id": "MISSING-MECH-%02d" % (index + 1),
            "kind": "mechanism_step_unsupported",
            "statement": step,
            "what_is_needed": (
                "Direct (source-linked) evidence that this stage behaves as described."
            ),
            "lead_only": round(lead, 3),
            "status": "MISSING",
            "note": (
                "An indirect item overlaps this stage" if lead >= 0.2
                else "Nothing in the stored knowledge base addresses this stage."
            ),
        })
    for index, uncertainty in enumerate(uncertainties):
        cover = max(
            [_jaccard(uncertainty, str(e.get("statement") or "")) for e in direct] or [0.0]
        )
        if cover >= 0.2:
            continue
        out.append({
            "gap_id": "MISSING-Q-%02d" % (index + 1),
            "kind": "open_question",
            "statement": uncertainty,
            "what_is_needed": (
                "A measurement or a sourced result that bounds this quantity."
            ),
            "lead_only": 0.0,
            "status": "MISSING",
            "note": "Recorded as an open question for this mechanism class.",
        })
    return out
