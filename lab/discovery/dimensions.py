"""
AARL Discovery — Descriptive Dimension Indicators
=================================================
Six **separate** descriptive indicators per direction, deliberately replacing any
single blended "scientific score" such as ``8.7/10 = good science``.

    potential novelty        is this combination already present in AARL's memory?
    mechanistic plausibility does a described mechanism exist, and does a run agree?
    evidence strength        how much provenance-linked evidence exists
    testability              can it be tested with the models AARL actually has?
    implementation risk      what the mechanism class costs to realise
    research uncertainty     how much is still unknown

Every level ships with its ``rationale`` and its ``basis`` (record ids and
library fields it was read from), so clicking a dimension can always answer
*why did AARL say that?*. Levels are descriptive, not claims of truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .status import EvidenceCounters

LEVELS = ("LOW", "MEDIUM", "HIGH")

_LABELS = {
    "potential_novelty": "Potential novelty",
    "mechanistic_plausibility": "Mechanistic plausibility",
    "evidence_strength": "Evidence strength",
    "testability": "Testability",
    "implementation_risk": "Implementation risk",
    "research_uncertainty": "Research uncertainty",
}

_NOTES = {
    "potential_novelty": (
        "How new this combination looks *relative to AARL's own memory*. Novelty is "
        "not a virtue by itself and not a claim of originality in the literature."
    ),
    "mechanistic_plausibility": (
        "Whether a mechanism can be described and whether anything AARL ran agreed "
        "with it. A described mechanism is not a demonstrated one."
    ),
    "evidence_strength": (
        "The amount of provenance-linked evidence recorded for this direction. "
        "Similar wording is never counted as evidence."
    ),
    "testability": (
        "Whether AARL could actually test this with the executable models it has."
    ),
    "implementation_risk": (
        "Library metadata for the mechanism class (what it typically costs to "
        "realise) — a design consideration, not a scientific judgement."
    ),
    "research_uncertainty": (
        "How much remains unknown about this direction, in plain terms."
    ),
}

ORDER = (
    "potential_novelty",
    "mechanistic_plausibility",
    "evidence_strength",
    "testability",
    "implementation_risk",
    "research_uncertainty",
)


def _entry(key: str, level: str, rationale: str, basis: List[str]) -> Dict[str, Any]:
    return {
        "key": key,
        "label": _LABELS[key],
        "level": level if level in LEVELS else "MEDIUM",
        "level_scale": list(LEVELS),
        "rationale": rationale,
        "basis": [b for b in basis if b],
        "descriptive_only": True,
        "what_this_is_not": "Not a score of scientific truth and not a probability.",
        "explanation": _NOTES[key],
    }


def dimensions_for(
    concept,
    facet,
    counters: EvidenceCounters,
    experiments: Optional[List[Dict[str, Any]]] = None,
    prior_failures: Optional[List[Dict[str, Any]]] = None,
    knowledge_overlap: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Build the six indicators from library metadata plus the stored records.

    ``experiments``      stored Simulation JSON records for this direction
    ``prior_failures``   stored FAIL records whose mechanism matches
    ``knowledge_overlap`` stored knowledge items whose wording overlaps (used only
                         to temper the novelty statement)
    """
    experiments = list(experiments or [])
    prior_failures = list(prior_failures or [])
    out: Dict[str, Dict[str, Any]] = {}

    # ---- potential novelty -------------------------------------------------
    if prior_failures:
        level = "LOW"
        rationale = (
            f"AARL already attempted a closely related mechanism ({len(prior_failures)}"
            " recorded failure(s): "
            + ", ".join(str(f.get("failure_id", "")) for f in prior_failures[:3])
            + "). The direction is not new to AARL's own memory."
        )
    elif knowledge_overlap >= 3:
        level = "LOW"
        rationale = (
            f"{knowledge_overlap} stored knowledge items already overlap this "
            "direction's wording, so AARL's memory already covers part of it."
        )
    else:
        level = str(getattr(concept, "novelty", "MEDIUM"))
        rationale = (
            f"Library metadata for the mechanism class rates it {level}; AARL's memory "
            f"holds {knowledge_overlap} overlapping item(s) and {len(prior_failures)} "
            "directly related failure record(s)."
        )
    out["potential_novelty"] = _entry(
        "potential_novelty", level, rationale,
        [f"concept:{concept.concept_id}"]
        + [str(f.get("failure_id", "")) for f in prior_failures],
    )

    # ---- mechanistic plausibility -----------------------------------------
    supported = [
        e for e in experiments if str(e.get("status")) in ("success", "promising")
    ]
    hard_fail = [
        e for e in experiments
        if str(e.get("status")) == "failure" or e.get("constraints_violated")
    ]
    if supported and not hard_fail:
        level = "HIGH"
        rationale = (
            f"A mechanism is fully described ({len(concept.mechanism_steps)} stages) "
            f"and {len(supported)} stored simulation(s) agreed with it."
        )
    elif supported and hard_fail:
        level = "MEDIUM"
        rationale = (
            f"A mechanism is described, but AARL's own runs disagree: {len(supported)} "
            f"supported it while {len(hard_fail)} failed or violated a constraint."
        )
    elif hard_fail:
        level = "LOW"
        rationale = (
            f"A mechanism is described, but {len(hard_fail)} stored run(s) failed or "
            "violated a declared constraint, so plausibility is currently weak."
        )
    else:
        level = "MEDIUM"
        rationale = (
            "A mechanism is fully described, but nothing has been run that could "
            "agree or disagree with it. Description is not demonstration."
        )
    out["mechanistic_plausibility"] = _entry(
        "mechanistic_plausibility", level, rationale,
        [str(e.get("experiment_id", "")) for e in experiments],
    )


    # ---- evidence strength -------------------------------------------------
    d, ind, con = counters.direct, counters.indirect, counters.contradicting
    if d >= 2 and con == 0:
        level = "HIGH"
    elif d >= 1 or ind >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"
    out["evidence_strength"] = _entry(
        "evidence_strength", level,
        f"{d} directly linked sourced item(s), {ind} indirectly related item(s), "
        f"{con} contradicting item(s). Text similarity is never counted.",
        [f"direct:{d}", f"indirect:{ind}", f"contradicting:{con}"],
    )

    # ---- testability -------------------------------------------------------
    if counters.simulations_unavailable and not experiments:
        level = "LOW"
        rationale = (
            "No executable model covers this domain, so the direction cannot be "
            "tested with the models AARL currently has."
        )
    else:
        level = str(getattr(concept, "testability", "MEDIUM"))
        rationale = (
            f"Library metadata rates the mechanism class {level} for testability"
            + (
                f", and {len(experiments)} run(s) were executed for this direction."
                if experiments else "."
            )
        )
    out["testability"] = _entry("testability", level, rationale,
                                [str(e.get("experiment_id", "")) for e in experiments])

    # ---- implementation risk ----------------------------------------------
    level = str(getattr(concept, "risk", "MEDIUM"))
    rationale = (
        f"Library metadata for the mechanism class rates realisation cost {level} "
        f"(source fields: {', '.join(concept.source_domains)})."
    )
    if prior_failures:
        level = "HIGH"
        rationale += (
            " Raised to HIGH because a closely related mechanism already failed in "
            "AARL's own record."
        )
    out["implementation_risk"] = _entry(
        "implementation_risk", level, rationale, [f"concept:{concept.concept_id}"]
    )

    # ---- research uncertainty ---------------------------------------------
    if out["evidence_strength"]["level"] == "HIGH" and supported:
        level = "MEDIUM"
        rationale = (
            "Supporting evidence and a supporting run exist, but no real-world "
            "replication is recorded, so uncertainty cannot be called LOW."
        )
    elif out["evidence_strength"]["level"] == "LOW" or counters.simulations_unavailable:
        level = "HIGH"
        rationale = (
            "Direct evidence is absent"
            + (" and no executable model exists"
               if counters.simulations_unavailable else "")
            + ", so most of what would need to be known is unknown."
        )
    else:
        level = "MEDIUM"
        rationale = "Partial support exists; the decisive questions remain open."
    out["research_uncertainty"] = _entry(
        "research_uncertainty", level, rationale,
        [f"missing_evidence:{counters.missing}",
         f"non_discriminating_runs:{counters.simulations_partial}"],
    )

    return {key: out[key] for key in ORDER}

