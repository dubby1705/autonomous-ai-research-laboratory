"""
AARL Discovery — Idea Status System
====================================
The five statuses AARL is allowed to assign to a research direction. The status
is **derived only from stored records** (provenance-linked knowledge items,
simulation outcomes, recorded contradictions). No status is ever assigned from
model prose, semantic similarity or a single blended "scientific score".

    EVIDENCE_BACKED  meaningful supporting evidence exists
    PROMISING        plausible mechanism + some supporting evidence
    SPECULATIVE      interesting possibility, evidence insufficient
    INCONCLUSIVE     current evidence cannot support or reject the idea
    FAILED_WEAK      contradictions, failed experiments, impossible assumptions

SPECULATIVE is a *valuable* outcome here: AARL exists to surface unusual
directions, and most genuinely new directions are speculative at discovery time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

STATUS_EVIDENCE_BACKED = "EVIDENCE_BACKED"
STATUS_PROMISING = "PROMISING"
STATUS_SPECULATIVE = "SPECULATIVE"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_FAILED_WEAK = "FAILED_WEAK"

STATUS_ORDER = (
    STATUS_EVIDENCE_BACKED,
    STATUS_PROMISING,
    STATUS_SPECULATIVE,
    STATUS_INCONCLUSIVE,
    STATUS_FAILED_WEAK,
)


def _meta(label, dot, css, meaning, does_not_mean) -> Dict[str, Any]:
    return {
        "label": label,
        "dot": dot,
        "css": css,
        "meaning": meaning,
        "does_not_mean": list(does_not_mean),
    }


#: Presentation + meaning of each status. ``does_not_mean`` exists so the UI can
#: state plainly what a status must never be read as.
STATUS_META: Dict[str, Dict[str, Any]] = {
    STATUS_EVIDENCE_BACKED: _meta(
        "EVIDENCE-BACKED", "#4ade80", "status-evidence",
        "Existing evidence provides meaningful support for this direction. The "
        "proposed combination or hypothesis may still require testing.",
        [
            "that the proposed combination has been tested as a combination",
            "that the direction is scientifically settled or 'proven'",
        ],
    ),
    STATUS_PROMISING: _meta(
        "PROMISING", "#60a5fa", "status-promising",
        "A plausible mechanism was identified and some supporting evidence or a "
        "supporting simulation exists, but substantial uncertainty remains.",
        [
            "that the mechanism has been validated outside a model",
            "that the gain will survive at real scale",
        ],
    ),
    STATUS_SPECULATIVE: _meta(
        "SPECULATIVE", "#c084fc", "status-speculative",
        "AARL identified an interesting research possibility, but current evidence "
        "is insufficient to establish that the proposed idea works. Speculative does "
        "NOT mean useless, fake or wrong.",
        [
            "'useless' — a speculative direction is a research opportunity",
            "'fake' — the mechanism is real; its transferred effect is not shown",
            "'wrong' — no contradiction has been established either",
        ],
    ),
    STATUS_INCONCLUSIVE: _meta(
        "INCONCLUSIVE", "#fbbf24", "status-inconclusive",
        "Current evidence does not clearly support or reject the hypothesis: signals "
        "are mixed, or the runs performed could not discriminate between the proposed "
        "mechanism and the baseline.",
        [
            "that the direction has been refuted",
            "that further work would be wasted",
        ],
    ),
    STATUS_FAILED_WEAK: _meta(
        "FAILED / WEAK", "#f87171", "status-failed",
        "The direction met strong contradictions, a failed experiment, an impossible "
        "assumption or clearly poor performance. It is kept as research memory, not "
        "deleted.",
        [
            "that the underlying mechanism is useless in every variant",
            "that the failure will not produce a better refined direction",
        ],
    ),
}

#: Rule ids, so every classification can be traced to the rule that fired.
RULE_DESCRIPTIONS: Dict[str, str] = {
    "R1_failed_simulation_no_support": (
        "A simulation of this direction did not succeed (failure status or violated "
        "constraint) and no simulation supported it."
    ),
    "R2_contradictions_outweigh_support": (
        "Recorded contradicting evidence outnumbers supporting evidence."
    ),
    "R3_mixed_simulation_outcomes": (
        "Some simulations of this direction succeeded and others failed: the current "
        "evidence does not determine whether the mechanism works."
    ),
    "R4_non_discriminating_run": (
        "A simulation ran but could not discriminate the mechanism from the baseline "
        "(partial result), and no direct evidence exists."
    ),
    "R5_supported_and_simulated": (
        "At least two independent sourced evidence items support the direction and a "
        "simulation supported it, with no recorded contradiction."
    ),
    "R6_partial_support_with_simulation": (
        "Some supporting evidence exists (direct or indirect) and a simulation "
        "supported it, with no recorded contradiction."
    ),
    "R7_conflicting_evidence": (
        "Both supporting and contradicting evidence are recorded, and no simulation "
        "result decides between them."
    ),
    "R8_no_executable_model": (
        "No executable model exists for this domain, so nothing was simulated. The "
        "direction is a possibility, not a result."
    ),
    "R9_insufficient_evidence_novel_combination": (
        "The direction is a novel combination with a described mechanism and no "
        "direct supporting evidence yet. Insufficient evidence — not refuted."
    ),
}


@dataclass
class EvidenceCounters:
    """Everything the classifier is allowed to look at (all from stored records)."""

    #: Knowledge items of claim_type 'sourced' linked to the direction's hypotheses.
    direct: int = 0
    #: Knowledge items linked at one remove (topic-matching, not provenance-linked).
    indirect: int = 0
    #: Knowledge items in the 'contradictions' section linked to the hypotheses.
    contradicting: int = 0
    #: Evidence the mechanism would need but that AARL does not have.
    missing: int = 0
    #: Simulations whose status was success / promising.
    simulations_supporting: int = 0
    #: Simulations whose status was failure or that violated a constraint.
    simulations_failed: int = 0
    #: Simulations whose status was partial (could not discriminate).
    simulations_partial: int = 0
    #: Simulations that could not run at all (no engine for the domain).
    simulations_unavailable: int = 0
    #: Human-provided real-world results.
    real_world_results: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direct_evidence": self.direct,
            "indirect_evidence": self.indirect,
            "contradicting_evidence": self.contradicting,
            "missing_evidence": self.missing,
            "simulations_supporting": self.simulations_supporting,
            "simulations_failed": self.simulations_failed,
            "simulations_partial": self.simulations_partial,
            "simulations_unavailable": self.simulations_unavailable,
            "real_world_results": self.real_world_results,
            "source_of_counters": (
                "Provenance-linked knowledge items and stored Simulation JSON "
                "records. Nothing else feeds this classification."
            ),
        }



def classify(counters: EvidenceCounters) -> Dict[str, Any]:
    """Assign one of the five statuses, with the rule and the numbers that fired."""
    d, ind, con = counters.direct, counters.indirect, counters.contradicting
    sup, fail = counters.simulations_supporting, counters.simulations_failed
    part, unav = counters.simulations_partial, counters.simulations_unavailable
    rwr = counters.real_world_results

    reasons: List[str] = []
    rule = "R9_insufficient_evidence_novel_combination"
    status = STATUS_SPECULATIVE

    if rwr >= 1 and fail == 0 and con == 0:
        # A human ran the real experiment: the strongest signal AARL accepts.
        status, rule = STATUS_EVIDENCE_BACKED, "R5_supported_and_simulated"
        reasons.append(f"{rwr} human-provided real-world result(s) are recorded.")
    elif fail >= 1 and sup == 0 and rwr == 0:
        status, rule = STATUS_FAILED_WEAK, "R1_failed_simulation_no_support"
        reasons.append(
            f"{fail} simulation(s) of this direction failed or violated a declared "
            "constraint, and none succeeded."
        )
    elif con >= 1 and con > d + ind:
        status, rule = STATUS_FAILED_WEAK, "R2_contradictions_outweigh_support"
        reasons.append(
            f"{con} contradicting evidence item(s) against {d + ind} supporting "
            "item(s)."
        )
    elif fail >= 1 and sup >= 1:
        status, rule = STATUS_INCONCLUSIVE, "R3_mixed_simulation_outcomes"
        reasons.append(
            f"{sup} simulation(s) supported the direction while {fail} failed."
        )
    elif part >= 1 and sup == 0 and d == 0 and rwr == 0:
        status, rule = STATUS_INCONCLUSIVE, "R4_non_discriminating_run"
        reasons.append(
            f"{part} simulation(s) ran but could not discriminate the mechanism from "
            "the baseline, and no direct evidence exists."
        )
    elif d >= 2 and con == 0 and (sup >= 1 or d >= 3):
        status, rule = STATUS_EVIDENCE_BACKED, "R5_supported_and_simulated"
        reasons.append(f"{d} directly linked sourced evidence item(s).")
        if sup:
            reasons.append(f"{sup} supporting simulation(s).")
    elif (d >= 1 or ind >= 2) and sup >= 1 and con == 0:
        status, rule = STATUS_PROMISING, "R6_partial_support_with_simulation"
        reasons.append(
            f"{d} direct and {ind} indirect supporting item(s), plus {sup} supporting "
            "simulation(s)."
        )
    elif d + ind >= 1 and con >= 1 and not (sup or fail):
        status, rule = STATUS_INCONCLUSIVE, "R7_conflicting_evidence"
        reasons.append(
            f"{d + ind} supporting item(s) and {con} contradicting item(s) are "
            "recorded; no simulation decides between them."
        )
    elif unav >= 1 and d == 0 and ind == 0:
        status, rule = STATUS_SPECULATIVE, "R8_no_executable_model"
        reasons.append(
            f"{unav} simulation(s) could not run: no executable model supports this "
            "domain. The mechanism is therefore untested, not refuted."
        )
    else:
        reasons.append(
            "Novel combination with a described mechanism and no directly linked "
            "supporting evidence yet."
        )

    reasons.append(
        "Classification uses recorded evidence and simulation records only — model "
        "prose and text similarity are never counted as evidence."
    )

    meta = STATUS_META[status]
    return {
        "status": status,
        "label": meta["label"],
        "dot": meta["dot"],
        "css": meta["css"],
        "meaning": meta["meaning"],
        "does_not_mean": list(meta["does_not_mean"]),
        "rule_id": rule,
        "rule_description": RULE_DESCRIPTIONS.get(rule, ""),
        "reasons": reasons,
        "counters": counters.to_dict(),
        "why_status_is_not_a_verdict": (
            "AARL statuses describe how much *stored* evidence currently exists for a "
            "direction. They are not a judgement that the idea is true or false."
        ),
        "what_would_change_it": what_would_change(status, counters),
    }



def what_would_change(status: str, counters: EvidenceCounters) -> List[str]:
    """Concrete next actions that would move this direction's status."""
    out: List[str] = []
    if counters.simulations_unavailable:
        out.append(
            "Build or register an executable model for this domain so the mechanism "
            "can be tested instead of merely described."
        )
    if not counters.simulations_supporting and not counters.simulations_unavailable:
        out.append(
            "Run the mechanism in a controlled simulation and record the result."
        )
    if counters.direct < 2:
        out.append(
            "Add at least two independent source-linked evidence items to the "
            "knowledge base (papers or researcher data)."
        )
    if counters.contradicting:
        out.append(
            f"Resolve the {counters.contradicting} recorded contradiction(s) "
            "experimentally before relying on either claim."
        )
    if status == STATUS_FAILED_WEAK:
        out.append(
            "Apply failure refinement: identify the bottleneck and generate a "
            "modified mechanism rather than discarding the idea."
        )
    if not out:
        out.append(
            "Independent replication outside the model, by a human researcher, is "
            "still required before any stronger wording is justified."
        )
    return out

