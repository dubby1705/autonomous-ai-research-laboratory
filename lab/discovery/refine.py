"""
AARL Discovery - Idea Refinement from Failure
=============================================
A failed direction is never deleted. It is turned into an input:

    FAILED IDEA -> failure analysis -> identify bottleneck -> modify mechanism
                -> refined hypothesis -> run again

Refinement operators are named, so the transformation is inspectable rather than
an unexplained new title. Each refined direction records the exact failure and
lesson records it was derived from.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

OPERATORS: Dict[str, Dict[str, Any]] = {
    "hybridize": {
        "label": "HYBRIDIZE",
        "rationale": (
            "Keep the local part of the mechanism but add a lightweight upper layer "
            "that bounds the coordination cost that caused the failure."
        ),
        "title_pattern": "Hybrid %s",
        "mechanism_edit": (
            "Split the mechanism: local decisions stay decentralized, while a thin "
            "global layer bounds drift and caps the signalling rate."
        ),
    },
    "bound_the_cost": {
        "label": "BOUND THE COST",
        "rationale": (
            "Accept that the failing quantity grows, and make its ceiling an explicit "
            "design constraint instead of an unstated hope."
        ),
        "title_pattern": "Budget-Bounded %s",
        "mechanism_edit": (
            "Add an explicit budget for the failing quantity, escalate only while the "
            "budget is unspent, and fall back to the conventional path when it is."
        ),
    },
    "re_scope": {
        "label": "RE-SCOPE",
        "rationale": (
            "Apply the mechanism only to the part of the workload where its premise "
            "actually holds, instead of uniformly."
        ),
        "title_pattern": "Selectively Applied %s",
        "mechanism_edit": (
            "Add a cheap classifier that routes only the eligible work through the "
            "mechanism and leaves the rest on the conventional path."
        ),
    },
    "measure_first": {
        "label": "MEASURE FIRST",
        "rationale": (
            "The run could not decide between the mechanism and the baseline, so the "
            "refinement is a deciding experiment rather than a new mechanism."
        ),
        "title_pattern": "Discriminating Test for %s",
        "mechanism_edit": (
            "Target a metric the mechanism must move and a workload where the "
            "baseline is demonstrably not already optimal."
        ),
    },
}

#: Failure type -> preferred operator, with the reason recorded.
OPERATOR_BY_FAILURE: Dict[str, str] = {
    "constraint_violation": "bound_the_cost",
    "unexpected_result": "re_scope",
    "inconclusive_result": "measure_first",
    "hypothesis_not_supported": "hybridize",
    "simulation_error": "hybridize",
}


def choose_operator(failure: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the refinement operator for a recorded failure (deterministic)."""
    failure_type = str(failure.get("failure_type", "") or "hypothesis_not_supported")
    key = OPERATOR_BY_FAILURE.get(failure_type, "hybridize")
    operator = dict(OPERATORS[key])
    operator["operator_id"] = key
    operator["chosen_because"] = (
        "Failure type '%s' is refined with the '%s' operator." % (failure_type, key)
    )
    return operator


def build_refinement(
    direction: Dict[str, Any],
    failure: Dict[str, Any],
    lesson: Optional[Dict[str, Any]] = None,
    operator: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Draft a refined direction derived from a failure record."""
    operator = operator or choose_operator(failure)
    lesson = lesson or {}
    parent_title = str(direction.get("title") or direction.get("direction_id") or "idea")
    bottleneck = str(
        failure.get("failed_assumption") or failure.get("cause")
        or failure.get("observed_behavior") or failure.get("failure_type", "")
    )
    changes = list(failure.get("recommended_changes") or [])
    if not changes:
        changes = [operator["mechanism_edit"]]
    title = str(operator["title_pattern"]) % parent_title
    return {
        "title": title,
        "refinement_operator": operator["operator_id"],
        "refinement_operator_label": operator["label"],
        "refinement_rationale": operator["rationale"],
        "refinement_chosen_because": operator["chosen_because"],
        "mechanism_edit": operator["mechanism_edit"],
        "parent_direction_id": direction.get("direction_id", ""),
        "derived_from_failure": failure.get("failure_id", ""),
        "derived_from_lesson": lesson.get("lesson_id", ""),
        "derived_from_experiment": failure.get("experiment_id", ""),
        "bottleneck": bottleneck,
        "core_question": (
            "If the mechanism is modified as follows - %s - does it recover its "
            "advantage without recreating the original bottleneck: %s?"
            % (operator["mechanism_edit"], bottleneck or "the recorded failure")
        ),
        "what_changed": changes,
        "what_stays_same": (
            "The research problem and the cross-domain inspiration are unchanged; "
            "only the mechanism is modified."
        ),
        "failure_memory_note": (
            "This direction only exists because %s is stored in AARL's failure "
            "memory. The same failure will not be re-generated silently."
            % str(failure.get("failure_id", "a previous failure"))
        ),
    }