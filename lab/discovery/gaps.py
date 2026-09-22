"""
AARL Discovery - Research Gap Generation
========================================
Gaps are derived from the *selected hypothesis and its records*, never from a
fixed template attached to the topic. A generic gap such as "optimal model
architecture" is not produced here; AARL asks instead what specifically is
unknown about this mechanism, at which parameter it might stop working, and what
measurement would decide the question.

Every gap records ``derived_from`` - the concrete record ids it came from - so a
reader can inspect the evidence behind the question.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evidence import NEVER_VALIDATED, classify_simulation


def _gap(kind, question, why_it_matters, derived_from, how_to_start) -> Dict[str, Any]:
    return {
        "kind": kind,
        "question": question,
        "why_it_matters": why_it_matters,
        "derived_from": [d for d in derived_from if d],
        "how_to_start": how_to_start,
        "status": "OPEN",
    }


def research_gaps(
    facet,
    concept,
    subject: str,
    hypothesis: Optional[Dict[str, Any]] = None,
    experiments: Optional[List[Dict[str, Any]]] = None,
    failures: Optional[List[Dict[str, Any]]] = None,
    panel: Optional[Dict[str, Any]] = None,
    lessons: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Generate the open questions that actually matter for this direction."""
    hypothesis = hypothesis or {}
    experiments = list(experiments or [])
    failures = list(failures or [])
    panel = panel or {}
    lessons = list(lessons or [])
    gaps: List[Dict[str, Any]] = []
    metric = facet.metric
    hyp_id = str(hypothesis.get("hypothesis_id", ""))

    # ---- 1. model-validity gap (assumption-based results) ------------------
    for sim in experiments:
        info = classify_simulation(sim)
        if info["result_class"] == "ASSUMPTION_BASED_RESULT":
            keys = list(info.get("assumed_modifiers", {}).keys())
            gaps.append(_gap(
                "model_validity",
                "The only result available for this direction is an ASSUMPTION-BASED "
                "RESULT driven by assumed multipliers%s. What measured data would "
                "replace those assumptions?" % (
                    (" (" + ", ".join(keys) + ")") if keys else ""
                ),
                "Until the multipliers are replaced by measurements, no claim about "
                "the mechanism's effect on %s can be supported." % metric,
                [info.get("experiment_id", ""), hyp_id],
                "Instrument a representative workload and measure the multipliers "
                "directly, then re-run the same comparison.",
            ))
            break

    # ---- 2. threshold gap --------------------------------------------------
    if failures:
        for failure in failures[:2]:
            gaps.append(_gap(
                "threshold",
                "At what value of the driving parameter does this mechanism stop "
                "paying off? A failure was recorded as: %s" % (
                    str(failure.get("observed_behavior")
                        or failure.get("cause") or failure.get("failure_type", ""))[:220]
                ),
                "A mechanism that works only inside an unstated range is a trap for "
                "whoever implements it next.",
                [str(failure.get("failure_id", "")), str(failure.get("experiment_id", "")),
                 hyp_id],
                "Sweep the suspected parameter across its plausible range and record "
                "where %s crosses over." % metric,
            ))
    elif experiments:
        gaps.append(_gap(
            "threshold",
            "How far does the recorded effect on %s extend before it reverses? "
            "Only %d configuration(s) were tested." % (metric, len(experiments)),
            "A single operating point cannot show where the benefit ends.",
            [str(e.get("experiment_id", "")) for e in experiments[:4]],
            "Sweep the parameter that the mechanism acts on and record %s at each "
            "point." % metric,
        ))

    # ---- 3. scaling gap ---------------------------------------------------
    scales: List[str] = []
    for sim in experiments:
        params = sim.get("parameters") or {}
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (int, float)) and key not in ("domain",):
                    scales.append("%s=%s" % (key, value))
    gaps.append(_gap(
        "scaling",
        "How does %s scale as the number of %s participants grows beyond the tested "
        "range%s?" % (metric, facet.label.lower(),
                      (" (" + ", ".join(scales[:3]) + ")") if scales else ""),
        "Coordination mechanisms usually show their real cost only once the "
        "participant count grows.",
        [str(e.get("experiment_id", "")) for e in experiments[:4]],
        "Repeat the comparison at several participant counts and fit the curve.",
    ))

    # ---- 4. robustness / uncertainty gaps ----------------------------------
    for uncertainty in list(concept.uncertainties)[:3]:
        gaps.append(_gap(
            "robustness",
            "%s - and how does that depend on the workload being uniform?"
            % uncertainty,
            "An unquantified uncertainty is the likeliest reason a plausible "
            "mechanism would fail in practice.",
            ["library:%s" % concept.concept_id, hyp_id],
            "Construct two workload classes (uniform and heterogeneous) and compare "
            "%s between them." % metric,
        ))

    # ---- 5. contradiction gaps --------------------------------------------
    for row in (panel.get("contradicting") or [])[:2]:
        gaps.append(_gap(
            "contradiction",
            "Two records disagree on a related claim. Which one holds under this "
            "direction's conditions? Conflicting statement: %s"
            % str(row.get("statement", ""))[:220],
            "Both claims are preserved in AARL's knowledge base; without resolution "
            "either could be used to argue in opposite directions.",
            [str(row.get("evidence_id", "")), hyp_id],
            "Design a measurement that distinguishes the two statements directly.",
        ))

    # ---- 6. evidence gaps --------------------------------------------------
    for row in (panel.get("missing") or [])[:3]:
        if row.get("kind") != "mechanism_step_unsupported":
            continue
        gaps.append(_gap(
            "evidence",
            "The mechanism stage '%s' has no source-linked support. What would "
            "demonstrate it?" % str(row.get("statement", ""))[:160],
            "An unsupported stage is where the proposed mechanism can break without "
            "anyone noticing.",
            [str(row.get("gap_id", "")), hyp_id],
            str(row.get("what_is_needed", "")),
        ))

    # ---- 7. non-discrimination gap -----------------------------------------
    for sim in experiments:
        status = str(sim.get("status", "")).lower()
        if status in ("partial", "simulation_unavailable"):
            gaps.append(_gap(
                "experiment_design",
                "The stored run could not discriminate the mechanism from the "
                "baseline (status '%s'). What experiment design would?" % status,
                "A run that cannot distinguish the mechanism cannot support or "
                "reject it.",
                [str(sim.get("experiment_id", "")), hyp_id],
                "Choose a metric the mechanism should move and a baseline it should "
                "move away from.",
            ))
            break

    # ---- 8. failure-refinement gap ----------------------------------------
    for lesson in lessons[:2]:
        gaps.append(_gap(
            "refinement",
            "The recorded lesson is: %s What modified mechanism would avoid it?"
            % str(lesson.get("lesson", ""))[:220],
            "AARL keeps failures so that the next direction is an improvement rather "
            "than a repeat.",
            [str(lesson.get("lesson_id", "")), str(lesson.get("failure_id", ""))],
            str((lesson.get("recommended_changes") or ["Modify the mechanism."])[0]),
        ))

    for index, gap in enumerate(gaps):
        gap["gap_id"] = "GAP-%02d" % (index + 1)
    gaps.append(_gap(
        "first_experiment",
        "What is the smallest experiment that would change this direction's status?",
        "AARL's statuses are updated only by records, so the fastest route to a "
        "stronger status is a deciding measurement.",
        [hyp_id] + [str(e.get("experiment_id", "")) for e in experiments[:2]],
        "See 'Suggested first experiment' in the research opportunity report.",
    ))
    return gaps