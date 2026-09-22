"""
AARL Showcase — Deep Research Analysis & Hypothesis Refinement
===============================================================
For every hypothesis with real experimental data, the LLM generates a deep
analysis that is strictly grounded in the measured numbers. The LLM may never
claim an improvement that is not supported by the data.

Promising hypotheses are then refined into H1-R1, H1-R2 ... -> a traceable
research history where AARL demonstrably learns from evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import ResearchConfig
from .llm_service import GroqService
from .records import Analysis, ExperimentResult, Hypothesis

# Parallel processing layer (shared with the main AARL pipeline), imported
# defensively so the showcase can still run on its own.
try:
    from Engine.Question_Engine.ParallelExecutor import llm_workers, parallel_map
    PARALLEL_AVAILABLE = True
except Exception:  # pragma: no cover - standalone deployment
    llm_workers = None
    parallel_map = None
    PARALLEL_AVAILABLE = False

log = logging.getLogger("aarl.analysis")


def _evidence_from(result: ExperimentResult) -> str:
    """A compact, factual summary of measurements for the LLM."""
    b = result.baseline
    p = result.proposed
    return "\n".join([
        f"Baseline  -> converged_rate={b.get('convergence_rate')}, "
        f"mean_iterations={b.get('mean_iterations')}, "
        f"mean_final_loss={b.get('mean_final_loss')}",
        f"Proposed  -> converged_rate={p.get('convergence_rate')}, "
        f"mean_iterations={p.get('mean_iterations')}, "
        f"mean_final_loss={p.get('mean_final_loss')}",
        f"Improvement (mean iterations) = {result.improvement_pct} %",
    ])


def deterministic_analysis(h: Hypothesis) -> Analysis:
    """Evidence-aware analysis without an LLM (clearly labelled)."""
    exp = h.experiment
    improved = exp.improvement_pct
    if improved is not None and improved > 0:
        conclusion = (
            f"Measured improvement of {improved:.1f}% mean iterations supports "
            f"this hypothesis on the controlled problem."
        )
        evidence = "moderate"
        strengths = ["Real measured reduction in iterations"]
    elif improved is not None:
        conclusion = (
            f"Measured {improved:.1f}% (i.e. a slowdown): evidence does not "
            "support a benefit on the controlled problem."
        )
        evidence = "weak"
        strengths = []
    else:
        conclusion = "No clean comparative claim possible: convergence was inconsistent."
        evidence = "none"
        strengths = []
    return Analysis(
        interpretation=(
            f"{h.mechanism} was tested against SGD on the controlled "
            "ill-conditioned quadratic problem with a fixed iteration budget."
        ),
        strengths=strengths,
        weaknesses=[
            "Single controlled problem",
            "Small number of runs",
            "Deterministic (no stochastic noise)",
        ],
        limitations=[
            "Only one objective family was tested",
            "No real dataset / model",
            "Fixed iteration budget caps very slow methods",
        ],
        improvements=[
            "Test on additional condition numbers",
            "Add stochastic gradients",
            "Vary dimensionality",
        ],
        evidence_strength=evidence,
        conclusion=conclusion,
        source="deterministic",
    )


def _analyze_job(payload):
    """
    Worker body: analyse ONE hypothesis (LLM if available, else deterministic).

    Module-level so it is safe to hand to a worker pool, and it never raises —
    an LLM failure degrades to the evidence-based deterministic analysis.
    """
    h, svc = payload
    if h.experiment.status != "SUCCESS":
        return Analysis(
            interpretation="No experiment results are available for this hypothesis.",
            conclusion="Experiment did not produce measurable data; analysis not possible.",
            evidence_strength="none",
            source="deterministic",
        )
    if svc.available:
        try:
            return _llm_analysis(h, svc)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "LLM analysis failed for %s (%s); using evidence-based fallback",
                h.hypothesis_id, exc,
            )
    return deterministic_analysis(h)


def analyze_hypotheses(hyps: List[Hypothesis], cfg: ResearchConfig, svc: GroqService) -> None:
    """
    Generate deep analysis for all hypotheses with real experimental data.

    PARALLEL: every hypothesis is analysed independently, so the LLM calls are
    issued concurrently (capped at the provider-friendly LLM worker limit)
    instead of one round-trip at a time. Results are written back in the
    original hypothesis order, so the run record is unchanged.
    """
    if not hyps:
        return

    if PARALLEL_AVAILABLE and getattr(cfg, "parallel", True) and len(hyps) > 1:
        cap = getattr(cfg, "max_workers", 0) or (llm_workers() if llm_workers else 0)
        analyses = parallel_map(
            _analyze_job,
            [(h, svc) for h in hyps],
            max_workers=cap,
            label="analysis",
        )
        for h, analysis in zip(hyps, analyses):
            h.analysis = (
                analysis if isinstance(analysis, Analysis)
                else deterministic_analysis(h)
            )
        return

    for h in hyps:
        h.analysis = _analyze_job((h, svc))


def _llm_analysis(h: Hypothesis, svc: GroqService) -> Analysis:
    system = (
        "You are the analysis engine of an autonomous AI research laboratory. "
        "You are given a hypothesis AND the real measured evidence from a "
        "controlled experiment. Ground every claim in the numbers. "
        "Never claim an improvement the data does not support. "
        "Return JSON only. No markdown. "
        'JSON: {"interpretation": str, "strengths": [str], "weaknesses": [str], '
        '"limitations": [str], "improvements": [str], '
        '"evidence_strength": "none|weak|moderate|strong", "conclusion": str}'
    )
    user = (
        f"HYPOTHESIS: {h.title}\n"
        f"Reasoning: {h.reasoning}\nMechanism: {h.mechanism}\n\n"
        f"EXPERIMENT: {h.experiment_spec.objective}\n"
        f"Proposed optimiser: {h.experiment_spec.proposed_method}\n\n"
        f"MEASURED EVIDENCE:\n{_evidence_from(h.experiment)}\n\n"
        "1. Does the data support the hypothesis?\n"
        "2. How strong is the evidence?\n"
        "3. What are the limitations?\n"
        "4. What should be tried next?"
    )
    data = svc.complete_json(
        system,
        user,
        required_fields=["interpretation", "evidence_strength", "conclusion"],
    )
    return Analysis(
        interpretation=str(data.get("interpretation", "")),
        strengths=[str(x) for x in (data.get("strengths") or [])],
        weaknesses=[str(x) for x in (data.get("weaknesses") or [])],
        limitations=[str(x) for x in (data.get("limitations") or [])],
        improvements=[str(x) for x in (data.get("improvements") or [])],
        evidence_strength=str(data.get("evidence_strength", "weak")),
        conclusion=str(data.get("conclusion", "")),
        source="llm",
    )
# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------
def refine_hypotheses(hyps: List[Hypothesis], cfg: ResearchConfig, svc: GroqService) -> None:
    """Refine the most promising hypotheses into HX-R1, HX-R2, ..."""
    passing = [
        h
        for h in hyps
        if h.experiment.status == "SUCCESS"
        and h.experiment.improvement_pct is not None
        and h.experiment.improvement_pct > 0
    ]
    passing.sort(key=lambda h: (h.experiment.improvement_pct or 0.0), reverse=True)

    for rank, h in enumerate(passing[:3], start=1):
        if svc.available:
            try:
                refined = _llm_refinement(h, rank, svc)
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM refinement failed for %s (%s)", h.hypothesis_id, exc)
                refined = None
        else:
            refined = _deterministic_refinement(h, rank)
        if refined is not None:
            h.refinements.append(refined)


def _llm_refinement(h: Hypothesis, rank: int, svc: GroqService) -> Optional[Dict[str, Any]]:
    system = (
        "You refine hypotheses for an autonomous AI research laboratory. "
        "Produce a single improved hypothesis that builds on the evidence. "
        "Return JSON only: "
        '{"title": str, "description": str, "mechanism": str, '
        '"change_summary": str, "expected_improvement": str}'
    )
    user = (
        f"Original: {h.title} (improvement {h.improvement_pct}% on measured data)\n"
        f"Analysis conclusion: {h.analysis.conclusion}\n"
        f"Limitations: {', '.join(h.analysis.limitations)}\n"
        "Derive one mature refinement that addresses a limitation."
    )
    data = svc.complete_json(system, user, required_fields=["title", "mechanism"])
    return {
        "hypothesis_id": h.hypothesis_id,
        "refined_id": f"{h.hypothesis_id}-R{rank}",
        "title": str(data.get("title", "")),
        "description": str(data.get("description", "")),
        "mechanism": str(data.get("mechanism", "")),
        "change_summary": str(data.get("change_summary", "")),
        "expected_improvement": str(data.get("expected_improvement", "")),
        "source": "llm",
    }


def _deterministic_refinement(h: Hypothesis, rank: int) -> Optional[Dict[str, Any]]:
    return {
        "hypothesis_id": h.hypothesis_id,
        "refined_id": f"{h.hypothesis_id}-R{rank}",
        "title": f"{h.title} (refined, evidence-tuned)",
        "description": h.description,
        "mechanism": h.mechanism,
        "change_summary": (
            f"Refinement informed by measured evidence "
            f"(improvement {h.experiment.improvement_pct:.1f}%)."
        ),
        "expected_improvement": "To be measured in future experiments.",
        "source": "deterministic",
    }