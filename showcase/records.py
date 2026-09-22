"""
AARL Showcase — Structured Hypothesis Record
===============================================
All research state is held in dataclasses (and persisted as JSON). Nothing is
recovered by parsing terminal output.

The full traceable record for every candidate hypothesis:

    hypothesis_id, title, description, research_question,
    reasoning, mechanism, assumptions,
    evaluation, tournament, experiment, analysis, refinement
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def to_dict(obj: Any) -> Any:
    """JSON-safe recursive serialisation of dataclasses / lists / dicts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _pop(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    return data.pop(key, default) if isinstance(data, dict) else default


# ---------------------------------------------------------------------------
# Evaluation (critical review / critic)
# ---------------------------------------------------------------------------
@dataclass
class Evaluation:
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    reasoning_score: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    reviewer_reasoning: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "Evaluation":
        d = d or {}
        return cls(
            novelty_score=float(d.get("novelty_score", 0.0) or 0.0),
            feasibility_score=float(d.get("feasibility_score", 0.0) or 0.0),
            reasoning_score=float(d.get("reasoning_score", 0.0) or 0.0),
            overall_score=float(d.get("overall_score", 0.0) or 0.0),
            passed=bool(d.get("passed", False)),
            reviewer_reasoning=str(d.get("reviewer_reasoning", "") or ""),
        )
# ---------------------------------------------------------------------------
# Experiment specification (designed by the LLM, validated before execution)
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    experiment_id: str = ""
    objective: str = ""
    baseline: Dict[str, Any] = field(default_factory=dict)
    proposed_method: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    controlled_variables: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    procedure: List[str] = field(default_factory=list)
    supported: bool = False  # whether it maps to a safe, executable experiment

    @classmethod
    def from_dict(cls, data: Any) -> "ExperimentSpec":
        data = data or {}
        return cls(
            experiment_id=str(data.get("experiment_id", "") or ""),
            objective=str(data.get("objective", "") or ""),
            baseline=dict(data.get("baseline", {}) or {}),
            proposed_method=str(data.get("proposed_method", "") or ""),
            parameters=dict(data.get("parameters", {}) or {}),
            controlled_variables=list(data.get("controlled_variables", []) or []),
            metrics=list(data.get("metrics", []) or []),
            expected_behavior=str(data.get("expected_behavior", "") or ""),
            procedure=list(data.get("procedure", []) or []),
            supported=bool(data.get("supported", False)),
        )


# ---------------------------------------------------------------------------
# Experiment results (real numerical values from executables)
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    status: str = "PENDING"  # SUCCESS / FAILED / UNSUPPORTED
    message: str = ""
    baseline: Dict[str, Any] = field(default_factory=dict)  # {mean_iters, final_loss, converged}
    proposed: Dict[str, Any] = field(default_factory=dict)
    improvement_pct: Optional[float] = None
    per_run_baseline: List[Dict[str, Any]] = field(default_factory=list)
    per_run_proposed: List[Dict[str, Any]] = field(default_factory=list)
    baseline_loss_curve: List[float] = field(default_factory=list)
    proposed_loss_curve: List[float] = field(default_factory=list)
    error: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "ExperimentResult":
        data = data or {}
        return cls(
            status=str(data.get("status", "PENDING") or "PENDING"),
            message=str(data.get("message", "") or ""),
            baseline=dict(data.get("baseline", {}) or {}),
            proposed=dict(data.get("proposed", {}) or {}),
            improvement_pct=data.get("improvement_pct", None),
            per_run_baseline=list(data.get("per_run_baseline", []) or []),
            per_run_proposed=list(data.get("per_run_proposed", []) or []),
            baseline_loss_curve=[float(x) for x in (data.get("baseline_loss_curve") or [])],
            proposed_loss_curve=[float(x) for x in (data.get("proposed_loss_curve") or [])],
            error=str(data.get("error", "") or ""),
        )


# ---------------------------------------------------------------------------
# LLM analysis (deep research analysis for a hypothesis)
# ---------------------------------------------------------------------------
@dataclass
class Analysis:
    interpretation: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    evidence_strength: str = ""  # none | weak | moderate | strong
    conclusion: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "Analysis":
        data = data or {}
        return cls(
            interpretation=str(data.get("interpretation", "") or ""),
            strengths=list(data.get("strengths", []) or []),
            weaknesses=list(data.get("weaknesses", []) or []),
            limitations=list(data.get("limitations", []) or []),
            improvements=list(data.get("improvements", []) or []),
            evidence_strength=str(data.get("evidence_strength", "") or ""),
            conclusion=str(data.get("conclusion", "") or ""),
            source=str(data.get("source", "") or ""),
        )


# ---------------------------------------------------------------------------
# Refined hypothesis (result of hypothesis refinement)
# ---------------------------------------------------------------------------
@dataclass
class RefinedHypothesis:
    hypothesis_id: str = ""  # e.g. "H1"
    refined_id: str = ""  # e.g. "H1-R1"
    title: str = ""
    description: str = ""
    mechanism: str = ""
    change_summary: str = ""
    expected_improvement: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "RefinedHypothesis":
        data = data or {}
        return cls(
            hypothesis_id=str(data.get("hypothesis_id", "") or ""),
            refined_id=str(data.get("refined_id", "") or ""),
            title=str(data.get("title", "") or ""),
            description=str(data.get("description", "") or ""),
            mechanism=str(data.get("mechanism", "") or ""),
            change_summary=str(data.get("change_summary", "") or ""),
            expected_improvement=str(data.get("expected_improvement", "") or ""),
            source=str(data.get("source", "") or ""),
        )
@dataclass
class Hypothesis:
    hypothesis_id: str = ""
    title: str = ""
    description: str = ""
    research_question: str = ""
    reasoning: str = ""
    mechanism: str = ""
    assumptions: List[str] = field(default_factory=list)
    source: str = "deterministic"  # 'llm' | 'deterministic'

    # Stage records
    evaluation: Evaluation = field(default_factory=Evaluation)
    tournament_status: str = "pending"  # pending | passed | rejected
    rejection_reason: str = ""
    experiment_spec: ExperimentSpec = field(default_factory=ExperimentSpec)
    experiment: ExperimentResult = field(default_factory=ExperimentResult)
    analysis: Analysis = field(default_factory=Analysis)
    refinements: List[Dict[str, Any]] = field(default_factory=list)

    # ---- convenience -------------------------------------------------------
    @property
    def passed(self) -> bool:
        return self.tournament_status == "passed"

    @property
    def experiment_succeeded(self) -> bool:
        return self.experiment.status == "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "description": self.description,
            "research_question": self.research_question,
            "reasoning": self.reasoning,
            "mechanism": self.mechanism,
            "assumptions": list(self.assumptions),
            "source": self.source,
            "evaluation": to_dict(self.evaluation),
            "tournament_status": self.tournament_status,
            "rejection_reason": self.rejection_reason,
            "experiment_spec": to_dict(self.experiment_spec),
            "experiment": to_dict(self.experiment),
            "analysis": to_dict(self.analysis),
            "refinements": [to_dict(r) for r in self.refinements],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Hypothesis":
        data = data or {}
        h = cls(
            hypothesis_id=str(data.get("hypothesis_id", "") or ""),
            title=str(data.get("title", "") or ""),
            description=str(data.get("description", "") or ""),
            research_question=str(data.get("research_question", "") or ""),
            reasoning=str(data.get("reasoning", "") or ""),
            mechanism=str(data.get("mechanism", "") or ""),
            assumptions=list(data.get("assumptions", []) or []),
            source=str(data.get("source", "deterministic") or "deterministic"),
        )
        ev = data.get("evaluation") or {}
        h.evaluation = Evaluation.from_dict(ev) if isinstance(ev, dict) else Evaluation()
        h.tournament_status = str(data.get("tournament_status", "pending") or "pending")
        h.rejection_reason = str(data.get("rejection_reason", "") or "")
        spec = data.get("experiment_spec") or {}
        h.experiment_spec = ExperimentSpec.from_dict(spec) if isinstance(spec, dict) else ExperimentSpec()
        exp = data.get("experiment") or {}
        h.experiment = ExperimentResult.from_dict(exp) if isinstance(exp, dict) else ExperimentResult()
        ana = data.get("analysis") or {}
        h.analysis = Analysis.from_dict(ana) if isinstance(ana, dict) else Analysis()
        refinements = data.get("refinements") or []
        h.refinements = list(refinements) if isinstance(refinements, list) else []
        return h


# ---------------------------------------------------------------------------
# The complete research run (research history)
# ---------------------------------------------------------------------------
@dataclass
class ResearchRun:
    run_id: str = ""
    research_question: str = ""
    domain: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    findings: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    llm_available: bool = False
    llm_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "research_question": self.research_question,
            "domain": self.domain,
            "config": to_dict(self.config),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "findings": to_dict(self.findings),
            "created_at": self.created_at,
            "llm_available": self.llm_available,
            "llm_model": self.llm_model,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ResearchRun":
        data = data or {}
        run = cls(
            run_id=str(data.get("run_id", "") or ""),
            research_question=str(data.get("research_question", "") or ""),
            domain=str(data.get("domain", "") or ""),
            config=dict(data.get("config", {}) or {}),
            created_at=str(data.get("created_at", "") or ""),
            llm_available=bool(data.get("llm_available", False)),
            llm_model=str(data.get("llm_model", "") or ""),
            findings=dict(data.get("findings", {}) or {}),
        )
        hyps = data.get("hypotheses") or []
        run.hypotheses = [Hypothesis.from_dict(h) for h in hyps if isinstance(h, dict)]
        return run