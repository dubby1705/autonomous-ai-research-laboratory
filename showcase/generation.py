"""
AARL Showcase — Research Planning Layer
========================================
Hypothesis generation, critical evaluation, tournament selection and experiment
design. The LLM is the *reasoning* layer; when it is unavailable the stage uses
a clearly-labelled deterministic fallback (`source='deterministic'`).

No experimental numbers are ever produced here — experiments always run in the
real simulation engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import optimizers as opt
from .config import ResearchConfig
from .records import Evaluation, ExperimentResult, ExperimentSpec, Hypothesis
from .llm_service import GroqService

# Parallel processing layer (shared with the main AARL pipeline).
# Imported defensively so the showcase remains runnable if it is ever copied
# out of the repository on its own.
try:
    from Engine.Question_Engine.ParallelExecutor import (
        describe_parallelism,
        parallel_starmap,
    )
    PARALLEL_AVAILABLE = True
except Exception:  # pragma: no cover - standalone/showcase-only deployment
    describe_parallelism = None
    parallel_starmap = None
    PARALLEL_AVAILABLE = False

log = logging.getLogger("aarl.plan")

# ---------------------------------------------------------------------------
# Curated deterministic pool — used ONLY when no LLM is available.
# Each entry maps to a safe, executable optimiser via "method_hint".
# ---------------------------------------------------------------------------
CURATED_HYPOTHESES: List[Dict[str, Any]] = [
    {
        "title": "Nesterov look-ahead momentum",
        "description": "Add Nesterov accelerated momentum: evaluate the gradient at the momentum look-ahead point to damp oscillation along steep curvature.",
        "research_question": "Does Nesterov look-ahead momentum improve gradient descent convergence on an ill-conditioned problem?",
        "reasoning": "Plain SGD zig-zags when curvature is imbalanced; look-ahead momentum corrects the step before overshoot.",
        "mechanism": "Take a momentum step, then compute the gradient at the look-ahead position and combine both contributions.",
        "assumptions": ["Smooth objective", "Noise-free gradients"],
        "method_hint": "nesterov",
    },
    {
        "title": "Adaptive per-coordinate scaling (RMSProp)",
        "description": "Normalise each gradient component by its RMS history so step sizes match local curvature direction-by-direction.",
        "research_question": "Does per-coordinate gradient scaling speed up convergence on an ill-conditioned problem?",
        "reasoning": "A single global step size is a poor match for anisotropic curvature; scaling each coordinate alone mitigates ill-conditioning.",
        "mechanism": "Maintain a moving average of squared gradients and scale the gradient by the inverse RMS plus epsilon.",
        "assumptions": ["Gradient statistics are locally stationary"],
        "method_hint": "rmsprop",
    },
    {
        "title": "Adam combines momentum with adaptive scaling",
        "description": "Adam keeps both first- and second-moment estimates and rescales each step, combining momentum and per-coordinate scaling.",
        "research_question": "Does Adam converge faster than plain SGD on a controlled quadratic problem?",
        "reasoning": "Combining momentum and adaptive scaling often yields fast and stable convergence across curvature regimes.",
        "mechanism": "Compute bias-corrected first and second moment estimates and scale the update by their ratio.",
        "assumptions": ["Smooth gradients", "Bounded variance"],
        "method_hint": "adam",
    },
    {
        "title": "Classic momentum with accumulated velocity",
        "description": "Accumulate velocity over past gradients so the optimiser rolls through shallow directions instead of oscillating across steep ones.",
        "research_question": "Does classic momentum reduce the number of iterations needed to converge?",
        "reasoning": "Momentum accumulates consistent gradient signal and damps high-frequency oscillation.",
        "mechanism": "Maintain velocity as an exponential moving average of gradients; step along the velocity.",
        "assumptions": ["Objective is smooth"],
        "method_hint": "momentum",
    },
    {
        "title": "Cosine learning-rate schedule",
        "description": "Decay the learning rate smoothly from a high value to near zero along a cosine curve, preventing late-stage oscillation.",
        "research_question": "Does a cosine schedule improve final convergence for plain SGD?",
        "reasoning": "An aggressive early rate with a stable late rate can balance rapid progress and precise convergence.",
        "mechanism": "Scale the base learning rate by the cosine of the elapsed training fraction.",
        "assumptions": ["Iteration budget known in advance"],
        "method_hint": "sgd_cosine",
    },
    {
        "title": "SGD with cosine warm restarts",
        "description": "Repeatedly anneal the learning rate down and back up in cycles so the iterate periodically explores larger steps.",
        "research_question": "Does cyclical strategies reduce loss more than a monotone schedule?",
        "reasoning": "Warm restarts periodically enlarge steps, helping the iterates escape poor basins and settle near the optimum.",
        "mechanism": "The learning rate cycles between high and low values following a cosine curve with a restart period.",
        "assumptions": ["The objective has multiple basins"],
        "method_hint": "sgd_warm_restart",
    },
    {
        "title": "AdaGrad adaptive sum-of-squares scaling",
        "description": "Scale the per-coordinate learning rate by the inverse square root of accumulated squared gradients.",
        "research_question": "Does AdaGrad's automatic step-size adaptation accelerate early convergence?",
        "reasoning": "Adaptive per-direction step sizes behave well when curvature varies widely across coordinates.",
        "mechanism": "Accumulate squared gradients over time and normalise each coordinate step by the accumulated sum.",
        "assumptions": ["Short horizon keeps step sizes generous"],
        "method_hint": "adagrad",
    },
    {
        "title": "AMSGrad running maximum second moment",
        "description": "AMSGrad keeps the running maximum of second-moment estimates instead of an exponential average, avoiding premature step-size decay.",
        "research_question": "Does a running maximum of the second moment stabilise Adam-style optimisation?",
        "reasoning": "Adam can aggressively shrink step sizes; tracking the maximum observed scale prevents stalls.",
        "mechanism": "Maintain max of second-moment estimates and use it as the scaling denominator.",
        "assumptions": ["Smooth gradients"],
        "method_hint": "amsgrad",
    },
]


# ---------------------------------------------------------------------------
# Deterministic planners (used when LLM is unavailable)
# ---------------------------------------------------------------------------
def deterministic_hypotheses(research_question: str, cfg: ResearchConfig) -> List[Hypothesis]:
    pool = CURATED_HYPOTHESES[: min(cfg.num_hypotheses, len(CURATED_HYPOTHESES))]
    hyps: List[Hypothesis] = []
    for i, item in enumerate(pool, start=1):
        hyps.append(
            Hypothesis(
                hypothesis_id=f"H{i}",
                title=item["title"],
                description=item["description"],
                research_question=item["research_question"],
                reasoning=item["reasoning"],
                mechanism=item["mechanism"],
                assumptions=list(item.get("assumptions", [])),
                source="deterministic",
            )
        )
    return hyps
# ---------------------------------------------------------------------------
# Deterministic critique / tournament / experiment design
# ---------------------------------------------------------------------------
def _deterministic_evaluation(h: Hypothesis) -> Evaluation:
    """Deterministic, rule-based critical evaluation."""
    text = (h.title + " " + h.description + " " + h.mechanism).lower()
    novelty = 60.0
    if any(k in text for k in ("adaptive", "nesterov", "adam", "amsgrad", "restart")):
        novelty += 15
    if any(k in text for k in ("momentum", "schedule", "scaling")):
        novelty += 10
    feasibility = 85.0 if h.mechanism else 60.0
    if method_hint(h) is None:
        feasibility -= 20.0
    reasoning_score = 70.0 + (10.0 if len(h.reasoning) > 60 else 0.0)
    overall = round(0.4 * novelty + 0.3 * feasibility + 0.3 * reasoning_score, 1)
    return Evaluation(
        novelty_score=round(novelty, 1),
        feasibility_score=round(feasibility, 1),
        reasoning_score=round(reasoning_score, 1),
        overall_score=overall,
        reviewer_reasoning=(
            "Deterministic heuristic: mechanism is plausible and maps to a safe, "
            "executable optimiser" if feasibility >= 70 else
            "Deterministic heuristic: weak mechanism mapping to a testable optimiser"
        ),
    )
def method_hint(h: Hypothesis) -> Optional[str]:
    """Return the optimiser name a hypothesis most plausibly maps to."""
    text = (h.title + " " + h.description + " " + h.mechanism).lower()
    hints = [
        ("nesterov", "nesterov"),
        ("adam", "adam"),
        ("amsgrad", "amsgrad"),
        ("rmsprop", "rmsprop"),
        ("adagrad", "adagrad"),
        ("warm restart", "sgd_warm_restart"),
        ("cosine", "sgd_cosine"),
        ("momentum", "momentum"),
        ("schedule", "sgd_cosine"),
        ("adaptive", "rmsprop"),
    ]
    for keyword, method in hints:
        if keyword in text:
            return method
    return None


def _deterministic_experiment_spec(h: Hypothesis, cfg: ResearchConfig) -> ExperimentSpec:
    method = method_hint(h) or "sgd"
    default_params = opt.sanitise_params(method, {})
    return ExperimentSpec(
        experiment_id=f"exp_{h.hypothesis_id}",
        objective=f"Test whether '{h.title}' improves gradient descent convergence relative to baseline SGD.",
        baseline={"method": "sgd", "parameters": {"lr": 0.05}},
        proposed_method=method,
        parameters=default_params,
        controlled_variables=["problem definition", "random seed", "starting point", "iteration budget"],
        metrics=["convergence_iterations", "final_loss", "convergence_rate", "improvement_pct"],
        expected_behavior="Faster convergence (fewer iterations) and unchanged or lower final loss.",
        procedure=[
            "Run baseline SGD on the controlled problem",
            "Run the proposed optimiser under identical seeds",
            "Aggregate metrics and compute improvement",
        ],
        supported=method in opt.known_optimizers(),
    )


def _apply_tournament(h: Hypothesis, evaluation: Evaluation, threshold: float) -> None:
    h.evaluation = evaluation
    passed = evaluation.overall_score >= threshold
    h.tournament_status = "passed" if passed else "rejected"
    h.rejection_reason = "" if passed else evaluation.reviewer_reasoning
# ---------------------------------------------------------------------------
# Public planning entry points (LLM-first, deterministic fallback)
# ---------------------------------------------------------------------------
def generate_hypotheses(
    research_question: str,
    cfg: ResearchConfig,
    svc: GroqService,
) -> List[Hypothesis]:
    """Generate N candidate hypotheses (LLM or clearly-labelled fallback)."""
    if not svc.available:
        log.info("LLM unavailable -> deterministic hypothesis generation")
        return deterministic_hypotheses(research_question, cfg)

    system = (
        "You are the hypothesis generator of an autonomous AI research laboratory. "
        "You must propose structurally sound, testable hypotheses. "
        "Return JSON only. No markdown. "
        "JSON object must be: "
        '{"hypotheses": [{"title": str, "description": str, "research_question": str, '
        '"reasoning": str, "mechanism": str, "assumptions": [str]}]}'
    )
    user = (
        f"Research problem: {research_question}\n\n"
        f"Propose exactly {cfg.num_hypotheses} distinct hypotheses, each describing "
        "a concrete, mechanistic way to improve gradient descent convergence. "
        "They must be testable on a controlled synthetic optimisation problem."
    )
    try:
        data = svc.complete_json(system, user, required_fields=["hypotheses"])
    except Exception as exc:  # noqa: BLE001
        log.warning("Hypothesis generation via LLM failed (%s); using fallback", exc)
        return deterministic_hypotheses(research_question, cfg)

    items = data.get("hypotheses") or []
    hyps: List[Hypothesis] = []
    for i, item in enumerate(items[: cfg.num_hypotheses], start=1):
        if not isinstance(item, dict) or not (item.get("title") or "").strip():
            continue
        hyps.append(
            Hypothesis(
                hypothesis_id=f"H{i}",
                title=str(item.get("title", "")).strip(),
                description=str(item.get("description", "")).strip(),
                research_question=str(item.get("research_question", "")).strip(),
                reasoning=str(item.get("reasoning", "")).strip(),
                mechanism=str(item.get("mechanism", "")).strip(),
                assumptions=[str(a) for a in (item.get("assumptions") or [])],
                source="llm",
            )
        )
    if not hyps:
        log.warning("LLM returned unusable hypotheses; using fallback")
        return deterministic_hypotheses(research_question, cfg)
    return hyps


def critique_and_tournament(
    hyps: List[Hypothesis],
    cfg: ResearchConfig,
    svc: GroqService,
) -> None:
    """Critically evaluate every hypothesis, then apply the tournament threshold.

    Multiple hypotheses survive: every PASS continues to experimentation.
    Rejected hypotheses are preserved with score, reasoning and rejection reason.
    """
    for h in hyps:
        if svc.available:
            evaluation = _llm_evaluate(h, svc)
        else:
            evaluation = _deterministic_evaluation(h)
        _apply_tournament(h, evaluation, cfg.tournament_pass_threshold)


def _llm_evaluate(h: Hypothesis, svc: GroqService) -> Evaluation:
    system = (
        "You are the critic of an autonomous AI research laboratory. "
        "Score the hypothesis for novelty, feasibility and reasoning quality. "
        "Return JSON only. No markdown. "
        'JSON: {"novelty_score": 0-100, "feasibility_score": 0-100, '
        '"reasoning_score": 0-100, "overall_score": 0-100, '
        '"passed": bool, "reviewer_reasoning": str}'
    )
    user = (
        f"HYPOTHESIS {h.hypothesis_id}: {h.title}\n\n"
        f"Description: {h.description}\nReasoning: {h.reasoning}\nMechanism: {h.mechanism}\n"
        "Could this be experimentally tested on a controlled optimisation problem?"
    )
    try:
        data = svc.complete_json(
            system,
            user,
            required_fields=["novelty_score", "feasibility_score", "reasoning_score", "overall_score", "passed"],
        )
        return Evaluation(
            novelty_score=_clamp_score(data.get("novelty_score")),
            feasibility_score=_clamp_score(data.get("feasibility_score")),
            reasoning_score=_clamp_score(data.get("reasoning_score")),
            overall_score=_clamp_score(data.get("overall_score")),
            passed=bool(data.get("passed", False)),
            reviewer_reasoning=str(data.get("reviewer_reasoning", "")),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM critique failed for %s (%s); using deterministic critic", h.hypothesis_id, exc)
        return _deterministic_evaluation(h)


def _clamp_score(val: Any) -> float:
    try:
        return float(min(max(float(val), 0.0), 100.0))
    except (TypeError, ValueError):
        return 50.0
def design_experiments(
    hyps: List[Hypothesis],
    cfg: ResearchConfig,
    svc: GroqService,
) -> None:
    """Design a structured experiment per surviving hypothesis (then validated)."""
    allowed = opt.known_optimizers()
    for h in hyps:
        if not h.passed:
            continue
        if svc.available:
            spec = _llm_experiment_spec(h, svc, allowed)
        else:
            spec = _deterministic_experiment_spec(h, cfg)
        h.experiment_spec = spec


def _llm_experiment_spec(h: Hypothesis, svc: GroqService, allowed: List[str]) -> ExperimentSpec:
    system = (
        "You design experiments for an autonomous AI research laboratory. "
        "Select EXACTLY one optimiser from the safe registry and tune only its "
        "documented numeric parameters. You may not generate code. "
        "Return JSON only. No markdown. "
        f"Registry (method -> allowed params): {_json_dumps(_opt_brief())} "
        'JSON: {"objective": str, "proposed_method": str, "parameters": {str: number}, '
        '"expected_behavior": str}'
    )
    user = (
        f"Hypothesis {h.hypothesis_id}: {h.title}\nMechanism: {h.mechanism}\n\n"
        "Pick the optimiser that best implements this hypothesis and set its parameters."
    )
    try:
        data = svc.complete_json(system, user, required_fields=["proposed_method"])
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM experiment design failed for %s (%s); using fallback", h.hypothesis_id, exc)
        return _deterministic_experiment_spec(h, ResearchConfig())

    method = str(data.get("proposed_method", "")).lower().strip()
    params = data.get("parameters") or {}
    if method not in allowed:
        log.warning("LLM proposed unsupported optimiser '%s' for %s; falling back", method, h.hypothesis_id)
        return _deterministic_experiment_spec(h, ResearchConfig())

    return ExperimentSpec(
        experiment_id=f"exp_{h.hypothesis_id}",
        objective=str(data.get("objective", "")),
        baseline={"method": "sgd", "parameters": {"lr": 0.05}},
        proposed_method=method,
        parameters=params if isinstance(params, dict) else {},
        controlled_variables=["problem definition", "random seed", "starting point", "iteration budget"],
        metrics=["convergence_iterations", "final_loss", "convergence_rate", "improvement_pct"],
        expected_behavior=str(data.get("expected_behavior", "")),
        procedure=[
            "Run baseline SGD on the controlled problem",
            "Run the proposed optimiser under identical seeds",
            "Aggregate metrics and compute improvement",
        ],
        supported=True,
    )


def _run_one_experiment(spec, cfg):
    """
    Worker body: execute a single hypothesis' experiment.

    Module-level so the job stays picklable (process-pool ready). Every job list
    (parallel *and* sequential) hands over the pair ``(spec, cfg)``, which the
    parallel layer unpacks into these two positional arguments — so both paths run
    exactly the same worker. It never raises: a failed experiment is returned as a
    FAILED ExperimentResult so one bad hypothesis can never lose the run.
    """
    from . import experiment_engine as exp_engine

    try:
        return exp_engine.run_experiment(spec, cfg)
    except Exception as exc:  # noqa: BLE001
        result = ExperimentResult()
        result.status = "FAILED"
        result.error = str(exc)
        result.message = "Experiment engine raised an unexpected exception."
        return result


def execute_experiments(
    hyps: List[Hypothesis],
    cfg: ResearchConfig,
) -> None:
    """
    Run the real simulation for every passing hypothesis; never crashes.

    PARALLEL: every survivor is an independent baseline-vs-proposed comparison,
    so they are all executed at once instead of one after another. Results are
    written back to the original Hypothesis objects in their original order, so
    the persisted run record is identical to the sequential version.
    """
    from . import experiment_engine as exp_engine
    from .report import format_improvement

    pending = [h for h in hyps if h.passed]
    if not pending:
        return

    for h in pending:
        method = h.experiment_spec.proposed_method or "?"
        print(f"[{h.hypothesis_id}] Running baseline vs {method} ...")

    use_parallel = (
        PARALLEL_AVAILABLE
        and getattr(cfg, "parallel", True)
        and len(pending) > 1
    )

    if use_parallel:
        if describe_parallelism:
            log.info("Experiments running in parallel: %s", describe_parallelism())
        results = parallel_starmap(
            _run_one_experiment,
            [(h.experiment_spec, cfg) for h in pending],
            max_workers=getattr(cfg, "max_workers", 0),
            label="experiments",
        )
    else:
        results = []
        for h in pending:
            try:
                results.append(exp_engine.run_experiment(h.experiment_spec, cfg))
            except Exception as exc:  # noqa: BLE001
                failed = ExperimentResult()
                failed.status = "FAILED"
                failed.error = str(exc)
                failed.message = "Experiment engine raised an unexpected exception."
                results.append(failed)

    # ---- write results back and report, in original hypothesis order -----
    for h, experiment in zip(pending, results):
        if experiment is None or not isinstance(experiment, ExperimentResult):
            experiment = ExperimentResult()
            experiment.status = "FAILED"
            experiment.error = "experiment worker returned no result"
            experiment.message = "Experiment engine raised an unexpected exception."

        h.experiment = experiment

        if h.experiment.status == "SUCCESS":
            print(
                f"[{h.hypothesis_id}] COMPLETE  "
                f"baseline={h.experiment.baseline.get('mean_iterations')} iters, "
                f"proposed={h.experiment.proposed.get('mean_iterations')} iters, "
                f"improvement={format_improvement(h.experiment.improvement_pct)}"
            )
        else:
            print(
                f"[{h.hypothesis_id}] {h.experiment.status} - "
                f"{h.experiment.error or h.experiment.message}"
            )
            h.experiment.message = "Experiment engine raised an unexpected exception."


def _opt_brief() -> Dict[str, Dict[str, float]]:
    from .optimizers import _OPT_DEFAULTS

    return {name: dict(params) for name, params in _OPT_DEFAULTS.items()}


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)