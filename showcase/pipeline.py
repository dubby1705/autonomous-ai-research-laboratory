"""
AARL Showcase — Research Pipeline Orchestrator
================================================
Runs one complete research cycle:

    research problem -> hypotheses -> critique -> tournament (multiple
    survivors) -> experiment design -> real simulation -> measured results
    -> LLM analysis -> hypothesis refinement -> report + storage

Failure isolation: a failed experiment, a malformed LLM response, or a missing
key never loses the whole run — the affected stage is marked and the pipeline
continues.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List

from . import analysis as analysis_mod
from . import generation as gen
from . import report as report_mod
from . import storage as storage_mod
from .config import ResearchConfig
from .llm_service import make_llm_service
from .records import ResearchRun

# Parallel processing layer (shared with the main AARL pipeline), imported
# defensively so the showcase can still run on its own.
try:
    from Engine.Question_Engine.ParallelExecutor import (
        failed_error,
        is_failed,
        run_parallel,
    )
    PARALLEL_AVAILABLE = True
except Exception:  # pragma: no cover - standalone deployment
    failed_error = None
    is_failed = None
    run_parallel = None
    PARALLEL_AVAILABLE = False

log = logging.getLogger("aarl.pipeline")


def _is_failed(value: Any) -> bool:
    """Safe wrapper: False when the parallel layer is not importable."""
    return bool(is_failed(value)) if is_failed else False


def _failure_detail(value: Any) -> Any:
    """Human-readable failure text (or the raw value as a fallback)."""
    return failed_error(value) if failed_error else value


def _summarise(run: ResearchRun) -> Dict[str, Any]:
    """Build the final findings dict from the actual experiment records."""
    passed = [h for h in run.hypotheses if h.passed]
    success = [h for h in passed if h.experiment.status == "SUCCESS"]
    improved = [
        h for h in success
        if h.experiment.improvement_pct is not None and h.experiment.improvement_pct > 0
    ]
    failed = [h for h in passed if h.experiment.status != "SUCCESS"]

    best = max(improved, key=lambda h: h.experiment.improvement_pct or 0.0) if improved else None

    if best is not None:
        summary = (
            f"Of {len(run.hypotheses)} generated hypotheses, {len(passed)} survived "
            f"the tournament. Real experiments were executed for {len(passed)} "
            f"survivors; {len(success)} succeeded and {len(failed)} were marked as "
            f"failed/unsupported. {len(improved)} showed a measurable improvement, "
            f"led by {best.hypothesis_id} "
            f"({best.experiment.improvement_pct:.1f}%)."
        )
    else:
        summary = (
            f"Of {len(run.hypotheses)} generated hypotheses, {len(passed)} survived "
            f"the tournament and were experimentally tested. No hypothesis showed "
            f"a clean measured improvement over the baseline on the controlled "
            f"problem, so no positive claim is made."
        )

    best_hyp = (
        {
            "hypothesis_id": best.hypothesis_id,
            "title": best.title,
            "improvement_pct": best.experiment.improvement_pct,
            "method": best.experiment_spec.proposed_method,
        }
        if best is not None
        else None
    )

    return {
        "summary": summary,
        "best_hypothesis": best_hyp,
        "num_generated": len(run.hypotheses),
        "num_survivors": len(passed),
        "num_success_experiments": len(success),
        "num_failed_experiments": len(failed),
        "num_improved": len(improved),
        "limitations": [
            "Controlled synthetic problem only (single objective family).",
            "Deterministic gradients (no stochastic noise).",
            "The LLM selects/tunes optimisers but never generates experimentation code.",
            "Improvement claims require both baseline and proposed to converge.",
            "Small number of runs and a fixed iteration budget.",
        ],
        "future_research": [
            "Add stochastic-gradient and mini-batch test beds.",
            "Expand the problem registry (Rosenbrock, logistic regression).",
            "Multi-stage refinement with follow-up experiments (HX-R1 ... Rk).",
            "Sensitivity analysis across learning rates and condition numbers.",
        ],
    }
def run_research_cycle(config: ResearchConfig) -> ResearchRun:
    """Execute the full research cycle for a given configuration."""
    log.info("Starting research cycle: %s", config.research_question)

    svc = make_llm_service(config)
    if not svc.available:
        print("\n[AARL] LLM unavailable. Using a transparent deterministic planning "
              "fallback; experiments remain real simulations.\n")

    run = ResearchRun(
        research_question=config.research_question,
        domain=config.domain,
        config=config.derive(),
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        llm_available=svc.available,
        llm_model=svc.model if svc.available else "deterministic-fallback",
    )
    run.config["run_dir"] = "(pending)"

    run.hypotheses = gen.generate_hypotheses(config.research_question, config, svc)
    print(report_mod.render_hypotheses(run.hypotheses))

    gen.critique_and_tournament(run.hypotheses, config, svc)
    print(report_mod.render_tournament(run.hypotheses))

    gen.design_experiments(run.hypotheses, config, svc)
    print(report_mod.section("PHASE 4 - EXPERIMENTATION (REAL SIMULATION)"))
    gen.execute_experiments(run.hypotheses, config)

    print(report_mod.section("PHASE 6 - DEEP RESEARCH ANALYSIS"))
    analysis_mod.analyze_hypotheses(run.hypotheses, config, svc)
    analysis_mod.refine_hypotheses(run.hypotheses, config, svc)
    refined = [r["refined_id"] for h in run.hypotheses for r in h.refinements]
    if refined:
        print(f"   Refined hypotheses: {', '.join(refined)}")

    run.findings = _summarise(run)
    return run


def persist_and_report(run: ResearchRun, config: ResearchConfig) -> Dict[str, Any]:
    """Persist the run; return {block, written, run_dir} for presentation."""
    directory = storage_mod.RunDirectory(root=config.run_root if config.run_root else "runs")
    run_dir = directory.create()
    run.config["run_dir"] = run_dir
    run.run_id = directory.run_id

    experiments_payload = storage_mod.build_experiments_payload(run)
    results_payload = storage_mod.build_results_payload(run)
    written = directory.save_run(run, experiments_payload, results_payload)

    # PARALLEL: the markdown report and the plots are independent artefacts, so
    # they are produced at the same time rather than one after the other.
    if PARALLEL_AVAILABLE and getattr(config, "parallel", True):
        stages = run_parallel(
            {
                "report": lambda: directory.save_text(
                    "report.md", report_mod.build_markdown_report(run)
                ),
                "plots": lambda: report_mod.generate_plots(run, directory.plots_dir),
            },
            max_workers=getattr(config, "max_workers", 0) or None,
            label="persist",
        )
        report_path = stages.get("report")
        plots = stages.get("plots")
    else:
        report_path = directory.save_text(
            "report.md", report_mod.build_markdown_report(run)
        )
        try:
            plots = report_mod.generate_plots(run, directory.plots_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("Plot generation failed: %s", exc)
            plots = []

    if report_path is None or _is_failed(report_path):
        raise RuntimeError(
            f"report generation failed: {_failure_detail(report_path)}"
        )
    written["report"] = report_path

    if plots is None or _is_failed(plots):
        log.warning("Plot generation failed: %s", _failure_detail(plots))
        written["plots"] = []
    else:
        written["plots"] = plots

    block = report_mod.build_terminal_block(run)
    return {"block": block, "written": written, "run_dir": run_dir}