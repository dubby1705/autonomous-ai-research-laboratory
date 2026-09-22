"""
AARL Showcase — Experiment Engine
==================================
Bridges a validated ExperimentSpec to the safe optimiser suite. Runs the
BASELINE vs PROPOSED comparison over multiple reproducible seeds and aggregates
real measured metrics.

One failed experiment never crashes the pipeline: exceptions are captured into
the hypothesis experiment record with status FAILED.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from . import optimizers as opt
from .config import ResearchConfig
from .records import ExperimentResult, ExperimentSpec

BASELINE = ("sgd", {"lr": 0.1})  # (method, params)


# ---------------------------------------------------------------------------
# Aggregation helpers (all inputs and outputs are real measured numbers)
# ---------------------------------------------------------------------------
def _round(v: float, ndigits: int = 4) -> float:
    if not np.isfinite(v):
        return float(v)
    return float(round(v, ndigits))


def _make_start(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(1.0, 1.5, dim)  # away from the optimum, so convergence is observable


def _append(per_run: List[Dict[str, Any]], seed: int, res: "opt.OptimiseResult") -> None:
    per_run.append(
        {
            "seed": seed,
            "converged": bool(res.converged),
            "iterations": int(res.iterations),
            "final_loss": _round(res.final_loss, 6),
        }
    )


def _aggregate(per_run: List[Dict[str, Any]], max_iterations: int) -> Dict[str, Any]:
    if not per_run:
        return {
            "converged_count": 0,
            "convergence_rate": 0.0,
            "mean_iterations": float("nan"),
            "std_iterations": 0.0,
            "mean_final_loss": float("nan"),
            "max_iters_cap": int(max_iterations),
        }
    conv = sum(1 for r in per_run if r["converged"])
    iters = [float(r["iterations"]) for r in per_run]
    losses = [float(r["final_loss"]) for r in per_run]
    return {
        "converged_count": int(conv),
        "convergence_rate": float(round(conv / len(per_run), 4)),
        "mean_iterations": _round(float(np.mean(iters)), 4),
        "std_iterations": _round(float(np.std(iters)), 4),
        "mean_final_loss": _round(float(np.mean(losses)), 6),
        "max_iters_cap": int(max_iterations),
    }


def _improvement(baseline: Dict[str, Any], proposed: Dict[str, Any]) -> float | None:
    """Improvement in mean iterations (%). Cautious: requires both to converge."""
    b_conv = baseline.get("converged_count", 0)
    p_conv = proposed.get("converged_count", 0)
    b_iters = baseline.get("mean_iterations")
    p_iters = proposed.get("mean_iterations")
    if b_conv == 0 or p_conv == 0:
        return None
    if not isinstance(b_iters, (int, float)) or not isinstance(p_iters, (int, float)):
        return None
    if b_iters <= 0:
        return None
    return float(round(100.0 * (float(b_iters) - float(p_iters)) / float(b_iters), 2))
# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_experiment(
    spec: ExperimentSpec,
    config: ResearchConfig,
    problem_name: str = "quadratic",
) -> ExperimentResult:
    """Execute a validated experiment spec. Never raises for a bad spec."""
    result = ExperimentResult()

    method = (spec.proposed_method or "").lower().strip()
    if not spec.supported:
        result.status = "UNSUPPORTED"
        result.message = "Experiment design did not map to a safe optimiser."
        result.error = spec.objective or ""
        return result
    if method not in opt.known_optimizers():
        result.status = "FAILED"
        result.message = f"Unknown optimiser technique '{method}'. Experiment paused."
        result.error = f"unknown optimiser: {method}"
        return result
    if problem_name not in opt.PROBLEMS:
        result.status = "FAILED"
        result.message = f"Problem '{problem_name}' is not available."
        result.error = f"unknown problem: {problem_name}"
        return result

    maker = opt.PROBLEMS[problem_name]["make"]
    dim = max(1, int(config.dimensionality))
    max_iterations = max(1, int(config.max_iterations))
    tol = config.convergence_tol
    num_runs = max(1, int(config.num_runs))

    # ----- validated, bounded configurations -------------------------------
    base_method, base_params = BASELINE
    baseline_cfg = opt.sanitise_params(base_method, base_params)
    proposed_cfg = opt.sanitise_params(method, dict(spec.parameters or {}))

    try:
        b_run: List[Dict[str, Any]] = []
        p_run: List[Dict[str, Any]] = []
        b_curve: List[float] = []
        p_curve: List[float] = []
        t0 = time.time()
        for run_idx in range(num_runs):
            seed = config.random_seed + run_idx * 7919  # reproducible per-run seeds
            problem = maker(dim, seed)
            start = _make_start(dim, seed)

            base_res = opt.run_optimiser(problem, base_method, baseline_cfg, max_iterations, tol, start)
            _append(b_run, seed, base_res)

            prop_res = opt.run_optimiser(problem, method, proposed_cfg, max_iterations, tol, start)
            _append(p_run, seed, prop_res)

            if run_idx == 0:  # representative loss curves for visualisation
                b_curve = [float(x) for x in base_res.loss_history]
                p_curve = [float(x) for x in prop_res.loss_history]
        elapsed_seconds = time.time() - t0
    except Exception as exc:  # noqa: BLE001 — never crash the run
        result.status = "FAILED"
        result.message = f"Experiment execution crashed: {exc}"
        result.error = str(exc)
        return result

    baseline_agg = _aggregate(b_run, max_iterations)
    proposed_agg = _aggregate(p_run, max_iterations)
    baseline_agg["elapsed_seconds"] = _round(elapsed_seconds, 4)
    proposed_agg["elapsed_seconds"] = _round(elapsed_seconds, 4)

    result.status = "SUCCESS"
    result.message = "Baseline vs proposed executed across controlled seeds."
    result.baseline = baseline_agg
    result.proposed = proposed_agg
    result.improvement_pct = _improvement(baseline_agg, proposed_agg)
    result.per_run_baseline = b_run
    result.per_run_proposed = p_run
    result.baseline_loss_curve = b_curve
    result.proposed_loss_curve = p_curve
    return result