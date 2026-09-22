"""
AARL Lab — ML optimisation simulation adapter
============================================
The lab's "ML experiment simulation": it runs the **existing, real** optimiser
suite (`showcase/optimizers.py` through `showcase/experiment_engine.py`) —
deterministic seeded trajectories, baseline vs proposed, aggregated metrics.

Nothing here fabricates numbers: every value in the Simulation JSON comes from
executing NumPy trajectories. Parameters are validated/clamped by the existing
registry, and any clamping is reported in the record (a silent clamp would hide
that the tested configuration differed from the requested one).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import SimulationEngine, SimulationRecord, SimulationRequest, _now

log = logging.getLogger("aarl.lab.sim.ml")

#: Rule-based hypothesis -> optimiser mapping (used only when no explicit
#: optimiser parameter was supplied; the derivation is written into the JSON).
_HYPOTHESIS_METHOD_RULES: Tuple[Tuple[str, str], ...] = (
    ("nesterov", "nesterov"),
    ("look-ahead", "nesterov"),
    ("lookahead", "nesterov"),
    ("amsgrad", "amsgrad"),
    ("adam", "adam"),
    ("rmsprop", "rmsprop"),
    ("per-coordinate", "rmsprop"),
    ("per coordinate", "rmsprop"),
    ("adaptive scaling", "rmsprop"),
    ("adagrad", "adagrad"),
    ("warm restart", "sgd_warm_restart"),
    ("restart", "sgd_warm_restart"),
    ("cosine", "sgd_cosine"),
    ("anneal", "sgd_cosine"),
    ("momentum", "momentum"),
)

#: Fallback when the hypothesis names no method (flagged as rule-derived).
_DEFAULT_METHOD = "momentum"


class MLOptimizationSimulation(SimulationEngine):
    """Real baseline-vs-proposed optimisation experiment, wrapped as an engine."""

    name = "ml_optimizer_simulation"
    version = "1.0"
    description = (
        "Seeded baseline(SGD) vs proposed optimiser comparison executed by the "
        "existing optimiser suite (showcase/experiment_engine.py)."
    )
    supported_domains = ("machine_learning",)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def derive_method(hypothesis_text: str) -> Tuple[str, str]:
        """Which optimiser the hypothesis implies, and how that was decided."""
        text = str(hypothesis_text or "").lower()
        for keyword, method in _HYPOTHESIS_METHOD_RULES:
            if keyword in text:
                return method, f"rule match on '{keyword}' in the hypothesis text"
        return _DEFAULT_METHOD, "rule-derived default (hypothesis named no method)"
# ------------------------------------------------------------------ run
    def run(self, request: SimulationRequest) -> SimulationRecord:
        from showcase import experiment_engine as exp_engine
        from showcase.config import ResearchConfig
        from showcase.optimizers import known_optimizers, sanitise_params
        from showcase.records import ExperimentSpec

        record = self.new_record(request, status="failure")
        params = dict(request.parameters or {})

        method = str(params.get("optimizer", "") or "").lower().strip()
        method_source = "request.parameters['optimizer']"
        if not method:
            method, method_source = self.derive_method(request.hypothesis_text)
        record.simulation_type = f"ml_optimizer:{method}"

        if method not in known_optimizers():
            record.status = "simulation_unavailable"
            record.simulation_type = "none"
            record.errors.append(
                f"optimiser '{method}' is not in the safe registry "
                f"({', '.join(known_optimizers())})"
            )
            return record

        problem_name = str(params.get("problem", "quadratic") or "quadratic")
        requested_params = dict(params.get("optimizer_params", {}) or {})

        # ---- validated, bounded configuration (existing registry rules) ------
        sanitised = sanitise_params(method, requested_params)
        clamped = {
            key: {"requested": requested_params[key], "applied": sanitised[key]}
            for key in requested_params
            if key in sanitised and float(sanitised[key]) != float(requested_params[key])
        }
        rejected = sorted(set(requested_params) - set(sanitised))

        cfg = ResearchConfig()
        cfg.num_runs = int(params.get("num_runs", cfg.num_runs))
        cfg.dimensionality = int(params.get("dimensionality", cfg.dimensionality))
        cfg.max_iterations = int(params.get("max_iterations", cfg.max_iterations))
        cfg.random_seed = int(params.get("seed", request.seed))
        cfg.parallel = False  # a single experiment; fan-out happens one level up

        record.parameters = {
            "optimizer": method,
            "optimizer_source": method_source,
            "optimizer_params_requested": requested_params,
            "optimizer_params_applied": sanitised,
            "problem": problem_name,
            "num_runs": cfg.num_runs,
            "dimensionality": cfg.dimensionality,
            "max_iterations": cfg.max_iterations,
            "convergence_tol": cfg.convergence_tol,
            "seed": cfg.random_seed,
            "baseline": {"optimizer": "sgd", "params": {"lr": 0.1}},
        }
        record.reproducibility = self.reproducibility(request)
        record.reproducibility["configuration"] = {
            "problem": problem_name,
            "dimensionality": cfg.dimensionality,
            "max_iterations": cfg.max_iterations,
            "num_runs": cfg.num_runs,
            "seed": cfg.random_seed,
            "optimizer": method,
            "optimizer_params": sanitised,
        }

        spec = ExperimentSpec(
            experiment_id=request.hypothesis_id,
            objective=request.hypothesis_text or "compare optimiser against SGD baseline",
            proposed_method=method,
            parameters=sanitised,
            metrics=["mean_iterations", "convergence_rate", "mean_final_loss"],
            supported=True,
        )

        try:
            result = exp_engine.run_experiment(spec, cfg, problem_name)
        except Exception as exc:  # noqa: BLE001 - never crash the loop
            record.status = "failure"
            record.errors.append(f"experiment engine raised: {exc}")
            return record

        if result.status != "SUCCESS":
            record.status = "failure"
            record.errors.append(result.error or result.message or result.status)
            record.observed_results = {
                "baseline": dict(result.baseline or {}),
                "proposed": dict(result.proposed or {}),
            }
            return record

        baseline, proposed = dict(result.baseline or {}), dict(result.proposed or {})
        def _metric(name: str, direction: str, unit: str) -> Optional[Dict[str, Any]]:
            base_value = baseline.get(name)
            new_value = proposed.get(name)
            if not isinstance(base_value, (int, float)) or not isinstance(new_value, (int, float)):
                return None
            if base_value == 0:
                change = 0.0
            else:
                change = 100.0 * (float(new_value) - float(base_value)) / abs(float(base_value))
            improvement = -change if direction == "lower_better" else change
            return {
                "baseline_value": round(float(base_value), 6),
                "observed_value": round(float(new_value), 6),
                "change_pct": round(change, 4),
                "improvement_pct": round(improvement, 4),
                "is_improvement": bool(improvement > 0),
                "direction": direction,
                "unit": unit,
                "description": f"{name} measured over {cfg.num_runs} seeded runs",
            }

        metrics: Dict[str, Any] = {}
        for name, direction, unit in (
            ("mean_iterations", "lower_better", "iterations"),
            ("mean_final_loss", "lower_better", "loss"),
            ("std_iterations", "lower_better", "iterations"),
            ("convergence_rate", "higher_better", "fraction"),
        ):
            metric = _metric(name, direction, unit)
            if metric:
                metrics[name] = metric
        metrics["iterations_improvement_pct"] = {
            "baseline_value": baseline.get("mean_iterations"),
            "observed_value": proposed.get("mean_iterations"),
            "improvement_pct": result.improvement_pct,
            "is_improvement": bool((result.improvement_pct or 0) > 0),
            "direction": "higher_better",
            "unit": "%",
            "description": (
                "Engine-computed improvement in mean iterations "
                "(requires both baseline and proposed to converge)"
            ),
        }
        record.metrics = metrics
        record.observed_results = {
            "baseline": baseline,
            "proposed": proposed,
            "per_run_baseline": list(result.per_run_baseline or []),
            "per_run_proposed": list(result.per_run_proposed or []),
            "loss_curve_baseline": list(result.baseline_loss_curve or []),
            "loss_curve_proposed": list(result.proposed_loss_curve or []),
        }

        record.status = "success"
        if clamped:
            record.notes.append(
                "Requested optimiser parameters were clamped to safe registry bounds: "
                + ", ".join(
                    f"{key}: {value['requested']} -> {value['applied']}"
                    for key, value in clamped.items()
                )
            )
        if rejected:
            record.notes.append(
                f"Ignored unknown optimiser parameters: {', '.join(rejected)}"
            )
        if method_source.startswith("rule-derived default"):
            record.status = "partial"
            record.notes.append(
                "The hypothesis named no optimiser, so a rule-derived default was tested; "
                "this result cannot distinguish between alternative mechanisms."
            )

        # ---- declared-constraint and expectation screening -------------------
        record.constraints_violated = self.check_constraints(
            request.constraints, {"mean_iterations": proposed.get("mean_iterations")}
        ) + [
            {
                "constraint": {
                    "name": key,
                    "op": "within safe registry bounds",
                    "value": value["applied"],
                    "source": "showcase.optimizers bounds (requested value was clamped)",
                },
                "observed": value["requested"],
                "violated": True,
                "reason": (
                    f"requested {key}={value['requested']} exceeded safe bounds; "
                    f"applied {value['applied']}"
                ),
            }
            for key, value in clamped.items()
        ]

        unexpected: List[Dict[str, Any]] = []
        for name, expected in dict(request.expected_results or {}).items():
            metric = record.metrics.get(str(name))
            if metric is None or not isinstance(expected, (int, float)):
                continue
            observed_value = metric.get("improvement_pct")
            if not isinstance(observed_value, (int, float)):
                continue
            deviation = abs(float(observed_value) - float(expected))
            denominator = abs(float(expected)) or 1.0
            if deviation / denominator > 0.25:
                unexpected.append(
                    {
                        "metric": name,
                        "expected": float(expected),
                        "observed": observed_value,
                        "deviation_pct": round(100.0 * deviation / denominator, 3),
                        "source": "declared expectation (generated hypothesis)",
                    }
                )
        record.unexpected_results = unexpected
        record.timestamp = _now()
        return record