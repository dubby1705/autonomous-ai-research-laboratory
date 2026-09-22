"""
AARL Lab — Domain-model simulation adapter
==========================================
Adapts the **existing** domain simulator (`Research/simulators/domain_metrics.py`,
9 domain models with real metric definitions and hypothesis→parameter maps) to the
lab's :class:`SimulationEngine` interface.

It does not re-implement any physics: it loads the existing module by file path
(tolerating either import path) and turns its comparison output into a
provenance-carrying Simulation JSON.

Honesty rules
-------------
* The **baseline** and the **hypothesis** run through the *same* domain model; the
  only difference is the hypothesis-derived parameter block (stated in the JSON).
* If the hypothesis maps to no domain parameter, the run is marked ``partial`` and
  says so — a run that could not discriminate the hypothesis is not a success.
* If the detected domain has no model, the status is ``simulation_unavailable``
  (never a fallback to a different domain's physics).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .base import SimulationEngine, SimulationRecord, SimulationRequest, _now
from .constraints import constraints_for

log = logging.getLogger("aarl.lab.sim.domain")

# Metric values are rounded for storage; comparisons use the raw numbers.
_REPORT_LIMIT = 14


def _load_domain_metrics():
    """Import the existing domain simulator (adds its directory to sys.path)."""
    try:
        from Research.simulators import domain_metrics as dm  # type: ignore

        return dm
    except Exception:  # pragma: no cover - flat path fallback
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sim_dir = os.path.join(root, "Research", "simulators")
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)
        try:
            import domain_metrics as dm  # type: ignore

            return dm
        except Exception:
            return None


class DomainSimulation(SimulationEngine):
    """Wrap the existing domain models as a pluggable simulation engine."""

    name = "domain_metrics_simulation"
    version = "1.0"
    description = (
        "Domain-model comparison using the project's existing domain_metrics.py "
        "simulators (baseline vs hypothesis-derived parameters)."
    )
    supported_domains = ("*",)

    def __init__(self, report_limit: int = _REPORT_LIMIT) -> None:
        self.report_limit = report_limit

    # ------------------------------------------------------------------ support
    def supports(self, domain: str, research_question: str, hypothesis_text: str) -> bool:
        dm = _load_domain_metrics()
        if dm is None:
            return False
        resolved = self.resolve_domain(dm, domain, research_question, hypothesis_text)
        return resolved in getattr(dm, "DOMAIN_SIMULATORS", {})

    @staticmethod
    def resolve_domain(dm, domain: str, research_question: str, hypothesis_text: str) -> str:
        """Detected domain: explicit request first, then ticker, then question."""
        if domain:
            return str(domain).lower()
        for text in (research_question, hypothesis_text):
            if text:
                return str(dm.detect_domain(text)).lower()
        return ""

    # ------------------------------------------------------------------ run
    def run(self, request: SimulationRequest) -> SimulationRecord:
        record = self.new_record(request, status="failure")
        dm = _load_domain_metrics()
        if dm is None:
            record.status = "simulation_unavailable"
            record.simulation_type = "none"
            record.errors.append(
                "domain metrics module unavailable (Research/simulators/domain_metrics.py)"
            )
            return record

        domain = self.resolve_domain(
            dm, request.domain, request.research_question, request.hypothesis_text
        )
        record.simulation_type = f"domain_model:{domain or 'unknown'}"
        simulators = getattr(dm, "DOMAIN_SIMULATORS", {})

        if domain not in simulators:
            record.status = "simulation_unavailable"
            record.simulation_type = "none"
            record.errors.append(
                f"no domain model exists for domain '{domain or 'undetected'}' "
                f"(available: {sorted(simulators.keys())})"
            )
            record.notes.append(
                "No domain physics model was available, so no simulation was run. "
                "AARL does not substitute another domain's model."
            )
            return record

        # ---- parameters: explicit block wins, otherwise extracted from text ----
        params: Dict[str, float] = {
            str(k): float(v)
            for k, v in dict(request.parameters or {}).items()
            if isinstance(v, (int, float))
        }
        extraction_source = "request.parameters"
        if not params:
            extracted = dm.extract_domain_params(domain, request.hypothesis_text)
            params = {str(k): float(v) for k, v in extracted.items()}
            extraction_source = "domain_metrics.extract_domain_params(hypothesis_text)"
        record.parameters = {
            "domain": domain,
            "domain_parameter_modifications": params,
            "extraction_source": extraction_source,
            "base_model": "domain_metrics baseline (no modifications)",
        }

        # ---- declared constraints (human-declared envelopes) ------------------
        constraints = list(request.constraints or []) or constraints_for(domain)
        record.reproducibility["constraints_used"] = constraints
        # ---- execute baseline and hypothesis on the SAME model ---------------
        try:
            baseline = dm.simulate_domain(domain, {}, is_baseline=True)
            hypothesis = dm.simulate_domain(domain, params, is_baseline=False)
        except Exception as exc:  # noqa: BLE001 - a broken model must not kill the run
            record.status = "failure"
            record.errors.append(f"domain model raised: {exc}")
            return record

        metric_defs = dm.get_domain_metrics(domain)
        metrics: Dict[str, Any] = {}
        for metric_name in dm.get_domain_metric_names(domain):
            if metric_name not in baseline or metric_name not in hypothesis:
                continue
            base_value = baseline[metric_name]
            new_value = hypothesis[metric_name]
            if not isinstance(base_value, (int, float)) or not isinstance(new_value, (int, float)):
                continue
            definition = metric_defs.get(metric_name, {})
            direction = str(definition.get("direction", "higher_better"))
            if base_value == 0:
                pct_change = 0.0
            else:
                pct_change = ((new_value - base_value) / abs(base_value)) * 100.0
            improvement = -pct_change if direction == "lower_better" else pct_change
            metrics[metric_name] = {
                "baseline_value": round(float(base_value), 6),
                "observed_value": round(float(new_value), 6),
                "change_pct": round(float(pct_change), 4),
                "improvement_pct": round(float(improvement), 4),
                "is_improvement": bool(improvement > 0),
                "direction": direction,
                "unit": str(definition.get("unit", "")),
                "description": str(definition.get("description", "")),
            }

        record.metrics = dict(sorted(metrics.items())[: self.report_limit])
        record.observed_results = {
            "baseline_metrics": {k: round(float(v), 6) for k, v in baseline.items()
                                 if isinstance(v, (int, float))},
            "hypothesis_metrics": {k: round(float(v), 6) for k, v in hypothesis.items()
                                   if isinstance(v, (int, float))},
            "comparison": record.metrics,
            "applied_modifications": hypothesis.get("applied_modifications", {}),
        }

        improved = [name for name, m in record.metrics.items() if m["is_improvement"]]
        worsened = [name for name, m in record.metrics.items() if m["improvement_pct"] < 0]
        record.status = "success"
        record.notes.append(
            f"{len(improved)} of {len(record.metrics)} metrics improved; "
            f"{len(worsened)} worsened."
        )
        if not params:
            record.status = "partial"
            record.notes.append(
                "No hypothesis keyword mapped to a domain parameter, so baseline and "
                "hypothesis are identical: the result cannot discriminate the hypothesis."
            )

        # ---- declared-constraint screening -----------------------------------
        record.constraints_violated = self.check_constraints(
            constraints, record.observed_results.get("hypothesis_metrics", {})
        )

        # ---- unexpected results: expectations that the model did not meet -----
        unexpected: List[Dict[str, Any]] = []
        for name, expected in dict(request.expected_results or {}).items():
            metric = record.metrics.get(str(name))
            if metric is None or not isinstance(expected, (int, float)):
                continue
            observed_value = metric["observed_value"]
            deviation = abs(observed_value - float(expected))
            reference = abs(float(expected)) or 1.0
            if deviation / reference > 0.25:
                unexpected.append(
                    {
                        "metric": name,
                        "expected": float(expected),
                        "observed": observed_value,
                        "deviation_pct": round(100.0 * deviation / reference, 3),
                        "source": "declared expectation (generated hypothesis)",
                    }
                )
        record.unexpected_results = unexpected
        record.timestamp = _now()
        return record