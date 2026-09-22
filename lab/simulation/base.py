"""
AARL Lab — Simulation Engine interface and Simulation JSON
==========================================================
A small, domain-agnostic contract so any simulator can be plugged into the
continuous research loop without touching the orchestrator:

    SimulationEngine (ABC)
    ├── DomainSimulation        — wraps Research/simulators/domain_metrics.py
    ├── MLOptimizationSim       — wraps showcase/experiment_engine.py + optimizers.py
    └── CustomSimulationAdapter — user drop-in modules from the research memory

Rules enforced here
-------------------
* Every run produces a **Simulation JSON** with the actual experimental
  conditions and results, so it is reproducible from the stored parameters.
* If no engine can run a hypothesis, the result is
  ``status="simulation_unavailable"`` — AARL never pretends a simulation happened
  and never substitutes model prose for a measurement.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Allowed simulation statuses.
SIMULATION_STATUSES = ("success", "failure", "partial", "simulation_unavailable")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass
class SimulationRequest:
    """Everything an engine needs, stated explicitly (inputs live in the JSON)."""

    hypothesis_id: str = ""
    hypothesis_text: str = ""
    research_question: str = ""
    domain: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    #: Declared limits, e.g. {"name": "power", "value": 300, "op": "<=", "source": "..."}
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    seed: int = 11

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_text": self.hypothesis_text,
            "research_question": self.research_question,
            "domain": self.domain,
            "parameters": dict(self.parameters),
            "expected_results": dict(self.expected_results),
            "constraints": [dict(c) for c in self.constraints],
            "seed": self.seed,
        }

    def input_hash(self) -> str:
        """Stable hash of the experimental inputs (reproducibility fingerprint)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class SimulationRecord:
    """The stored Simulation JSON for one hypothesis experiment."""

    experiment_id: str = ""
    hypothesis_id: str = ""
    simulation_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    observed_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "failure"
    errors: List[str] = field(default_factory=list)
    constraints_violated: List[Dict[str, Any]] = field(default_factory=list)
    unexpected_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    adapter: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis_id": self.hypothesis_id,
            "simulation_type": self.simulation_type,
            "parameters": dict(self.parameters),
            "expected_results": dict(self.expected_results),
            "observed_results": dict(self.observed_results),
            "metrics": dict(self.metrics),
            "status": self.status,
            "errors": list(self.errors),
            "constraints_violated": [dict(c) for c in self.constraints_violated],
            "unexpected_results": [dict(u) for u in self.unexpected_results],
            "timestamp": self.timestamp,
            "adapter": dict(self.adapter),
            "reproducibility": dict(self.reproducibility),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationRecord":
        data = data or {}
        return cls(
            experiment_id=str(data.get("experiment_id", "") or ""),
            hypothesis_id=str(data.get("hypothesis_id", "") or ""),
            simulation_type=str(data.get("simulation_type", "") or ""),
            parameters=dict(data.get("parameters") or {}),
            expected_results=dict(data.get("expected_results") or {}),
            observed_results=dict(data.get("observed_results") or {}),
            metrics=dict(data.get("metrics") or {}),
            status=str(data.get("status", "failure") or "failure"),
            errors=list(data.get("errors") or []),
            constraints_violated=list(data.get("constraints_violated") or []),
            unexpected_results=list(data.get("unexpected_results") or []),
            timestamp=str(data.get("timestamp", "") or ""),
            adapter=dict(data.get("adapter") or {}),
            reproducibility=dict(data.get("reproducibility") or {}),
            notes=list(data.get("notes") or []),
        )

    # convenience ------------------------------------------------------------
    def improvement_pct(self) -> Optional[float]:
        """Best measured improvement across the record's metrics (or None)."""
        best: Optional[float] = None
        for metric in self.metrics.values():
            if not isinstance(metric, dict):
                continue
            value = metric.get("improvement_pct")
            if isinstance(value, (int, float)):
                best = float(value) if best is None else max(best, float(value))
        return best

    def is_usable(self) -> bool:
        """True when the record holds real measured numbers."""
        return self.status in ("success", "partial") and bool(self.metrics)
class SimulationEngine(ABC):
    """Base class for every simulator AARL can run."""

    name: str = "simulation_engine"
    version: str = "1.0"
    description: str = ""
    #: Domains the engine serves (use ("*",) for "any domain").
    supported_domains: tuple = ()

    def supports(self, domain: str, research_question: str, hypothesis_text: str) -> bool:
        """Whether this engine can meaningfully simulate the hypothesis."""
        if not self.supported_domains:
            return False
        if "*" in self.supported_domains:
            return True
        return (domain or "").lower() in {d.lower() for d in self.supported_domains}

    @abstractmethod
    def run(self, request: SimulationRequest) -> SimulationRecord:
        """Execute the simulation and return the Simulation JSON."""

    # ---- helpers ----------------------------------------------------------
    def adapter_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "module": type(self).__module__,
            "description": self.description,
        }

    def reproducibility(
        self, request: SimulationRequest, deterministic: bool = True
    ) -> Dict[str, Any]:
        return {
            "seed": request.seed,
            "input_hash": request.input_hash(),
            "deterministic": bool(deterministic),
            "how_to_reproduce": (
                f"Re-run hypothesis {request.hypothesis_id} with the stored parameter "
                f"block and seed {request.seed} through adapter '{self.name}'."
            ),
            "inputs": request.to_dict(),
        }

    def new_record(
        self, request: SimulationRequest, status: str = "failure"
    ) -> SimulationRecord:
        return SimulationRecord(
            hypothesis_id=request.hypothesis_id,
            simulation_type=self.name,
            parameters=dict(request.parameters),
            expected_results=dict(request.expected_results),
            status=status,
            timestamp=_now(),
            adapter=self.adapter_info(),
            reproducibility=self.reproducibility(request),
        )

    # ---- constraint bookkeeping ------------------------------------------
    @staticmethod
    def check_constraints(
        constraints: List[Dict[str, Any]], observed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate declared constraints against observed values.

        A constraint is ``{"name": <metric>, "op": "<=", "value": <number>,
        "source": <where it comes from>}``. Only numeric metrics are checked;
        anything unverifiable is reported, not silently assumed to hold.
        """
        violations: List[Dict[str, Any]] = []
        for constraint in constraints or []:
            name = str(constraint.get("name", "") or "")
            op = str(constraint.get("op", "<=") or "<=")
            limit = constraint.get("value")
            value = observed.get(name)
            if not isinstance(value, (int, float)) or not isinstance(limit, (int, float)):
                violations.append(
                    {
                        "constraint": dict(constraint),
                        "observed": value,
                        "violated": None,
                        "reason": "not verifiable from this simulation's metrics",
                    }
                )
                continue
            ops = {
                "<=": value <= limit,
                "<": value < limit,
                ">=": value >= limit,
                ">": value > limit,
                "==": value == limit,
            }
            held = ops.get(op, True)
            if not held:
                violations.append(
                    {
                        "constraint": dict(constraint),
                        "observed": value,
                        "violated": True,
                        "reason": f"{name} = {value} does not satisfy {op} {limit}",
                    }
                )
        return violations


def unavailable_record(request: SimulationRequest, reason: str) -> SimulationRecord:
    """Simulation JSON for the honest "no simulator could run this" case."""
    return SimulationRecord(
        hypothesis_id=request.hypothesis_id,
        simulation_type="none",
        parameters=dict(request.parameters),
        expected_results=dict(request.expected_results),
        observed_results={},
        metrics={},
        status="simulation_unavailable",
        errors=[reason],
        timestamp=_now(),
        adapter={"name": "none", "version": "n/a", "module": None},
        reproducibility={
            "seed": request.seed,
            "input_hash": request.input_hash(),
            "deterministic": False,
            "how_to_reproduce": "Not reproducible: no simulator was available.",
        },
        notes=[
            "simulation_unavailable — AARL refuses to fabricate a simulation result.",
            "Enable a matching adapter (or add one under custom_simulators/) to test "
            "this hypothesis.",
        ],
    )