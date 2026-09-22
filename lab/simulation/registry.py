"""
AARL Lab — Simulation engine registry
=====================================
Chooses the engine that should test a hypothesis and guarantees an honest result
either way:

* an **exact** domain match beats a wildcard engine (so a real ML experiment is
  preferred over a generic parametric model for ML questions);
* researcher-provided custom engines are registered first and win ties;
* if nothing matches, the caller receives a ``simulation_unavailable`` Simulation
  JSON with the reason — never a fabricated run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import SimulationEngine, SimulationRecord, SimulationRequest, unavailable_record
from .custom_adapter import discover_custom_simulators
from .domain_adapter import DomainSimulation
from .optimizer_adapter import MLOptimizationSimulation

log = logging.getLogger("aarl.lab.sim.registry")


def default_engines() -> List[SimulationEngine]:
    """Built-in engines in preference order (exact-domain ML first)."""
    return [MLOptimizationSimulation(), DomainSimulation()]


class SimulationRegistry:
    """Holds the available simulation engines and routes requests to one."""

    def __init__(
        self,
        engines: Optional[List[SimulationEngine]] = None,
        custom_simulator_dir: Optional[str] = None,
    ) -> None:
        self.engines: List[SimulationEngine] = []
        for engine in discover_custom_simulators(custom_simulator_dir or ""):
            self.register(engine)
        for engine in engines if engines is not None else default_engines():
            self.register(engine)

    # ------------------------------------------------------------------ registry
    def register(self, engine: SimulationEngine) -> None:
        self.engines.append(engine)

    def available(self) -> List[Dict[str, Any]]:
        """Capability manifest for reporting."""
        manifest: List[Dict[str, Any]] = []
        for engine in self.engines:
            manifest.append(
                {
                    "name": engine.name,
                    "version": engine.version,
                    "domains": list(engine.supported_domains),
                    "description": engine.description,
                    "source": (
                        "researcher-provided"
                        if engine.__class__.__name__ == "CustomSimulationAdapter"
                        else "built-in"
                    ),
                }
            )
        return manifest

    # ------------------------------------------------------------------ selection
    @staticmethod
    def _priority(engine: SimulationEngine, domain: str) -> Optional[int]:
        """0 = exact domain, 1 = wildcard, None = does not support."""
        domains = {str(d).lower() for d in engine.supported_domains}
        if not domains:
            return None
        if "*" in domains:
            return 1
        return 0 if (domain or "").lower() in domains else None

    def select(self, request: SimulationRequest) -> Optional[SimulationEngine]:
        """First engine by (priority, registration order) that can run this request."""
        best: Optional[SimulationEngine] = None
        best_priority = 99
        for engine in self.engines:
            try:
                if not engine.supports(
                    request.domain, request.research_question, request.hypothesis_text
                ):
                    continue
            except Exception as exc:  # noqa: BLE001 - a broken engine is skipped, visibly
                log.warning("engine %s.supports() failed: %s", engine.name, exc)
                continue
            priority = self._priority(engine, request.domain or "")
            if priority is None:
                priority = 2  # engine claims support without declaring the domain
            if priority < best_priority:
                best, best_priority = engine, priority
        return best

    # ------------------------------------------------------------------ execution
    def run(
        self, request: SimulationRequest, experiment_id: str = ""
    ) -> SimulationRecord:
        """Run the selected engine (or return a simulation_unavailable record)."""
        engine = self.select(request)
        if engine is None:
            record = unavailable_record(
                request,
                (
                    f"no simulation engine supports domain "
                    f"'{request.domain or 'undetected'}' "
                    f"(available: {[e.name for e in self.engines]})"
                ),
            )
            record.experiment_id = experiment_id
            return record
        try:
            record = engine.run(request)
        except Exception as exc:  # noqa: BLE001 - never lose the run
            record = unavailable_record(request, f"engine '{engine.name}' raised: {exc}")
        record.experiment_id = experiment_id or record.experiment_id
        return record