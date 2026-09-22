"""
AARL Lab — Experiment Manager (Simulation JSON layer)
=====================================================
Selects a SimulationEngine for a hypothesis (or records an honest
``simulation_unavailable`` when none supports the problem), runs it, and
persists the full Simulation JSON into research memory (EXP-###).  Results are
never invented: everything in the record comes from the executed engine.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Dict, List, Optional

from lab.simulation.base import SimulationRequest
from lab.simulation.registry import (
    default_engines,
    discover_custom_simulators,
    unavailable_record,
)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class ExperimentManager:
    def __init__(self, memory, config):
        self.memory = memory
        self.config = config
        self.engines: List[Any] = list(default_engines())
        custom_dir = getattr(config, "custom_simulator_dir", "") or ""
        if custom_dir:
            try:
                self.engines.extend(discover_custom_simulators(custom_dir))
            except Exception as exc:  # noqa: BLE001 - custom sims are optional
                self.memory.append_history({
                    "event": "custom_simulator_discovery_failed",
                    "directory": custom_dir,
                    "error": str(exc),
                })

    # ------------------------------------------------------------------
    def _select_engine(self, domain: str, research_question: str, hypothesis_text: str):
        for engine in self.engines:
            try:
                if engine.supports(domain, research_question, hypothesis_text):
                    return engine
            except Exception:  # noqa: BLE001 - a broken engine must not kill the cycle
                continue
        return None

    def available_domains(self) -> List[str]:
        names = []
        for engine in self.engines:
            try:
                names.append(str(engine.adapter_info().get("name", type(engine).__name__)))
            except Exception:  # noqa: BLE001
                names.append(type(engine).__name__)
        return names

    # ------------------------------------------------------------------
    def run(self, hypothesis: Dict[str, Any], research_question: str, iteration: int) -> Dict[str, Any]:
        domain = str(hypothesis.get("domain", "") or "")
        hyp_text = hypothesis.get("hypothesis") or hypothesis.get("statement") or ""
        hyp_id = hypothesis.get("hypothesis_id", "")

        request = SimulationRequest(
            hypothesis_id=hyp_id,
            hypothesis_text=hyp_text,
            research_question=research_question,
            domain=domain,
            parameters=dict(hypothesis.get("parameters") or {}),
            expected_results=dict(hypothesis.get("expected_outcome") or {}),
            constraints=list(hypothesis.get("constraints") or []),
            seed=int(getattr(self.config, "random_seed", 11)),
        )

        engine = self._select_engine(domain, research_question, hyp_text)
        if engine is None:
            record = unavailable_record(
                request,
                reason=(
                    "No registered simulation engine supports this research domain. "
                    f"Available engines: {', '.join(self.available_domains())}. "
                    "Recorded honestly as 'simulation_unavailable' — no simulated results are claimed."
                ),
            )
        else:
            try:
                record = engine.run(request)
            except Exception as exc:  # noqa: BLE001 - engine crash is a recorded failure
                record = unavailable_record(request, reason=f"Simulation engine raised: {exc}")
                record.status = "failure"
                record.errors.append(f"engine_exception: {exc}")

        record.iteration = iteration  # type: ignore[attr-defined]
        data = dataclasses.asdict(record) if dataclasses.is_dataclass(record) and not isinstance(record, type) else dict(record.__dict__)
        data["iteration"] = iteration
        experiment_id = data.get("experiment_id") or self.memory.next_id("simulations", "EXP")
        data["experiment_id"] = experiment_id

        path = self.memory.append_record("simulations", experiment_id, data)
        self.memory.append_history({
            "event": "simulation_complete",
            "experiment_id": experiment_id,
            "hypothesis_id": hyp_id,
            "iteration": iteration,
            "status": data.get("status", ""),
            "simulation_type": data.get("simulation_type", ""),
        })
        data["path"] = path
        return data
