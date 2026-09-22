"""
AARL Lab — Simulation package
=============================
Modular simulation layer for the continuous research-learning loop.

    SimulationEngine (ABC)          base.py       — Simulation JSON contract
    ├── MLOptimizationSimulation    optimizer_adapter.py — real ML experiments
    ├── DomainSimulation            domain_adapter.py    — existing domain models
    └── CustomSimulationAdapter     custom_adapter.py    — researcher drop-ins
    SimulationRegistry              registry.py   — selection + honest fallback
    constraints_for(domain)         constraints.py — declared envelopes
"""

from .base import (  # noqa: F401
    SIMULATION_STATUSES,
    SimulationEngine,
    SimulationRecord,
    SimulationRequest,
    unavailable_record,
)
from .custom_adapter import CustomSimulationAdapter, discover_custom_simulators  # noqa: F401
from .domain_adapter import DomainSimulation  # noqa: F401
from .optimizer_adapter import MLOptimizationSimulation  # noqa: F401
from .registry import SimulationRegistry, default_engines  # noqa: F401

__all__ = [
    "SIMULATION_STATUSES",
    "SimulationEngine",
    "SimulationRecord",
    "SimulationRequest",
    "unavailable_record",
    "CustomSimulationAdapter",
    "discover_custom_simulators",
    "DomainSimulation",
    "MLOptimizationSimulation",
    "SimulationRegistry",
    "default_engines",
]