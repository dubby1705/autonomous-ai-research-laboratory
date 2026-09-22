"""
AARL Lab — Custom simulation adapter (domain-extensibility hook)
===============================================================
So the lab is *not* hardcoded to the domains that ship with AARL, a researcher can
drop a module into ``research_memory/custom_simulators/`` and it becomes a
first-class simulation engine.

Contract for a custom module
----------------------------
It must expose one of these callables::

    def simulate(parameters: dict, seed: int | None = None) -> dict
    def run_simulation(hypothesis: str, problem: str = "") -> dict

and may describe itself (optional but recommended)::

    NAME = "my_simulator"          # engine name
    VERSION = "0.1"
    DOMAINS = ["networking", "photonics"]   # domains this engine can serve
    CONSTRAINTS = [{"name": ..., "op": ..., "value": ..., "source": ...}]

The returned dict is split honestly:

* numbers under ``baseline``/``observed`` (or a flat numeric dict) become
  **metrics** (stored as ``simulation_observation``);
* any ``constraints_violated`` / ``errors`` the module reports are preserved;
* everything else is kept verbatim under ``observed_results.raw``.

A module that cannot be imported or called yields ``simulation_unavailable`` — the
loop reports a missing capability instead of inventing one.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional

from .base import SimulationEngine, SimulationRecord, SimulationRequest, _now

log = logging.getLogger("aarl.lab.sim.custom")


def _split_numeric(payload: Dict[str, Any]) -> Dict[str, float]:
    return {
        str(key): float(value)
        for key, value in dict(payload or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class CustomSimulationAdapter(SimulationEngine):
    """Adapter around one researcher-provided simulator module."""

    def __init__(self, module_path: str) -> None:
        self.module_path = os.path.abspath(module_path)
        self.name = os.path.splitext(os.path.basename(module_path))[0]
        self.version = "0.1"
        self.description = f"researcher-provided simulator at {self.module_path}"
        self.supported_domains: tuple = ()
        self.declared_constraints: List[Dict[str, Any]] = []
        self._module = None

    # ------------------------------------------------------------------ loading
    def load(self):
        """Import the custom module (cached). Returns None when unusable."""
        if self._module is not None:
            return self._module
        try:
            spec = importlib.util.spec_from_file_location(
                f"aarl_custom_sim_{self.name}", self.module_path
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - a broken drop-in must not kill the loop
            log.warning("custom simulator %s failed to import: %s", self.module_path, exc)
            return None

        self._module = module
        self.name = str(getattr(module, "NAME", self.name))
        self.version = str(getattr(module, "VERSION", self.version))
        self.description = str(getattr(module, "DESCRIPTION", self.description))
        domains = getattr(module, "DOMAINS", None)
        if isinstance(domains, (list, tuple)) and domains:
            self.supported_domains = tuple(str(d) for d in domains)
        self.declared_constraints = [
            dict(c) for c in (getattr(module, "CONSTRAINTS", None) or []) if isinstance(c, dict)
        ]
        return self._module

    def is_usable(self) -> bool:
        module = self.load()
        if module is None:
            return False
        return callable(getattr(module, "simulate", None)) or callable(
            getattr(module, "run_simulation", None)
        )

    def supports(self, domain: str, research_question: str, hypothesis_text: str) -> bool:
        if not self.is_usable():
            return False
        if not self.supported_domains:
            return False  # an undeclared scope is never assumed to cover anything
        return super().supports(domain, research_question, hypothesis_text)
# ------------------------------------------------------------------ run
    def run(self, request: SimulationRequest) -> SimulationRecord:
        record = self.new_record(request, status="failure")
        record.simulation_type = f"custom:{self.name}"
        module = self.load()
        if module is None or not self.is_usable():
            record.status = "simulation_unavailable"
            record.simulation_type = "none"
            record.errors.append(
                f"custom simulator '{self.module_path}' could not be loaded or exposes "
                "no simulate()/run_simulation() callable"
            )
            return record

        sim_fn = getattr(module, "simulate", None)
        try:
            if callable(sim_fn):
                raw = sim_fn(dict(request.parameters or {}), request.seed)
            else:
                raw = module.run_simulation(
                    request.hypothesis_text, request.research_question
                )
        except Exception as exc:  # noqa: BLE001
            record.status = "failure"
            record.errors.append(f"custom simulator raised: {exc}")
            return record

        if not isinstance(raw, dict):
            record.status = "failure"
            record.errors.append("custom simulator returned a non-dict result")
            return record

        baseline_numbers = _split_numeric(raw.get("baseline") or {})
        observed_numbers = _split_numeric(raw.get("observed") or {})
        if not observed_numbers:
            # a flat numeric dict is accepted as the observed result
            observed_numbers = _split_numeric(
                {k: v for k, v in raw.items() if not isinstance(v, (dict, list))}
            )

        metrics: Dict[str, Dict[str, Any]] = {}
        for name, value in observed_numbers.items():
            baseline_value: Optional[float] = baseline_numbers.get(name)
            if baseline_value is None:
                metrics[name] = {
                    "baseline_value": None,
                    "observed_value": round(value, 6),
                    "change_pct": None,
                    "improvement_pct": None,
                    "is_improvement": None,
                    "direction": "unknown",
                    "unit": "",
                    "description": "custom simulator metric (no baseline supplied)",
                }
                continue
            change = (
                0.0
                if baseline_value == 0
                else 100.0 * (value - baseline_value) / abs(baseline_value)
            )
            metrics[name] = {
                "baseline_value": round(baseline_value, 6),
                "observed_value": round(value, 6),
                "change_pct": round(change, 4),
                "improvement_pct": round(change, 4),
                "is_improvement": bool(change > 0),
                "direction": "higher_better",
                "unit": "",
                "description": "custom simulator metric (direction assumed higher_better)",
            }

        record.metrics = metrics
        record.observed_results = {
            "raw": raw,
            "baseline": baseline_numbers,
            "observed": observed_numbers,
        }
        record.status = "success" if metrics else "failure"
        if not metrics:
            record.errors.append("custom simulator produced no numeric metrics")
        if not baseline_numbers:
            record.notes.append(
                "The custom simulator supplied no baseline, so its numbers are not "
                "comparable with an AARL baseline."
            )

        constraints = list(request.constraints or []) or list(self.declared_constraints)
        record.constraints_violated = self.check_constraints(constraints, observed_numbers)
        reported = raw.get("constraints_violated")
        if isinstance(reported, list):
            record.constraints_violated.extend(
                {"constraint": item, "source": "custom simulator report"}
                for item in reported
                if isinstance(item, (dict, str))
            )
        errors = raw.get("errors")
        if isinstance(errors, list):
            record.errors.extend(str(item) for item in errors)

        record.timestamp = _now()
        return record


def discover_custom_simulators(directory: str) -> List[CustomSimulationAdapter]:
    """Every usable ``*.py`` drop-in in a directory (``_``-prefixed files ignored)."""
    if not directory or not os.path.isdir(directory):
        return []
    found: List[CustomSimulationAdapter] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        adapter = CustomSimulationAdapter(os.path.join(directory, name))
        if adapter.is_usable():
            found.append(adapter)
    return found