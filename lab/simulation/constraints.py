"""
AARL Lab — Declared constraint envelopes for the domain simulator
=================================================================
These limits are **human-declared engineering envelopes**, not physics that AARL
derived. Each one carries its own ``source`` string so the Simulation JSON and the
report can always show where a limit came from, and each is treated as an
*assumption* (never as a fact).

They exist so a simulation can honestly report ``constraints_violated`` — e.g. a
hypothesis that pushes a chip design past a practical clock or power envelope —
instead of silently reporting a "great" number.
"""

from __future__ import annotations

from typing import Any, Dict, List

_CONSTRAINTS: Dict[str, List[Dict[str, Any]]] = {
    "computer_architecture": [
        {
            "name": "clock_frequency_ghz",
            "op": "<=",
            "value": 8.0,
            "source": "human-declared practical CMOS clock envelope (lab constraint set v1)",
        },
        {
            "name": "power_consumption_w",
            "op": "<=",
            "value": 400.0,
            "source": "human-declared datacentre-socket power envelope (lab constraint set v1)",
        },
        {
            "name": "area_mm2",
            "op": "<=",
            "value": 900.0,
            "source": "human-declared reticle-limited die area (lab constraint set v1)",
        },
    ],
    "battery": [
        {
            "name": "temperature_rise_c",
            "op": "<=",
            "value": 60.0,
            "source": "human-declared thermal safety envelope (lab constraint set v1)",
        },
    ],
    "aerospace": [
        {
            "name": "drag_coefficient",
            "op": "<=",
            "value": 0.8,
            "source": "human-declared drag plausibility envelope (lab constraint set v1)",
        },
    ],
    "machine_learning": [
        {
            "name": "inference_latency_ms",
            "op": "<=",
            "value": 500.0,
            "source": "human-declared latency budget (lab constraint set v1)",
        },
    ],
}

#: Constraints that apply to every domain (empty by design, kept explicit).
DEFAULT_CONSTRAINTS: List[Dict[str, Any]] = []


def constraints_for(domain: str) -> List[Dict[str, Any]]:
    """Declared constraints for a domain (a copy, so callers can extend them)."""
    return [dict(item) for item in _CONSTRAINTS.get((domain or "").lower(), DEFAULT_CONSTRAINTS)]


def known_domains() -> List[str]:
    return sorted(_CONSTRAINTS.keys())