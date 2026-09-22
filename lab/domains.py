"""
AARL Lab — Domain resolution for multi-domain research
=====================================================
Reuses the project's existing domain knowledge instead of duplicating it:

* ``Engine/Question_Engine/DomainDetector.py`` — multi-domain scoring with topics,
  matched keywords and math engines (step 1 of the loop: "identify domains").
* ``Research/simulators/domain_metrics.py`` — the 9 domains that actually have
  executable models, used to route simulation.

The two registries use different names, so the mapping lives here in one place.
If a detected domain has no executable model the lab records
``simulation_unavailable`` rather than borrowing another domain's physics.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

log = logging.getLogger("aarl.lab.domains")

#: Engine domain name -> domain_metrics model name ("" = no executable model).
ENGINE_TO_MODEL: Dict[str, str] = {
    "chemistry": "chemistry",
    "biology": "medicine",
    "materials_science": "materials",
    "computer_architecture": "computer_architecture",
    "physics": "",   # mechanics topics are routed through domain_metrics detection
    "quantum": "",   # no quantum model exists yet -> simulation_unavailable
}


def _load_engine_detector():
    """Import the existing multi-domain detector (both package and flat paths)."""
    try:
        from Engine.Question_Engine import DomainDetector as detector  # type: ignore

        return detector
    except Exception:  # pragma: no cover - flat path fallback
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        engine_dir = os.path.join(root, "Engine", "Question_Engine")
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        try:
            import DomainDetector as detector  # type: ignore

            return detector
        except Exception:
            return None


def _load_domain_metrics():
    try:
        from Research.simulators import domain_metrics as dm  # type: ignore

        return dm
    except Exception:  # pragma: no cover - flat path fallback
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sim_dir = os.path.join(root, "Research", "simulators")
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)
        try:
            import domain_metrics as dm  # type: ignore

            return dm
        except Exception:
            return None


def relevant_domains(research_question: str) -> List[Dict[str, Any]]:
    """Relevant domains for a research question (multi-domain, scored).

    Returns the existing detector's records (domain, score, matched keywords,
    topics, math engine, simulator); a best-effort list when it is unavailable.
    """
    detector = _load_engine_detector()
    records: List[Dict[str, Any]] = []
    if detector is not None:
        try:
            records = list(detector.detect_domains(research_question) or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("domain detection failed: %s", exc)
            records = []

    for record in records:
        model = ENGINE_TO_MODEL.get(str(record.get("domain", "")), "")
        record["model_domain"] = model
        record["simulation_available"] = bool(model)

    if not records:
        dm = _load_domain_metrics()
        if dm is not None:
            detected = str(dm.detect_domain(research_question) or "")
            records = [
                {
                    "domain": detected or "undetected",
                    "score": 0,
                    "matched_keywords": [],
                    "topics": [],
                    "math_engine": "",
                    "simulator": "",
                    "model_domain": detected,
                    "simulation_available": bool(detected),
                    "detection_source": "domain_metrics.detect_domain (fallback)",
                }
            ]
    return records


def primary_domain(research_question: str, hint: str = "") -> str:
    """The domain used to route simulation (explicit hint wins).

    Order: explicit hint -> Engine detector's best *simulatable* domain ->
    ``domain_metrics.detect_domain``.
    """
    if hint:
        return ENGINE_TO_MODEL.get(str(hint).lower(), str(hint).lower())

    for record in relevant_domains(research_question):
        model = str(record.get("model_domain", "") or "")
        if model:
            return model

    dm = _load_domain_metrics()
    if dm is not None:
        return str(dm.detect_domain(research_question) or "")
    return ""


def domain_context(research_question: str, limit: int = 8) -> List[str]:
    """Human-readable domain lines for reporting."""
    lines: List[str] = []
    for record in relevant_domains(research_question)[:limit]:
        topics = ", ".join(str(t) for t in (record.get("topics") or [])[:3])
        lines.append(
            f"{record.get('domain')} (score={record.get('score')}"
            f"{', topics: ' + topics if topics else ''}"
            f"{', simulation: ' + str(record.get('model_domain')) if record.get('model_domain') else ', no simulator'})"
        )
    return lines


def has_simulator(domain: str) -> bool:
    dm = _load_domain_metrics()
    if dm is None or not domain:
        return False
    return domain in getattr(dm, "DOMAIN_SIMULATORS", {})