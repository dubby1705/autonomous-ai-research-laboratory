"""
AARL Lab — Failure Analysis & Lesson Memory
============================================
When a simulation fails or behaves unexpectedly, ``FailureAnalyzer`` produces a
structured failure record (FAIL-###) that answers *why* it failed: which
assumption broke, which parameter caused the problem, and what to change.
``LessonManager`` distils each failure into a reusable lesson record (LESSON-###)
that future hypothesis generation retrieves. Nothing is ever deleted.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

_SUCCESS_STATUSES = {"success", "promising", "completed"}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    """Render a JSON value (dict/list/other) as a compact string for records."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        import json
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


class FailureAnalyzer:
    """Analyse a Simulation JSON record into a structured failure record."""

    def __init__(self, memory):
        self.memory = memory

    # ------------------------------------------------------------------
    def is_failure(self, simulation: Dict[str, Any]) -> bool:
        """A failure record is created for non-success simulations or any
        experiment with violated constraints or unexpected results."""
        status = str(simulation.get("status", "") or "").lower()
        if status and status not in _SUCCESS_STATUSES:
            return True
        if simulation.get("constraints_violated"):
            return True
        if simulation.get("unexpected_results"):
            return True
        if simulation.get("errors"):
            return True
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_failure_type(simulation: Dict[str, Any]) -> str:
        if simulation.get("constraints_violated"):
            return "constraint_violation"
        if simulation.get("errors"):
            return "simulation_error"
        if simulation.get("unexpected_results"):
            return "unexpected_result"
        status = str(simulation.get("status", "") or "").lower()
        if status in {"partial", "inconclusive"}:
            return "inconclusive_result"
        return "hypothesis_not_supported"

    @staticmethod
    def _infer_cause_and_assumption(simulation: Dict[str, Any]):
        """Return (cause, failed_assumption) inferred from the actual record.

        Everything is derived from the simulation record itself — no invented
        causes. An explanation supplied by the engine is used verbatim.
        """
        cause_parts: List[str] = []
        failed_assumption = ""

        observed = simulation.get("observed_results") or {}
        if isinstance(observed, dict):
            explanation = observed.get("explanation") or observed.get("cause") or ""
            if explanation:
                cause_parts.append(str(explanation))

        for c in simulation.get("constraints_violated") or []:
            constraint = c if isinstance(c, str) else json_safe(c)
            cause_parts.append(f"constraint violated: {constraint}")
            if not failed_assumption:
                failed_assumption = (
                    f"The constraint '{constraint}' was assumed to hold under the "
                    "hypothesis parameters, but was violated in simulation."
                )

        for u in (simulation.get("unexpected_results") or [])[:2]:
            cause_parts.append(f"unexpected behavior: {u if isinstance(u, str) else json_safe(u)}")
        for e in (simulation.get("errors") or [])[:2]:
            cause_parts.append(f"error: {e if isinstance(e, str) else json_safe(e)}")

        notes = str(simulation.get("notes", "") or "")
        if notes and not cause_parts:
            cause_parts.append(notes)

        if not failed_assumption and simulation.get("expected_results"):
            failed_assumption = (
                "Expected results were assumed achievable with the hypothesis "
                "parameters; observed results deviated from them."
            )

        cause = "; ".join(cause_parts) if cause_parts else "cause not determinable from the simulation record"
        return cause, failed_assumption

    @staticmethod
    def _infer_changes(simulation: Dict[str, Any]) -> List[str]:
        """Recommended parameter/assumption changes derived from the record."""
        changes: List[str] = []
        params = simulation.get("parameters") or {}
        for c in simulation.get("constraints_violated") or []:
            text = c if isinstance(c, str) else json_safe(c)
            matched = False
            for key in params:
                if str(key) in text:
                    changes.append(f"Reduce or relax parameter '{key}' (violated constraint: {text})")
                    matched = True
            if not matched:
                changes.append(f"Re-design the configuration so that '{text}' is satisfied")
        for u in simulation.get("unexpected_results") or []:
            changes.append(
                "Investigate and model the unexpected behavior in the next "
                f"iteration: {u if isinstance(u, str) else json_safe(u)}"
            )
        if not changes:
            changes.append("Adjust hypothesis parameters and re-run; avoid repeating this configuration")
        seen, out = set(), []
        for ch in changes:
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
        return out

    @staticmethod
    def _severity(simulation: Dict[str, Any]) -> str:
        if simulation.get("errors") or simulation.get("constraints_violated"):
            return "high"
        if str(simulation.get("status", "") or "").lower() == "failure":
            return "high"
        if simulation.get("unexpected_results"):
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    def analyze(self, simulation: Dict[str, Any]) -> Dict[str, Any]:
        failure_type = self._infer_failure_type(simulation)
        cause, failed_assumption = self._infer_cause_and_assumption(simulation)

        failure_id = self.memory.next_id("failures", "FAIL")
        record = {
            "failure_id": failure_id,
            "experiment_id": simulation.get("experiment_id", ""),
            "hypothesis_id": simulation.get("hypothesis_id", ""),
            "iteration": simulation.get("iteration", ""),
            "failure_type": failure_type,
            "cause": cause,
            "failed_assumption": failed_assumption,
            "observed_behavior": json_safe(simulation.get("observed_results")),
            "expected_behavior": json_safe(simulation.get("expected_results")),
            "lesson": "",  # filled by LessonManager.from_failure
            "recommended_changes": self._infer_changes(simulation),
            "severity": self._severity(simulation),
            "evidence": [
                f"simulation record: {simulation.get('experiment_id', '')}",
                f"simulation status: {simulation.get('status', '')}",
                f"metrics: {json_safe(simulation.get('metrics'))}",
            ],
            "provenance": {
                "derived_from": "simulation_record",
                "experiment_id": simulation.get("experiment_id", ""),
                "simulation_type": simulation.get("simulation_type", ""),
                "generated_by": "aarl.lab.failures.FailureAnalyzer",
            },
            "timestamp": _now(),
        }
        self.memory.append_record("failures", failure_id, record)
        self.memory.append_history({
            "event": "failure_analyzed",
            "failure_id": failure_id,
            "experiment_id": record["experiment_id"],
            "hypothesis_id": record["hypothesis_id"],
            "failure_type": failure_type,
            "severity": record["severity"],
        })
        return record


class LessonManager:




    """Distil failures into reusable lesson records (LESSON-###)."""

    def __init__(self, memory):
        self.memory = memory

    def from_failure(self, failure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rec_changes = failure.get("recommended_changes") or []
        failed_assumption = failure.get("failed_assumption") or ""
        observed = failure.get("observed_behavior") or ""

        if failed_assumption:
            lesson = (
                f"Do not assume the failed assumption holds unconditionally: {failed_assumption}. "
                f"Observed: {observed}."
            )
        elif rec_changes:
            lesson = f"From {failure['failure_type']}: apply the recommended changes ({rec_changes[0]})"
        else:
            lesson = f"{failure['failure_type']} observed; avoid repeating this configuration: {observed[:200]}"

        lesson_id = self.memory.next_id("lessons", "LESSON")
        record = {
            "lesson_id": lesson_id,
            "failure_id": failure.get("failure_id", ""),
            "experiment_id": failure.get("experiment_id", ""),
            "hypothesis_id": failure.get("hypothesis_id", ""),
            "iteration": failure.get("iteration", ""),
            "failure_type": failure.get("failure_type", ""),
            "lesson": lesson,
            "recommended_changes": list(rec_changes),
            "applied_in": [],          # hypothesis ids that later used this lesson
            "severity": failure.get("severity", "medium"),
            "provenance": {
                "derived_from": "simulation_failure_analysis",
                "failure_id": failure.get("failure_id", ""),
                "simulation_record": failure.get("experiment_id", ""),
                "generated_by": "aarl.lab.failures.LessonManager",
            },
            "timestamp": _now(),
        }
        self.memory.append_record("lessons", lesson_id, record)
        self.memory.append_history({
            "event": "lesson_recorded",
            "lesson_id": lesson_id,
            "failure_id": record["failure_id"],
            "hypothesis_id": record["hypothesis_id"],
        })
        return record

    def mark_applied(self, lesson_id: str, hypothesis_id: str) -> None:
        lesson = self.memory.get("lessons", lesson_id)
        if not lesson:
            return
        applied = lesson.get("applied_in") or []
        if hypothesis_id not in applied:
            applied.append(hypothesis_id)
            lesson["applied_in"] = applied
            # append_record never destroys the previous version (revisioning)
            self.memory.append_record("lessons", lesson_id, lesson, revision_of="lesson applied by new hypothesis")

    def retrieve(self, k: int = 5) -> List[Dict[str, Any]]:
        """Most recent lessons, newest first (used by hypothesis generation)."""
        lessons = self.memory.load_collection("lessons")
        lessons.sort(key=lambda r: str(r.get("stored_at", "")), reverse=True)
        return lessons[:k]
