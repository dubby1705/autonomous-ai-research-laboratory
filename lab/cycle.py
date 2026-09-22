"""
AARL Lab — Continuous Research Cycle Orchestrator
==================================================
Runs the full research-learning loop:

    research question -> domains -> knowledge -> hypotheses -> simulation
    -> failure analysis -> lessons -> improved hypotheses -> re-simulation
    -> best candidate -> real-world experiment PROPOSAL (human approved)
    -> master research state (single source of truth for reporting)

Every iteration is stored; nothing is deleted. The master research state is
written to ``research_memory/master_state.json`` and is the ONLY input of the
reporting layer.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from lab.config import LabConfig
from lab.domains import relevant_domains, primary_domain
from lab.knowledge import KnowledgeManager
from lab.memory import ResearchMemory
from lab.hypotheses import HypothesisEngine
from lab.experiments import ExperimentManager
from lab.failures import FailureAnalyzer, LessonManager
from lab.realworld import RealWorldManager

_SUCCESS = {"success", "promising", "completed"}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class ResearchCycle:
    def __init__(
        self,
        config: Optional[LabConfig] = None,
        llm_service: Optional[Any] = None,
        memory: Optional[ResearchMemory] = None,
    ):
        self.config = config or LabConfig()
        self.memory = memory or ResearchMemory(self.config.memory_dir)
        self.knowledge = KnowledgeManager(self.memory)
        self.llm = llm_service
        self.hypotheses = HypothesisEngine(self.memory, self.knowledge, self.config, llm_service)
        self.experiments = ExperimentManager(self.memory, self.config)
        self.failure_analyzer = FailureAnalyzer(self.memory)
        self.lessons = LessonManager(self.memory)
        self.realworld = RealWorldManager(self.memory, self.knowledge, self.config)

    # ------------------------------------------------------------------ run
    def run(self, research_question: Optional[str] = None) -> Dict[str, Any]:
        question = research_question or self.config.research_question
        if not question:
            raise ValueError("A research question is required (config or argument).")

        domains = relevant_domains(question)
        domain_entries = [d if isinstance(d, dict) else {"domain": d} for d in domains]
        primary = primary_domain(question)

        self.memory.append_history({
            "event": "research_cycle_started",
            "research_question": question,
            "primary_domain": primary,
            "domains": [str(d.get("domain", d)) for d in domain_entries],
        })

        iteration_log: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        threshold = float(getattr(self.config, "success_threshold_pct", 0.0) or 0.0)

        for iteration in range(1, int(getattr(self.config, "num_iterations", 3) or 3) + 1):
            hyps = self.hypotheses.generate(question, iteration, primary)
            round_record: Dict[str, Any] = {"iteration": iteration, "hypotheses": [], "experiments": []}

            for hyp in hyps:
                simulation = self.experiments.run(hyp, question, iteration)
                round_record["experiments"].append(simulation["experiment_id"])

                failure = None
                lesson = None
                if self.failure_analyzer.is_failure(simulation):
                    failure = self.failure_analyzer.analyze(simulation)
                    lesson = self.lessons.from_failure(failure)

                round_record["hypotheses"].append({
                    "hypothesis_id": hyp["hypothesis_id"],
                    "experiment_id": simulation["experiment_id"],
                    "status": simulation.get("status", ""),
                    "failure_id": (failure or {}).get("failure_id", ""),
                    "lesson_id": (lesson or {}).get("lesson_id", ""),
                })

                best = self._maybe_update_best(best, hyp, simulation)
                if best is not None and best.get("best_improvement_pct", -1e9) >= threshold:
                    round_record["threshold_met"] = True

            iteration_log.append(round_record)
            if round_record.get("threshold_met"):
                break

        # ---- real-world proposal for the best simulated candidate ----------
        proposal = None
        if best is not None:
            sim = self.memory.get("simulations", best["experiment_id"])
            proposal = self.realworld.propose(best, sim)

        state = self._master_state(
            question, domain_entries, primary, iteration_log, best, proposal
        )
        self._save_master_state(state)
        self.memory.append_history({
            "event": "research_cycle_completed",
            "research_question": question,
            "iterations": len(iteration_log),
            "best_experiment_id": (best or {}).get("experiment_id", ""),
            "proposal_id": (proposal or {}).get("proposal_id", ""),
        })
        return state

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _improvement_of(simulation: Dict[str, Any]) -> Optional[float]:
        """Best improvement_pct across the simulation's metrics (or None)."""
        best: Optional[float] = None
        for _, entry in (simulation.get("metrics") or {}).items():
            if isinstance(entry, dict):
                value = entry.get("improvement_pct")
                if isinstance(value, (int, float)):
                    best = value if best is None else max(best, value)
        return best

    def _maybe_update_best(
        self,
        best: Optional[Dict[str, Any]],
        hyp: Dict[str, Any],
        simulation: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        status = str(simulation.get("status", "") or "").lower()
        if status not in _SUCCESS:
            return best
        improvement = self._improvement_of(simulation)
        if improvement is None:
            return best
        candidate = {
            "hypothesis_id": hyp["hypothesis_id"],
            "hypothesis": hyp.get("hypothesis", ""),
            "experiment_id": simulation["experiment_id"],
            "status": status,
            "best_improvement_pct": improvement,
            "metrics": simulation.get("metrics", {}),
            "iteration": simulation.get("iteration", ""),
        }
        if best is None or improvement > best.get("best_improvement_pct", -1e9):
            return candidate
        return best

    def _master_state(
        self,
        question: str,
        domain_entries: List[Dict[str, Any]],
        primary: str,
        iteration_log: List[Dict[str, Any]],
        best: Optional[Dict[str, Any]],
        proposal: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        simulations = self.memory.load_collection("simulations")
        failures = self.memory.load_collection("failures")
        lessons = self.memory.load_collection("lessons")
        hypotheses = self.memory.load_collection("hypotheses")
        papers = self.memory.load_collection("papers")
        rwe = self.memory.load_collection("real_world_experiments")
        knowledge_summary = self.knowledge.summary()
        knowledge_items = [
            i.to_dict() if hasattr(i, "to_dict") else dict(i)
            for i in self.knowledge.items()
        ]

        return {
            "generated_at": _now(),
            "research_question": question,
            "primary_domain": primary,
            "relevant_domains": domain_entries,
            "papers": papers,
            "hypotheses": hypotheses,
            "simulations": simulations,
            "failures": failures,
            "lessons": lessons,
            "real_world": {
                "proposals": [r for r in rwe if str(r.get("record_id", "")).startswith("RWE")],
                "results": [r for r in rwe if str(r.get("record_id", "")).startswith("RWR")],
                "best_proposal_id": (proposal or {}).get("proposal_id", ""),
                "validated": bool(
                    any(str(r.get("record_id", "")).startswith("RWR") for r in rwe)
                ),
            },
            "iterations": iteration_log,
            "best_candidate": best or {},
            "knowledge_summary": knowledge_summary,
            "knowledge_items": knowledge_items,
            "simulation_availability": {
                "engines": self.experiments.available_domains(),
                "note": (
                    "Experiments without a supporting engine are recorded with "
                    "status 'simulation_unavailable' — no simulated results are claimed."
                ),
            },
        }

    def _save_master_state(self, state: Dict[str, Any]) -> None:
        import os
        ResearchMemory._atomic_write_json(
            os.path.join(self.memory.root, "master_state.json"), state
        )


