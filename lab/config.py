"""
AARL Lab — Configuration for the continuous research-learning loop
==================================================================
Additive to the existing stack: this dataclass holds every value the lab needs
for a reproducible cycle. It is snapshotted into the stored records, so an
experiment can be re-run from its stored parameters alone.

Nothing here mutates ``showcase.config.ResearchConfig`` — the showcase keeps its
own configuration and behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from showcase.config import asdict_flat, get_groq_api_key, get_groq_model


@dataclass
class LabConfig:
    """Typed configuration for one continuous research-learning cycle."""

    # ---- Research framing -------------------------------------------------
    research_question: str = "How can gradient descent convergence be improved?"
    #: Optional domain override ("" = let the simulation registry decide).
    domain_hint: str = ""

    # ---- Persistent research memory ---------------------------------------
    memory_dir: str = "research_memory"

    # ---- Iteration budget -------------------------------------------------
    num_cycles: int = 1
    num_iterations: int = 3
    hypotheses_per_iteration: int = 3
    #: A cycle stops early once an iteration improves on the baseline by at
    #: least this percentage (simulation-measured).
    success_threshold_pct: float = 25.0
    lesson_retrieval_k: int = 5

    # ---- Real simulation (optimiser adapter) ------------------------------
    num_runs: int = 3
    dimensionality: int = 12
    max_experiment_iterations: int = 4000
    convergence_tol: float = 1e-4

    # ---- Determinism ------------------------------------------------------
    random_seed: int = 11

    # ---- Parallelism (shared AARL worker pool) ----------------------------
    parallel: bool = True
    max_workers: int = 0

    # ---- LLM (planning / explanation only — never numbers) ----------------
    temperature: float = 0.4
    max_llm_retries: int = 3
    llm_timeout_seconds: int = 60
    llm_budget_tokens: int = 4096

    # ---- Custom simulator drop-ins ----------------------------------------
    #: "" -> <memory_dir>/custom_simulators
    custom_simulator_dir: str = ""

    # ---- derived helpers --------------------------------------------------
    def derive(self) -> Dict[str, Any]:
        """JSON-safe replica, snapshotted into records for reproducibility."""
        return asdict_flat(self)

    def llm_available(self) -> bool:
        """True when a Groq key is configured (LLM is optional everywhere)."""
        return bool(get_groq_api_key())

    def llm_model(self) -> str:
        return get_groq_model()

    def to_research_config(self):
        """Map onto the existing showcase config for the real experiment engine."""
        from showcase.config import ResearchConfig

        cfg = ResearchConfig()
        cfg.research_question = self.research_question
        cfg.num_runs = int(self.num_runs)
        cfg.dimensionality = int(self.dimensionality)
        cfg.max_iterations = int(self.max_experiment_iterations)
        cfg.convergence_tol = float(self.convergence_tol)
        cfg.random_seed = int(self.random_seed)
        cfg.temperature = float(self.temperature)
        cfg.max_llm_retries = int(self.max_llm_retries)
        cfg.llm_timeout_seconds = int(self.llm_timeout_seconds)
        cfg.llm_budget_tokens = int(self.llm_budget_tokens)
        cfg.parallel = bool(self.parallel)
        cfg.max_workers = int(self.max_workers)
        return cfg