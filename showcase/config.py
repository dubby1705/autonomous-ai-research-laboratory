"""
AARL Showcase — Central Configuration
=======================================
Single source of truth for all configurable values.

- Loads environment variables from a `.env` file (via python-dotenv) and from
  the process environment.
- Never stores an API key in any source file; the key must live in `.env` or the
  environment.
- Exposes typed, bounded configuration so magic numbers are not scattered across
  the research code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load `.env` from the repository root (the current working directory).
load_dotenv(os.path.join(os.getcwd(), ".env"))

# ---------------------------------------------------------------------------
# LLM / Groq settings
# ---------------------------------------------------------------------------
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL_ENV = "GROQ_MODEL"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Model names known to work with Groq. Used for validation before a request is sent.
_KNOWN_GROQ_MODELS: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


def get_groq_api_key() -> str:
    """Return the configured Groq API key, or an empty string if unset."""
    return os.environ.get(GROQ_API_KEY_ENV, "").strip()


def get_groq_model() -> str:
    """Return the configured Groq model name."""
    model = os.environ.get(GROQ_MODEL_ENV, "").strip() or DEFAULT_GROQ_MODEL
    return model


def is_known_groq_model(model: str) -> bool:
    return model in _KNOWN_GROQ_MODELS


@dataclass
class ResearchConfig:
    """Typed, centralised configuration for a research run."""

    # ---- Topic / domain ------------------------------------------------
    research_question: str = (
        "How can gradient descent convergence be improved?"
    )
    domain: str = "machine_learning_optimization"

    # ---- Run storage ---------------------------------------------------
    run_root: str = "runs"

    # ---- Pipeline scale -----------------------------------------------
    num_hypotheses: int = 8
    tournament_pass_threshold: float = 70.0  # overall score out of 100 to PASS

    # ---- Experiments -----------------------------------------------------
    num_runs: int = 5
    dimensionality: int = 12
    max_iterations: int = 4000
    convergence_tol: float = 1e-4
    time_tolerance: float = 1e-6

    # ---- Determinism -----------------------------------------------------
    random_seed: int = 42

    # ---- Parallelism -----------------------------------------------------
    # The showcase fans independent work (experiments, report + plots) out over
    # the shared AARL worker pool. max_workers=0 means "all CPU cores"; set
    # AARL_MAX_WORKERS / AARL_PARALLEL=0 in the environment to pin or disable.
    parallel: bool = True
    max_workers: int = 0

    # ---- LLM ---------------------------------------------------------------
    temperature: float = 0.4
    max_llm_retries: int = 3
    llm_timeout_seconds: int = 60
    llm_budget_tokens: int = 4096

    # ---- safe experiment registry ------------------------------------------
    # Which optimisation methods the experiment designer is allowed to select.
    allowed_optimizers: List[str] = field(
        default_factory=lambda: [
            "sgd",
            "momentum",
            "nesterov",
            "adagrad",
            "rmsprop",
            "adam",
            "amsgrad",
            "adam_cosine",
            "sgd_cosine",
            "sgd_warm_restart",
        ]
    )

    def derive(self) -> Dict[str, Any]:
        """Return a dict replica for persistence / reproducibility."""
        return asdict_flat(self)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "ResearchConfig":
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k) and k not in ("allowed_optimizers",):
                setattr(cfg, k, v)
        return cfg


def asdict_flat(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass (recursively) into a JSON-safe dict."""
    from dataclasses import is_dataclass

    if is_dataclass(obj):
        return {k: asdict_flat(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {str(k): asdict_flat(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [asdict_flat(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback for arbitrary objects (e.g. tuples, enums) -> string
    return str(obj)