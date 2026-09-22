"""
AARL Showcase — an autonomous AI research laboratory unit.

Pipeline:
  research problem -> hypothesis generation -> critical evaluation
  -> research tournament (multiple survivors) -> experiment design
    -> real simulation -> measured results -> LLM analysis
    -> hypothesis refinement -> final research report
"""

__version__ = "1.0.0"

from .config import ResearchConfig  # noqa: F401
from .records import (  # noqa: F401
    Hypothesis,
    ResearchRun,
    ExperimentSpec,
    ExperimentResult,
    Evaluation,
    Analysis,
    RefinedHypothesis,
)

__all__ = [
    "ResearchConfig",
    "Hypothesis",
    "ResearchRun",
    "ExperimentSpec",
    "ExperimentResult",
    "Evaluation",
    "Analysis",
    "RefinedHypothesis",
]