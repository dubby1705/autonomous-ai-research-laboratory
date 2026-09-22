"""
ConfidenceEngine.py — Confidence Scoring Engine
================================================
Every surviving idea receives a confidence score based on:

  - literature support
  - engineering feasibility
  - validation results
  - simulation
  - manufacturing
  - consistency
  - comparison with state of the art

The system does NOT claim that an idea is correct.
It estimates how promising each idea is based on available evidence.
"""

import os
import json
import math
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from Engine.Question_Engine.Validation.validation_engine import (
    get_validation_engine,
    PASS,
    WARNING,
    FAIL,
)


# =========================================================
# CONFIDENCE DIMENSIONS
# =========================================================
CONFIDENCE_DIMENSIONS = [
    {
        "name": "literature_support",
        "description": "How much existing literature/prior art supports this idea",
        "weight": 0.20,
        "max_score": 1.0,
    },
    {
        "name": "engineering_feasibility",
        "description": "Can this be engineered with current technology",
        "weight": 0.20,
        "max_score": 1.0,
    },
    {
        "name": "validation_results",
        "description": "Results from the validation engine",
        "weight": 0.25,
        "max_score": 1.0,
    },
    {
        "name": "simulation",
        "description": "Simulation results and predicted performance",
        "weight": 0.10,
        "max_score": 1.0,
    },
    {
        "name": "manufacturing",
        "description": "Manufacturability and scalability",
        "weight": 0.10,
        "max_score": 1.0,
    },
    {
        "name": "consistency",
        "description": "Consistency with known scientific laws",
        "weight": 0.10,
        "max_score": 1.0,
    },
    {
        "name": "state_of_art_comparison",
        "description": "Comparison with current state of the art",
        "weight": 0.05,
        "max_score": 1.0,
    },
]


# =========================================================
# CONFIDENCE ENGINE
# =========================================================
class ConfidenceEngine:
    """Computes confidence scores for candidate ideas."""

    def __init__(self, problem_statement: str):
        self.problem = problem_statement
        self.validation_engine = get_validation_engine()

    def _score_literature_support(self, idea: str) -> float:
        """Score literature support based on keyword coverage of established concepts."""
        idea_lower = idea.lower()
        score = 0.0

        # Established scientific concepts indicate literature support
        established_concepts = [
            "graphene", "silicon", "lithium", "polymer", "ceramic", "composite",
            "nanoparticle", "nanotube", "electrolyte", "catalyst", "semiconductor",
            "transistor", "quantum", "neural", "algorithm", "optimization",
            "electrochemical", "thermodynamic", "aerodynamic", "structural",
            "bioavailability", "pharmacokinetic", "control", "feedback",
        ]
        hits = sum(1 for c in established_concepts if c in idea_lower)
        score += min(0.6, hits * 0.1)

        # Specificity indicates deeper literature grounding
        if re.search(r'\d+', idea):
            score += 0.2
        if len(idea.split()) >= 12:
            score += 0.2

        return min(1.0, score)

    def _score_engineering_feasibility(self, idea: str) -> float:
        """Score engineering feasibility based on practical implementation keywords."""
        idea_lower = idea.lower()
        score = 0.0

        # Feasibility indicators
        feasible_terms = [
            "implement", "build", "construct", "design", "develop", "engineer",
            "manufactur", "fabricat", "integrate", "deploy", "produce",
            "scale", "prototype", "test", "validate", "optimize",
        ]
        hits = sum(1 for t in feasible_terms if t in idea_lower)
        score += min(0.5, hits * 0.1)

        # Specific materials/techniques indicate feasibility
        specific_terms = [
            "graphene", "silicon", "polymer", "ceramic", "composite",
            "nanoparticle", "nanotube", "electrolyte", "catalyst",
            "machine learning", "neural", "algorithm", "sensor",
            "actuator", "control", "battery", "solar", "thermal",
        ]
        hits = sum(1 for t in specific_terms if t in idea_lower)
        score += min(0.5, hits * 0.1)

        # Length penalty
        words = idea.split()
        if 8 <= len(words) <= 30:
            score += 0.2
        elif 5 <= len(words) <= 40:
            score += 0.1

        return min(1.0, score)

    def _score_validation_results(self, idea: str) -> float:
        """Score based on validation engine results."""
        result = self.validation_engine.validate_idea(idea, self.problem)
        return result.validation_score

    def _score_simulation(self, idea: str) -> float:
        """Score based on simulation potential."""
        idea_lower = idea.lower()
        score = 0.0

        # Simulation-friendly keywords
        sim_terms = [
            "simulation", "model", "modeling", "computational", "numerical",
            "finite element", "cfd", "molecular dynamics", "monte carlo",
            "optimization", "machine learning", "neural", "algorithm",
            "analysis", "prediction", "forecast",
        ]
        hits = sum(1 for t in sim_terms if t in idea_lower)
        score += min(0.6, hits * 0.15)

        # Quantifiable metrics
        if re.search(r'\d+', idea):
            score += 0.2
        if any(m in idea_lower for m in ["efficiency", "performance", "capacity",
                                          "density", "strength", "accuracy", "speed"]):
            score += 0.2

        return min(1.0, score)

    def _score_manufacturing(self, idea: str) -> float:
        """Score manufacturability and scalability."""
        idea_lower = idea.lower()
        score = 0.0

        manuf_terms = [
            "manufactur", "fabricat", "production", "scale", "scalable",
            "process", "assembly", "yield", "cost", "affordable",
            "industrial", "commercial", "mass production",
        ]
        hits = sum(1 for t in manuf_terms if t in idea_lower)
        score += min(0.7, hits * 0.15)

        # Established materials are more manufacturable
        if any(m in idea_lower for m in ["graphene", "silicon", "polymer", "ceramic",
                                          "composite", "steel", "aluminum", "copper"]):
            score += 0.3

        return min(1.0, score)

    def _score_consistency(self, idea: str) -> float:
        """Score consistency with known scientific laws."""
        idea_lower = idea.lower()
        score = 0.5  # Default: assume consistent unless red flags

        # Red flags for inconsistency
        red_flags = [
            "perpetual motion", "free energy", "over unity", "violates",
            "contradicts", "impossible", "breaks physics", "ignores",
            "negates", "defies", "without energy", "zero energy",
        ]
        for flag in red_flags:
            if flag in idea_lower:
                score -= 0.3

        # Established scientific grounding
        grounded_terms = [
            "thermodynamic", "electrochemical", "quantum", "aerodynamic",
            "structural", "kinetic", "dynamic", "mechanical", "chemical",
            "physical", "electrical", "thermal", "optical",
        ]
        hits = sum(1 for t in grounded_terms if t in idea_lower)
        score += min(0.5, hits * 0.1)

        return max(0.0, min(1.0, score))

    def _score_state_of_art_comparison(self, idea: str) -> float:
        """Score comparison with state of the art."""
        idea_lower = idea.lower()
        score = 0.0

        # Improvement indicators
        improvement_terms = [
            "improve", "enhance", "increase", "reduce", "optimize",
            "better", "faster", "stronger", "more efficient", "higher",
            "lower", "superior", "advanced", "novel", "innovative",
        ]
        hits = sum(1 for t in improvement_terms if t in idea_lower)
        score += min(0.6, hits * 0.1)

        # Specific targets
        if re.search(r'\d+%|\d+x|\d+ times', idea):
            score += 0.2
        if any(m in idea_lower for m in ["vs", "compared", "baseline", "state of the art",
                                          "current", "existing", "conventional"]):
            score += 0.2

        return min(1.0, score)

    def compute_confidence(self, idea: str, validation_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute the confidence score for an idea.

        Args:
            idea: The candidate idea text
            validation_result: Optional pre-computed validation result

        Returns:
            Dict with confidence score and dimension breakdown
        """
        # Compute each dimension
        dimensions = {}

        # Literature support
        lit_score = self._score_literature_support(idea)
        dimensions["literature_support"] = {
            "score": round(lit_score, 3),
            "weight": 0.20,
            "description": "How much existing literature/prior art supports this idea",
        }

        # Engineering feasibility
        feas_score = self._score_engineering_feasibility(idea)
        dimensions["engineering_feasibility"] = {
            "score": round(feas_score, 3),
            "weight": 0.20,
            "description": "Can this be engineered with current technology",
        }

        # Validation results
        if validation_result:
            val_score = validation_result.get("validation_score", 0.0)
        else:
            val_score = self._score_validation_results(idea)
        dimensions["validation_results"] = {
            "score": round(val_score, 3),
            "weight": 0.25,
            "description": "Results from the validation engine",
        }

        # Simulation
        sim_score = self._score_simulation(idea)
        dimensions["simulation"] = {
            "score": round(sim_score, 3),
            "weight": 0.10,
            "description": "Simulation results and predicted performance",
        }

        # Manufacturing
        manuf_score = self._score_manufacturing(idea)
        dimensions["manufacturing"] = {
            "score": round(manuf_score, 3),
            "weight": 0.10,
            "description": "Manufacturability and scalability",
        }

        # Consistency
        cons_score = self._score_consistency(idea)
        dimensions["consistency"] = {
            "score": round(cons_score, 3),
            "weight": 0.10,
            "description": "Consistency with known scientific laws",
        }

        # State of art comparison
        soa_score = self._score_state_of_art_comparison(idea)
        dimensions["state_of_art_comparison"] = {
            "score": round(soa_score, 3),
            "weight": 0.05,
            "description": "Comparison with current state of the art",
        }

        # Weighted total
        total_score = sum(
            dim["score"] * dim["weight"]
            for dim in dimensions.values()
        )

        # Confidence level
        if total_score >= 0.8:
            level = "HIGH"
        elif total_score >= 0.6:
            level = "MODERATE"
        elif total_score >= 0.4:
            level = "LOW"
        else:
            level = "SPECULATIVE"

        return {
            "idea": idea,
            "confidence_score": round(total_score, 3),
            "confidence_level": level,
            "dimensions": dimensions,
            "computed_at": time.time(),
        }

    def compute_batch(self, ideas: List[Dict[str, Any]],
                      validation_results: Optional[Dict[str, Dict[str, Any]]] = None,
                      max_workers: int = 8) -> List[Dict[str, Any]]:
        """Compute confidence scores for a batch of ideas in parallel."""
        if not ideas:
            return []

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idea_entry in ideas:
                idea_text = idea_entry.get("idea", "")
                val_result = None
                if validation_results:
                    idea_id = idea_entry.get("idea_id", "")
                    val_result = validation_results.get(idea_id)
                futures.append(
                    executor.submit(self.compute_confidence, idea_text, val_result)
                )

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    continue

        # Sort by confidence score
        results.sort(key=lambda x: x["confidence_score"], reverse=True)
        return results


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================
def compute_confidence(idea: str, problem: str,
                       validation_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute confidence score for a single idea."""
    engine = ConfidenceEngine(problem)
    return engine.compute_confidence(idea, validation_result)


def compute_confidence_batch(ideas: List[Dict[str, Any]], problem: str,
                             max_workers: int = 8) -> List[Dict[str, Any]]:
    """Compute confidence scores for a batch of ideas."""
    engine = ConfidenceEngine(problem)
    return engine.compute_batch(ideas, max_workers=max_workers)


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    test_ideas = [
        {"idea": "Graphene-enhanced thermal management system for GPU with improved cooling and power efficiency"},
        {"idea": "Solid-state battery with high energy density and safety features for electric vehicles"},
        {"idea": "Floating vehicle with buoyancy control and stability systems for marine transport"},
        {"idea": "Nanoparticle drug delivery with improved bioavailability and reduced toxicity"},
    ]
    test_problem = "Design a better battery with higher energy density"

    results = compute_confidence_batch(test_ideas, test_problem)
    for r in results:
        print(f"\n  Idea: {r['idea'][:60]}...")
        print(f"  Confidence: {r['confidence_score']:.3f} ({r['confidence_level']})")
        for dim_name, dim in r["dimensions"].items():
            print(f"    {dim_name:30s}: {dim['score']:.3f}")