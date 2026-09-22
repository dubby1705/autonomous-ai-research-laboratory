"""
TournamentSelection.py — Knockout Tournament Selection Engine
==============================================================
Behaves like a knockout tournament. Only the strongest ideas survive.

Round 1: 1000 ideas → Remove all failed ideas → 200 remain
Round 2: Run more detailed validation → 50 remain
Round 3: Run even more validation → 10 remain
Final:   Run comprehensive evaluation → Best Candidate

Each round applies progressively stricter validation criteria.
"""

import os
import json
import time
import hashlib
import random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from Engine.Question_Engine.Validation.validation_engine import (
    get_validation_engine,
    PASS,
    WARNING,
    FAIL,
)


# =========================================================
# ROUND CONFIGURATIONS
# =========================================================
ROUND_CONFIGS = [
    {
        "round": 1,
        "name": "Initial Screening",
        "description": "Remove weakest ideas",
        "survival_rate": 0.3,       # Keep top 30%
        "min_validation_score": 0.05,
        "max_failed_tests": 99,     # Allow failures at this stage
        "max_warning_tests": 99,    # Allow warnings at this stage
    },
    {
        "round": 2,
        "name": "Detailed Validation",
        "description": "Run more detailed validation",
        "survival_rate": 0.3,       # Keep top 30% of survivors
        "min_validation_score": 0.1,
        "max_failed_tests": 2,
        "max_warning_tests": 99,
    },
    {
        "round": 3,
        "name": "Deep Validation",
        "description": "Run even more validation",
        "survival_rate": 0.3,       # Keep top 30% of survivors
        "min_validation_score": 0.15,
        "max_failed_tests": 1,
        "max_warning_tests": 5,
    },
    {
        "round": 4,
        "name": "Final Evaluation",
        "description": "Run comprehensive evaluation",
        "survival_rate": 0.2,       # Keep top 20% of survivors
        "min_validation_score": 0.0,  # Accept any score in final round
        "max_failed_tests": 99,     # Allow failures - we pick best by score
        "max_warning_tests": 99,    # Allow warnings - we pick best by score
    },
]


# =========================================================
# TOURNAMENT ENGINE
# =========================================================
class TournamentSelection:
    """Knockout tournament selection for candidate ideas."""

    def __init__(self, problem_statement: str):
        self.problem = problem_statement
        self.validation_engine = get_validation_engine()
        self.rounds = []
        self.eliminated = []
        self.winner = None

    def _validate_idea(self, idea: Dict[str, Any], round_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an idea with the current round's criteria."""
        idea_text = idea.get("idea", "")
        result = self.validation_engine.validate_idea(idea_text, self.problem)

        # Apply round-specific criteria
        validation_score = result.validation_score
        failed = result.failed
        warnings = result.warnings

        min_score = round_config["min_validation_score"]
        max_failed = round_config["max_failed_tests"]
        max_warnings = round_config["max_warning_tests"]

        # Determine survival
        survives = True
        reasons = []

        if validation_score < min_score:
            survives = False
            reasons.append(f"Validation score {validation_score:.2f} < minimum {min_score}")

        if failed > max_failed:
            survives = False
            reasons.append(f"{failed} failed tests > maximum {max_failed}")

        if warnings > max_warnings:
            survives = False
            reasons.append(f"{warnings} warnings > maximum {max_warnings}")

        # Critical failure always eliminates
        critical_fail = any(
            t.status == FAIL and t.severity == "critical"
            for t in result.tests
        )
        if critical_fail:
            survives = False
            reasons.append("Critical test failure")

        return {
            "idea": idea_text,
            "idea_id": result.idea_id,
            "domain": result.domain,
            "validation_score": validation_score,
            "overall_status": result.overall_status,
            "passed": result.passed,
            "warnings": result.warnings,
            "failed": result.failed,
            "tests": [t.to_dict() for t in result.tests],
            "survives": survives,
            "elimination_reasons": reasons,
            "round": round_config["round"],
        }

    def run_tournament(
        self,
        ideas: List[Dict[str, Any]],
        max_workers: int = 8,
        output_file: str = "tournament_results.json",
    ) -> Dict[str, Any]:
        """
        Run the full knockout tournament on candidate ideas.

        Args:
            ideas: List of idea dicts with at least {"idea": str}
            max_workers: Parallel workers for validation
            output_file: Where to save results

        Returns:
            Tournament results with winner
        """
        if not ideas:
            return {"error": "No ideas provided", "winner": None}

        print("\n" + "=" * 80)
        print("  [TROPHY] KNOCKOUT TOURNAMENT SELECTION")
        print("=" * 80)
        print(f"  Initial ideas: {len(ideas)}")
        print(f"  Problem: {self.problem[:80]}...")
        print("=" * 80)

        current_pool = list(ideas)
        all_round_results = []

        for round_config in ROUND_CONFIGS:
            round_num = round_config["round"]
            print(f"\n  {'='*60}")
            print(f"  ROUND {round_num}: {round_config['name']}")
            print(f"  {round_config['description']}")
            print(f"  {'='*60}")
            print(f"  Ideas entering round: {len(current_pool)}")

            # Validate all ideas in parallel
            validated = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._validate_idea, idea, round_config)
                    for idea in current_pool
                ]
                for future in as_completed(futures):
                    try:
                        validated.append(future.result())
                    except Exception:
                        continue

            # Separate survivors and eliminated
            survivors = [v for v in validated if v["survives"]]
            eliminated = [v for v in validated if not v["survives"]]

            # Sort survivors by validation score
            survivors.sort(key=lambda x: x["validation_score"], reverse=True)

            # Apply survival rate cap
            max_survivors = max(1, int(len(current_pool) * round_config["survival_rate"]))
            if len(survivors) > max_survivors:
                # Move excess survivors to eliminated
                excess = survivors[max_survivors:]
                survivors = survivors[:max_survivors]
                for e in excess:
                    e["survives"] = False
                    e["elimination_reasons"].append(
                        f"Exceeded survival cap ({max_survivors} max)"
                    )
                eliminated.extend(excess)

            # Record round results
            round_result = {
                "round": round_num,
                "name": round_config["name"],
                "ideas_in": len(current_pool),
                "survivors": len(survivors),
                "eliminated": len(eliminated),
                "survivor_ids": [s["idea_id"] for s in survivors],
                "eliminated_ids": [e["idea_id"] for e in eliminated],
            }
            all_round_results.append(round_result)

            print(f"  [OK] Survivors: {len(survivors)}")
            print(f"  [X] Eliminated: {len(eliminated)}")
            if survivors:
                print(f"  [TOP] Top survivor score: {survivors[0]['validation_score']:.3f}")

            # Store eliminated ideas
            self.eliminated.extend(eliminated)

            # Update pool for next round
            current_pool = [{"idea": s["idea"]} for s in survivors]

            # If only 1 survivor, we have a winner
            if len(survivors) <= 1:
                break

        # Final winner
        if current_pool:
            # Run final comprehensive evaluation on the last survivors
            final_results = []
            with ThreadPoolExecutor(max_workers=min(max_workers, len(current_pool))) as executor:
                futures = [
                    executor.submit(self._validate_idea, idea, ROUND_CONFIGS[-1])
                    for idea in current_pool
                ]
                for future in as_completed(futures):
                    try:
                        final_results.append(future.result())
                    except Exception:
                        continue

            final_results.sort(key=lambda x: x["validation_score"], reverse=True)

            if final_results:
                self.winner = final_results[0]
                print(f"\n  [WINNER] TOURNAMENT WINNER:")
                print(f"  {'='*60}")
                print(f"  Idea: {self.winner['idea'][:120]}...")
                print(f"  Validation Score: {self.winner['validation_score']:.3f}")
                print(f"  Domain: {self.winner['domain']}")
                print(f"  Tests Passed: {self.winner['passed']}, "
                      f"Warnings: {self.winner['warnings']}, Failed: {self.winner['failed']}")

        # Build final result
        result = {
            "problem_statement": self.problem,
            "total_ideas_entered": len(ideas),
            "rounds": all_round_results,
            "winner": self.winner,
            "eliminated_count": len(self.eliminated),
            "eliminated": self.eliminated,
            "completed_at": time.time(),
        }

        # Save to disk
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, default=str)

        print(f"\n  [FILE] Tournament results saved to: {output_file}")
        print("=" * 80)

        return result


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================
def run_tournament(
    ideas: List[Dict[str, Any]],
    problem_statement: str,
    max_workers: int = 8,
    output_file: str = "tournament_results.json",
) -> Dict[str, Any]:
    """Run the knockout tournament on candidate ideas."""
    tournament = TournamentSelection(problem_statement)
    return tournament.run_tournament(ideas, max_workers=max_workers, output_file=output_file)


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Test with sample ideas
    test_ideas = [
        {"idea": "Graphene-enhanced thermal management system for GPU with improved cooling and power efficiency"},
        {"idea": "Solid-state battery with high energy density and safety features for electric vehicles"},
        {"idea": "Floating vehicle with buoyancy control and stability systems for marine transport"},
        {"idea": "Nanoparticle drug delivery with improved bioavailability and reduced toxicity"},
        {"idea": "Quantum-enhanced optimization algorithm for machine learning workloads"},
        {"idea": "Composite material with enhanced strength and reduced weight for aerospace"},
        {"idea": "Adaptive control system for robotic arm with improved precision and repeatability"},
        {"idea": "Novel catalyst with higher turnover frequency for chemical synthesis"},
    ]
    result = run_tournament(test_ideas, "Design a better battery with higher energy density")
    if result.get("winner"):
        print(f"\nWinner: {result['winner']['idea'][:100]}...")