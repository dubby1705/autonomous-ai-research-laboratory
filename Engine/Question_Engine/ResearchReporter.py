"""
ResearchReporter.py — Selective Reporting System
=================================================
Only generates a complete report for the final winner.
For the remaining ideas only stores:
  - confidence
  - reason for rejection
  - failed validation stage
  - important observations
"""

import os
import json
import time
from typing import List, Dict, Any, Optional


def _load_json(path: str) -> Any:
    """Safely load a JSON file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_json(path: str, data: Any) -> None:
    """Safely save data to a JSON file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def generate_winner_report(
    winner: Dict[str, Any],
    problem_statement: str,
    confidence: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Generate a complete report for the final tournament winner.

    Args:
        winner: The winning idea dict from tournament results
        confidence: Confidence score dict from ConfidenceEngine
        problem_statement: The research problem
        output_dir: Directory to save the report

    Returns:
        Path to the generated report file
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "winner_report.md")

    lines = []
    lines.append("# 🏆 AARL WINNER RESEARCH REPORT")
    lines.append("")
    lines.append(f"**Research Problem:** {problem_statement}")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Winner idea
    lines.append("## 1. WINNING CANDIDATE")
    lines.append("")
    lines.append(f"**Idea:** {winner.get('idea', 'N/A')}")
    lines.append("")
    lines.append(f"**Domain:** {winner.get('domain', 'N/A')}")
    lines.append(f"**Validation Score:** {winner.get('validation_score', 0):.3f}")
    lines.append(f"**Overall Status:** {winner.get('overall_status', 'N/A')}")
    lines.append("")

    # Confidence
    lines.append("## 2. CONFIDENCE ASSESSMENT")
    lines.append("")
    lines.append(f"**Confidence Score:** {confidence.get('confidence_score', 0):.3f}")
    lines.append(f"**Confidence Level:** {confidence.get('confidence_level', 'N/A')}")
    lines.append("")
    lines.append("### Dimension Breakdown")
    lines.append("")
    lines.append("| Dimension | Score | Weight |")
    lines.append("|-----------|-------|--------|")
    for dim_name, dim in confidence.get("dimensions", {}).items():
        lines.append(f"| {dim_name.replace('_', ' ').title()} | {dim['score']:.3f} | {dim['weight']:.2f} |")
    lines.append("")

    # Validation tests
    lines.append("## 3. VALIDATION RESULTS")
    lines.append("")
    lines.append(f"**Tests Passed:** {winner.get('passed', 0)}")
    lines.append(f"**Warnings:** {winner.get('warnings', 0)}")
    lines.append(f"**Failed:** {winner.get('failed', 0)}")
    lines.append("")
    lines.append("### Test Details")
    lines.append("")
    for test in winner.get("tests", []):
        status_icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(test.get("status", ""), "❓")
        lines.append(f"- {status_icon} **{test.get('test_name', 'N/A')}**: {test.get('details', '')}")
    lines.append("")

    # Important observations
    lines.append("## 4. IMPORTANT OBSERVATIONS")
    lines.append("")
    lines.append("- This candidate survived all tournament rounds of knockout validation.")
    lines.append("- The confidence score reflects available evidence, not certainty.")
    lines.append("- Further experimental validation is required before claiming correctness.")
    lines.append("- The system estimates promise based on evidence, not truth.")
    lines.append("")

    # Next steps
    lines.append("## 5. RECOMMENDED NEXT STEPS")
    lines.append("")
    lines.append("1. **Prototype Development:** Build a proof-of-concept prototype")
    lines.append("2. **Simulation:** Run detailed domain-specific simulations")
    lines.append("3. **Experimental Validation:** Design experiments to test key claims")
    lines.append("4. **Literature Review:** Deep-dive into supporting prior art")
    lines.append("5. **Iterative Refinement:** Use feedback to improve the design")
    lines.append("")

    lines.append("---")
    lines.append("*This report was generated automatically by the AARL Autonomous Research Laboratory.*")
    lines.append("*The system estimates how promising each idea is based on available evidence.*")
    lines.append("*It does NOT claim that any idea is correct.*")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path


def generate_eliminated_summary(
    eliminated: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Generate a compact summary of all eliminated ideas.
    Only stores confidence, reason for rejection, failed validation stage,
    and important observations.

    Args:
        eliminated: List of eliminated idea dicts from tournament
        output_dir: Directory to save the summary

    Returns:
        Path to the generated summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "eliminated_ideas_summary.json")

    summary = []
    for idea in eliminated:
        # Determine failed validation stage
        failed_stage = None
        for test in idea.get("tests", []):
            if test.get("status") == "FAIL":
                failed_stage = test.get("test_name", "unknown")
                break

        summary.append({
            "idea": idea.get("idea", ""),
            "confidence": idea.get("validation_score", 0),
            "reason_for_rejection": "; ".join(idea.get("elimination_reasons", ["Unknown"])),
            "failed_validation_stage": failed_stage,
            "important_observations": [
                f"Domain: {idea.get('domain', 'unknown')}",
                f"Overall status: {idea.get('overall_status', 'unknown')}",
                f"Tests passed: {idea.get('passed', 0)}, "
                f"Warnings: {idea.get('warnings', 0)}, Failed: {idea.get('failed', 0)}",
            ],
        })

    data = {
        "total_eliminated": len(summary),
        "eliminated_ideas": summary,
        "generated_at": time.time(),
    }

    _save_json(summary_path, data)
    return summary_path


def generate_tournament_report(
    tournament_result: Dict[str, Any],
    confidence_results: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generate all reports for the tournament.

    Args:
        tournament_result: Result from TournamentSelection
        confidence_results: Confidence scores for surviving ideas
        output_dir: Directory to save reports

    Returns:
        Dict with paths to generated reports
    """
    os.makedirs(output_dir, exist_ok=True)
    reports = {}

    # Winner report
    winner = tournament_result.get("winner")
    winner_confidence = None
    if winner:
        # Find matching confidence
        for conf in confidence_results:
            if conf.get("idea") == winner.get("idea"):
                winner_confidence = conf
                break
        if winner_confidence is None and confidence_results:
            winner_confidence = confidence_results[0]

        if winner_confidence:
            reports["winner"] = generate_winner_report(
                winner,
                tournament_result.get("problem_statement", ""),
                winner_confidence,
                output_dir,
            )

    # Eliminated summary
    eliminated = tournament_result.get("eliminated", [])
    if eliminated:
        reports["eliminated"] = generate_eliminated_summary(eliminated, output_dir)

    # Tournament overview
    overview_path = os.path.join(output_dir, "tournament_overview.json")
    overview = {
        "problem_statement": tournament_result.get("problem_statement", ""),
        "total_ideas_entered": tournament_result.get("total_ideas_entered", 0),
        "rounds": tournament_result.get("rounds", []),
        "winner_idea": winner.get("idea", "") if winner else None,
        "winner_confidence": winner_confidence.get("confidence_score", 0) if winner_confidence else None,
        "eliminated_count": tournament_result.get("eliminated_count", 0),
        "generated_at": time.time(),
    }
    _save_json(overview_path, overview)
    reports["overview"] = overview_path

    return reports


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Test with sample data
    test_winner = {
        "idea": "Solid-state battery with high energy density and safety features",
        "domain": "battery",
        "validation_score": 0.85,
        "overall_status": "PASS",
        "passed": 4,
        "warnings": 0,
        "failed": 0,
        "tests": [
            {"test_name": "energy_density", "status": "PASS", "details": "Energy density addressed"},
            {"test_name": "stability", "status": "PASS", "details": "Stability addressed"},
            {"test_name": "cycle_life", "status": "PASS", "details": "Cycle life addressed"},
            {"test_name": "safety", "status": "PASS", "details": "Safety addressed"},
        ],
    }
    test_confidence = {
        "idea": "Solid-state battery with high energy density and safety features",
        "confidence_score": 0.78,
        "confidence_level": "MODERATE",
        "dimensions": {
            "literature_support": {"score": 0.8, "weight": 0.20},
            "engineering_feasibility": {"score": 0.7, "weight": 0.20},
            "validation_results": {"score": 0.85, "weight": 0.25},
            "simulation": {"score": 0.6, "weight": 0.10},
            "manufacturing": {"score": 0.7, "weight": 0.10},
            "consistency": {"score": 0.9, "weight": 0.10},
            "state_of_art_comparison": {"score": 0.8, "weight": 0.05},
        },
    }
    test_eliminated = [
        {
            "idea": "Floating vehicle with buoyancy control",
            "validation_score": 0.3,
            "elimination_reasons": ["Validation score 0.30 < minimum 0.50"],
            "overall_status": "FAIL",
            "domain": "floating",
            "passed": 2,
            "warnings": 3,
            "failed": 4,
            "tests": [
                {"test_name": "lift", "status": "FAIL", "details": "No lift considerations"},
            ],
        }
    ]

    reports = generate_tournament_report(
        {
            "problem_statement": "Design a better battery",
            "total_ideas_entered": 2,
            "rounds": [],
            "winner": test_winner,
            "eliminated": test_eliminated,
            "eliminated_count": 1,
        },
        [test_confidence],
        "test_reports",
    )
    print(f"Generated reports: {reports}")