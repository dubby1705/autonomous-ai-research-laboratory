"""Offline regression tests for the AARL evidence / confidence fixes.

These lock in the behaviours that stop AARL presenting assumptions, predictions,
mathematical validity or preliminary simulations as proven evidence:

  * an unconstrained LLM ``final_score`` (e.g. 78.0) is clamped to [0, 10],
  * the documented weighted formula is re-applied rather than trusted,
  * evidence is gated on RELEVANCE before it can raise confidence,
  * simulation type is classified explicitly instead of assumed "validated".

Skipped automatically when the stored research artefacts are absent.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "Research", "dessign_a_advance_gpu_for_ai_and_use_new_consept")
PROBLEM = "Design a new advanced GPU architecture for AI"
REQUIRED = ("verified_hypotheses.json", "evidence_scoring_db.json",
            "derived_equations.json", "comparison_report.json")
_HAVE_DATA = all(os.path.isfile(os.path.join(DATA_DIR, n)) for n in REQUIRED)


def _load(name: str):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as handle:
        return json.load(handle)


@unittest.skipUnless(_HAVE_DATA, f"research artefacts missing in {DATA_DIR}")
class TestEvidenceAndConfidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from Engine.Question_Engine import ResearchOutputGenerator as R

        cls.R = R
        cls.verified_raw = _load("verified_hypotheses.json")
        cls.evidence = _load("evidence_scoring_db.json")
        cls.equations = _load("derived_equations.json")
        cls.comparison = _load("comparison_report.json")
        cls.kb: dict = {}

    # ---------------------------------------------------------------- scores
    def test_runaway_final_score_is_clamped(self):
        raw_scores = [h.get("final_score") for h in self.verified_raw]
        cleaned = self.R.validate_verified_hypotheses(self.verified_raw)
        scores = [h.get("final_score") for h in cleaned]
        self.assertTrue(all(isinstance(s, (int, float)) for s in scores))
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 10.0)
        if any(isinstance(s, (int, float)) and s > 10 for s in raw_scores):
            self.assertFalse(
                any(s > 10 for s in scores),
                "an out-of-range raw score survived sanitization",
            )

    def test_confidence_formula_is_recomputed(self):
        cleaned = self.R.validate_verified_hypotheses(self.verified_raw)
        hyp = cleaned[0]
        breakdown = self.R._compute_confidence_breakdown(hyp)
        dims = breakdown["dimensions"]
        for name, value in dims.items():
            self.assertGreaterEqual(value, 1, f"{name} below clamp floor")
            self.assertLessEqual(value, 10, f"{name} above clamp ceiling")
        expected = (
            dims["novelty"] * 0.15
            + dims["evidence"] * 0.25
            + dims["feasibility"] * 0.15
            + dims["consistency"] * 0.20
            + dims["testability"] * 0.15
            + (10.0 - dims["risk"]) * 0.10
        )
        expected = max(0.0, min(10.0, expected))
        self.assertAlmostEqual(breakdown["final_score"], round(expected, 2), places=2)
        self.assertLessEqual(breakdown["final_score"], 10.0)

    def test_evidence_gate_caps_and_downgrades(self):
        cleaned = self.R.validate_verified_hypotheses(self.verified_raw)
        hyp = cleaned[0]
        text = hyp.get("refined_hypothesis") or hyp.get("original_prediction", "")
        score, status, _, _ = self.R._evidence_score_for_hypothesis(
            self.evidence, text, PROBLEM
        )
        breakdown = self.R._compute_confidence_breakdown(hyp)
        gate = self.R._apply_evidence_gate(breakdown, score, status)

        cap = self.R.EVIDENCE_CONFIDENCE_CAP[status]
        self.assertLessEqual(gate["final_score"], cap)
        self.assertEqual(gate["evidence_dimension"], max(1, min(10, round(score))))
        self.assertEqual(gate["llm_evidence_dimension"],
                         breakdown["dimensions"]["evidence"])
        self.assertEqual(gate["llm_score"], breakdown["final_score"])
        # a verdict can never exceed what the evidence status allows
        ceiling = self.R.EVIDENCE_VERDICT_CEILING[status]
        self.assertLessEqual(self.R._VERDICT_ORDER.index(gate["verdict"]),
                             self.R._VERDICT_ORDER.index(ceiling))
        if status != "SUPPORTED_BY_EVIDENCE":
            self.assertNotEqual(gate["verdict"], "Strongly Supported")

    def test_equation_relevance_accounting_is_exact(self):
        cleaned = self.R.validate_verified_hypotheses(self.verified_raw)
        summary = self.R._equation_relevance_summary(
            self.equations, cleaned[0], PROBLEM
        )
        self.assertEqual(summary["relevant_to_hypothesis"]
                         + summary["irrelevant_but_valid"], summary["inspected"])
        self.assertEqual(summary["inspected"] + summary["uninspected"],
                         summary["total_validated"])
        # a trivially valid but unrelated equation must not be called relevant
        weird = {"equation": "t = k * k", "domain": "computer_architecture",
                 "variables": {"t": "time", "k": "constant"}}
        _, is_rel, label = self.R._equation_relevance_to_hypothesis(
            weird, cleaned[0].get("original_prediction", ""), PROBLEM
        )
        self.assertFalse(is_rel)
        self.assertEqual(label, "VALID_BUT_NOT_RELEVANT")

    # -------------------------------------------------------------- evidence
    def test_evidence_is_gated_on_relevance(self):
        cleaned = self.R.validate_verified_hypotheses(self.verified_raw)
        hyp = cleaned[0]
        text = hyp.get("refined_hypothesis") or hyp.get("original_prediction", "")
        score, status, relevant, total = self.R._evidence_score_for_hypothesis(
            self.evidence, text, PROBLEM
        )
        self.assertIn(status, ("INSUFFICIENT_DIRECT_EVIDENCE", "NO_EVIDENCE_FOUND",
                               "UNVERIFIED_DIRECT_EVIDENCE", "SUPPORTED_BY_EVIDENCE"))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)
        self.assertLessEqual(relevant, total)
        if relevant == 0:
            self.assertIn(status, ("INSUFFICIENT_DIRECT_EVIDENCE", "NO_EVIDENCE_FOUND"))
            self.assertLessEqual(score, 4.0)

    def test_simulation_type_is_classified(self):
        sim_type, detail = self.R._classify_simulation_type(self.comparison)
        self.assertIn(sim_type, ("EXECUTED_SIMULATION", "ASSUMPTION_BASED_PROJECTION",
                                 "SIMULATION_UNAVAILABLE"))
        self.assertTrue(detail)

    # ---------------------------------------------------------------- report
    def test_report_07_is_gated_and_in_range(self):
        cleaned = self.R.validate_verified_hypotheses(self.verified_raw)
        report = self.R.generate_report_07_evidence_and_confidence(
            kb_data=self.kb,
            equations_data=self.equations,
            comparison_data=self.comparison,
            evidence_data=self.evidence,
            verified_data=cleaned,
            user_problem=PROBLEM,
        )
        self.assertGreater(len(report), 500)
        self.assertIn("AWARDED Confidence Score", report)
        self.assertIn("Evidence Status", report)
        self.assertIn("not used, reference only", report)
        # the awarded score must be the evidence-gated one, never the LLM's
        awarded = re.search(r"AWARDED Confidence Score:\s*([0-9.]+)/10", report)
        self.assertIsNotNone(awarded)
        value = float(awarded.group(1))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 6.0)  # unverified evidence ceiling
        for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", report):
            number = float(match.group(1))
            self.assertGreaterEqual(number, 0.0)
            self.assertLessEqual(number, 10.0)
        self.assertNotIn("78.0/10", report)
        self.assertNotIn("Strongly Supported\n", report.split("Awarded verdict")[0]
                         .split("LLM self-assessment")[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)