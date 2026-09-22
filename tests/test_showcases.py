"""Unit tests for the AARL research showcase (offline; no API key needed)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from showcase import generation as gen
from showcase import optimizers as opt
from showcase.config import ResearchConfig
from showcase.experiment_engine import run_experiment
from showcase.llm_service import extract_json
from showcase.records import ExperimentSpec, Hypothesis, ResearchRun


class TestLLMParsing(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fenced_json(self):
        text = "Here you go:\n```json\n{\"title\": \"X\", \"n\": 2}\n```\nDone."
        self.assertEqual(extract_json(text), {"title": "X", "n": 2})

    def test_json_with_surrounding_prose(self):
        text = 'Sure! {"hypotheses": []} hope that helps'
        self.assertEqual(extract_json(text), {"hypotheses": []})

    def test_malformed_returns_none(self):
        self.assertIsNone(extract_json("not json at all {"))

    def test_missing_fields_detected(self):
        from showcase.llm_service import validate_fields

        missing = validate_fields({"a": 1, "b": ""}, ["a", "b", "c"])
        self.assertEqual(missing, ["b", "c"])


class TestOptimizerRegistry(unittest.TestCase):
    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            opt.sanitise_params("does_not_exist", {})

    def test_params_clamped_to_bounds(self):
        p = opt.sanitise_params("momentum", {"lr": 99.0, "momentum": -5})
        self.assertLessEqual(p["lr"], 1.0)
        self.assertGreaterEqual(p["momentum"], 0.0)

    def test_all_registered_optimisers_run(self):
        problem = opt.make_quadratic_problem(6, seed=5)
        start = __import__("numpy").zeros(6)
        for m in opt.known_optimizers():
            r = opt.run_optimiser(problem, m, {}, 500, 1e9, start)
            self.assertTrue(r.converged)  # tol=1e9 -> immediate convergence
            self.assertIsInstance(r.final_loss, float)


class TestExperimentEngine(unittest.TestCase):
    def test_measured_metrics_are_real_and_reproducible(self):
        cfg = ResearchConfig(num_runs=2, dimensionality=8,
                             max_iterations=1500, random_seed=3)
        spec = ExperimentSpec(experiment_id="e1", objective="t",
                              proposed_method="momentum", parameters={},
                              supported=True)
        r1 = run_experiment(spec, cfg)
        r2 = run_experiment(spec, cfg)
        self.assertEqual(r1.status, "SUCCESS")
        # identical seeds => identical measured numbers
        self.assertEqual(r1.proposed["mean_iterations"],
                         r2.proposed["mean_iterations"])
        self.assertEqual(len(r1.per_run_proposed), 2)

    def test_unsupported_design_is_flagged_not_crashed(self):
        cfg = ResearchConfig(num_runs=1, dimensionality=8, max_iterations=100)
        spec = ExperimentSpec(objective="t", proposed_method="quantum_flux",
                              supported=True)
        r = run_experiment(spec, cfg)
        self.assertEqual(r.status, "FAILED")

    def test_unsupported_spec_is_paused(self):
        cfg = ResearchConfig()
        spec = ExperimentSpec(objective="t", proposed_method="sgd",
                              supported=False)
        r = run_experiment(spec, cfg)
        self.assertEqual(r.status, "UNSUPPORTED")


class TestTournament(unittest.TestCase):
    def test_multiple_survivors_and_preserved_rejections(self):
        cfg = ResearchConfig(num_hypotheses=8, tournament_pass_threshold=100.0)
        hyps = gen.deterministic_hypotheses(cfg.research_question, cfg)
        svc = _OfflineServiceStub()
        gen.critique_and_tournament(hyps, cfg, svc)
        survivors = [h for h in hyps if h.passed]
        rejected = [h for h in hyps if not h.passed]
        self.assertEqual(len(rejected), len(hyps))
        for h in rejected:
            self.assertTrue(h.rejection_reason or h.evaluation.reviewer_reasoning)

        # now a permissive threshold: multiple survive
        cfg2 = ResearchConfig(num_hypotheses=8, tournament_pass_threshold=10.0)
        hyps2 = gen.deterministic_hypotheses(cfg2.research_question, cfg2)
        gen.critique_and_tournament(hyps2, cfg2, svc)
        survivors2 = [h for h in hyps2 if h.passed]
        self.assertGreaterEqual(len(survivors2), 2)


class _OfflineServiceStub:
    """Minimal stand-in exposing available=False (deterministic path)."""

    available = False
    model = "stub"

    def complete_json(self, *a, **k):  # pragma: no cover
        raise AssertionError("LLM should not be used in offline tests")


class _BrokenLLMStub:
    """Simulates an LLM that always returns malformed responses."""

    available = True
    model = "broken-stub"
    init_error = None

    def complete_json(self, *a, **k):
        raise RuntimeError("simulated malformed LLM response")

    def complete_text(self, *a, **k):  # pragma: no cover
        raise RuntimeError("simulated malformed LLM response")


class TestLLMFailureResilience(unittest.TestCase):
    def test_malformed_llm_responses_fall_back_and_continue(self):
        cfg = ResearchConfig(num_hypotheses=5, num_runs=1, dimensionality=6,
                             max_iterations=800, random_seed=4)
        svc = _BrokenLLMStub()
        hyps = gen.generate_hypotheses(cfg.research_question, cfg, svc)
        self.assertGreaterEqual(len(hyps), 2)  # fallback hypotheses produced
        gen.critique_and_tournament(hyps, cfg, svc)
        survivors = [h for h in hyps if h.passed]
        self.assertGreaterEqual(len(survivors), 1)
        gen.design_experiments(hyps, cfg, svc)
        for h in survivors:
            self.assertTrue(h.experiment_spec.supported)  # safe design fallback


class _RecordingLLMStub:
    """Returns valid canned analyses while recording every prompt it receives."""

    available = True
    model = "recording-stub"
    init_error = None

    def __init__(self):
        self.prompts = []

    def complete_json(self, system_prompt, user_prompt, *a, **k):
        self.prompts.append(user_prompt)
        return {
            "interpretation": "stub interpretation",
            "strengths": [], "weaknesses": [], "limitations": [],
            "improvements": [],
            "evidence_strength": "moderate",
            "conclusion": "stub conclusion",
        }

    def complete_text(self, *a, **k):  # pragma: no cover
        return ""


class TestLLMReceivesRealData(unittest.TestCase):
    def test_analysis_prompt_contains_actual_measurements(self):
        from showcase.analysis import analyze_hypotheses

        cfg = ResearchConfig(num_hypotheses=3, num_runs=2, dimensionality=8,
                             max_iterations=1500, random_seed=9)
        svc = _RecordingLLMStub()
        hyps = gen.deterministic_hypotheses(cfg.research_question, cfg)
        gen.critique_and_tournament(hyps, cfg, _OfflineServiceStub())
        survivors = [h for h in hyps if h.passed]
        gen.design_experiments(survivors, cfg, _OfflineServiceStub())
        gen.execute_experiments(survivors, cfg)
        ok = [h for h in survivors if h.experiment.status == "SUCCESS"]
        self.assertGreaterEqual(len(ok), 1)

        analyze_hypotheses(ok, cfg, svc)
        evidence_prompts = [p for p in svc.prompts if "MEASURED EVIDENCE" in p]
        self.assertGreaterEqual(len(evidence_prompts), len(ok))
        # the real numbers must appear verbatim in the prompt given to the LLM
        for h in ok:
            match = [p for p in evidence_prompts if h.title in p]
            self.assertTrue(match, f"no analysis prompt contained {h.hypothesis_id}")
            self.assertIn(str(h.experiment.baseline["mean_iterations"]), match[0])
            self.assertIn(str(h.experiment.proposed["mean_iterations"]), match[0])
        # stored analysis is labelled with its source
        for h in ok:
            self.assertEqual(h.analysis.source, "llm")
            self.assertEqual(h.analysis.interpretation, "stub interpretation")


class TestPipelineEndToEnd(unittest.TestCase):
    """Full research cycle without an LLM key (deterministic planning)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="aarl_test_runs_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_full_cycle_offline(self):
        from showcase.pipeline import run_research_cycle, persist_and_report

        cfg = ResearchConfig(
            num_hypotheses=6, num_runs=2, dimensionality=8,
            max_iterations=1200, random_seed=5, run_root=self._tmp,
        )
        run = run_research_cycle(cfg)
        result = persist_and_report(run, cfg)

        # multiple survivors
        survivors = [h for h in run.hypotheses if h.passed]
        self.assertGreaterEqual(len(survivors), 2)
        # every survivor has a validated spec and a terminal experiment status
        for h in survivors:
            self.assertTrue(h.experiment_spec.supported)
            self.assertIn(h.experiment.status, ("SUCCESS", "FAILED", "UNSUPPORTED"))
        # at least one real measurement with actual numbers
        ok = [h for h in survivors if h.experiment.status == "SUCCESS"]
        self.assertGreaterEqual(len(ok), 1)
        for h in ok:
            self.assertIsInstance(h.experiment.baseline["mean_iterations"], float)
            self.assertGreaterEqual(h.experiment.baseline["mean_iterations"], 0.0)
        # findings + report + artifacts
        self.assertIn("summary", run.findings)
        with open(result["written"]["report"], encoding="utf-8") as fh:
            md = fh.read()
        for heading in ("## 1. Research Problem", "## 9. Experimental Results",
                        "## 13. Final Findings"):
            self.assertIn(heading, md)
        for f in ("research.json", "hypotheses.json", "experiments.json",
                  "results.json", "report.md"):
            self.assertTrue(os.path.exists(os.path.join(result["run_dir"], f)))

    def test_previous_runs_preserved(self):
        from showcase.pipeline import run_research_cycle, persist_and_report

        cfg = ResearchConfig(num_hypotheses=4, num_runs=1, dimensionality=6,
                             max_iterations=800, random_seed=2, run_root=self._tmp)
        r1 = persist_and_report(run_research_cycle(cfg), cfg)
        r2 = persist_and_report(run_research_cycle(cfg), cfg)
        self.assertNotEqual(r1["run_dir"], r2["run_dir"])
        self.assertTrue(os.path.exists(os.path.join(r1["run_dir"], "research.json")))
        self.assertTrue(os.path.exists(os.path.join(r2["run_dir"], "research.json")))


class TestRecordsRoundTrip(unittest.TestCase):
    def test_research_run_serialises_and_restores(self):
        h = Hypothesis(hypothesis_id="H1", title="T", mechanism="M")
        h.tournament_status = "passed"
        run = ResearchRun(run_id="r1", hypotheses=[h],
                          research_question="q", config={"seed": 1})
        data = run.to_dict()
        restored = ResearchRun.from_dict(data)
        self.assertEqual(restored.run_id, "r1")
        self.assertEqual(restored.hypotheses[0].title, "T")
        self.assertEqual(restored.hypotheses[0].tournament_status, "passed")


class TestReportGeneration(unittest.TestCase):
    def test_markdown_report_structure(self):
        from showcase.report import build_markdown_report

        cfg = ResearchConfig(num_hypotheses=4, num_runs=1, dimensionality=6,
                             max_iterations=800, random_seed=2)
        from showcase.pipeline import run_research_cycle

        run = run_research_cycle(cfg)
        md = build_markdown_report(run)
        for heading in ("## 3. Hypotheses Generated", "## 4. Critical Evaluation",
                        "## 5. Tournament Results", "## 7. Experimental Methodology"):
            self.assertIn(heading, md)
        # no API key material ever appears
        self.assertNotIn("gsk_", md)


if __name__ == "__main__":
    unittest.main()