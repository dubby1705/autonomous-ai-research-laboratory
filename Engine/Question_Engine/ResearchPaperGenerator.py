"""
ResearchPaperGenerator.py — Comprehensive Research Paper Generator
==================================================================
Generates full academic-style research papers from AARL research outputs.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional

# Parallel processing layer. Paper generation touches ten independent
# data sources and twelve independent section renderers; both stages are
# fanned out so the paper builds in the time of its slowest part.
try:  # script execution (Engine/Question_Engine on sys.path)
    from ParallelExecutor import is_failed, run_parallel
except Exception:  # pragma: no cover - imported as a package
    from Engine.Question_Engine.ParallelExecutor import (
        is_failed,
        run_parallel,
    )


def _load_json(path: str) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _safe_get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def _format_date(dt: Optional[str] = None) -> str:
    if dt:
        try:
            return datetime.datetime.fromisoformat(dt).strftime("%B %d, %Y")
        except Exception:
            return dt
    return datetime.datetime.now().strftime("%B %d, %Y")


def _truncate(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."



class ResearchDataCollector:
    """Collects and organizes all research data from AARL outputs."""
    
    def __init__(self, root_dir: str, run_dir: Optional[str] = None):
        self.root_dir = root_dir
        self.run_dir = run_dir
        self.data = {}
        
    def collect_all(self, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Gather every data source concurrently.

        The ten loaders are independent file reads, so they run at the same
        time instead of as ten sequential disk round-trips. The returned dict
        keeps a fixed key order, so the generated paper remains identical to
        the sequential version.
        """
        loaders = {
            "problem_statement": self._get_problem_statement,
            "knowledge_base": self._load_knowledge_base,
            "hypotheses": self._load_hypotheses,
            "experiments": self._load_experiments,
            "tournament_results": self._load_tournament_results,
            "final_solution": self._load_final_solution,
            "validation_results": self._load_validation_results,
            "confidence_scores": self._load_confidence_scores,
            "derived_equations": self._load_equations,
            "metadata": self._collect_metadata,
        }
        collected = run_parallel(loaders, max_workers=max_workers, label="paper-data")

        fallbacks = {
            "problem_statement": "Research Problem (Not Specified)",
            "knowledge_base": {},
            "hypotheses": [],
            "experiments": {},
            "tournament_results": {},
            "final_solution": {},
            "validation_results": [],
            "confidence_scores": [],
            "derived_equations": {},
            "metadata": {},
        }
        self.data = {
            key: (fallbacks[key] if is_failed(value) or value is None else value)
            for key, value in collected.items()
        }
        return self.data

    def _get_problem_statement(self) -> str:
        sources = [
            os.path.join(self.root_dir, "deep_research_knowledge_base.json"),
            os.path.join(self.root_dir, "Engine", "Question_Engine", "deep_research_knowledge_base.json"),
        ]
        for src in sources:
            kb = _load_json(src)
            if kb and "core_research_thesis" in kb:
                return kb.get("core_research_thesis", "")
        
        fs = _load_json(os.path.join(self.root_dir, "final_research_solution.json"))
        if fs and "original_problem" in fs:
            return fs.get("original_problem", "")
        
        return "Research Problem (Not Specified)"
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        sources = [
            os.path.join(self.root_dir, "deep_research_knowledge_base.json"),
            os.path.join(self.root_dir, "Engine", "Question_Engine", "deep_research_knowledge_base.json"),
        ]
        for src in sources:
            kb = _load_json(src)
            if kb:
                return kb
        return {}
    
    def _load_hypotheses(self) -> List[Dict[str, Any]]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "hypotheses.json"),
            os.path.join(self.root_dir, "hypotheses.json"),
        ]
        if self.run_dir:
            sources.extend([os.path.join(self.run_dir, "hypotheses.json")])
        
        for src in sources:
            hyps = _load_json(src)
            if hyps and isinstance(hyps, list):
                return hyps
            elif hyps and isinstance(hyps, dict):
                for key in ["hypotheses", "ideas", "items"]:
                    if key in hyps:
                        return hyps[key]
        return []
    
    def _load_experiments(self) -> Dict[str, Any]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "results.json"),
            os.path.join(self.root_dir, "results.json"),
        ]
        if self.run_dir:
            sources.extend([
                os.path.join(self.run_dir, "results.json"),
                os.path.join(self.run_dir, "experiments.json"),
            ])
        
        for src in sources:
            exp = _load_json(src)
            if exp:
                return exp
        return {}
    
    def _load_tournament_results(self) -> Dict[str, Any]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "tournament_results.json"),
            os.path.join(self.root_dir, "tournament_results.json"),
        ]
        if self.run_dir:
            sources.extend([os.path.join(self.run_dir, "tournament_results.json")])
        
        for src in sources:
            tour = _load_json(src)
            if tour:
                return tour
        return {}
    
    def _load_final_solution(self) -> Dict[str, Any]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "final_research_solution.json"),
            os.path.join(self.root_dir, "final_research_solution.json"),
        ]
        for src in sources:
            fs = _load_json(src)
            if fs:
                return fs
        return {}
    
    def _load_validation_results(self) -> List[Dict[str, Any]]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "validation_results.json"),
        ]
        for src in sources:
            val = _load_json(src)
            if val and isinstance(val, list):
                return val
        return []
    
    def _load_confidence_scores(self) -> List[Dict[str, Any]]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "confidence_scores.json"),
        ]
        for src in sources:
            conf = _load_json(src)
            if conf and isinstance(conf, list):
                return conf
        return []
    
    def _load_equations(self) -> Dict[str, Any]:
        sources = [
            os.path.join(self.root_dir, "Engine", "Question_Engine", "derived_equations.json"),
            os.path.join(self.root_dir, "derived_equations.json"),
        ]
        for src in sources:
            eqs = _load_json(src)
            if eqs:
                return eqs
        return {}
    
    def _collect_metadata(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.datetime.now().isoformat(),
            "generator_version": "AARL Research Paper Generator v1.0",
            "root_dir": self.root_dir,
            "run_dir": self.run_dir,
        }



class ResearchPaperGenerator:
    """Generates comprehensive research papers from collected data."""
    
    def __init__(self, data: Dict[str, Any], max_workers: Optional[int] = None):
        self.data = data
        # Worker cap used when rendering the paper sections in parallel.
        self.max_workers = max_workers

    def generate_full_paper(self, output_dir: str, format: str = "markdown") -> str:
        os.makedirs(output_dir, exist_ok=True)
        
        if format == "latex":
            content = self._generate_latex_paper()
            ext = ".tex"
        else:
            content = self._generate_markdown_paper()
            ext = ".md"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_paper_{timestamp}{ext}"
        filepath = os.path.join(output_dir, filename)
        _save_text(filepath, content)
        
        return filepath
    
    def _section_renderers(self):
        """The twelve paper sections, in publication order."""
        return [
            ("title_page", self._generate_title_page),
            ("abstract", self._generate_abstract),
            ("table_of_contents", self._generate_table_of_contents),
            ("introduction", self._generate_introduction),
            ("literature_review", self._generate_literature_review),
            ("methodology", self._generate_methodology),
            ("results", self._generate_results),
            ("discussion", self._generate_discussion),
            ("conclusion", self._generate_conclusion),
            ("references", self._generate_references),
            ("appendices", self._generate_appendices),
            ("final_idea_summary", self._generate_final_idea_summary),
        ]

    def _generate_markdown_paper(self) -> str:
        """
        Render every section concurrently, then assemble them in publication
        order.

        Each section is an independent pure function of ``self.data``, so the
        produced document is identical to the sequential result while the
        wall-clock cost becomes that of the slowest section rather than the sum
        of all twelve.
        """
        renderers = self._section_renderers()
        rendered = run_parallel(
            dict(renderers),
            max_workers=self.max_workers,
            label="paper-sections",
        )

        lines: List[str] = []
        for name, _renderer in renderers:
            section = rendered.get(name)
            if section is None or is_failed(section):
                continue
            lines.extend(section)
            lines.append("")
        return "\n".join(lines)

    def _generate_title_page(self) -> List[str]:
        lines = []
        problem = self.data.get("problem_statement", "Research Study")
        lines.append("# " + problem)
        lines.append("")
        lines.append("## A Comprehensive Research Study")
        lines.append("")
        lines.append(f"**Generated by:** Autonomous AI Research Laboratory (AARL)")
        lines.append(f"**Date:** {_format_date()}")
        lines.append(f"**Version:** 1.0")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{problem}*")
        lines.append("")
        lines.append("This research paper presents a comprehensive analysis generated through")
        lines.append("autonomous AI-driven research methodologies, including hypothesis generation,")
        lines.append("critical evaluation, experimental validation, and evidence-based conclusion.")
        lines.append("")
        return lines
    
    def _generate_abstract(self) -> List[str]:
        lines = []
        lines.append("## Abstract")
        lines.append("")
        abstract_text = self._generate_abstract_text()
        for para in abstract_text.split("\n"):
            if para.strip():
                lines.append(para)
                lines.append("")
        lines.append("**Keywords:** " + self._generate_keywords())
        lines.append("")
        return lines
    
    def _generate_abstract_text(self) -> str:
        problem = self.data.get("problem_statement", "the research problem")
        final_sol = self.data.get("final_solution", {})
        hypotheses = self.data.get("hypotheses", [])
        parts = []
        parts.append(f"This paper presents a comprehensive research study addressing **{problem}**.")
        parts.append("")
        if hypotheses:
            parts.append(f"The research employed an autonomous hypothesis generation and evaluation framework,")
            parts.append(f"producing **{len(hypotheses)} competing hypotheses** that were critically evaluated and experimentally validated.")
        else:
            parts.append("The research employed a systematic methodology combining theoretical analysis with experimental validation.")
        parts.append("")
        if final_sol and "executive_summary" in final_sol:
            summary = final_sol.get("executive_summary", "")
            if summary:
                parts.append(f"**Key Findings:** {summary}")
                parts.append("")
        evidence = final_sol.get("key_evidence", []) if final_sol else []
        if evidence:
            parts.append("The study provides the following evidence supporting the proposed solution:")
            for ev in evidence[:3]:
                parts.append(f"- {ev}")
            parts.append("")
        parts.append("The research demonstrates a systematic approach to problem-solving through")
        parts.append("autonomous exploration of multiple solution pathways, critical evaluation,")
        parts.append("and empirical validation.")
        return "\n".join(parts)
    
    def _generate_keywords(self) -> str:
        problem = self.data.get("problem_statement", "")
        kb = self.data.get("knowledge_base", {})
        # Insertion-ordered de-duplication. A plain ``set`` here made the
        # keyword list — and therefore the whole paper — change between runs,
        # because Python randomises string hashing per process.
        keywords: List[str] = []

        def _add(term) -> None:
            token = str(term)
            if token and token not in keywords:
                keywords.append(token)
        if problem:
            words = problem.lower().split()
            for w in words:
                if len(w) > 4 and w not in ["this", "that", "with", "from", "have"]:
                    _add(w)
        if kb:
            if "domain" in kb:
                _add(kb["domain"])
            proven = kb.get("proven_facts", [])
            for fact in proven[:2]:
                for word in fact.lower().split():
                    if len(word) > 5:
                        _add(word)
        default_kw = ["autonomous research", "AI-driven discovery", "hypothesis testing"]
        for kw in default_kw:
            _add(kw)
        return ", ".join(keywords[:8])
    
    def _generate_table_of_contents(self) -> List[str]:
        lines = []
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("1. [Introduction](#1-introduction)")
        lines.append("2. [Literature Review / State of the Art](#2-literature-review)")
        lines.append("3. [Research Methodology](#3-research-methodology)")
        lines.append("4. [Results and Analysis](#4-results-and-analysis)")
        lines.append("5. [Discussion](#5-discussion)")
        lines.append("6. [Conclusion](#6-conclusion)")
        lines.append("7. [References](#7-references)")
        lines.append("8. [Appendices](#8-appendices)")
        lines.append("9. [Final Idea Summary](#9-final-idea-summary)")
        lines.append("")
        return lines

    
    def _generate_introduction(self) -> List[str]:
        lines = []
        lines.append("## 1. Introduction")
        lines.append("")
        problem = self.data.get("problem_statement", "the research problem")
        kb = self.data.get("knowledge_base", {})
        lines.append("### 1.1 Motivation and Background")
        lines.append("")
        lines.append(f"This research addresses the fundamental challenge of **{problem}**.")
        lines.append("")
        if kb:
            domain = kb.get("domain", "")
            domain_desc = kb.get("domain_description", "")
            if domain:
                lines.append(f"The research falls within the domain of **{domain}** {domain_desc}.")
                lines.append("")
        lines.append("Understanding and solving this problem has significant implications for:")
        lines.append("")
        lines.append("- Advancing theoretical knowledge in the field")
        lines.append("- Developing practical solutions and applications")
        lines.append("- Addressing current limitations and challenges")
        lines.append("- Enabling future innovations and discoveries")
        lines.append("")
        lines.append("### 1.2 Problem Statement")
        lines.append("")
        lines.append(f"**Research Question:** {problem}")
        lines.append("")
        lines.append("### 1.3 Research Objectives")
        lines.append("")
        lines.append("The primary objectives of this research are:")
        lines.append("")
        lines.append("1. **Hypothesis Generation**: Generate multiple competing hypotheses")
        lines.append("2. **Critical Evaluation**: Critically evaluate each hypothesis")
        lines.append("3. **Experimental Validation**: Design and execute controlled experiments")
        lines.append("4. **Evidence-Based Conclusion**: Synthesize experimental results")
        lines.append("5. **Knowledge Contribution**: Document findings comprehensively")
        lines.append("")
        return lines
    
    def _generate_literature_review(self) -> List[str]:
        lines = []
        lines.append("## 2. Literature Review / State of the Art")
        lines.append("")
        kb = self.data.get("knowledge_base", {})
        prior_art = kb.get("state_of_the_art_prior_art", [])
        if prior_art:
            lines.append("### 2.1 Current State of the Art")
            lines.append("")
            lines.append("The following approaches represent the current state of the art:")
            lines.append("")
            for i, item in enumerate(prior_art, 1):
                lines.append(f"{i}. **{item}**")
            lines.append("")
        else:
            lines.append("### 2.1 Current State of the Art")
            lines.append("")
            lines.append("The research builds upon existing knowledge and established principles.")
            lines.append("")
        proven = kb.get("proven_facts", [])
        if proven:
            lines.append("### 2.2 Established Knowledge")
            lines.append("")
            lines.append("The following facts are well-established in the field:")
            lines.append("")
            for i, fact in enumerate(proven, 1):
                lines.append(f"{i}. {fact}")
            lines.append("")
        unknowns = kb.get("critical_unanswered_unknowns", [])
        if unknowns:
            lines.append("### 2.3 Open Questions and Challenges")
            lines.append("")
            lines.append("Despite advances, several critical questions remain unanswered:")
            lines.append("")
            for i, q in enumerate(unknowns, 1):
                lines.append(f"{i}. {q}")
            lines.append("")
        failures = kb.get("failure_modes_and_risks", [])
        if failures:
            lines.append("### 2.4 Known Failure Modes and Risks")
            lines.append("")
            lines.append("Understanding potential failure modes is crucial:")
            lines.append("")
            for i, f in enumerate(failures, 1):
                lines.append(f"{i}. **{f}**")
            lines.append("")
        return lines

    
    def _generate_methodology(self) -> List[str]:
        lines = []
        lines.append("## 3. Research Methodology")
        lines.append("")
        lines.append("This research employed a comprehensive, multi-phase autonomous research methodology.")
        lines.append("")
        lines.append("### 3.1 Problem Analysis")
        lines.append("")
        lines.append("The research problem was systematically analyzed to understand:")
        lines.append("")
        lines.append("- Core requirements and constraints")
        lines.append("- Key variables and parameters")
        lines.append("- Success criteria and metrics")
        lines.append("- Potential solution approaches")
        lines.append("")
        lines.append("### 3.2 Hypothesis Generation")
        lines.append("")
        hypotheses = self.data.get("hypotheses", [])
        lines.append(f"The system generated **{len(hypotheses)} competing hypotheses** through autonomous generation.")
        lines.append("")
        if hypotheses:
            lines.append("**Generated Hypotheses:**")
            lines.append("")
            for i, hyp in enumerate(hypotheses[:10], 1):
                hyp_text = hyp.get("hypothesis", hyp.get("idea", hyp.get("title", str(hyp))))
                lines.append(f"{i}. {hyp_text}")
            if len(hypotheses) > 10:
                lines.append(f"... and {len(hypotheses) - 10} more hypotheses")
            lines.append("")
        lines.append("### 3.3 Critical Evaluation and Tournament Selection")
        lines.append("")
        lines.append("Each hypothesis underwent rigorous critical evaluation based on:")
        lines.append("")
        lines.append("- **Novelty**: Originality and innovative aspects")
        lines.append("- **Feasibility**: Practical implementability")
        lines.append("- **Reasoning Quality**: Logical soundness and evidence support")
        lines.append("- **Potential Impact**: Expected benefits and significance")
        lines.append("")
        tournament = self.data.get("tournament_results", {})
        if tournament:
            survivors = tournament.get("survivors", [])
            eliminated = tournament.get("eliminated", [])
            if survivors is not None and eliminated is not None:
                lines.append(f"After evaluation, **{len(survivors)} hypotheses survived** the tournament")
                lines.append(f"and **{len(eliminated)} were eliminated** based on evaluation criteria.")
                lines.append("")
        lines.append("### 3.4 Experimental Validation")
        lines.append("")
        lines.append("Surviving hypotheses were tested through controlled experiments designed to:")
        lines.append("")
        lines.append("- Measure performance against baseline methods")
        lines.append("- Quantify improvements and trade-offs")
        lines.append("- Validate theoretical predictions empirically")
        lines.append("- Assess robustness under varying conditions")
        lines.append("")
        return lines

    
    def _generate_results(self) -> List[str]:
        lines = []
        lines.append("## 4. Results and Analysis")
        lines.append("")
        experiments = self.data.get("experiments", {})
        tournament = self.data.get("tournament_results", {})
        final_sol = self.data.get("final_solution", {})
        validation = self.data.get("validation_results", [])
        lines.append("### 4.1 Tournament Evaluation Results")
        lines.append("")
        if tournament:
            winner = tournament.get("winner", {})
            if winner:
                lines.append(f"**Tournament Winner:** {winner.get('idea', winner.get('hypothesis', 'N/A'))}")
                lines.append("")
                lines.append(f"**Validation Score:** {winner.get('validation_score', 'N/A')}")
                lines.append(f"**Overall Status:** {winner.get('overall_status', 'N/A')}")
                lines.append("")
        lines.append("### 4.2 Experimental Results")
        lines.append("")
        if experiments and experiments.get("results"):
            results = experiments["results"]
            if isinstance(results, list) and results:
                lines.append("The following experimental results were obtained:")
                lines.append("")
                lines.append("| Hypothesis | Method | Status | Baseline | Proposed | Improvement |")
                lines.append("|------------|--------|--------|----------|----------|-------------|")
                for result in results[:10]:
                    hyp_id = result.get("hypothesis_id", result.get("id", "N/A"))
                    method = result.get("method", result.get("proposed_method", "N/A"))
                    status = result.get("status", "N/A")
                    baseline = result.get("baseline", "N/A")
                    proposed = result.get("proposed", "N/A")
                    improvement = result.get("improvement_pct", result.get("improvement", "N/A"))
                    lines.append(f"| {hyp_id} | {method} | {status} | {baseline} | {proposed} | {improvement} |")
                lines.append("")
        if validation:
            lines.append("### 4.3 Validation Analysis")
            lines.append("")
            lines.append("Each hypothesis underwent comprehensive validation testing:")
            lines.append("")
            for val in validation[:5]:
                idea = val.get("idea", val.get("hypothesis", "N/A"))
                score = val.get("validation_score", val.get("score", "N/A"))
                status = val.get("overall_status", "N/A")
                lines.append(f"- **{idea[:80]}**")
                lines.append(f"  - Score: {score}")
                lines.append(f"  - Status: {status}")
                lines.append("")
        lines.append("### 4.4 Evidence Summary")
        lines.append("")
        evidence = final_sol.get("key_evidence", []) if final_sol else []
        if evidence:
            lines.append("The research accumulated the following key evidence:")
            lines.append("")
            for i, ev in enumerate(evidence, 1):
                lines.append(f"{i}. {ev}")
            lines.append("")
        return lines

    
    def _generate_discussion(self) -> List[str]:
        lines = []
        lines.append("## 5. Discussion")
        lines.append("")
        final_sol = self.data.get("final_solution", {})
        kb = self.data.get("knowledge_base", {})
        lines.append("### 5.1 Interpretation of Results")
        lines.append("")
        if final_sol and "core_mechanism" in final_sol:
            lines.append(f"The research reveals that the most promising approach involves:")
            lines.append("")
            lines.append(f"**{final_sol.get('core_mechanism', '')}**")
            lines.append("")
        lines.append("Key insights from this research include:")
        lines.append("")
        if final_sol:
            falsification = final_sol.get("falsification_criteria", [])
            if falsification:
                lines.append("**Falsification Criteria:** The proposed solution would be considered invalid if:")
                lines.append("")
                for f in falsification[:3]:
                    lines.append(f"- {f}")
                lines.append("")
        lines.append("### 5.2 Implications")
        lines.append("")
        lines.append("**Theoretical Implications:**")
        lines.append("")
        lines.append("- Contributes to understanding of the research problem domain")
        lines.append("- Provides new perspectives on solution approaches")
        lines.append("- Identifies gaps in current knowledge")
        lines.append("")
        lines.append("**Practical Implications:**")
        lines.append("")
        lines.append("- Offers actionable insights for practitioners")
        lines.append("- Demonstrates viable solution pathways")
        lines.append("- Highlights implementation considerations")
        lines.append("")
        lines.append("### 5.3 Comparison with Existing Approaches")
        lines.append("")
        prior_art = kb.get("state_of_the_art_prior_art", [])
        if prior_art:
            lines.append("Compared to existing approaches in the field:")
            lines.append("")
            for i, approach in enumerate(prior_art, 1):
                lines.append(f"{i}. **{approach}**")
            lines.append("")
            lines.append("The proposed solution offers potential advantages:")
            lines.append("")
            lines.append("- Novel combination of techniques")
            lines.append("- Addressed limitations of prior approaches")
            lines.append("- Empirical validation of effectiveness")
            lines.append("")
        return lines

    
    def _generate_conclusion(self) -> List[str]:
        lines = []
        lines.append("## 6. Conclusion")
        lines.append("")
        final_sol = self.data.get("final_solution", {})
        problem = self.data.get("problem_statement", "the research problem")
        lines.append("### 6.1 Summary of Findings")
        lines.append("")
        lines.append(f"This comprehensive research study addressed the challenge of **{problem}**")
        lines.append("through an autonomous, systematic research methodology.")
        lines.append("")
        if final_sol and "executive_summary" in final_sol:
            lines.append(final_sol.get("executive_summary", ""))
            lines.append("")
        lines.append("### 6.2 Contributions")
        lines.append("")
        lines.append("The main contributions of this research include:")
        lines.append("")
        lines.append("1. **Systematic Methodology**: Demonstrated an effective autonomous research pipeline")
        lines.append("2. **Multiple Solutions Explored**: Generated and evaluated diverse hypotheses")
        lines.append("3. **Empirical Validation**: Conducted controlled experiments to validate findings")
        lines.append("4. **Evidence-Based Conclusions**: Derived conclusions supported by experimental evidence")
        lines.append("5. **Knowledge Documentation**: Comprehensive documentation of the research process")
        lines.append("")
        lines.append("### 6.3 Open Questions for Future Research")
        lines.append("")
        open_qs = final_sol.get("open_questions", []) if final_sol else []
        if open_qs:
            lines.append("This research identified several open questions for future investigation:")
            lines.append("")
            for i, q in enumerate(open_qs, 1):
                lines.append(f"{i}. {q}")
            lines.append("")
        else:
            lines.append("Several questions remain open for future investigation:")
            lines.append("")
            lines.append("- Optimization of key parameters")
            lines.append("- Validation under diverse conditions")
            lines.append("- Long-term performance and reliability")
            lines.append("")
        lines.append("### 6.4 Proposed Future Experiments")
        lines.append("")
        proposed_exp = final_sol.get("proposed_experiment", "") if final_sol else ""
        if proposed_exp:
            lines.append(f"**Recommended Next Step:** {proposed_exp}")
            lines.append("")
        return lines

    
    def _generate_references(self) -> List[str]:
        lines = []
        lines.append("## 7. References")
        lines.append("")
        lines.append("This research draws upon the following knowledge sources and methodologies:")
        lines.append("")
        kb = self.data.get("knowledge_base", {})
        prior_art = kb.get("state_of_the_art_prior_art", [])
        if prior_art:
            lines.append("**State of the Art and Prior Work:**")
            lines.append("")
            for i, item in enumerate(prior_art, 1):
                lines.append(f"[{i}] {item}")
            lines.append("")
        proven = kb.get("proven_facts", [])
        if proven:
            lines.append("**Established Principles:**")
            lines.append("")
            for i, fact in enumerate(proven, 1):
                lines.append(f"[{i + len(prior_art)}] {fact}")
            lines.append("")
        lines.append("**Methodological References:**")
        lines.append("")
        lines.append("[1] Autonomous AI Research Laboratory (AARL) Methodology")
        lines.append("[2] Systematic Hypothesis Generation and Evaluation Framework")
        lines.append("[3] Experimental Design and Validation Protocols")
        lines.append("[4] Evidence-Based Research Synthesis Methods")
        lines.append("")
        lines.append("---")
        lines.append("*Note: References are automatically generated from the research knowledge base.*")
        lines.append("")
        return lines
    
    def _generate_appendices(self) -> List[str]:
        lines = []
        lines.append("## 8. Appendices")
        lines.append("")
        lines.append("### Appendix A: Complete Hypothesis List")
        lines.append("")
        hypotheses = self.data.get("hypotheses", [])
        if hypotheses:
            lines.append(f"The following **{len(hypotheses)} hypotheses** were generated:")
            lines.append("")
            for i, hyp in enumerate(hypotheses, 1):
                hyp_text = hyp.get("hypothesis", hyp.get("idea", hyp.get("title", str(hyp))))
                lines.append(f"**H{i}.** {hyp_text}")
            lines.append("")
        else:
            lines.append("No hypotheses were generated during this research session.")
            lines.append("")
        lines.append("### Appendix B: Experimental Data Summary")
        lines.append("")
        experiments = self.data.get("experiments", {})
        if experiments:
            lines.append("**Experiment Configuration:**")
            lines.append("")
            config = experiments.get("configuration", {})
            if config:
                for key, value in config.items():
                    lines.append(f"- {key}: {value}")
            lines.append("")
            results = experiments.get("results", [])
            if results:
                lines.append("**Detailed Results:**")
                lines.append("")
                for result in results:
                    lines.append(f"- Hypothesis: {result.get('hypothesis_id', 'N/A')}")
                    lines.append(f"  - Method: {result.get('method', 'N/A')}")
                    lines.append(f"  - Status: {result.get('status', 'N/A')}")
                    if 'baseline' in result:
                        lines.append(f"  - Baseline Performance: {result['baseline']}")
                    if 'proposed' in result:
                        lines.append(f"  - Proposed Performance: {result['proposed']}")
                    if 'improvement_pct' in result:
                        lines.append(f"  - Improvement: {result['improvement_pct']}%")
                    lines.append("")
        else:
            lines.append("No experimental data available for this research session.")
            lines.append("")
        return lines

    
    def _generate_final_idea_summary(self) -> List[str]:
        lines = []
        lines.append("=" * 80)
        lines.append("")
        lines.append("# 9. FINAL IDEA SUMMARY — Comprehensive Synthesis")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        final_sol = self.data.get("final_solution", {})
        kb = self.data.get("knowledge_base", {})
        evidence = self.data.get("validation_results", [])
        tournament = self.data.get("tournament_results", {})
        problem = self.data.get("problem_statement", "Research Problem")
        
        # Executive Summary Box
        lines.append("## 9.1 Executive Summary")
        lines.append("")
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "FINAL RESEARCH SOLUTION — EXECUTIVE SUMMARY".center(78) + "│")
        lines.append("└" + "─" * 78 + "┘")
        lines.append("")
        
        if final_sol and "executive_summary" in final_sol:
            summary = final_sol.get("executive_summary", "")
            for para in summary.split(". "):
                if para.strip():
                    lines.append(f"> {para.strip()}.")
                    lines.append("")
        
        # The Final Idea
        lines.append("## 9.2 The Final Research Idea")
        lines.append("")
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "PROPOSED SOLUTION".center(78) + "│")
        lines.append("└" + "─" * 78 + "┘")
        lines.append("")
        lines.append(f"**Research Problem:** {problem}")
        lines.append("")
        
        if final_sol and "core_mechanism" in final_sol:
            lines.append("**Core Mechanism:**")
            lines.append("")
            lines.append(final_sol.get("core_mechanism", ""))
            lines.append("")
        
        # Detailed breakdown
        lines.append("## 9.3 Detailed Solution Breakdown")
        lines.append("")
        lines.append("### 9.3.1 Problem Understanding")
        lines.append("")
        lines.append(f"The research addressed the following core challenge:")
        lines.append("")
        lines.append(f"> **{problem}**")
        lines.append("")
        
        if kb:
            domain = kb.get("domain", "")
            if domain:
                lines.append(f"**Domain:** {domain}")
                lines.append("")
        
        # Solution Components
        lines.append("### 9.3.2 Solution Architecture")
        lines.append("")
        lines.append("The proposed solution consists of the following key components:")
        lines.append("")
        
        components = []
        if kb:
            if "proven_facts" in kb:
                for fact in kb["proven_facts"][:3]:
                    components.append(("Scientific Principle", fact))
            if "state_of_the_art_prior_art" in kb:
                for art in kb["state_of_the_art_prior_art"][:2]:
                    components.append(("Prior Art Integration", art))
        
        for ev in evidence[:3]:
            if "idea" in ev or "hypothesis" in ev:
                components.append(("Validated Concept", ev.get("idea", ev.get("hypothesis", ""))))
        
        if not components:
            if final_sol and "core_mechanism" in final_sol:
                components.append(("Core Approach", final_sol.get("core_mechanism", "")))
        
        for i, (comp_type, comp_desc) in enumerate(components, 1):
            lines.append(f"{i}. **{comp_type}:** {comp_desc}")
        lines.append("")
        
        # Evidence
        lines.append("### 9.3.3 Evidence Supporting the Solution")
        lines.append("")
        key_evidence = final_sol.get("key_evidence", []) if final_sol else []
        if key_evidence:
            lines.append("The following evidence supports the proposed solution:")
            lines.append("")
            for i, ev in enumerate(key_evidence, 1):
                lines.append(f"{i}. {ev}")
            lines.append("")
        else:
            lines.append("The solution is supported by systematic analysis and validation.")
            lines.append("")
        
        # Why this solution
        lines.append("### 9.3.4 Why This Solution Was Selected")
        lines.append("")
        reasoning_points = []
        
        winner = tournament.get("winner", {})
        if winner:
            reasoning_points.append(f"Winner of autonomous tournament evaluation with validation score of {winner.get('validation_score', 'N/A')}")
        
        if evidence:
            passed = [e for e in evidence if e.get("overall_status") == "PASS"]
            if passed:
                reasoning_points.append(f"Successfully passed {len(passed)} validation tests")
        
        if final_sol:
            falsification = final_sol.get("falsification_criteria", [])
            if falsification:
                reasoning_points.append("Meets clear falsification criteria for scientific validation")
        
        if not reasoning_points:
            reasoning_points.append("Selected through systematic evaluation of multiple alternatives")
        
        for i, point in enumerate(reasoning_points, 1):
            lines.append(f"{i}. {point}")
        lines.append("")
        
        # Implementation Guidance
        lines.append("## 9.4 Implementation Guidance")
        lines.append("")
        lines.append("### 9.4.1 Key Considerations")
        lines.append("")
        
        considerations = []
        failures = kb.get("failure_modes_and_risks", [])
        if failures:
            considerations.append("**Risk Mitigation:**")
            for f in failures[:2]:
                considerations.append(f"- Address potential risk: {f}")
        
        unknowns = kb.get("critical_unanswered_unknowns", [])
        if unknowns:
            considerations.append("**Open Questions to Address:**")
            for u in unknowns[:2]:
                considerations.append(f"- {u}")
        
        if not considerations:
            considerations.append("Implement and validate according to the proposed experimental design")
        
        for cons in considerations:
            lines.append(cons)
        lines.append("")
        
        lines.append("### 9.4.2 Recommended Validation Experiment")
        lines.append("")
        proposed_exp = final_sol.get("proposed_experiment", "") if final_sol else ""
        if proposed_exp:
            lines.append(f"**Experiment:** {proposed_exp}")
            lines.append("")
        else:
            lines.append("**Recommended Approach:**")
            lines.append("")
            lines.append("1. Implement the proposed solution")
            lines.append("2. Design controlled experiments to measure performance")
            lines.append("3. Compare against baseline methods")
            lines.append("4. Document and analyze results")
            lines.append("")
        
        lines.append("### 9.4.3 Success Criteria")
        lines.append("")
        lines.append("The solution can be considered successful if:")
        lines.append("")
        
        criteria = []
        if final_sol:
            falsification = final_sol.get("falsification_criteria", [])
            for f in falsification[:2]:
                criteria.append(f"- Does NOT exhibit: {f}")
        criteria.append("- Demonstrates measurable improvement over baseline")
        criteria.append("- Is reproducible under controlled conditions")
        criteria.append("- Aligns with established scientific principles")
        
        for c in criteria:
            lines.append(c)
        lines.append("")
        
        # Limitations
        lines.append("## 9.5 Limitations and Caveats")
        lines.append("")
        lines.append("This research and the proposed solution have the following limitations:")
        lines.append("")
        limitations = [
            "The solution is based on current knowledge and may require updates as new information emerges",
            "Empirical validation may be needed under diverse conditions",
            "Implementation details may require domain-specific expertise",
            "Long-term performance characteristics need further investigation",
        ]
        for i, lim in enumerate(limitations, 1):
            lines.append(f"{i}. {lim}")
        lines.append("")
        
        # Next steps
        lines.append("## 9.6 Next Steps and Future Directions")
        lines.append("")
        next_steps = []
        open_qs = final_sol.get("open_questions", []) if final_sol else []
        for q in open_qs[:3]:
            next_steps.append(f"Investigate: {q}")
        if not next_steps:
            next_steps.append("Conduct empirical validation experiments")
            next_steps.append("Refine solution based on experimental results")
            next_steps.append("Explore alternative approaches for comparison")
        lines.append("Recommended next steps:")
        lines.append("")
        for i, step in enumerate(next_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")
        
        # Final statement
        lines.append("## 9.7 Concluding Statement")
        lines.append("")
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "FINAL RESEARCH IDEA — COMPREHENSIVE SUMMARY".center(78) + "│")
        lines.append("└" + "─" * 78 + "┘")
        lines.append("")
        lines.append("")
        lines.append(f"**Research Problem:** {problem}")
        lines.append("")
        if final_sol and "core_mechanism" in final_sol:
            lines.append(f"**Proposed Solution:** {final_sol.get('core_mechanism', '')}")
            lines.append("")
        lines.append("**Key Evidence:**")
        lines.append("")
        if key_evidence:
            for ev in key_evidence[:3]:
                lines.append(f"✓ {ev}")
        else:
            lines.append("✓ Systematically generated and evaluated through autonomous research pipeline")
        lines.append("")
        lines.append("**This research demonstrates the effectiveness of autonomous AI-driven research")
        lines.append("methodology in exploring complex problems, generating innovative hypotheses, and")
        lines.append("deriving evidence-based solutions through systematic evaluation and validation.**")
        lines.append("")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        
        return lines

    
    def _generate_latex_paper(self) -> str:
        lines = [
            "\\documentclass[12pt,a4paper]{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{geometry}",
            "\\usepackage{amsmath}",
            "\\usepackage{graphicx}",
            "\\usepackage{hyperref}",
            "\\usepackage{setspace}",
            "",
            "\\geometry{margin=1in}",
            "\\onehalfspacing",
            "",
            "\\title{A Comprehensive Study on " + self.data.get('problem_statement', 'Research') + "}",
            "\\author{Autonomous AI Research Laboratory (AARL)}",
            "\\date{" + _format_date() + "}",
            "",
            "\\begin{document}",
            "",
            "\\maketitle",
            "",
            "\\begin{abstract}",
            "",
        ]
        abstract = self._generate_abstract_text()
        for para in abstract.split("\n"):
            if para.strip():
                lines.append("\\noindent " + para)
        lines.extend([
            "",
            "\\end{abstract}",
            "",
            "\\tableofcontents",
            "",
            "\\newpage",
            "",
            "\\section{Introduction}",
            "",
            "This research addresses the fundamental challenge of " + self.data.get('problem_statement', 'the research problem') + ".",
            "",
            "\\section{Literature Review}",
            "",
            "The research builds upon existing knowledge and established principles.",
            "",
            "\\section{Methodology}",
            "",
            "The system generated " + str(len(self.data.get('hypotheses', []))) + " competing hypotheses through autonomous generation.",
            "",
            "\\section{Results}",
            "",
        ])
        tournament = self.data.get("tournament_results", {})
        if tournament:
            winner = tournament.get("winner", {})
            if winner:
                lines.extend([
                    "\\textbf{Winner: } " + winner.get('idea', 'N/A'),
                    "",
                ])
        lines.extend([
            "\\section{Discussion}",
            "",
            "The research reveals important insights about the problem domain.",
            "",
            "\\section{Conclusion}",
            "",
            "This comprehensive research study addressed the research problem through an autonomous, systematic methodology.",
            "",
            "\\section*{References}",
            "",
            "\\begin{enumerate}",
            "\\item Autonomous AI Research Laboratory (AARL) Methodology",
            "\\item Systematic Hypothesis Generation Framework",
            "\\item Experimental Design and Validation Protocols",
            "\\end{enumerate}",
            "",
            "\\end{document}",
        ])
        return "\n".join(lines)



def generate_research_paper(
    root_dir: str,
    output_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
    format: str = "markdown",
    filename: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> str:
    if output_dir is None:
        output_dir = os.path.join(root_dir, "Research_Papers")
    os.makedirs(output_dir, exist_ok=True)
    
    collector = ResearchDataCollector(root_dir, run_dir)
    data = collector.collect_all(max_workers=max_workers)
    
    generator = ResearchPaperGenerator(data, max_workers=max_workers)
    filepath = generator.generate_full_paper(output_dir, format)
    
    if filename:
        ext = ".md" if format == "markdown" else ".tex"
        if not filename.endswith(ext):
            filename += ext
        custom_path = os.path.join(output_dir, filename)
        os.rename(filepath, custom_path)
        filepath = custom_path
    
    return filepath


def generate_paper_from_pipeline_data(
    pipeline_output_dir: str,
    problem_statement: str,
    output_dir: Optional[str] = None,
) -> str:
    data = {
        "problem_statement": problem_statement,
        "knowledge_base": {},
        "hypotheses": [],
        "experiments": {},
        "tournament_results": {},
        "final_solution": {},
        "validation_results": [],
        "confidence_scores": [],
        "derived_equations": {},
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "source": "AARL Pipeline",
        }
    }
    
    tournament_file = os.path.join(pipeline_output_dir, "tournament_results.json")
    if os.path.exists(tournament_file):
        tournament = _load_json(tournament_file)
        if tournament:
            data["tournament_results"] = tournament
    
    final_sol_file = os.path.join(pipeline_output_dir, "final_research_solution.json")
    if os.path.exists(final_sol_file):
        final_sol = _load_json(final_sol_file)
        if final_sol:
            data["final_solution"] = final_sol
    
    hyp_file = os.path.join(pipeline_output_dir, "hypotheses.json")
    if os.path.exists(hyp_file):
        hyps = _load_json(hyp_file)
        if hyps:
            data["hypotheses"] = hyps if isinstance(hyps, list) else []
    
    generator = ResearchPaperGenerator(data, max_workers=max_workers)
    
    if output_dir is None:
        output_dir = os.path.join(pipeline_output_dir, "research_papers")
    
    return generator.generate_full_paper(output_dir, "markdown")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AARL Research Paper Generator - Generate comprehensive research papers"
    )
    parser.add_argument("--root-dir", default=".", help="Root directory of AARL project")
    parser.add_argument("--output-dir", default=None, help="Output directory for papers")
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown", help="Output format")
    parser.add_argument("--filename", default=None, help="Custom filename for the paper")
    parser.add_argument("--run-dir", default=None, help="Specific run directory to use")
    parser.add_argument("--workers", type=int, default=None,
                        help="max parallel workers (default: all CPU cores, "
                             "env: AARL_MAX_WORKERS)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AARL Research Paper Generator")
    print("=" * 70)
    print(f"Root Directory: {args.root_dir}")
    print(f"Output Format: {args.format}")
    print(f"Output Directory: {args.output_dir or os.path.join(args.root_dir, 'Research_Papers')}")
    print("-" * 70)
    
    try:
        paper_path = generate_research_paper(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            run_dir=args.run_dir,
            format=args.format,
            filename=args.filename,
            max_workers=args.workers,
        )
        print(f"[OK] Research paper generated successfully!")
        print(f"[FILE] Paper location: {paper_path}")
        print("=" * 70)
    except Exception as e:
        print(f"[FAIL] Error generating paper: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


print("ResearchPaperGenerator module loaded successfully.")


