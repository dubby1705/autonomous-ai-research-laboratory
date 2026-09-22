"""
FINAL DESIGN EXPLAINED — researcher-readable deep-dive document.

Phase 13 of the AARL pipeline. Takes every session artifact (knowledge base,
verified hypotheses, derived equations, evidence scores, simulation comparison,
final solution) and explains THE NEW IDEA in full: what is being proposed, the
complete design part by part, how it works, the mathematics behind it, the
predicted performance, the evidence for it, and how it could fail.

The generator is FULLY DOMAIN-AGNOSTIC and DETERMINISTIC (no LLM): every
sentence is assembled from the current session's outputs, so a researcher can
read the document and trace each statement back to a session file (Appendix).
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # script execution (Engine/Question_Engine is on sys.path)
    from ResearchOutputGenerator import (
        _infer_architecture, _infer_calculations, _infer_components,
        _infer_engineering_metrics, _infer_materials, _infer_technologies,
        _load_json, _safe_get, _truncate,
    )
    _ROG_HELPERS = True
except Exception:  # pragma: no cover - package import
    try:
        from Engine.Question_Engine.ResearchOutputGenerator import (
            _infer_architecture, _infer_calculations, _infer_components,
            _infer_engineering_metrics, _infer_materials, _infer_technologies,
            _load_json, _safe_get, _truncate,
        )
        _ROG_HELPERS = True
    except Exception:
        _ROG_HELPERS = False

if not _ROG_HELPERS:  # minimal local fallbacks so the module always works
    def _load_json(path: str) -> Any:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _safe_get(data: Any, key: str, default: Any = None) -> Any:
        return data.get(key, default) if isinstance(data, dict) else default

    def _truncate(text: str, max_len: int = 200) -> str:
        text = str(text or "").strip()
        return text if len(text) <= max_len else text[:max_len].rstrip() + "..."

    def _no_infer(*args, **kwargs) -> List[str]:
        return []

    _infer_components = _infer_materials = _infer_technologies = _no_infer
    _infer_architecture = _infer_engineering_metrics = _infer_calculations = _no_infer


# Plain-English reading of each relationship type found in derived equations.
_RELATIONSHIP_READINGS = {
    "proportional": "rises in direct proportion when the right-hand side grows",
    "inverse": "falls as the denominator grows (opposite movement)",
    "exponential": "changes exponentially — small input shifts compound rapidly",
    "logarithmic": "changes logarithmically — large inputs give diminishing returns",
    "power_law": "follows a power law — scaling behaviour with an exponent",
    "polynomial": "follows a polynomial curve of the given degree",
    "linear": "changes at a constant rate",
    "quadratic": "changes quadratically — the effect grows with the square",
    "threshold": "stays flat until a tipping point, then changes sharply",
    "sigmoid": "starts slow, accelerates, then saturates (S-shaped curve)",
}

_CLASSIFICATION_NOTES = {
    "Excellent": "strong evidence, clear mechanism, testable and consistent with known laws",
    "Plausible": "reasonable idea with some evidence; testable although challenging",
    "Speculative": "interesting but lacking evidence or mechanism — worth exploring",
}


def _load_from(data_dir: str, filename: str,
               fallback_dirs: Optional[List[str]] = None) -> Any:
    """Load a JSON artifact: session data dir first, then fallback dirs."""
    data = _load_json(os.path.join(data_dir, filename))
    if data is not None:
        return data
    for d in (fallback_dirs or []):
        data = _load_json(os.path.join(d, filename))
        if data is not None:
            return data
    return None


def _bullets(items: List[Any], indent: str = "- ") -> List[str]:
    out = []
    for it in items or []:
        text = it.strip() if isinstance(it, str) else _truncate(it, 300)
        if text:
            out.append(f"{indent}{text}")
    return out


def _read_relationship(rel_type: str) -> str:
    return _RELATIONSHIP_READINGS.get(str(rel_type or "").lower(),
                                      "relates the variables as written")


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)

# =========================================================
# DOCUMENT SECTIONS (1-4)
# =========================================================
def _sec_header(problem: str, domain: str, runtime_str: str,
                stats: Dict[str, Any]) -> List[str]:
    out = [
        "# FINAL DESIGN & NEW IDEA — FULLY EXPLAINED",
        "",
        f"- **Research problem:** {problem or 'N/A'}",
        f"- **Domain:** {domain or 'general'}",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        + (f"  |  **Pipeline runtime:** {runtime_str}" if runtime_str else ""),
    ]
    if stats:
        out.append(
            f"- **Session scale:** {stats.get('kb_nodes', 0)} knowledge items · "
            f"{stats.get('hypotheses_approved', 0)}/"
            f"{stats.get('hypotheses_generated', 0)} hypotheses approved · "
            f"{stats.get('math_validated', 0)} equations validated"
        )
    out += [
        "",
        "> **How to read this document** — sections 1–3 explain the new idea and",
        "> why it is credible; sections 4–7 are the full design (every part, the",
        "> mathematics and the predicted performance); sections 8–11 give the",
        "> evidence, the risks and what to test next. The appendix maps every",
        "> statement back to the session file it came from.",
        "",
        "---",
        "",
    ]
    return out


def _sec_summary(solution: Dict[str, Any]) -> List[str]:
    title = _safe_get(solution, "research_title") or "Proposed research solution"
    summary = _safe_get(solution, "executive_summary") or ""
    mechanism = _safe_get(solution, "core_mechanism") or ""
    out = [
        "## 1. EXECUTIVE SUMMARY — WHAT AARL PROPOSES",
        "",
        f"**{title}**",
        "",
    ]
    if summary:
        out += [summary, ""]
    if mechanism:
        out += ["**Core mechanism in one paragraph:**", "", mechanism, ""]
    out += [
        "The remainder of this document unpacks that mechanism: which parts make up",
        "the design, what each part does, the equations that govern it, how much",
        "improvement is predicted against the current baseline, and how strongly the",
        "evidence supports each claim.",
        "",
        "---",
        "",
    ]
    return out


def _sec_problem(kb: Dict[str, Any], user_problem: str) -> List[str]:
    thesis = _safe_get(kb, "core_research_thesis") or ""
    prior_art = _safe_get(kb, "state_of_the_art_prior_art") or []
    facts = _safe_get(kb, "proven_facts") or []
    out = ["## 2. THE PROBLEM AND WHAT EXISTS TODAY", ""]
    if user_problem:
        out += [f"**Problem statement.** {user_problem}", ""]
    if thesis:
        out += [f"**Knowledge-base thesis.** {thesis}", ""]
    if prior_art:
        out += ["**State of the art — prior approaches found during research:**", ""]
        out += _bullets(prior_art)
        out += [""]
    if facts:
        out += [
            "**Established facts the design builds on** (treated as verified ground",
            "truth):",
            "",
        ]
        out += _bullets(facts)
        out += [""]
    out += ["---", ""]
    return out

def _sec_new_idea(verified: List[Dict[str, Any]],
                  kb: Dict[str, Any]) -> List[str]:
    out = [
        "## 3. THE NEW IDEA — CANDIDATE CONCEPTS AND HOW THEY WERE CHOSEN",
        "",
        "The hypothesis engine generated candidates, scored every one across six",
        "dimensions (novelty, feasibility, evidence, consistency, testability,",
        "risk), and kept only the concepts that scored well enough. The surviving",
        "concepts follow, best first.",
        "",
    ]
    ranked = sorted(
        verified,
        key=lambda v: float(_safe_get(v, "final_score", 0.0) or 0.0),
        reverse=True,
    )
    if ranked:
        for i, v in enumerate(ranked, 1):
            cls = _safe_get(v, "classification", "Unknown")
            out += [
                f"### 3.{i} Concept {i} — {cls} "
                f"(overall {float(_safe_get(v, 'final_score', 0) or 0):.1f}/10)",
                "",
                "**The idea:** "
                + (_safe_get(v, "refined_hypothesis")
                   or _safe_get(v, "original_prediction", "N/A")),
                "",
                "**Six-dimension review:**",
                "",
                "| Dimension | Score | Meaning |",
                "|---|---|---|",
                f"| Novelty | {_safe_get(v, 'novelty', 0)}/10 | how new the idea is |",
                f"| Feasibility | {_safe_get(v, 'feasibility', 0)}/10 | testable with current technology |",
                f"| Evidence | {_safe_get(v, 'evidence', 0)}/10 | existing supporting evidence |",
                f"| Consistency | {_safe_get(v, 'consistency', 0)}/10 | agreement with known laws |",
                f"| Testability | {_safe_get(v, 'testability', 0)}/10 | clarity of the testing experiment |",
                f"| Risk | {_safe_get(v, 'risk', 0)}/10 | chance of fatal flaw (lower is better) |",
                "",
            ]
            reasoning = _safe_get(v, "reasoning") or ""
            if reasoning:
                out += [f"**Reviewer's reasoning.** {reasoning}", ""]
            weakness = _safe_get(v, "primary_weakness") or ""
            if weakness:
                out += [f"**Honest weakness flagged by the reviewer.** {weakness}", ""]
            note = _CLASSIFICATION_NOTES.get(cls)
            if note:
                out += [f"*Classification meaning: {note}.*", ""]
    else:
        hyps = _safe_get(kb, "proposed_testable_hypotheses") or []
        out += [
            "The verifier did not return scored concepts this session, so the raw",
            "testable hypotheses mined from the knowledge base are listed instead:",
            "",
        ]
        out += _bullets(hyps)
        out += [""]
    out += ["---", ""]
    return out

def _sec_design(solution: Dict[str, Any], kb: Dict[str, Any],
                verified: List[Dict[str, Any]],
                comparison: Dict[str, Any],
                param_changes: List[Dict[str, Any]]) -> List[str]:
    mechanism = _safe_get(solution, "core_mechanism") or ""
    experiment = _safe_get(solution, "proposed_experiment") or ""
    out = [
        "## 4. FULL DESIGN — EVERY PART EXPLAINED",
        "",
        "### 4.1 The core mechanism, step by step",
        "",
    ]
    if mechanism:
        out += [mechanism, ""]
    else:
        out += [
            "The session did not record an explicit mechanism statement; the",
            "mechanism is implied by the equations in section 6 and the causal",
            "chains in section 5.",
            "",
        ]
    if experiment:
        out += ["**How the design would be realised and measured:**", "", experiment, ""]

    kb_data = kb if isinstance(kb, dict) else {}
    verified_data = verified if isinstance(verified, list) else []
    comparison_data = comparison if isinstance(comparison, dict) else {}
    architecture = _infer_architecture(kb_data, verified_data, comparison_data)
    components = _infer_components(kb_data, verified_data, comparison_data)
    materials = _infer_materials(kb_data, verified_data)
    technologies = _infer_technologies(kb_data, verified_data)

    if architecture:
        out += ["### 4.2 Architecture", ""]
        out += _bullets(architecture)
        out += [""]
    if components:
        out += [
            "### 4.3 Components — what each part is and does",
            "",
            "Each component below was inferred from the knowledge base and the",
            "approved concepts. Together they realise the mechanism of 4.1, and",
            "each one's behaviour is constrained by the equations in section 6.",
            "",
        ]
        for i, comp in enumerate(components, 1):
            out += [f"{i}. **{comp}**"]
        out += [""]
    if materials:
        out += ["### 4.4 Materials", ""]
        out += _bullets(materials)
        out += [""]
    if technologies:
        out += ["### 4.5 Technologies and tooling", ""]
        out += _bullets(technologies)
        out += [""]
    if param_changes:
        out += [
            "### 4.6 Design parameters and why each value was chosen",
            "",
            "The simulation phase tuned these parameters; a selection reason is",
            "recorded for every one.",
            "",
            "| Parameter | Change | Reason |",
            "|---|---|---|",
        ]
        for p in param_changes[:40]:
            name = _safe_get(p, "parameter_name", "?")
            mod = _safe_get(p, "modification_type", "?")
            factor = _safe_get(p, "modification_factor")
            change = f"{mod} ×{factor}" if factor is not None else str(mod)
            reason = _truncate(_safe_get(p, "selection_reason", ""), 120)
            out.append(f"| {name} | {change} | {reason} |")
        out += [""]
    out += ["---", ""]
    return out

def _sec_how_it_works(doscan: List[Dict[str, Any]],
                      kb: Dict[str, Any]) -> List[str]:
    out = [
        "## 5. HOW IT WORKS — CAUSAL CHAINS FROM THE CLUSTERING PHASE",
        "",
        "DOSCAN clustered the knowledge base and generated cause-and-effect",
        "predictions of the form “if X is changed, then Y responds”. These are the",
        "operating principles of the design: they say what happens when a part of",
        "the system is pushed.",
        "",
    ]
    chains: List[str] = []
    for entry in doscan or []:
        for pred in entry.get("predictions", []) or []:
            text = pred.strip() if isinstance(pred, str) else _truncate(pred, 250)
            if text and text not in chains:
                chains.append(text)
    if chains:
        out += _bullets(chains[:25])
        out += [""]
        out.append(
            "Reading guide: each line is a lever. The condition (\"if …\") is the part "
            "of the design you would act on; the consequence (\"then …\") is the "
            "response the lab predicts. Section 7 quantifies those responses."
        )
        out += [""]
    facts = _safe_get(kb, "proven_facts") or []
    if facts:
        out += ["**Physical/logical constraints the chains must respect:**", ""]
        out += _bullets(facts[:12])
        out += [""]
    out += ["---", ""]
    return out


def _sec_mathematics(equations: Dict[str, Any]) -> List[str]:
    validated = _safe_get(equations, "validated_equations") or []
    out = [
        "## 6. THE MATHEMATICS — EVERY EQUATION EXPLAINED",
        "",
        f"The mathematics engine generated and validated {len(validated)}",
        "relationship(s) for this problem. For each one: the formula, what it",
        "means in plain language, and the variables involved.",
        "",
    ]
    if validated:
        for i, eq in enumerate(validated, 1):
            formula = _safe_get(eq, "equation", "?")
            rel = _safe_get(eq, "relationship_type", "relationship")
            hyp = _truncate(_safe_get(eq, "hypothesis", ""), 160)
            variables = _safe_get(eq, "variables") or {}
            hint = _safe_get(eq, "dimension_hint", "")
            reasoning = _truncate(_safe_get(eq, "validation_reasoning", ""), 220)
            out += [
                f"### 6.{i} `{formula}`",
                "",
                f"- **What it says:** the quantity on the left "
                f"{_read_relationship(rel)} as the quantities on the right change.",
            ]
            if hyp:
                out.append(f"- **Which concept it belongs to:** {hyp}")
            if variables:
                glossary = ", ".join(
                    f"`{k}` ({_truncate(v, 80)})" if isinstance(v, str) and v
                    else f"`{k}`"
                    for k, v in list(variables.items())[:10]
                )
                out.append(f"- **Variable glossary:** {glossary}")
            if hint and str(hint).lower() != "standard":
                out.append(f"- **Dimensional hint:** {hint}")
            if reasoning:
                out.append(f"- **Why it passed validation:** {reasoning}")
            out += [""]
        out.append(
            "These equations are the quantitative skeleton of section 4: the design "
            "parameters (4.6) plug into them, and the predicted performance in "
            "section 7 is what the simulation produced when it evaluated them."
        )
        out += [""]
    else:
        out += ["No equations survived validation in this session.", ""]
    out += ["---", ""]
    return out

def _sec_performance(comparison: Dict[str, Any]) -> List[str]:
    metrics = _safe_get(comparison, "metrics") or {}
    summary = _safe_get(comparison, "summary") or {}
    baseline = _safe_get(comparison, "baseline_summary") or {}
    hypothesis = _safe_get(comparison, "hypothesis_summary") or {}
    sim = _safe_get(comparison, "simulation_results") or {}
    out = [
        "## 7. PREDICTED PERFORMANCE VS THE CURRENT BASELINE",
        "",
        "The simulation engine ran the baseline design and the new design under",
        "identical conditions. Every measured metric:",
        "",
    ]
    if metrics:
        out += [
            "| Metric | Baseline | New design | Change | Improvement |",
            "|---|---|---|---|---|",
        ]
        for name, m in metrics.items():
            if not isinstance(m, dict):
                continue
            base = _fmt_metric(m.get("baseline_value", "?"))
            new = _fmt_metric(m.get("hypothesis_value", "?"))
            pct = m.get("percentage_change")
            imp = m.get("improvement_pct")
            change = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else "?"
            improvement = (
                f"{imp:+.2f}%" if isinstance(imp, (int, float))
                else ("yes" if m.get("is_improvement") else "no")
            )
            out.append(f"| {name} | {base} | {new} | {change} | {improvement} |")
        out += [""]
    if summary:
        verdict = _safe_get(summary, "verdict", "")
        avg = _safe_get(summary, "average_improvement_pct")
        line = f"**Verdict:** {verdict or 'n/a'}"
        if isinstance(avg, (int, float)):
            line += (f" — average improvement {avg:.2f}% across "
                     f"{_safe_get(summary, 'metrics_total', '?')} metric(s).")
        out += [line, ""]
    if baseline and hypothesis:
        out += [
            "### 7.1 Side-by-side operating points",
            "",
            "| Metric | Baseline design | New design |",
            "|---|---|---|",
        ]
        for k in baseline:
            if k in hypothesis:
                out.append(
                    f"| {k} | {_fmt_metric(baseline[k])} | {_fmt_metric(hypothesis[k])} |"
                )
        out += [""]
    sim_meta = _safe_get(sim, "metadata") or {}
    if sim_meta:
        engine = _safe_get(sim_meta, "simulation_engine", "simulation engine")
        domain = _safe_get(sim_meta, "domain", "")
        out += [
            f"*Simulation provenance: {engine}"
            + (f" ({domain} domain model)" if domain else "")
            + f", run at {_safe_get(sim_meta, 'timestamp', 'n/a')}.*",
            "",
        ]
    out += ["---", ""]
    return out

def _sec_evidence(evidence: Dict[str, Any]) -> List[str]:
    scores = _safe_get(evidence, "scores") or {}
    out = [
        "## 8. EVIDENCE & CONFIDENCE FOR EVERY CLAIM",
        "",
        "Each scientific claim made during the session was scored by the evidence",
        "engine: literature mining, knowledge-graph support, equation validation",
        "and simulation success are weighted into one confidence number, and the",
        "engine states what must be done next to raise it.",
        "",
    ]
    if scores:
        for i, (claim, s) in enumerate(scores.items(), 1):
            if not isinstance(s, dict):
                continue
            verdict = _safe_get(s, "verdict", "unscored")
            conf = _safe_get(s, "confidence", 0)
            factors = _safe_get(s, "factor_breakdown") or {}
            missing = _safe_get(s, "missing_evidence") or []
            next_action = _safe_get(s, "next_action", "")
            out += [
                f"### 8.{i} Claim {i} — {verdict} ({float(conf or 0):.1%} confidence)",
                "",
                f"> {_truncate(claim, 300)}",
                "",
            ]
            if factors:
                pretty = ", ".join(f"{k} {v:.2f}" for k, v in factors.items()
                                   if isinstance(v, (int, float)))
                out += [f"**Confidence factors:** {pretty}", ""]
            if missing:
                out += ["**Missing evidence (gaps):**", ""]
                out += _bullets(missing)
                out += [""]
            if next_action:
                out += [f"**Recommended next action:** {next_action}", ""]
            out += [""]
    else:
        out += ["No evidence-scoring session data was found.", ""]
    out += ["---", ""]
    return out


def _sec_key_evidence(solution: Dict[str, Any]) -> List[str]:
    key = _safe_get(solution, "key_evidence") or []
    if not key:
        return []
    out = [
        "## 9. KEY EVIDENCE SUPPORTING THE IDEA",
        "",
        "The Socratic reasoning phase recorded the strongest arguments in favour",
        "of the proposed design:",
        "",
    ]
    out += _bullets(key)
    out += ["", "---", ""]
    return out

def _sec_risks(solution: Dict[str, Any], kb: Dict[str, Any],
               verified: List[Dict[str, Any]]) -> List[str]:
    falsify = _safe_get(solution, "falsification_criteria") or []
    failure_modes = _safe_get(kb, "failure_modes_and_risks") or []
    weaknesses = [
        _safe_get(v, "primary_weakness") for v in verified
        if _safe_get(v, "primary_weakness")
    ]
    out = [
        "## 10. HOW IT COULD FAIL — RISKS & FALSIFICATION TESTS",
        "",
        "A design is only trustworthy when its failure conditions are written",
        "down. Three independent sources feed this section: the Socratic phase's",
        "falsification criteria, the knowledge base's known failure modes, and",
        "the weaknesses the hypothesis reviewer flagged.",
        "",
    ]
    if falsify:
        out += ["**The idea would be falsified if:**", ""]
        out += _bullets(falsify)
        out += [""]
    if failure_modes:
        out += ["**Known failure modes and risks in this domain:**", ""]
        out += _bullets(failure_modes)
        out += [""]
    if weaknesses:
        out += ["**Weaknesses admitted by the hypothesis reviewer:**", ""]
        for w in weaknesses:
            out.append(f"- {_truncate(w, 280)}")
        out += [""]
    out += ["---", ""]
    return out


def _sec_next_steps(solution: Dict[str, Any], kb: Dict[str, Any],
                    evidence: Dict[str, Any]) -> List[str]:
    open_q = _safe_get(solution, "open_questions") or []
    unknowns = _safe_get(kb, "critical_unanswered_unknowns") or []
    data_inputs = _safe_get(kb, "required_empirical_data_inputs") or []
    experiment = _safe_get(solution, "proposed_experiment") or ""
    actions = []
    for s in (_safe_get(evidence, "scores") or {}).values():
        na = _safe_get(s, "next_action")
        if na and na not in actions:
            actions.append(na)
    out = [
        "## 11. OPEN QUESTIONS & THE NEXT EXPERIMENTS TO RUN",
        "",
    ]
    if experiment:
        out += ["**The proposed experiment (from the Socratic phase):**", "",
                experiment, ""]
    if open_q:
        out += ["**Open questions about the design:**", ""]
        out += _bullets(open_q)
        out += [""]
    if unknowns:
        out += ["**Unanswered questions the knowledge base flagged:**", ""]
        out += _bullets(unknowns)
        out += [""]
    if data_inputs:
        out += ["**Data that must be collected to close the gaps:**", ""]
        out += _bullets(data_inputs)
        out += [""]
    if actions:
        out += ["**Evidence engine's recommended actions, in priority order:**", ""]
        out += _bullets(actions)
        out += [""]
    out += ["---", ""]
    return out


def _sec_appendix(stats: Dict[str, Any]) -> List[str]:
    provenance = [
        ("final_research_solution.json",
         "executive summary, core mechanism, key evidence, falsification "
         "criteria, open questions, proposed experiment (sections 1, 4.1, 9, 10, 11)"),
        ("deep_research_knowledge_base.json",
         "research thesis, state of the art, proven facts, unknowns, failure "
         "modes, required data (sections 2, 5, 10, 11)"),
        ("verified_hypotheses.json",
         "six-dimension scores, reviewer reasoning, weaknesses (section 3)"),
        ("doscan_breakthroughs.json",
         "clustered causal predictions (section 5)"),
        ("derived_equations.json",
         "validated equations, variable glossary, validation reasoning (section 6)"),
        ("comparison_report.json / parameter_changes.json",
         "baseline vs new-design metrics, parameter tuning reasons (sections 4.6, 7)"),
        ("evidence_scoring_db.json",
         "per-claim confidence, factor breakdown, next actions (section 8)"),
    ]
    out = [
        "## 12. APPENDIX — WHERE EVERY STATEMENT CAME FROM",
        "",
        "This document is generated deterministically from the session's own",
        "artifacts (no LLM is involved in writing it). Statement sources:",
        "",
        "| Session file | Used for |",
        "|---|---|",
    ]
    for fname, use in provenance:
        out.append(f"| `{fname}` | {use} |")
    out += ["", "**Pipeline statistics for this session:**", ""]
    if stats:
        for k, v in stats.items():
            out.append(f"- {k}: {v}")
        out += [""]
    out += ["---", "", "*End of document.*", ""]
    return out

def generate_final_design_explained(
    data_dir: str,
    root_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    user_problem: str = "",
    runtime_str: str = "",
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build FINAL_DESIGN_EXPLAINED.md from the session artifacts in ``data_dir``.

    All loads prefer ``data_dir`` (the single problem output folder) and fall
    back to ``root_dir`` / its ``Research/`` subfolder for legacy layouts.
    Returns the path of the written document.
    """
    root_dir = root_dir or os.path.dirname(data_dir)
    research_dir = os.path.join(root_dir, "Research")
    fallbacks = [root_dir, research_dir]

    solution = _load_from(data_dir, "final_research_solution.json", fallbacks) or {}
    kb = _load_from(data_dir, "deep_research_knowledge_base.json", fallbacks) or {}
    verified = _load_from(data_dir, "verified_hypotheses.json", fallbacks) or []
    equations = _load_from(data_dir, "derived_equations.json", fallbacks) or {}
    evidence = _load_from(data_dir, "evidence_scoring_db.json", fallbacks) or {}
    comparison = _load_from(data_dir, "comparison_report.json", fallbacks) or {}
    params = _load_from(data_dir, "parameter_changes.json", fallbacks) or []
    doscan = _load_from(data_dir, "doscan_breakthroughs.json", fallbacks) or []

    if not isinstance(solution, dict):
        solution = {}
    # The Socratic phase nests the actual solution payload under the
    # "final_research_solution" key — unwrap it when present.
    inner = _safe_get(solution, "final_research_solution")
    if isinstance(inner, dict):
        solution = inner
    if not isinstance(kb, dict):
        kb = {}
    if not isinstance(verified, list):
        verified = []
    if not isinstance(equations, dict):
        equations = {}
    if not isinstance(evidence, dict):
        evidence = {}
    if not isinstance(comparison, dict):
        comparison = {}
    if not isinstance(params, list):
        params = []
    if not isinstance(doscan, list):
        doscan = []
    if not isinstance(stats, dict):
        stats = {}

    domain = _safe_get(kb, "domain", "") or ""
    problem = user_problem or _safe_get(solution, "original_problem", "") \
        or _safe_get(kb, "core_research_thesis", "")

    doc: List[str] = []
    doc += _sec_header(problem, domain, runtime_str, stats)
    doc += _sec_summary(solution)
    doc += _sec_problem(kb, user_problem)
    doc += _sec_new_idea(verified, kb)
    doc += _sec_design(solution, kb, verified, comparison, params)
    doc += _sec_how_it_works(doscan, kb)
    doc += _sec_mathematics(equations)
    doc += _sec_performance(comparison)
    doc += _sec_evidence(evidence)
    doc += _sec_key_evidence(solution)
    doc += _sec_risks(solution, kb, verified)
    doc += _sec_next_steps(solution, kb, evidence)
    doc += _sec_appendix(stats)

    out_dir = output_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "FINAL_DESIGN_EXPLAINED.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc).rstrip() + "\n")
    return path


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    print(generate_final_design_explained(data_dir=os.getcwd(), root_dir=_root))
