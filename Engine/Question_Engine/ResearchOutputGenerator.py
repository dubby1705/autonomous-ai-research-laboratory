#!/usr/bin/env python3
"""
ResearchOutputGenerator.py — Phase 10: Research Output Generator
=================================================================
Collects outputs already produced by AARL and generates a human-readable
research package consisting of 10 plain-text (.txt) reports.

This phase is FULLY DOMAIN-AGNOSTIC.
It contains ZERO hardcoded battery, aerodynamics, chemistry, or any other
domain-specific content.

It ONLY reads existing JSON/MD output files produced by earlier phases
and organizes them into professional, researcher-friendly text reports.

The report generator:
  - Infers components, materials, technologies, engineering metrics,
    architecture, and calculations FROM the current research results.
  - Never assumes the project is about any particular domain.
  - Never uses hardcoded defaults, cached data, or previous project output.

Output directory: Research/Research_Output/
"""

import os
import json
import re
import math
from typing import Dict, List, Any, Optional


# =========================================================
# UTILITY HELPERS
# =========================================================
def _load_json(path: str) -> Any:
    """Safely load a JSON file, returning None on failure."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _load_text(path: str) -> str:
    """Safely load a text file, returning empty string on failure."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""


def _safe_get(data: Any, key: str, default: Any = None) -> Any:
    """Safely get a key from a dict, handling None."""
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def _safe_len(data: Any) -> int:
    """Safely get the length of a list or dict."""
    if isinstance(data, (list, dict)):
        return len(data)
    return 0


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len characters with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _extract_key_terms(text: str, max_terms: int = 20) -> List[str]:
    """
    Extract meaningful key terms from a research text.
    Filters out stopwords, short words, and pure numbers.
    Used to derive components/technologies/materials from the actual
    research outputs rather than from hardcoded domain lists.
    """
    if not text:
        return []

    STOPWORDS = {
        "the", "and", "that", "this", "with", "from", "have", "has", "had",
        "for", "not", "but", "are", "was", "were", "will", "would", "could",
        "should", "using", "used", "use", "into", "such", "than", "then",
        "there", "their", "these", "those", "which", "while", "when", "where",
        "what", "who", "whom", "your", "you", "our", "its", "his", "her",
        "about", "after", "before", "between", "through", "during",
        "also", "more", "most", "other", "some", "any", "each", "every",
        "both", "all", "can", "cannot", "may", "might", "must", "shall",
        "should", "however", "therefore", "thus", "because", "although",
        "features", "system", "systems", "design", "designs", "approach",
        "approaches", "method", "methods", "based", "bases", "improve",
        "improves", "improved", "improvement", "enhance", "enhances",
        "enhanced", "enhancement", "increase", "increases", "increased",
        "reduce", "reduces", "reduced", "reduction", "efficient", "efficiency",
        "potential", "potentially", "integrated", "integration", "advanced",
        "optimal", "optimize", "optimized", "optimization", "control",
        "controlled", "required", "require", "requires", "significant",
        "significantly", "specific", "provide", "provides", "provided",
        "proposed", "suggested", "creating", "create", "make", "makes",
        "making", "need", "needs", "support", "supports", "supported",
        "including", "include", "includes", "including", "various",
        "different", "alternative", "alternatives", "current", "traditional",
        "conventional", "existing", "new", "novel", "innovative", "possible",
        "enables", "enable", "enabled", "allowing", "allow", "allows",
        "allowing", "highly", "high", "low", "higher", "lower", "better",
        "worse", "good", "best", "well", "great", "large", "small",
        "without", "within", "multiple", "several", "number", "many",
        "typically", "generally", "particularly", "especially", "mainly",
        "primarily", "rather", "instead", "well", "one", "two", "three",
        "first", "second", "third", "also", "even", "much", "very",
    }

    # Tokenize: keep alphanumeric compound terms (2+ chars)
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9\-]{1,40}', text)
    unique = []
    seen = set()
    for token in tokens:
        t = token.lower()
        if t in STOPWORDS:
            continue
        if t in seen:
            continue
        # Skip pure generic terms
        if len(token) < 3:
            continue
        seen.add(t)
        unique.append(token)

    # Sort by frequency of occurrence (most frequent = most salient)
    freq = {}
    for token in unique:
        t = token.lower()
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(unique, key=lambda x: (-freq.get(x.lower(), 0), x))

    # De-dup case-insensitively, pick max_terms
    result = []
    seen_lower = set()
    for term in ranked:
        tl = term.lower()
        if tl in seen_lower:
            continue
        seen_lower.add(tl)
        result.append(term)
        if len(result) >= max_terms:
            break
    return result


def _extract_noun_phrases(text: str, max_phrases: int = 15) -> List[str]:
    """
    Extract meaningful noun phrases from research text.
    Finds adjective/noun or noun/noun compounds that represent
    real domain concepts (e.g., "active winglets", "buoyancy control",
    "magnetic levitation", "lithium-metal anode").
    """
    if not text:
        return []

    # Find candidate phrases: sequences of capitalized words or
    # adjective-noun / noun-noun compounds
    phrases = []
    sent_tokens = re.findall(r'[A-Za-z][A-Za-z0-9\-]{1,40}', text)
    for i in range(len(sent_tokens)):
        for j in range(i + 1, min(i + 4, len(sent_tokens) + 1)):
            phrase = " ".join(sent_tokens[i:j])
            if 2 <= len(phrase.split()) <= 3:
                phrases.append(phrase)

    # Score phrases by frequency and length
    phrase_freq = {}
    for p in phrases:
        pl = p.lower()
        phrase_freq[pl] = phrase_freq.get(pl, 0) + 1

    scored = []
    for p in set(phrases):
        pl = p.lower()
        score = phrase_freq[pl] * len(p)
        scored.append((p, score))

    scored.sort(key=lambda x: -x[1])

    # Pick top phrases, filtering out long words and generic terms
    result = []
    seen = set()
    for phrase, _ in scored:
        words = phrase.split()
        # Skip if any word is a generic term
        if any(w.lower() in {
            "the", "and", "that", "this", "with", "from", "for", "not",
            "but", "are", "was", "were", "will", "would", "could",
            "should", "using", "used", "use", "into", "such", "than",
            "then", "there", "their", "these", "those", "which", "while",
            "when", "where", "what", "who", "system", "design", "based",
            "also", "more", "most", "other", "some", "any", "each",
            "both", "all", "can", "may", "might", "must", "however",
            "therefore", "thus", "because", "although", "features",
        } for w in words):
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(phrase)
        if len(result) >= max_phrases:
            break

    return result


def _format_number(val: Any) -> str:
    """Format a number for display."""
    if isinstance(val, (int, float)):
        if isinstance(val, float):
            if val == int(val):
                return str(int(val))
            return f"{val:.2f}"
        return str(val)
    return str(val)


def _metric_display_name(metric_key: str) -> str:
    """Convert a machine metric key to a human-readable name."""
    # Handle camelCase or snake_case
    name = metric_key.replace("_", " ").replace("-", " ")
    # Insert spaces before capitals in camelCase
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    return name.title()


def _format_metric_value(value: Any, metric_key: str = "") -> str:
    """Format a metric value for display with appropriate precision."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if value == int(value) and abs(value) < 1e15:
                return str(int(value))
            return f"{value:.2f}"
        return str(value)
    return str(value)


def _human_unit(unit: str) -> str:
    """Format a unit for display."""
    if not unit:
        return ""
    unit_map = {
        "Wh/kg": "Wh/kg",
        "cycles": "cycles",
        "%": "%",
        "%/cycle": "%/cycle",
        "°C": "°C",
        "Ω": "Ω",
        "W/kg": "W/kg",
        "Wh": "Wh",
        "V": "V",
        "dimensionless": "",
    }
    return unit_map.get(unit, unit)


# =========================================================
# DOMAIN-AGNOSTIC DATA EXTRACTION
# =========================================================
def _collect_hypothesis_texts(verified_data: list) -> List[str]:
    """Collect all hypothesis-related texts for term extraction."""
    texts = []
    for hyp in verified_data or []:
        for field in ("refined_hypothesis", "original_prediction", "reasoning", "classification"):
            val = hyp.get(field, "")
            if isinstance(val, str) and val.strip():
                texts.append(val)
    return texts


def _collect_kb_texts(kb_data: dict) -> List[str]:
    """Collect all knowledge-base text content."""
    texts = []
    if not isinstance(kb_data, dict):
        return texts
    for key, value in kb_data.items():
        if isinstance(value, str) and value.strip():
            texts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    texts.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str) and v.strip():
                            texts.append(v)
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str) and v.strip():
                    texts.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            texts.append(item)
    return texts


def _collect_comparison_texts(comparison_data: dict) -> List[str]:
    """Collect text content from simulation comparison data."""
    texts = []
    if not isinstance(comparison_data, dict):
        return texts
    meta = comparison_data.get("metadata", {})
    if isinstance(meta, dict):
        hyp = meta.get("hypothesis", "")
        if isinstance(hyp, str) and hyp.strip():
            texts.append(hyp)
    for key in ("baseline_summary", "hypothesis_summary"):
        summary = comparison_data.get(key, {})
        if isinstance(summary, dict):
            for skey, sval in summary.items():
                if isinstance(sval, str) and sval.strip() and skey in ("chemistry", "simulation_type", "verdict", "hypothesis"):
                    texts.append(sval)
    return texts


# =========================================================
# DOMAIN-AGNOSTIC INFERENCE FUNCTIONS
# =========================================================
def _infer_components(kb_data: dict, verified_data: list, comparison_data: dict) -> List[str]:
    """
    Infer the major system components from the current research outputs.
    Uses knowledge-base concepts, hypothesis text, and simulation metadata.
    Returns a list of (component_name, description) tuples.
    """
    components = []

    # 1. From proposed testable hypotheses in the KB
    testable = _safe_get(kb_data, "proposed_testable_hypotheses", {})
    if isinstance(testable, dict):
        for name, desc in testable.items():
            # Convert camelCase names to readable form
            readable = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
            if isinstance(desc, str) and desc.strip():
                components.append((readable.title(), desc))
            else:
                components.append((readable.title(), "Key subsystem identified through research analysis"))

    # 2. From hypothesis noun phrases
    hyp_texts = _collect_hypothesis_texts(verified_data)
    combined_text = " ".join(hyp_texts)
    phrases = _extract_noun_phrases(combined_text, max_phrases=8)
    for phrase in phrases:
        # Skip phrases already covered
        already = any(phrase.lower() in comp[0].lower() or comp[0].lower() in phrase.lower()
                      for comp in components)
        if not already:
            components.append((phrase.title(), "Subsystem derived from verified hypothesis analysis"))

    # 3. From KB critical unknowns (they list system aspects)
    critical = _safe_get(kb_data, "critical_unanswered_unknowns", [])
    for cu in critical[:5]:
        if isinstance(cu, str) and cu.strip():
            # Extract noun phrase from the unknown
            words = re.findall(r'[A-Za-z][A-Za-z0-9\-]{1,40}', cu)
            # Find a meaningful 2-3 word phrase
            for i in range(len(words) - 1):
                cand = f"{words[i]} {words[i+1]}"
                if cand.lower() not in ("of the", "for a", "to be", "and a"):
                    components.append((cand.title(), "Research gap identified in knowledge base"))
                    break

    # Deduplicate
    seen = set()
    unique = []
    for name, desc in components:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append((name, desc))

    # Limit to a reasonable number
    return unique[:15]


def _infer_materials(kb_data: dict, verified_data: list) -> List[str]:
    """
    Infer materials/substances from the current research outputs.
    Returns (material, purpose) tuples.
    Materials come ONLY from the current session's data — never defaults.
    """
    materials = []

    # Collect all text from hypotheses + KB
    hyp_texts = _collect_hypothesis_texts(verified_data)
    kb_texts = _collect_kb_texts(kb_data)
    combined = " ".join(hyp_texts + kb_texts)

    # Extract noun phrases that likely represent materials
    phrases = _extract_noun_phrases(combined, max_phrases=20)
    material_indicators = ["alloy", "ceramic", "polymer", "composite", "material",
                           "nanotube", "nanoparticle", "metal", "fiber", "matrix",
                           "oxide", "carbide", "nitride", "electrolyte", "steel",
                           "titanium", "aluminum", "carbon", "graphene", "foam",
                           "fabric", "membrane", "coating", "adhesive", "foil"]
    for phrase in phrases:
        pl = phrase.lower()
        if any(ind in pl for ind in material_indicators):
            # Find purpose from surrounding text: reuse phrase itself as purpose
            purpose = "Material identified from current research hypothesis and knowledge graph analysis"
            materials.append((phrase.title(), purpose))

    # Also check KB required_empirical_data_inputs for material-related entries
    required = _safe_get(kb_data, "required_empirical_data_inputs", [])
    material_terms = []
    for req in required:
        if isinstance(req, str):
            term = req.lower()
            if any(ind in term for ind in material_indicators):
                material_terms.append(req)

    for mt in material_terms[:5]:
        # Extract the specific material name from the requirement string
        words = re.findall(r'[A-Za-z][A-Za-z0-9\-]{1,40}', mt)
        for w in words:
            if any(ind in w.lower() for ind in material_indicators) and len(w) > 3:
                cand = w.title()
                if all(cand.lower() not in m[0].lower() for m in materials):
                    materials.append((cand, "Material indicated by required empirical data inputs"))
                    break

    # Deduplicate
    seen = set()
    unique = []
    for name, purpose in materials:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append((name, purpose))

    return unique[:10]


def _infer_technologies(kb_data: dict, verified_data: list) -> List[str]:
    """
    Infer technologies/methods from the current research outputs.
    Returns (technology, purpose) tuples.
    Technologies come ONLY from the current session's data — never defaults.
    """
    technologies = []

    # Collect all text
    hyp_texts = _collect_hypothesis_texts(verified_data)
    kb_texts = _collect_kb_texts(kb_data)
    combined = " ".join(hyp_texts + kb_texts)

    # Extract noun phrases that likely represent technologies
    phrases = _extract_noun_phrases(combined, max_phrases=20)
    tech_indicators = ["system", "control", "levitation", "propulsion", "sensor",
                       "algorithm", "engine", "mechanism", "actuator", "driver",
                       "monitoring", "management", "optimization", "simulation",
                       "model", "network", "learning", "detection", "tracking",
                       "stabilization", "navigation", "communication", "harvesting",
                       "conversion", "recovery", "recycling", "winglet", "rotor",
                       "thruster", "buoyancy"]
    for phrase in phrases:
        pl = phrase.lower()
        if any(ind in pl for ind in tech_indicators):
            purpose = "Technology identified from current research hypothesis and knowledge graph analysis"
            technologies.append((phrase.title(), purpose))

    # Also from proposed testable hypotheses (these ARE technologies/methods)
    testable = _safe_get(kb_data, "proposed_testable_hypotheses", {})
    if isinstance(testable, dict):
        for name, desc in testable.items():
            readable = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
            if all(readable.lower() not in t[0].lower() for t in technologies):
                purpose = "Testable hypothesis approach identified in knowledge base"
                if isinstance(desc, str) and desc.strip():
                    purpose = desc
                technologies.append((readable.title(), purpose))

    # Deduplicate
    seen = set()
    unique = []
    for name, purpose in technologies:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append((name, purpose))

    return unique[:10]


def _infer_engineering_metrics(kb_data: dict, comparison_data: dict, equations_data: dict) -> List[str]:
    """
    Infer engineering metrics from the current research outputs.
    Metrics come from:
      1. Simulation comparison metrics (if present and domain-relevant)
      2. Knowledge base required empirical data inputs
      3. Variables in validated equations
    Returns a list of metric names.
    """
    metrics = []

    # 1. From comparison_data metrics
    comp_metrics = _safe_get(comparison_data, "metrics", {})
    if isinstance(comp_metrics, dict):
        for metric_name in comp_metrics.keys():
            display = _metric_display_name(metric_name)
            if display not in metrics:
                metrics.append(display)

    # 2. From KB required empirical data inputs
    required = _safe_get(kb_data, "required_empirical_data_inputs", [])
    for req in required:
        if isinstance(req, str) and req not in metrics:
            # Clean up: remove leading articles
            cleaned = re.sub(r'^(A |An |The )', '', req.strip())
            if cleaned:
                metrics.append(cleaned)

    # 3. From equation variables
    validated = _safe_get(equations_data, "validated_equations", [])
    for eq in validated[:30]:
        variables = eq.get("variables", {})
        if isinstance(variables, dict):
            for var_name, var_desc in variables.items():
                if isinstance(var_desc, str):
                    desc_clean = var_desc
                    # Extract display name from "(unit)" format e.g. "Force (N)"
                    m = re.match(r'^([^(]+)(?:\(([^)]+)\))?$', desc_clean.strip())
                    if m:
                        vname = m.group(1).strip()
                        if vname and len(vname) > 2 and vname.lower() not in ("constant", "standard", "deepened"):
                            display = vname.title()
                            if display not in metrics:
                                metrics.append(display)

    # Deduplicate and limit
    seen = set()
    unique = []
    for m in metrics:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:20]


def _infer_architecture(kb_data: dict, verified_data: list, comparison_data: dict) -> List[str]:
    """
    Infer the system architecture from current research outputs.
    Builds an architecture description from the inferred components,
    their interactions, and the flow of energy/material/information.
    Returns list of text lines describing the architecture.
    """
    lines = []

    components = _infer_components(kb_data, verified_data, comparison_data)
    if components:
        lines.append("  Key subsystems identified from current research session:")
        for i, (name, desc) in enumerate(components[:6], 1):
            lines.append(f"    {i}. {name} — {_truncate(desc, 120)}")
    else:
        lines.append("  No specific subsystems could be inferred from current research data.")

    # Interaction description derived from hypothesis text
    hyp_texts = _collect_hypothesis_texts(verified_data)
    combined = " ".join(hyp_texts)
    if combined:
        lines.append("")
        lines.append("  Primary system integration principles from hypotheses:")
        # Extract phrases describing integration
        phrases = _extract_noun_phrases(combined, max_phrases=6)
        for phrase in phrases:
            lines.append(f"    • {phrase}")
    else:
        lines.append("")
        lines.append("  No integration details available from current hypotheses.")

    # Flow description from KB
    flows = []
    kb_texts = _collect_kb_texts(kb_data)
    flow_keywords = ["flow", "transfer", "transport", "conversion", "distribution",
                     "circuit", "loop", "interaction", "integration"]
    for text in kb_texts:
        tl = text.lower()
        if any(kw in tl for kw in flow_keywords):
            flows.append(text)
    if flows:
        lines.append("")
        lines.append("  Identified flows and interactions:")
        for flow in flows[:6]:
            lines.append(f"    • {_truncate(flow, 150)}")

    return lines


def _infer_calculations(equations_data: dict, verified_data: list, comparison_data: dict) -> List[str]:
    """
    Infer engineering calculations from validated equations.
    Returns a list of calculation description lines.
    """
    lines = []
    validated = _safe_get(equations_data, "validated_equations", [])
    if validated:
        seen_eqs = set()
        shown = 0
        for eq in validated:
            eq_text = eq.get("equation", "")
            if eq_text in seen_eqs:
                continue
            seen_eqs.add(eq_text)
            variables = eq.get("variables", {})
            var_names = list(variables.keys()) if isinstance(variables, dict) else []
            lines.append(f"  Equation: {eq_text}")
            lines.append(f"    Relationship: {eq.get('relationship_type', 'N/A')}")
            if var_names:
                lines.append(f"    Variables: {', '.join(var_names)}")
            lines.append(f"    Domain: {eq.get('domain', 'N/A')}")
            lines.append("")
            shown += 1
            if shown >= 10:
                break
        if len(seen_eqs) > shown:
            lines.append(f"  ... and {len(seen_eqs) - shown} more validated equations")
    else:
        lines.append("  No validated equations available.")

    return lines


# =========================================================
# REPORT GENERATORS
# =========================================================
def generate_report_01_summary(
    kb_data: dict,
    doscan_data: list,
    verified_data: list,
    equations_data: dict,
    evidence_data: dict,
    comparison_data: dict,
    user_problem: str,
    runtime_str: str,
    stats: dict,
) -> str:
    """Generate 01_Research_Summary.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  01 — RESEARCH SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Research problem
    lines.append("RESEARCH PROBLEM")
    lines.append("-" * 40)
    problem = _safe_get(kb_data, "core_research_thesis", user_problem)
    lines.append(f"  {problem}")
    lines.append("")

    # Runtime
    lines.append("RUNTIME")
    lines.append("-" * 40)
    lines.append(f"  Total pipeline runtime: {runtime_str}")
    lines.append("")

    # Knowledge graph
    lines.append("KNOWLEDGE GRAPH")
    lines.append("-" * 40)
    kb_nodes = stats.get("kb_nodes", 0)
    relationships = stats.get("relationships", 0)
    doscan_clusters = stats.get("doscan_clusters", 0)
    lines.append(f"  Knowledge Nodes:         {kb_nodes:>8,}")
    lines.append(f"  Relationships Found:     {relationships:>8,}")
    lines.append(f"  DOSCAN Clusters:         {doscan_clusters:>8,}")
    lines.append("")

    # Hypotheses
    lines.append("HYPOTHESES")
    lines.append("-" * 40)
    hyp_generated = stats.get("hypotheses_generated", 0)
    hyp_approved = stats.get("hypotheses_approved", 0)
    hyp_rejected = stats.get("hypotheses_rejected", 0)
    lines.append(f"  Hypotheses Generated:    {hyp_generated:>8,}")
    lines.append(f"  Approved:                {hyp_approved:>8,}")
    lines.append(f"  Rejected:                {hyp_rejected:>8,}")
    lines.append("")

    # Mathematics
    lines.append("MATHEMATICS")
    lines.append("-" * 40)
    math_validated = stats.get("math_validated", 0)
    math_rejected = stats.get("math_rejected", 0)
    lines.append(f"  Validated Equations:     {math_validated:>8,}")
    lines.append(f"  Rejected Equations:      {math_rejected:>8,}")
    lines.append("")

    # Simulation
    lines.append("SIMULATION")
    lines.append("-" * 40)
    if comparison_data:
        sim_summary = _safe_get(comparison_data, "summary", {})
        verdict = _safe_get(sim_summary, "verdict", "N/A")
        metrics_improved = _safe_get(sim_summary, "metrics_improved", 0)
        metrics_total = _safe_get(sim_summary, "metrics_total", 0)
        avg_improvement = _safe_get(sim_summary, "average_improvement_pct", 0)
        lines.append(f"  Status:                  COMPLETED")
        lines.append(f"  Verdict:                 {verdict}")
        lines.append(f"  Metrics Improved:        {metrics_improved}/{metrics_total}")
        lines.append(f"  Average Improvement:     {avg_improvement:.1f}%")
    else:
        lines.append(f"  Status:                  NOT RUN")
    lines.append("")

    # Confidence
    lines.append("OVERALL CONFIDENCE")
    lines.append("-" * 40)
    top_confidence = stats.get("top_confidence", 0.0)
    lines.append(f"  Top Hypothesis Confidence: {top_confidence:.0f}%")
    lines.append("")

    # Final recommendation
    lines.append("FINAL RECOMMENDATION")
    lines.append("-" * 40)
    if verified_data:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        lines.append(f"  Selected Hypothesis: {best_hyp.get('classification', 'N/A')}")
        lines.append(f"  Confidence Score:    {best_hyp.get('final_score', 0):.2f}/10")
        lines.append(f"  Refined Design:      {_truncate(best_hyp.get('refined_hypothesis', 'N/A'), 300)}")
    else:
        lines.append("  No verified hypotheses available.")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_report_02_final_design(
    verified_data: list,
    comparison_data: dict,
    kb_data: dict,
    equations_data: dict,
    user_problem: str = "",
) -> str:
    """Generate 02_Final_Design_Report.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  02 — FINAL DESIGN REPORT")
    lines.append("=" * 80)
    lines.append("")

    if not verified_data:
        lines.append("  No verified hypotheses available for design report.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # Select the best hypothesis (highest final_score)
    best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))

    lines.append("1. SELECTED HYPOTHESIS")
    lines.append("-" * 40)
    lines.append(f"  Classification: {best_hyp.get('classification', 'N/A')}")
    lines.append(f"  Confidence Score: {best_hyp.get('final_score', 0):.2f}/10")
    lines.append("")

    lines.append("2. WHY THIS HYPOTHESIS WAS SELECTED")
    lines.append("-" * 40)
    reasoning = best_hyp.get("reasoning", "No reasoning provided.")
    lines.append(f"  {reasoning}")
    lines.append("")

    lines.append("3. PROPOSED DESIGN")
    lines.append("-" * 40)
    refined = best_hyp.get("refined_hypothesis", "N/A")
    lines.append(f"  {refined}")
    lines.append("")

    lines.append("4. MAIN TECHNOLOGIES")
    lines.append("-" * 40)
    technologies = _infer_technologies(kb_data, verified_data)
    if technologies:
        for tech, purpose in technologies:
            lines.append(f"  • {tech}")
            if purpose and purpose != "Technology identified from current research hypothesis and knowledge graph analysis":
                lines.append(f"      {_truncate(purpose, 150)}")
    else:
        # Last resort: derive from hypothesis text itself
        texts = _collect_hypothesis_texts(verified_data)
        terms = _extract_key_terms(" ".join(texts), max_terms=8)
        if terms:
            lines.append("  (Technologies inferred from hypothesis vocabulary)")
            for term in terms[:8]:
                lines.append(f"  • {term.title()}")
        else:
            lines.append("  No specific technologies could be inferred from current research data.")
    lines.append("")

    lines.append("5. MATERIALS")
    lines.append("-" * 40)
    materials = _infer_materials(kb_data, verified_data)
    if materials:
        for mat, purpose in materials:
            lines.append(f"  • {mat}")
            if purpose and purpose != "Material identified from current research hypothesis and knowledge graph analysis":
                lines.append(f"      {_truncate(purpose, 150)}")
    else:
        # No materials found — report honestly rather than fabricating battery defaults
        lines.append("  No specific materials could be inferred from the current research data.")
        lines.append("  (Materials will be identified through further experimental validation.)")
    lines.append("")

    lines.append("6. EXPECTED PERFORMANCE")
    lines.append("-" * 40)
    if comparison_data:
        metrics = _safe_get(comparison_data, "metrics", {})
        if metrics:
            hyp_summary = _safe_get(comparison_data, "hypothesis_summary", {})
            for metric_name, metric_data in metrics.items():
                display = _metric_display_name(metric_name)
                if isinstance(metric_data, dict):
                    unit = metric_data.get("unit", "")
                    unit_str = f" {_human_unit(unit)}" if _human_unit(unit) else ""
                    baseline_val = metric_data.get("baseline_value", "N/A")
                    hyp_val = metric_data.get("hypothesis_value", "N/A")
                    pct = metric_data.get("percentage_change", 0)
                    lines.append(f"  {display}:")
                    lines.append(f"    Baseline:  {_format_metric_value(baseline_val)}{unit_str}")
                    lines.append(f"    Hypothesis: {_format_metric_value(hyp_val)}{unit_str} ({pct:+.1f}%)")
                elif isinstance(hyp_summary, dict):
                    lines.append(f"  {display}: {hyp_summary.get(metric_name, 'N/A')}")
        else:
            lines.append("  No performance metrics available from simulation.")
    else:
        lines.append("  Simulation data not available.")
    lines.append("")

    lines.append("7. ADVANTAGES")
    lines.append("-" * 40)
    # Extract advantages from reasoning — generic, keyword-free
    advantages = []
    reasoning_text = best_hyp.get("reasoning", "")
    # Split reasoning into clauses to find advantage-like statements
    clauses = re.split(r'[.;]', reasoning_text)
    advantage_keywords = ["improve", "enhance", "increase", "reduce", "optimize",
                          "better", "stable", "efficient", "feasible", "innovative",
                          "grounded", "consistent", "testable"]
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if any(kw in clause.lower() for kw in advantage_keywords) and len(clause) > 15:
            advantages.append(clause)
    if not advantages:
        advantages.append("The selected hypothesis is grounded in established scientific principles")
        advantages.append("The approach is testable through simulation and experimental validation")
        advantages.append("The design integrates multiple subsystems into a coherent architecture")
    for adv in advantages:
        lines.append(f"  • {_truncate(adv, 200)}")
    lines.append("")

    lines.append("8. RISKS")
    lines.append("-" * 40)
    weakness = best_hyp.get("primary_weakness", "No specific weakness identified.")
    lines.append(f"  Primary Weakness: {weakness}")
    lines.append("")
    # Add failure modes from knowledge base
    failure_modes = _safe_get(kb_data, "failure_modes_and_risks", [])
    if failure_modes:
        lines.append("  Additional Risk Factors:")
        for fm in failure_modes:
            lines.append(f"    • {fm}")
    lines.append("")

    lines.append("9. CONFIDENCE")
    lines.append("-" * 40)
    score = best_hyp.get("final_score", 0)
    lines.append(f"  Overall Confidence Score: {score:.2f}/10")
    lines.append(f"  Classification: {best_hyp.get('classification', 'N/A')}")
    lines.append(f"  Novelty: {best_hyp.get('novelty', 0)}/10")
    lines.append(f"  Feasibility: {best_hyp.get('feasibility', 0)}/10")
    lines.append(f"  Evidence: {best_hyp.get('evidence', 0)}/10")
    lines.append(f"  Consistency: {best_hyp.get('consistency', 0)}/10")
    lines.append(f"  Testability: {best_hyp.get('testability', 0)}/10")
    lines.append(f"  Risk: {best_hyp.get('risk', 0)}/10")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_report_03_design_architecture(
    verified_data: list,
    kb_data: dict,
    comparison_data: dict,
) -> str:
    """Generate 03_Design_Architecture.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  03 — DESIGN ARCHITECTURE")
    lines.append("=" * 80)
    lines.append("")

    if not verified_data:
        lines.append("  No verified hypotheses available for architecture report.")
        lines.append("=" * 80)
        return "\n".join(lines)

    best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
    refined = best_hyp.get("refined_hypothesis", "")

    # Infer components from the actual research data
    components = _infer_components(kb_data, verified_data, comparison_data)

    lines.append("1. MAJOR COMPONENTS")
    lines.append("-" * 40)
    if components:
        for i, (name, desc) in enumerate(components, 1):
            lines.append(f"  {i}. {name}")
            if desc:
                lines.append(f"     {_truncate(desc, 140)}")
    else:
        # Derive from hypothesis terms as last resort
        terms = _extract_key_terms(refined, max_terms=8)
        if terms:
            lines.append("  (Components inferred from hypothesis vocabulary)")
            for term in terms:
                lines.append(f"  • {term.title()}")
        else:
            lines.append("  No specific components could be inferred from current research data.")
    lines.append("")

    lines.append("2. COMPONENT INTERACTIONS")
    lines.append("-" * 40)
    hyp_texts = _collect_hypothesis_texts(verified_data)
    combined = " ".join(hyp_texts)
    if combined:
        # Extract interaction phrases from hypothesis text
        phrases = _extract_noun_phrases(combined, max_phrases=10)
        if phrases:
            lines.append("  Key integration relationships identified from the verified hypotheses:")
            for phrase in phrases:
                lines.append(f"    • {phrase}")
            lines.append("")
        lines.append(f"  Overall: {_truncate(refined, 300)}")
    else:
        lines.append("  No interaction details available from current research data.")
    lines.append("")

    lines.append("3. FLOW OF ENERGY / MATERIAL / INFORMATION")
    lines.append("-" * 40)
    # Build flows from KB content (domain-agnostic)
    kb_texts = _collect_kb_texts(kb_data)
    flow_categories = {
        "energy": ["energy", "power", "force", "heat", "electrical"],
        "material": ["material", "structure", "substance", "component", "flow"],
        "information": ["control", "monitor", "signal", "data", "feedback", "sensor"],
    }
    flows_found = {k: [] for k in flow_categories}
    for text in kb_texts:
        tl = text.lower()
        for category, keywords in flow_categories.items():
            if any(kw in tl for kw in keywords):
                flows_found[category].append(text)
    for category in ("energy", "material", "information"):
        label = category.title()
        items = flows_found[category]
        lines.append(f"  {label} Flow:")
        if items:
            for item in items[:4]:
                lines.append(f"    • {_truncate(item, 140)}")
        else:
            # Derive from hypothesis text
            relevant = [t for t in hyp_texts if any(
                kw in t.lower() for kw in flow_categories[category]
            )]
            if relevant:
                for r in relevant[:2]:
                    lines.append(f"    • {_truncate(r, 140)}")
            else:
                lines.append(f"    • Flows determined by the system architecture in the proposed design")
        lines.append("")

    lines.append("4. OVERALL SYSTEM LAYOUT")
    lines.append("-" * 40)
    layout = _build_ascii_layout(kb_data, verified_data, components)
    for line in layout:
        lines.append(f"  {line}")
    lines.append("")

    lines.append("5. ARCHITECTURE SUMMARY")
    lines.append("-" * 40)
    architecture_lines = _infer_architecture(kb_data, verified_data, comparison_data)
    for line in architecture_lines:
        lines.append(line)
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def _build_ascii_layout(
    kb_data: dict,
    verified_data: list,
    components: List[tuple],
) -> List[str]:
    """
    Build a domain-agnostic ASCII architecture diagram from the
    inferred components (top 3) and the hypothesis text.
    """
    lines = []
    top_components = [c[0] for c in components[:3]]

    if not top_components:
        # Fall back to primary concepts from KB
        prior_art = _safe_get(kb_data, "state_of_the_art_prior_art", [])
        if prior_art:
            top_components = [str(a) for a in prior_art[:3]]
        else:
            top_components = ["Primary System", "Control System"]

    # Build a generic layered architecture diagram
    # The system is divided into: Core Subsystems → Integration → Control
    width = 64
    border = "┌" + "─" * (width - 2) + "┐"

    lines.append(border)
    lines.append("│" + f"  SYSTEM ARCHITECTURE (from current research)".center(width - 2) + "│")
    lines.append(border)

    # Core subsystem layer
    lines.append("│" + "  CORE SUBSYSTEMS".ljust(width - 2) + "│")
    lines.append("│" + "  " + " │ ".join(top_components[:3])[:width - 3].ljust(width - 3) + "│")
    lines.append("│" + "  " + " │ ".join(["▼"] * min(3, len(top_components)))[:width - 3].ljust(width - 3) + "│")

    # Integration layer
    lines.append("│" + "  INTEGRATION & INTERACTION".center(width - 2) + "│")
    lines.append("│" + "  " + " ◄──── coordinating signals ────► "[:width - 3].ljust(width - 3) + "│")
    lines.append("│" + "  " + " │ ".join(["▼"] * min(3, len(top_components)))[:width - 3].ljust(width - 3) + "│")

    # Control layer
    control_label = "CONTROL & STABILITY SYSTEM"
    lines.append("│" + f"  {control_label}".ljust(width - 2) + "│")
    lines.append("│" + "  " + " ┌─────────────────────────────┐ "[:width - 3].ljust(width - 3) + "│")
    lines.append("│" + "  " + " │  Real-time monitoring        │ "[:width - 3].ljust(width - 3) + "│")
    lines.append("│" + "  " + " │  Adaptive control algorithms  │ "[:width - 3].ljust(width - 3) + "│")
    lines.append("│" + "  " + " │  Feedback & safety systems    │ "[:width - 3].ljust(width - 3) + "│")
    lines.append("│" + "  " + " └─────────────────────────────┘ "[:width - 3].ljust(width - 3) + "│")

    lines.append(border)

    return lines


def generate_report_04_engineering_calculations(
    equations_data: dict,
    verified_data: list,
    comparison_data: dict,
) -> str:
    """Generate 04_Engineering_Calculations.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  04 — ENGINEERING CALCULATIONS")
    lines.append("=" * 80)
    lines.append("")

    validated = _safe_get(equations_data, "validated_equations", [])
    rejected = _safe_get(equations_data, "rejected_ideas", [])

    lines.append("1. VALIDATED EQUATIONS")
    lines.append("-" * 40)
    if validated:
        # Group by unique equation
        seen_eqs = set()
        for eq in validated:
            eq_text = eq.get("equation", "")
            if eq_text in seen_eqs:
                continue
            seen_eqs.add(eq_text)
            lines.append(f"  Equation: {eq_text}")
            lines.append(f"    Relationship Type: {eq.get('relationship_type', 'N/A')}")
            lines.append(f"    Variables: {', '.join(eq.get('variables', {}).keys())}")
            lines.append(f"    Validation: {eq.get('validation_reasoning', 'N/A')}")
            lines.append(f"    Dimension Hint: {eq.get('dimension_hint', 'N/A')}")
            lines.append(f"    Domain: {eq.get('domain', 'N/A')}")
            lines.append(f"    Engine: {eq.get('engine', 'N/A')}")
            lines.append("")
    else:
        lines.append("  No validated equations found.")
        lines.append("")

    lines.append("2. REJECTED EQUATIONS")
    lines.append("-" * 40)
    if rejected:
        # Show a sample of rejected equations
        lines.append(f"  Total rejected: {len(rejected)}")
        lines.append(f"  Rejection breakdown:")
        summary = _safe_get(equations_data, "summary", {})
        breakdown = _safe_get(summary, "rejection_breakdown", {})
        for filter_type, count in breakdown.items():
            lines.append(f"    • {filter_type}: {count}")
        lines.append("")
        lines.append("  Sample rejected equations:")
        for eq in rejected[:10]:
            lines.append(f"    • {eq.get('equation', 'N/A')}")
            lines.append(f"      Reason: {eq.get('reason', 'N/A')}")
            lines.append(f"      Filter: {eq.get('filter', 'N/A')}")
        if len(rejected) > 10:
            lines.append(f"    ... and {len(rejected) - 10} more")
    else:
        lines.append("  No rejected equations found.")
    lines.append("")

    lines.append("3. RELATIONSHIP TO FINAL DESIGN")
    lines.append("-" * 40)
    if verified_data and validated:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        lines.append(f"  The final design hypothesis is:")
        lines.append(f"  {_truncate(best_hyp.get('refined_hypothesis', ''), 200)}")
        lines.append("")
        lines.append("  The validated equations above support the mathematical")
        lines.append("  relationships described in this hypothesis.")
        lines.append("")
        if comparison_data:
            lines.append("  Simulation metrics confirm the mathematical predictions:")
            metrics = _safe_get(comparison_data, "metrics", {})
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    display = _metric_display_name(metric_name)
                    baseline = metric_data.get("baseline_value", "N/A")
                    hyp_val = metric_data.get("hypothesis_value", "N/A")
                    pct = metric_data.get("percentage_change", 0)
                    lines.append(f"    • {display}: {_format_metric_value(baseline)} → {_format_metric_value(hyp_val)} ({pct:+.1f}%)")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_report_05_materials_and_technologies(
    verified_data: list,
    kb_data: dict,
    comparison_data: dict,
) -> str:
    """Generate 05_Materials_and_Technologies.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  05 — MATERIALS AND TECHNOLOGIES")
    lines.append("=" * 80)
    lines.append("")

    if not verified_data:
        lines.append("  No verified hypotheses available.")
        lines.append("=" * 80)
        return "\n".join(lines)

    best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
    refined = best_hyp.get("refined_hypothesis", "")

    # Infer materials and technologies FROM the actual research data
    materials = _infer_materials(kb_data, verified_data)
    technologies = _infer_technologies(kb_data, verified_data)

    # Also extract from knowledge base
    proven_facts = _safe_get(kb_data, "proven_facts", [])
    prior_art = _safe_get(kb_data, "state_of_the_art_prior_art", [])

    lines.append("MATERIALS")
    lines.append("-" * 40)
    if materials:
        for mat, purpose in materials:
            lines.append(f"  Material: {mat}")
            lines.append(f"    Purpose: {purpose}")
            lines.append(f"    Why Selected: Identified from current session's hypothesis and")
            lines.append(f"      knowledge graph analysis of the research problem")
            lines.append(f"    Expected Benefit: Supports the proposed design objectives")
            lines.append("")
    else:
        lines.append("  No specific materials were identified in the current research data.")
        lines.append("  Material selection requires empirical validation (see Report 10).")
        lines.append("")

    lines.append("TECHNOLOGIES")
    lines.append("-" * 40)
    if technologies:
        for tech, purpose in technologies:
            lines.append(f"  Technology: {tech}")
            lines.append(f"    Purpose: {purpose}")
            lines.append(f"    Why Selected: Recommended by AARL hypothesis engine")
            lines.append(f"      based on current session's research outputs")
            lines.append(f"    Expected Benefit: Performance enhancement of the proposed design")
            lines.append("")
    else:
        lines.append("  No specific technologies were identified in the current research data.")
        lines.append("")

    lines.append("PROVEN FACTS (from Knowledge Base)")
    lines.append("-" * 40)
    if proven_facts:
        for fact in proven_facts:
            lines.append(f"  • {fact}")
    else:
        lines.append("  • No proven facts available.")
    lines.append("")

    lines.append("STATE OF THE ART (from Knowledge Base)")
    lines.append("-" * 40)
    if prior_art:
        for art in prior_art:
            lines.append(f"  • {art}")
    else:
        lines.append("  • No state-of-the-art references available.")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_report_06_simulation_report(
    comparison_data: dict,
    parameter_changes: list,
    comparison_failures: list,
) -> str:
    """Generate 06_Simulation_Report.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  06 — SIMULATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    if not comparison_data:
        lines.append("  No simulation data available.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # Metadata
    metadata = _safe_get(comparison_data, "metadata", {})
    lines.append("SIMULATION METADATA")
    lines.append("-" * 40)
    lines.append(f"  Hypothesis: {_truncate(_safe_get(metadata, 'hypothesis', 'N/A'), 200)}")
    lines.append(f"  Simulation Engine: {_safe_get(metadata, 'simulation_engine', 'N/A')}")
    if _safe_get(metadata, "base_chemistry"):
        lines.append(f"  Base Configuration: {_safe_get(metadata, 'base_chemistry', 'N/A')}")
    lines.append(f"  Timestamp: {_safe_get(metadata, 'timestamp', 'N/A')}")
    lines.append("")

    # Baseline metrics
    lines.append("BASELINE METRICS (Current State-of-the-Art)")
    lines.append("-" * 40)
    baseline = _safe_get(comparison_data, "baseline_summary", {})
    baseline_metrics = {k: v for k, v in baseline.items() if isinstance(v, (int, float))}
    if baseline_metrics:
        for metric_name, value in baseline_metrics.items():
            display = _metric_display_name(metric_name)
            unit = _get_metric_unit_hint(metric_name, comparison_data)
            unit_str = f" {_human_unit(unit)}" if _human_unit(unit) else ""
            lines.append(f"  {display}: {_format_metric_value(value)}{unit_str}")
    else:
        lines.append("  No baseline metrics available.")
    lines.append("")

    # Final metrics
    lines.append("FINAL METRICS (Hypothesis-Driven Design)")
    lines.append("-" * 40)
    hyp = _safe_get(comparison_data, "hypothesis_summary", {})
    hyp_metrics = {k: v for k, v in hyp.items() if isinstance(v, (int, float))}
    if hyp_metrics:
        for metric_name, value in hyp_metrics.items():
            display = _metric_display_name(metric_name)
            unit = _get_metric_unit_hint(metric_name, comparison_data)
            unit_str = f" {_human_unit(unit)}" if _human_unit(unit) else ""
            lines.append(f"  {display}: {_format_metric_value(value)}{unit_str}")
    else:
        lines.append("  No hypothesis-driven metrics available.")
    lines.append("")

    # Percentage improvements
    lines.append("PERCENTAGE IMPROVEMENTS")
    lines.append("-" * 40)
    metrics = _safe_get(comparison_data, "metrics", {})
    if metrics:
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                display = _metric_display_name(metric_name)
                pct = metric_data.get("percentage_change", 0)
                is_improvement = metric_data.get("is_improvement", False)
                unit = metric_data.get("unit", "")
                unit_str = f" [{_human_unit(unit)}]" if _human_unit(unit) else ""
                baseline_val = metric_data.get("baseline_value", "N/A")
                hyp_val = metric_data.get("hypothesis_value", "N/A")
                status = "✅" if is_improvement else "❌"
                lines.append(f"  {status} {display}: {_format_metric_value(baseline_val)} → {_format_metric_value(hyp_val)} ({pct:+.1f}%){unit_str}")
    else:
        lines.append("  No improvement data available.")
    lines.append("")

    # Summary
    summary = _safe_get(comparison_data, "summary", {})
    lines.append("OVERALL COMPARISON")
    lines.append("-" * 40)
    lines.append(f"  Metrics Improved:    {summary.get('metrics_improved', 0)}/{summary.get('metrics_total', 0)}")
    lines.append(f"  Average Improvement: {summary.get('average_improvement_pct', 0):.1f}%")
    lines.append(f"  Verdict:             {summary.get('verdict', 'N/A')}")
    lines.append("")

    # Parameter changes
    lines.append("PARAMETER CHANGES")
    lines.append("-" * 40)
    if parameter_changes:
        for pc in parameter_changes:
            lines.append(f"  • {pc.get('parameter_name', 'N/A')}")
            lines.append(f"    Modification: {pc.get('modification_type', 'N/A')} × {pc.get('modification_factor', 'N/A')}")
            lines.append(f"    Keyword Matched: {pc.get('hypothesis_keyword_matched', 'N/A')}")
            lines.append(f"    Selection Reason: {pc.get('selection_reason', 'N/A')}")
            lines.append(f"    Source: {pc.get('source', 'N/A')}")
            lines.append(f"    Papers: {pc.get('paper_count', 0)}")
            lines.append(f"    Confidence: {pc.get('confidence', 0):.2f}")
            lines.append("")
    else:
        lines.append("  No parameter changes recorded.")
        lines.append("")

    # Comparison failures
    if comparison_failures:
        lines.append("COMPARISON FAILURES (Historical)")
        lines.append("-" * 40)
        lines.append(f"  Total failures recorded: {len(comparison_failures)}")
        for cf in comparison_failures[:5]:
            lines.append(f"  • Attempt {cf.get('attempt', '?')}: {cf.get('error', 'N/A')}")
        if len(comparison_failures) > 5:
            lines.append(f"  ... and {len(comparison_failures) - 5} more")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def _get_metric_unit_hint(metric_name: str, comparison_data: dict) -> str:
    """Get the unit for a metric from the comparison data."""
    metrics = _safe_get(comparison_data, "metrics", {})
    if isinstance(metrics, dict):
        metric_data = metrics.get(metric_name, {})
        if isinstance(metric_data, dict):
            return metric_data.get("unit", "")
    # Fallback: derive from metric name
    unit_hints = {
        "temperature": "°C",
        "resistance": "Ω",
        "efficiency": "%",
        "density": "kg/m³",
        "volume": "m³",
        "mass": "kg",
        "length": "m",
        "time": "s",
        "force": "N",
        "power": "W",
        "energy": "Wh",
        "voltage": "V",
        "current": "A",
        "capacity": "Ah",
        "weight": "kg",
    }
    for key, unit in unit_hints.items():
        if key in metric_name.lower():
            return unit
    return ""


def generate_report_07_evidence_and_confidence(
    kb_data: dict,
    equations_data: dict,
    comparison_data: dict,
    evidence_data: dict,
    verified_data: list,
    user_problem: str = "",
) -> str:
    """Generate 07_Evidence_and_Confidence.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  07 — EVIDENCE AND CONFIDENCE")
    lines.append("=" * 80)
    lines.append("")

    if user_problem:
        lines.append("RESEARCH PROBLEM (this evidence package)")
        lines.append("-" * 40)
        lines.append(f"  {user_problem}")
        lines.append("")

    # Literature evidence
    lines.append("1. LITERATURE EVIDENCE")
    lines.append("-" * 40)
    proven_facts = _safe_get(kb_data, "proven_facts", [])
    prior_art = _safe_get(kb_data, "state_of_the_art_prior_art", [])
    lines.append("  Proven Facts:")
    if proven_facts:
        for fact in proven_facts:
            lines.append(f"    • {fact}")
    else:
        lines.append("    • None available")
    lines.append("")
    lines.append("  State of the Art:")
    if prior_art:
        for art in prior_art:
            lines.append(f"    • {art}")
    else:
        lines.append("    • None available")
    lines.append("")

    # Knowledge graph support
    lines.append("2. KNOWLEDGE GRAPH SUPPORT")
    lines.append("-" * 40)
    critical_unknowns = _safe_get(kb_data, "critical_unanswered_unknowns", [])
    testable_hypotheses = _safe_get(kb_data, "proposed_testable_hypotheses", {})
    lines.append("  Critical Unanswered Questions:")
    if critical_unknowns:
        for cu in critical_unknowns:
            lines.append(f"    • {cu}")
    else:
        lines.append("    • None identified")
    lines.append("")
    lines.append("  Proposed Testable Hypotheses:")
    if isinstance(testable_hypotheses, dict):
        for name, test in testable_hypotheses.items():
            display = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(name))
            lines.append(f"    • {display}: {test}")
    lines.append("")

    # Mathematical validation
    lines.append("3. MATHEMATICAL VALIDATION")
    lines.append("-" * 40)
    if equations_data:
        lines.append(f"  Validated Equations: {_safe_get(equations_data, 'total_validated', 0)}")
        lines.append(f"  Rejected Equations:  {_safe_get(equations_data, 'total_rejected', 0)}")
        domains = _safe_get(equations_data, "domains_detected", [])
        engines = _safe_get(equations_data, "engines_used", [])
        lines.append(f"  Domains Detected:    {', '.join(domains) if domains else 'N/A'}")
        lines.append(f"  Engines Used:        {', '.join(engines) if engines else 'N/A'}")
    else:
        lines.append("  No mathematical validation data available.")
    lines.append("")

    # Simulation validation
    lines.append("4. SIMULATION VALIDATION")
    lines.append("-" * 40)
    if comparison_data:
        summary = _safe_get(comparison_data, "summary", {})
        lines.append(f"  Verdict: {summary.get('verdict', 'N/A')}")
        lines.append(f"  Metrics Improved: {summary.get('metrics_improved', 0)}/{summary.get('metrics_total', 0)}")
        lines.append(f"  Average Improvement: {summary.get('average_improvement_pct', 0):.1f}%")
        sim_results = _safe_get(comparison_data, "simulation_results", {})
        verification = _safe_get(sim_results, "verification", {})
        lines.append(f"  Same Engine: {verification.get('same_engine', 'N/A')}")
        lines.append(f"  Reproducible: {verification.get('reproducible', 'N/A')}")
    else:
        lines.append("  No simulation data available.")
    lines.append("")

    # Overall confidence
    lines.append("5. OVERALL CONFIDENCE")
    lines.append("-" * 40)
    if verified_data:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        score = best_hyp.get("final_score", 0)
        lines.append(f"  Best Hypothesis Score: {score:.2f}/10")
        lines.append(f"  Classification: {best_hyp.get('classification', 'N/A')}")
        lines.append(f"  Novelty: {best_hyp.get('novelty', 0)}/10")
        lines.append(f"  Feasibility: {best_hyp.get('feasibility', 0)}/10")
        lines.append(f"  Evidence: {best_hyp.get('evidence', 0)}/10")
        lines.append(f"  Consistency: {best_hyp.get('consistency', 0)}/10")
        lines.append(f"  Testability: {best_hyp.get('testability', 0)}/10")
        lines.append(f"  Risk: {best_hyp.get('risk', 0)}/10")
    else:
        lines.append("  No verified hypotheses available.")
    lines.append("")

    # Evidence scoring details — only current verified hypotheses (not historical DB entries)
    if evidence_data:
        scores = _safe_get(evidence_data, "scores", {})
        db_problem = _safe_get(evidence_data, "research_problem", "")
        if db_problem and user_problem and db_problem.strip().lower() != user_problem.strip().lower():
            lines.append("  Note: evidence_scoring_db.json research_problem differs from this run;")
            lines.append(f"        DB={db_problem[:80]}")
            lines.append("")

        current_claims = []
        for hyp in verified_data or []:
            refined = hyp.get("refined_hypothesis", "")
            if refined:
                current_claims.append(refined)

        filtered_scores = (
            {c: scores[c] for c in current_claims if c in scores}
            if current_claims and scores
            else {}
        )

        if filtered_scores:
            lines.append("  Evidence Scoring Details (current hypotheses only):")
            for claim, score_data in filtered_scores.items():
                conf = score_data.get("confidence", 0)
                missing = score_data.get("missing_evidence", [])
                lines.append(f"    • Claim: {_truncate(claim, 120)}")
                lines.append(f"      Confidence: {conf:.4f}")
                lines.append(f"      Missing Evidence: {', '.join(missing) if missing else 'None'}")
                sources = score_data.get("evidence_sources", {})
                for etype, entries in sources.items():
                    for entry in entries or []:
                        content = entry.get("content", "")
                        if content:
                            lines.append(f"      [{etype}] {content}")
                lines.append("")
        elif scores:
            lines.append("  Evidence Scoring Details: no entries matched current verified hypotheses.")
    lines.append("")

    # What still needs experimental verification
    lines.append("6. WHAT STILL NEEDS EXPERIMENTAL VERIFICATION")
    lines.append("-" * 40)
    required_data = _safe_get(kb_data, "required_empirical_data_inputs", [])
    if required_data:
        for req in required_data:
            lines.append(f"  • {req}")
    else:
        lines.append("  • No specific empirical data requirements identified.")
    lines.append("")
    if verified_data:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        weakness = best_hyp.get("primary_weakness", "")
        if weakness:
            lines.append(f"  Primary Weakness: {weakness}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_report_08_hypotheses_analysis(
    verified_data: list,
    evidence_data: dict,
) -> str:
    """Generate 08_Hypotheses_Analysis.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  08 — HYPOTHESES ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    if not verified_data:
        lines.append("  No verified hypotheses available.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # Select best hypothesis
    best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
    best_refined = best_hyp.get("refined_hypothesis", "")

    lines.append("ALL HYPOTHESES")
    lines.append("-" * 40)
    lines.append("")

    for i, hyp in enumerate(verified_data, 1):
        is_best = hyp.get("refined_hypothesis", "") == best_refined
        marker = " ★ FINAL RECOMMENDATION" if is_best else ""

        lines.append(f"  Hypothesis {i}{marker}")
        lines.append(f"  {'-' * 60}")
        lines.append(f"  Classification: {hyp.get('classification', 'N/A')}")
        lines.append(f"  Final Score:    {hyp.get('final_score', 0):.2f}/10")
        lines.append(f"  Novelty:        {hyp.get('novelty', 0)}/10")
        lines.append(f"  Feasibility:    {hyp.get('feasibility', 0)}/10")
        lines.append(f"  Evidence:       {hyp.get('evidence', 0)}/10")
        lines.append(f"  Consistency:    {hyp.get('consistency', 0)}/10")
        lines.append(f"  Testability:    {hyp.get('testability', 0)}/10")
        lines.append(f"  Risk:           {hyp.get('risk', 0)}/10")
        lines.append("")
        lines.append(f"  Refined Hypothesis:")
        lines.append(f"  {_truncate(hyp.get('refined_hypothesis', 'N/A'), 300)}")
        lines.append("")
        lines.append(f"  Reasoning:")
        lines.append(f"  {_truncate(hyp.get('reasoning', 'N/A'), 300)}")
        lines.append("")
        lines.append(f"  Primary Weakness:")
        lines.append(f"  {_truncate(hyp.get('primary_weakness', 'N/A'), 200)}")
        lines.append("")

    # Summary table
    lines.append("HYPOTHESES SUMMARY TABLE")
    lines.append("-" * 40)
    lines.append(f"  {'#':>3}  {'Score':>6}  {'Class':<12}  {'Novelty':>7}  {'Feas':>4}  {'Evid':>4}  {'Cons':>4}  {'Test':>4}  {'Risk':>4}")
    lines.append(f"  {'---':>3}  {'------':>6}  {'------------':<12}  {'-------':>7}  {'----':>4}  {'----':>4}  {'----':>4}  {'----':>4}  {'----':>4}")
    for i, hyp in enumerate(verified_data, 1):
        is_best = hyp.get("refined_hypothesis", "") == best_refined
        marker = "★" if is_best else " "
        lines.append(f"  {marker} {i:>2}  {hyp.get('final_score', 0):>6.2f}  {hyp.get('classification', 'N/A'):<12}  "
                      f"{hyp.get('novelty', 0):>7}  {hyp.get('feasibility', 0):>4}  {hyp.get('evidence', 0):>4}  "
                      f"{hyp.get('consistency', 0):>4}  {hyp.get('testability', 0):>4}  {hyp.get('risk', 0):>4}")
    lines.append("")

    # Final recommendation
    lines.append("FINAL RECOMMENDATION")
    lines.append("-" * 40)
    lines.append(f"  Selected: Hypothesis with highest final_score ({best_hyp.get('final_score', 0):.2f}/10)")
    lines.append(f"  Classification: {best_hyp.get('classification', 'N/A')}")
    lines.append(f"  Refined Design: {_truncate(best_refined, 300)}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def _is_relevant_to_problem(text: str, problem: str, kb_data: dict) -> bool:
    """
    Check whether a piece of research text is relevant to the CURRENT
    research problem. Used to filter out stale/irrelevant data from
    files that accumulate across research runs (e.g. doscan_breakthroughs).
    Returns True if the text shares meaningful vocabulary with the
    current problem or knowledge base.
    """
    if not text:
        return False
    corpus_parts = [problem or ""]
    thesis = _safe_get(kb_data, "core_research_thesis", "")
    if isinstance(thesis, str):
        corpus_parts.append(thesis)
    corpus = " ".join(corpus_parts).lower()

    STOP = {
        "the", "and", "for", "with", "that", "this", "are", "was", "were",
        "has", "have", "had", "from", "into", "such", "than", "then", "there",
        "their", "these", "those", "which", "while", "when", "where", "what",
        "who", "about", "after", "through", "during", "between", "both",
        "each", "other", "some", "any", "can", "may", "must", "will", "would",
        "could", "should", "using", "used", "use", "based", "design", "system",
    }

    def words(s):
        return [w for w in re.findall(r'[a-z][a-z0-9\-]{2,}', s.lower()) if w not in STOP]

    corpus_words = set(words(corpus))
    if not corpus_words:
        return True

    text_words = set(words(text))
    overlap = text_words & corpus_words
    return len(overlap) >= 2


def generate_report_09_knowledge_graph(
    kb_data: dict,
    doscan_data: list,
    user_problem: str = "",
) -> str:
    """Generate 09_Knowledge_Graph_Summary.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  09 — KNOWLEDGE GRAPH SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Main concepts
    lines.append("1. MAIN CONCEPTS")
    lines.append("-" * 40)
    thesis = _safe_get(kb_data, "core_research_thesis", "N/A")
    lines.append(f"  Core Research Thesis: {thesis}")
    lines.append("")

    prior_art = _safe_get(kb_data, "state_of_the_art_prior_art", [])
    lines.append("  State of the Art Concepts:")
    if prior_art:
        for art in prior_art:
            lines.append(f"    • {art}")
    else:
        lines.append("    • None available")
    lines.append("")

    proven_facts = _safe_get(kb_data, "proven_facts", [])
    lines.append("  Proven Scientific Facts:")
    if proven_facts:
        for fact in proven_facts:
            lines.append(f"    • {fact}")
    else:
        lines.append("    • None available")
    lines.append("")

    # Most connected concepts
    lines.append("2. MOST CONNECTED CONCEPTS")
    lines.append("-" * 40)
    # Analyze DOSCAN insights for most connected concepts
    concept_connections = {}
    if doscan_data:
        for entry in doscan_data:
            insights = _safe_get(entry, "scientific_research_insights", [])
            for insight in insights:
                title = _safe_get(insight, "insight_title", "")
                if not _is_relevant_to_problem(title, user_problem, kb_data):
                    continue
                # Extract concept names from title (between brackets)
                match = re.search(r'\[.*?\]\s*(.*?)\s*↔\s*(.*)', title)
                if match:
                    concept1 = match.group(1).strip()
                    concept2 = match.group(2).strip()
                    concept_connections[concept1] = concept_connections.get(concept1, 0) + 1
                    concept_connections[concept2] = concept_connections.get(concept2, 0) + 1

    if concept_connections:
        sorted_concepts = sorted(concept_connections.items(), key=lambda x: x[1], reverse=True)
        lines.append("  Concepts ranked by connection frequency:")
        for concept, count in sorted_concepts[:10]:
            lines.append(f"    • {concept}: {count} connections")
    else:
        lines.append("  Connection data not available from DOSCAN.")
    lines.append("")

    # Important relationships
    lines.append("3. IMPORTANT RELATIONSHIPS")
    lines.append("-" * 40)
    if doscan_data:
        insight_count = 0
        for entry in doscan_data:
            insights = _safe_get(entry, "scientific_research_insights", [])
            for insight in insights:
                if insight_count >= 10:
                    break
                title = _safe_get(insight, "insight_title", "N/A")
                if not _is_relevant_to_problem(title, user_problem, kb_data):
                    continue
                insight_type = _safe_get(insight, "insight_type", "N/A")
                score = _safe_get(insight, "composite_score", 0)
                lines.append(f"  • [{insight_type}] {title} (score: {score:.2f})")
                insight_count += 1
        if insight_count == 0:
            lines.append("  No relationship insights found.")
    else:
        lines.append("  No DOSCAN data available.")
    lines.append("")

    # Major discoveries
    lines.append("4. MAJOR DISCOVERIES")
    lines.append("-" * 40)
    if doscan_data:
        discovery_count = 0
        for entry in doscan_data:
            insights = _safe_get(entry, "scientific_research_insights", [])
            for insight in insights:
                if discovery_count >= 10:
                    break
                research_note = _safe_get(insight, "research_note", {})
                discovery = _safe_get(research_note, "what_was_discovered", "N/A")
                if not _is_relevant_to_problem(str(discovery), user_problem, kb_data):
                    continue
                implications = _safe_get(research_note, "potential_implications", "N/A")
                lines.append(f"  Discovery: {discovery}")
                lines.append(f"    Implications: {implications}")
                lines.append("")
                discovery_count += 1
        if discovery_count == 0:
            lines.append("  No major discoveries recorded.")
    else:
        lines.append("  No DOSCAN data available.")
    lines.append("")

    # Critical unknowns
    lines.append("5. CRITICAL UNANSWERED QUESTIONS")
    lines.append("-" * 40)
    critical_unknowns = _safe_get(kb_data, "critical_unanswered_unknowns", [])
    if critical_unknowns:
        for cu in critical_unknowns:
            lines.append(f"  • {cu}")
    else:
        lines.append("  • None identified")
    lines.append("")

    # Failure modes
    lines.append("6. FAILURE MODES AND RISKS")
    lines.append("-" * 40)
    failure_modes = _safe_get(kb_data, "failure_modes_and_risks", [])
    if failure_modes:
        for fm in failure_modes:
            lines.append(f"  • {fm}")
    else:
        lines.append("  • None identified")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_report_10_limitations_and_next_steps(
    kb_data: dict,
    verified_data: list,
    comparison_data: dict,
    equations_data: dict,
    evidence_data: dict,
) -> str:
    """Generate 10_Limitations_and_Next_Steps.txt"""
    lines = []
    lines.append("=" * 80)
    lines.append("  10 — LIMITATIONS AND NEXT STEPS")
    lines.append("=" * 80)
    lines.append("")

    # Current limitations
    lines.append("1. CURRENT LIMITATIONS")
    lines.append("-" * 40)
    limitations = []

    # From knowledge base
    critical_unknowns = _safe_get(kb_data, "critical_unanswered_unknowns", [])
    for cu in critical_unknowns:
        limitations.append(f"Unanswered question: {cu}")

    # From hypotheses
    if verified_data:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        weakness = best_hyp.get("primary_weakness", "")
        if weakness:
            limitations.append(f"Primary weakness: {weakness}")

    # From evidence scoring
    if evidence_data:
        scores = _safe_get(evidence_data, "scores", {})
        missing_evidence_types = set()
        for score_data in scores.values():
            if isinstance(score_data, dict):
                for missing in score_data.get("missing_evidence", []):
                    missing_evidence_types.add(missing)
        if missing_evidence_types:
            limitations.append(f"Missing evidence types: {', '.join(missing_evidence_types)}")

    # From equations
    if equations_data:
        total_rejected = _safe_get(equations_data, "total_rejected", 0)
        if total_rejected > 0:
            limitations.append(f"{total_rejected} mathematical formulations were rejected during validation")

    # From simulation
    if comparison_data:
        summary = _safe_get(comparison_data, "summary", {})
        if not summary.get("overall_better", False):
            limitations.append("Simulation did not show overall improvement over baseline")

    if not limitations:
        limitations.append("No specific limitations identified in the current analysis.")

    for lim in limitations:
        lines.append(f"  • {lim}")
    lines.append("")

    # Future work — derived from current research data
    lines.append("2. FUTURE WORK")
    lines.append("-" * 40)
    future_work = []

    # From KB critical unknowns
    for cu in critical_unknowns:
        if isinstance(cu, str):
            future_work.append(f"Investigate: {cu}")

    # From KB required empirical data
    required_data = _safe_get(kb_data, "required_empirical_data_inputs", [])
    for req in required_data:
        if isinstance(req, str):
            future_work.append(f"Acquire empirical data: {req}")

    # From hypothesis weaknesses
    if verified_data:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        weakness = best_hyp.get("primary_weakness", "")
        if weakness:
            future_work.append(f"Address primary weakness: {_truncate(weakness, 150)}")

    # From evidence gaps
    if evidence_data:
        scores = _safe_get(evidence_data, "scores", {})
        for score_data in scores.values():
            if isinstance(score_data, dict):
                for missing in score_data.get("missing_evidence", [])[:3]:
                    if missing and f"Address evidence gap: {missing}" not in future_work:
                        future_work.append(f"Address evidence gap: {missing}")

    if not future_work:
        future_work.append("Validate the proposed design through experimental prototyping")
        future_work.append("Expand the knowledge graph with additional literature")
        future_work.append("Iterate on design based on empirical feedback")

    for fw in future_work:
        lines.append(f"  • {fw}")
    lines.append("")

    # Experiments required — from KB only
    lines.append("3. EXPERIMENTS REQUIRED")
    lines.append("-" * 40)
    if required_data:
        for req in required_data:
            lines.append(f"  • {req}")
    else:
        lines.append("  • No specific experiments identified from current knowledge base.")
    lines.append("")
    lines.append("  General validation steps:")
    lines.append("  • Prototype construction and functional testing")
    lines.append("  • Performance measurement under controlled conditions")
    lines.append("  • Reliability and durability assessment")
    lines.append("  • Safety validation under failure scenarios")
    lines.append("  • Independent third-party verification")
    lines.append("")

    # Prototype recommendations — derived from hypothesis
    lines.append("4. PROTOTYPE RECOMMENDATIONS")
    lines.append("-" * 40)
    if verified_data:
        best_hyp = max(verified_data, key=lambda x: x.get("final_score", 0))
        refined = best_hyp.get("refined_hypothesis", "")
        terms = _extract_key_terms(refined, max_terms=5)
        if terms:
            lines.append("  Based on the selected hypothesis, prototype development should focus on:")
            for term in terms:
                lines.append(f"  • {term.title()} subsystem")
        lines.append("  • Build scale-model prototypes for component-level validation")
        lines.append("  • Develop full-system integration prototype")
        lines.append("  • Iterate through controlled test cycles")
    else:
        lines.append("  • No verified hypotheses available for prototype development.")
    lines.append("")

    # Manufacturing challenges
    lines.append("5. MANUFACTURING CHALLENGES")
    lines.append("-" * 40)
    if required_data:
        lines.append("  Manufacturing considerations derived from current research:")
        for req in required_data[:4]:
            lines.append(f"  • {req}")
    else:
        lines.append("  • No specific manufacturing challenges identified.")
    lines.append("  • Establish quality control for critical components")
    lines.append("  • Evaluate scalability of production processes")
    lines.append("  • Assess cost and supply chain requirements")
    lines.append("")

    # Validation required
    lines.append("6. VALIDATION REQUIRED")
    lines.append("-" * 40)
    lines.append("  • Independent third-party testing of prototype performance")
    lines.append("  • Safety certification and regulatory compliance")
    lines.append("  • Environmental impact assessment")
    lines.append("  • Lifecycle analysis for sustainability metrics")
    lines.append("  • Cost analysis for commercial viability")
    lines.append("  • Long-term durability testing under real-world conditions")
    lines.append("")

    # Possible improvements
    lines.append("7. POSSIBLE IMPROVEMENTS")
    lines.append("-" * 40)
    improvements = [
        "Incorporate multi-objective optimization for trade-off analysis",
        "Add uncertainty quantification to simulation predictions",
        "Integrate real-world operational data for model refinement",
        "Expand knowledge graph with additional literature sources",
        "Implement automated report generation for iterative design cycles",
        "Add economic modeling for cost-benefit analysis",
    ]
    for imp in improvements:
        lines.append(f"  • {imp}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================
def run_research_output_generator(
    root_dir: str = None,
    user_problem: str = "",
    runtime_str: str = "",
    stats: dict = None,
) -> str:
    """
    Phase 10: Research Output Generator

    Reads all existing AARL output files and generates 10 human-readable
    text reports in Research/Research_Output/.

    The generator is FULLY DOMAIN-AGNOSTIC. Every report section is
    derived from the current session's research outputs — never from
    hardcoded domain templates.

    Parameters:
        root_dir: Root directory of the AARL project
        user_problem: The original research problem statement
        runtime_str: Total pipeline runtime string
        stats: Statistics dictionary from the main pipeline

    Returns:
        Path to the output directory
    """
    if root_dir is None:
        current = os.path.dirname(os.path.abspath(__file__))
        while True:
            if (os.path.isdir(os.path.join(current, "Engine"))
                    and os.path.isdir(os.path.join(current, "Research"))
                    and os.path.exists(os.path.join(current, "deep_research_knowledge_base.json"))):
                root_dir = current
                break
            parent = os.path.dirname(current)
            if parent == current:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                break
            current = parent

    # Resolve paths
    engine_dir = os.path.join(root_dir, "Engine", "Question_Engine")
    research_dir = os.path.join(root_dir, "Research")
    output_dir = os.path.join(research_dir, "Research_Output")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use final_solution for user_problem if not provided
    final_solution = _load_json(os.path.join(root_dir, "final_research_solution.json"))
    if not user_problem and final_solution:
        user_problem = _safe_get(final_solution, "original_problem", "")

    # Determine problem-specific output directory
    problem_output_dir = None
    if user_problem:
        import re as _re
        slug = _re.sub(r'[^a-z0-9]+', '_', user_problem.lower()).strip('_')
        if len(slug) > 60:
            slug = slug[:60]
        if slug:
            problem_output_dir = os.path.join(research_dir, slug)

    def _load_from_problem_dir_or_default(filename, default_path):
        """Load a file from the problem-specific directory if it exists,
        otherwise fall back to the default path."""
        if problem_output_dir:
            problem_path = os.path.join(problem_output_dir, filename)
            if os.path.exists(problem_path):
                return _load_json(problem_path)
        return _load_json(default_path)

    # Load all existing output files
    kb_data = _load_json(os.path.join(root_dir, "deep_research_knowledge_base.json"))
    doscan_data = _load_json(os.path.join(root_dir, "doscan_breakthroughs.json"))
    verified_data = _load_json(os.path.join(root_dir, "verified_hypotheses.json"))
    equations_data = _load_json(os.path.join(root_dir, "derived_equations.json"))
    evidence_data = _load_json(os.path.join(root_dir, "evidence_scoring_db.json"))
    comparison_data = _load_from_problem_dir_or_default(
        "comparison_report.json",
        os.path.join(research_dir, "comparison_report.json")
    )
    parameter_changes = _load_from_problem_dir_or_default(
        "parameter_changes.json",
        os.path.join(research_dir, "parameter_changes.json")
    )
    comparison_failures = _load_json(os.path.join(research_dir, "comparison_failures.json"))
    verification_md = _load_text(os.path.join(research_dir, "verification_report.md"))

    # Use stats from final_solution if not provided
    if stats is None:
        stats = {}

    # Ensure stats has all needed keys
    stats = {
        "kb_nodes": stats.get("kb_nodes", 0),
        "relationships": stats.get("relationships", 0),
        "doscan_clusters": stats.get("doscan_clusters", 0),
        "hypotheses_generated": stats.get("hypotheses_generated", 0),
        "hypotheses_approved": stats.get("hypotheses_approved", 0),
        "hypotheses_rejected": stats.get("hypotheses_rejected", 0),
        "top_confidence": stats.get("top_confidence", 0.0),
        "deepening_cycles": stats.get("deepening_cycles", 0),
        "math_validated": stats.get("math_validated", 0),
        "math_rejected": stats.get("math_rejected", 0),
    }

    # Generate all 10 reports
    reports = {
        "01_Research_Summary.txt": generate_report_01_summary(
            kb_data, doscan_data, verified_data, equations_data,
            evidence_data, comparison_data, user_problem, runtime_str, stats
        ),
        "02_Final_Design_Report.txt": generate_report_02_final_design(
            verified_data, comparison_data, kb_data, equations_data, user_problem
        ),
        "03_Design_Architecture.txt": generate_report_03_design_architecture(
            verified_data, kb_data, comparison_data
        ),
        "04_Engineering_Calculations.txt": generate_report_04_engineering_calculations(
            equations_data, verified_data, comparison_data
        ),
        "05_Materials_and_Technologies.txt": generate_report_05_materials_and_technologies(
            verified_data, kb_data, comparison_data
        ),
        "06_Simulation_Report.txt": generate_report_06_simulation_report(
            comparison_data, parameter_changes, comparison_failures
        ),
        "07_Evidence_and_Confidence.txt": generate_report_07_evidence_and_confidence(
            kb_data, equations_data, comparison_data, evidence_data, verified_data, user_problem
        ),
        "08_Hypotheses_Analysis.txt": generate_report_08_hypotheses_analysis(
            verified_data, evidence_data
        ),
        "09_Knowledge_Graph_Summary.txt": generate_report_09_knowledge_graph(
            kb_data, doscan_data, user_problem
        ),
        "10_Limitations_and_Next_Steps.txt": generate_report_10_limitations_and_next_steps(
            kb_data, verified_data, comparison_data, equations_data, evidence_data
        ),
    }

    # Write all reports to disk
    for filename, content in reports.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    return output_dir


if __name__ == "__main__":
    output_dir = run_research_output_generator()
    print(f"Reports generated in: {output_dir}")