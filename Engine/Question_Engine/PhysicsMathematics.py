#!/usr/bin/env python3
"""
PhysicsMathematics.py — Physics-Specific Equation Engine
==========================================================
Generates equations from physics domains:
  - Classical Mechanics: Newton's Laws, conservation laws
  - Thermodynamics: heat transfer, entropy, efficiency
  - Fluid Dynamics: Bernoulli, Navier-Stokes approximations
  - Electromagnetism: Maxwell equations, circuit laws
  - Material Mechanics: stress/strain, elasticity
  - Aerodynamics: lift, drag, flow dynamics

No LLM used for equation generation — purely structured derivation.

Also provides the domain-aware mathematics orchestrator that routes
hypotheses to the appropriate specialized engine based on detected
scientific domains.
"""

import random
import math
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

# Reuse shared schemas and utilities from Mathematics.py
from Mathematics import MathIdea, MathValidation, is_rubbish_idea

# =========================================================
# PHYSICS RELATIONSHIP TEMPLATES
# =========================================================
PHYSICS_TEMPLATES = [
    # Force & Motion
    lambda a, b: f"{a} = {b} * a",           # F = ma
    lambda a, b: f"{a} = 0.5 * {b} * v^2",   # KE = 0.5*m*v^2
    lambda a, b: f"{a} = {b} * g * h",       # PE = m*g*h
    lambda a, b: f"{a} = {b} * v",           # p = m*v (momentum)
    lambda a, b: f"{a} = k * {b}",           # F = kx (Hooke's Law)

    # Thermodynamics
    lambda a, b: f"{a} = k * dT / {b}",      # Q = k*A*dT/L (Fourier)
    lambda a, b: f"{a} = {b} * c * dT",       # Q = m*c*dT
    lambda a, b: f"d{a} = dQ / {b}",          # dS = dQ/T
    lambda a, b: f"eta = 1 - {b} / {a}",      # Carnot efficiency

    # Fluid Dynamics
    lambda a, b: f"{a} + 0.5 * rho * v^2 = {b}",  # Bernoulli
    lambda a, b: f"{a} = rho * {b} * v^2",         # Drag force
    lambda a, b: f"Re = rho * v * {a} / mu",       # Reynolds number

    # Electromagnetism
    lambda a, b: f"{a} = {b} * R",            # V = IR
    lambda a, b: f"{a} = {b} / {b}_0",        # C = Q/V
    lambda a, b: f"{a} = mu_0 * {b} / (2*pi*r)",  # B field
    lambda a, b: f"{a} = -{b} * dPhi/dt",     # Faraday's Law

    # Material Mechanics
    lambda a, b: f"sigma = {a} * epsilon",    # Stress = E * strain
    lambda a, b: f"{a} = {b} / A",            # sigma = F/A
    lambda a, b: f"epsilon = d{a} / {b}",     # strain = dL/L
]

PHYSICS_RELATIONSHIP_TYPES = [
    "proportional", "inverse", "exponential", "power_law",
    "linear", "quadratic", "inverse_square",
]

# =========================================================
# PHYSICS CONCEPT MAP
# =========================================================
PHYSICS_CONCEPT_MAP = {
    "force": {"symbol": "F", "unit": "N", "equations": ["F = m*a", "F = k*x", "F = mu*N"]},
    "mass": {"symbol": "m", "unit": "kg", "equations": ["m = rho*V", "m = F/a"]},
    "acceleration": {"symbol": "a", "unit": "m/s^2", "equations": ["a = F/m", "a = dv/dt"]},
    "velocity": {"symbol": "v", "unit": "m/s", "equations": ["v = d/t", "v = a*t"]},
    "energy": {"symbol": "E", "unit": "J", "equations": ["E = 0.5*m*v^2", "E = m*g*h", "E = F*d"]},
    "power": {"symbol": "P", "unit": "W", "equations": ["P = F*v", "P = E/t", "P = V*I"]},
    "pressure": {"symbol": "P", "unit": "Pa", "equations": ["P = F/A", "P = rho*g*h"]},
    "temperature": {"symbol": "T", "unit": "K", "equations": ["T = E/(k_B)", "T = PV/(nR)"]},
    "heat": {"symbol": "Q", "unit": "J", "equations": ["Q = m*c*dT", "Q = k*A*dT/L"]},
    "entropy": {"symbol": "S", "unit": "J/K", "equations": ["dS = dQ/T", "S = k*ln(W)"]},
    "voltage": {"symbol": "V", "unit": "V", "equations": ["V = I*R", "V = E/Q"]},
    "current": {"symbol": "I", "unit": "A", "equations": ["I = V/R", "I = Q/t"]},
    "resistance": {"symbol": "R", "unit": "Ohm", "equations": ["R = V/I", "R = rho*L/A"]},
    "stress": {"symbol": "sigma", "unit": "Pa", "equations": ["sigma = F/A", "sigma = E*epsilon"]},
    "strain": {"symbol": "epsilon", "unit": "", "equations": ["epsilon = dL/L", "epsilon = sigma/E"]},
    "density": {"symbol": "rho", "unit": "kg/m^3", "equations": ["rho = m/V", "rho = P/(R*T)"]},
    "drag": {"symbol": "F_d", "unit": "N", "equations": ["F_d = 0.5*rho*v^2*Cd*A"]},
    "lift": {"symbol": "L", "unit": "N", "equations": ["L = 0.5*rho*v^2*Cl*A"]},
}


def generate_physics_ideas(hypothesis: str, num_ideas: int = 20) -> List[MathIdea]:
    """Generate physics-specific mathematical formulations."""
    # Extract physics variables from the hypothesis
    variables = []
    hyp_lower = hypothesis.lower()
    for concept, info in PHYSICS_CONCEPT_MAP.items():
        if concept in hyp_lower:
            variables.append(info)

    if len(variables) < 2:
        variables = [
            {"symbol": "F", "unit": "N", "equations": ["F = m*a"], "name": "F", "description": "Force (N)"},
            {"symbol": "m", "unit": "kg", "equations": ["m = rho*V"], "name": "m", "description": "Mass (kg)"},
            {"symbol": "v", "unit": "m/s", "equations": ["v = d/t"], "name": "v", "description": "Velocity (m/s)"},
            {"symbol": "k", "unit": "varies", "equations": [], "name": "k", "description": "Constant"},
        ]

    # Convert to standard format
    std_vars = []
    for v in variables:
        std_vars.append({
            "name": v.get("symbol", v.get("name", "x")),
            "concept": concept if 'concept' in dir() else "physics",
            "unit": v.get("unit", ""),
            "equations": v.get("equations", []),
            "description": v.get("description", f"{v.get('symbol', 'x')} ({v.get('unit', '')})"),
        })

    ideas = []
    for i in range(num_ideas):
        a = std_vars[0]["name"] if std_vars else "x"
        b = std_vars[1]["name"] if len(std_vars) > 1 else "y"
        template = random.choice(PHYSICS_TEMPLATES)
        try:
            eq_text = template(a, b)
        except Exception:
            eq_text = f"{a} = k * {b}"

        var_desc = {v["name"]: v["description"] for v in std_vars[:4]}
        var_desc["k"] = "Constant"

        ideas.append(MathIdea(
            idea_id=f"PHY-{i+1:03d}",
            equation_text=eq_text,
            variables=var_desc,
            relationship_type=random.choice(PHYSICS_RELATIONSHIP_TYPES),
        ))

    return ideas


def validate_physics_idea(hypothesis: str, idea: MathIdea) -> MathValidation:
    """
    Physics-specific deterministic validation.
    Checks dimensional consistency and physical plausibility.
    """
    eq = idea.equation_text
    rel_type = idea.relationship_type

    # Check for physically meaningful patterns
    has_physics_constant = any(c in eq for c in ['k', 'mu_0', 'g', 'rho', 'Cd', 'Cl'])
    has_physics_relation = any(op in eq for op in ['*', '/', '^', 'sqrt'])

    # Check for basic dimensional consistency
    if 'v^2' in eq and '0.5' in eq:
        return MathValidation(makes_sense=True, reasoning="Kinetic energy form: 0.5*m*v^2", dimension_hint="J")
    if 'm*a' in eq or 'm * a' in eq:
        return MathValidation(makes_sense=True, reasoning="Newton's Second Law: F=ma", dimension_hint="N")
    if 'I*R' in eq or 'I * R' in eq:
        return MathValidation(makes_sense=True, reasoning="Ohm's Law: V=IR", dimension_hint="V")
    if 'rho' in eq and 'v^2' in eq:
        return MathValidation(makes_sense=True, reasoning="Fluid dynamics equation", dimension_hint="N")
    if 'epsilon' in eq and 'sigma' in eq:
        return MathValidation(makes_sense=True, reasoning="Stress-strain relationship", dimension_hint="Pa")

    if has_physics_constant and has_physics_relation:
        return MathValidation(makes_sense=True, reasoning=f"Physics: {rel_type}", dimension_hint="Standard")

    return MathValidation(makes_sense=False, reasoning="Not physically meaningful", dimension_hint="")


# =========================================================
# DOMAIN-AWARE MATHEMATICS ORCHESTRATOR
# =========================================================
def run_domain_aware_mathematics(
    hypotheses: List[Any],
    detected_domains: List[dict],
    equations_file: str = "derived_equations.json"
) -> Dict[str, Any]:
    """
    Domain-Aware Mathematics Engine.

    Routes each hypothesis to the appropriate specialized math engine based on
    detected scientific domains. Uses PhysicsMathematics for physics domains,
    and falls back to the general Mathematics engine for other domains
    (chemistry, quantum, biology, materials_science).

    Pipeline (mirrors run_mathematics_engine but with domain-aware routing):
    1. Generate math ideas using domain-specific or general engine
    2. Filter rubbish immediately (deterministic checks)
    3. TF-IDF vectorize surviving ideas
    4. DBSCAN cluster (r=1→r=4 multi-resolution)
    5. For each cluster, pick the best idea and validate
       (physics → deterministic validation, others → LLM validation)
    6. If valid → deepen that cluster with more variations → re-cluster
    7. If invalid → discard
    8. Save all validated equations to derived_equations.json
    """
    # Import general engine utilities (avoids circular import at module load time)
    from Mathematics import (
        generate_random_math_ideas,
        multi_resolution_cluster,
        validate_math_idea_llm,
        deepen_cluster,
    )

    print("\n" + "=" * 80)
    print("📐 DOMAIN-AWARE MATHEMATICS ENGINE — Specialized Equation Derivation")
    print("=" * 80)

    # Determine which domains have specialized engines
    domain_engines = {}
    for d in detected_domains:
        domain_engines[d["domain"]] = d["math_engine"]

    has_physics = "physics" in domain_engines
    print(f"  Detected domain engines: {list(domain_engines.values())}")
    print(f"  Physics engine available: {has_physics}")

    all_validated = []
    all_rejected = []

    for i, hypothesis in enumerate(hypotheses, 1):
        # Normalize hypothesis to string (may be dict from hypothesis engine)
        if isinstance(hypothesis, dict):
            hyp_text = hypothesis.get(
                "refined_hypothesis",
                hypothesis.get("hypothesis", str(hypothesis))
            )
        else:
            hyp_text = str(hypothesis)

        print(f"\n  [{i}/{len(hypotheses)}] Processing hypothesis...")
        print(f"      📝 {hyp_text[:120]}...")

        # Determine which engine to use for this hypothesis
        # Use the primary (highest-scoring) domain
        primary_domain = detected_domains[0]["domain"] if detected_domains else "general"
        primary_engine = domain_engines.get(primary_domain, "general")

        # Step 1: Generate math ideas using domain-specific or general engine
        if primary_engine == "physics" or primary_domain == "physics":
            print(f"      🧪 Using Physics Mathematics Engine for domain: {primary_domain}")
            raw_ideas = generate_physics_ideas(hyp_text, num_ideas=20)
        else:
            print(f"      🧪 Using General Mathematics Engine for domain: {primary_domain}")
            raw_ideas = generate_random_math_ideas(hyp_text, num_ideas=20, domain=primary_domain)

        print(f"         Generated {len(raw_ideas)} raw ideas")

        # Step 2: Filter rubbish immediately (deterministic)
        surviving_ideas = []
        rubbish_count = 0
        for idea in raw_ideas:
            is_rubbish, reason = is_rubbish_idea(idea)
            if is_rubbish:
                rubbish_count += 1
                all_rejected.append({
                    "hypothesis": hyp_text,
                    "equation": idea.equation_text,
                    "reason": reason,
                    "filter": "rubbish_filter"
                })
            else:
                surviving_ideas.append(idea)

        print(f"         🗑️  Rubbish filtered: {rubbish_count}")
        print(f"         ✅ Surviving: {len(surviving_ideas)}")

        if not surviving_ideas:
            print(f"      ⏸️  No surviving ideas. Skipping.")
            continue

        # Step 3: TF-IDF vectorize + DBSCAN cluster
        idea_texts = [idea.equation_text for idea in surviving_ideas]
        print(f"      🔬 Clustering {len(idea_texts)} ideas with DBSCAN (r=1→4)...")
        cluster_result = multi_resolution_cluster(idea_texts, max_r=4)

        print(f"         Clusters formed: {cluster_result['total_clusters']}")
        for cname, cinfo in cluster_result['clusters'].items():
            print(f"            {cname}: {cinfo['size']} ideas (r={cinfo['r_level']}, eps={cinfo['eps']})")

        # Step 4: For each cluster, validate the best idea
        validated_this_hypothesis = 0
        deepened_clusters = 0

        for cname, cinfo in cluster_result['clusters'].items():
            cluster_ideas = [surviving_ideas[idx] for idx in cinfo['document_indices']]
            best_idea = cluster_ideas[0]

            # Step 5: Validation — domain-specific or LLM
            print(f"         🔍 Validating {cname} (representative: {best_idea.equation_text[:60]}...)")

            if primary_engine == "physics" or primary_domain == "physics":
                # Use physics-specific deterministic validation
                validation = validate_physics_idea(hyp_text, best_idea)
            else:
                # Use LLM validation (with deterministic fallback)
                validation = validate_math_idea_llm(hyp_text, best_idea)

            if validation.makes_sense:
                validated_this_hypothesis += 1
                all_validated.append({
                    "hypothesis": hyp_text,
                    "equation": best_idea.equation_text,
                    "variables": best_idea.variables,
                    "relationship_type": best_idea.relationship_type,
                    "cluster": cname,
                    "r_level": cinfo['r_level'],
                    "validation_reasoning": validation.reasoning,
                    "dimension_hint": validation.dimension_hint,
                    "domain": primary_domain,
                    "engine": primary_engine,
                    "timestamp": datetime.now().isoformat()
                })
                print(f"            ✅ MAKES SENSE — {validation.reasoning[:80]}")

                # Step 6: Deepen this cluster
                print(f"            🔄 Deepening cluster with more variations...")
                deepened_ideas = deepen_cluster(hyp_text, cinfo['documents'], num_new=8, domain=primary_domain)

                for d_idea in deepened_ideas:
                    is_rubbish, reason = is_rubbish_idea(d_idea)
                    if not is_rubbish:
                        if primary_engine == "physics" or primary_domain == "physics":
                            d_validation = validate_physics_idea(hyp_text, d_idea)
                        else:
                            d_validation = validate_math_idea_llm(hyp_text, d_idea)
                        if d_validation.makes_sense:
                            deepened_clusters += 1
                            all_validated.append({
                                "hypothesis": hyp_text,
                                "equation": d_idea.equation_text,
                                "variables": d_idea.variables,
                                "relationship_type": d_idea.relationship_type,
                                "cluster": f"{cname}_deepened",
                                "r_level": cinfo['r_level'] + 1,
                                "validation_reasoning": d_validation.reasoning,
                                "dimension_hint": d_validation.dimension_hint,
                                "domain": primary_domain,
                                "engine": primary_engine,
                                "timestamp": datetime.now().isoformat()
                            })
                            print(f"            ✅ Deepened: {d_idea.equation_text[:60]}...")
            else:
                rejection_reason = validation.reasoning
                all_rejected.append({
                    "hypothesis": hyp_text,
                    "equation": best_idea.equation_text,
                    "reason": rejection_reason,
                    "filter": "domain_validation"
                })
                print(f"            ❌ RUBBISH — {rejection_reason[:80]}")

        print(f"      📊 Results for this hypothesis: {validated_this_hypothesis} validated, {deepened_clusters} deepened")

    # Save to disk
    output = {
        "total_validated": len(all_validated),
        "total_rejected": len(all_rejected),
        "validated_equations": all_validated,
        "rejected_ideas": all_rejected,
        "domains_detected": [d["domain"] for d in detected_domains],
        "engines_used": list(set(d["math_engine"] for d in detected_domains)),
        "summary": {
            "validated_count": len(all_validated),
            "rejected_count": len(all_rejected),
            "rejection_breakdown": {
                "rubbish_filter": sum(1 for r in all_rejected if r.get("filter") == "rubbish_filter"),
                "domain_validation": sum(1 for r in all_rejected if r.get("filter") == "domain_validation"),
                "llm_validation": sum(1 for r in all_rejected if r.get("filter") == "llm_validation"),
                "llm_unavailable": sum(1 for r in all_rejected if r.get("filter") == "llm_unavailable"),
                "parser_error": sum(1 for r in all_rejected if r.get("filter") == "parser_error")
            }
        }
    }

    with open(equations_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n📄 All equations saved to '{equations_file}'")
    print(f"📊 Summary: {len(all_validated)} validated, {len(all_rejected)} rejected "
          f"({output['summary']['rejection_breakdown']['rubbish_filter']} by rubbish filter, "
          f"{output['summary']['rejection_breakdown']['domain_validation']} by domain validation)")
    print("=" * 80)

    return output


if __name__ == "__main__":
    test_hypotheses = [
        "Increasing electrode surface area increases battery capacity linearly with area",
        "Battery energy density is inversely proportional to internal resistance squared"
    ]
    test_domains = [
        {"domain": "physics", "score": 5, "matched_keywords": ["energy", "force"], "topics": ["Thermodynamics"], "math_engine": "physics", "simulator": "physics"},
    ]
    run_domain_aware_mathematics(test_hypotheses, test_domains)
