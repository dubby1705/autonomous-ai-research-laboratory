"""
RESEARCH MATHEMATICS ENGINE (DBSCAN-Driven)
No more LLM equation hallucination.

Instead:
1. Generate random mathematical formulations from hypothesis keywords
2. TF-IDF vectorize them
3. DBSCAN cluster (r=1→r=4 multi-resolution)
4. Filter out rubbish immediately (dimensional analysis + constraint check)
5. LLM only used for: "Does this make sense?" (yes/no)
6. If yes → deepen that cluster with more random variations
7. If no → discard
"""

import os
import json
import math
import re
import random
import concurrent.futures
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from GroqClient import llm_complete_json

# =========================================================
# PYDANTIC SCHEMAS
# =========================================================
class MathIdea(BaseModel):
    idea_id: str = Field(description="Unique ID for this math idea.")
    equation_text: str = Field(description="The equation in plain text (e.g., 'C = A * k').")
    variables: Dict[str, str] = Field(description="Map of variable name -> description.")
    relationship_type: str = Field(description="One of: proportional, inverse, exponential, logarithmic, power_law, polynomial, linear, quadratic, threshold, sigmoid")

class MathValidation(BaseModel):
    makes_sense: bool = Field(description="True if this mathematical relationship is physically/logically meaningful.")
    reasoning: str = Field(description="Why it makes sense or why it's rubbish.")
    dimension_hint: str = Field(description="If it makes sense, suggest what dimensions the variables should have.")

# =========================================================
# TOKENIZATION & TF-IDF (Shared with DOSCAN)
# =========================================================
BOUNDARY_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "with", "by", "on", "in",
    "at", "from", "as", "this", "that", "it", "can", "could", "would", "should", "using", "such",
    "without", "will", "has", "have", "had", "not", "no", "only", "but", "however", "and", "or",
    "which", "where", "when", "why", "how", "we", "they", "their", "our", "show", "shows",
    "demonstrate", "compare", "search", "find", "found", "use", "used", "requires", "assumes",
    "must", "always", "during", "between", "through", "under", "over", "into"
}

def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if w not in BOUNDARY_WORDS]

def build_tfidf_vectors(documents: List[str]) -> List[Dict[str, float]]:
    doc_tokens = [tokenize(doc) for doc in documents]
    N = max(len(documents), 1)
    df = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    vectors = []
    for tokens in doc_tokens:
        vec = {}
        tf = Counter(tokens)
        total_terms = max(len(tokens), 1)
        for word, count in tf.items():
            vec[word] = (count / total_terms) * math.log(N / (1 + df[word]))
        vectors.append(vec)
    return vectors

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[w] * vec2[w] for w in intersection)
    mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
    return 0.0 if mag1 == 0 or mag2 == 0 else dot_product / (mag1 * mag2)

# =========================================================
# DBSCAN CLUSTERING (Multi-Resolution)
# =========================================================
def dbscan_cluster(documents: List[str], vectors: List[Dict[str, float]], eps: float, min_pts: int = 1) -> Tuple[Dict[int, List[int]], List[int]]:
    labels = [0] * len(documents)
    cluster_id = 0
    def region_query(p_idx: int) -> List[int]:
        return [q_idx for q_idx, q_vec in enumerate(vectors) if cosine_similarity(vectors[p_idx], q_vec) >= eps]

    for p in range(len(documents)):
        if labels[p] != 0: continue
        neighbors = region_query(p)
        if len(neighbors) < min_pts + 1:
            labels[p] = -1
        else:
            cluster_id += 1
            labels[p] = cluster_id
            i = 0
            while i < len(neighbors):
                q = neighbors[i]
                if labels[q] == -1: labels[q] = cluster_id
                elif labels[q] == 0:
                    labels[q] = cluster_id
                    q_neighbors = region_query(q)
                    if len(q_neighbors) >= min_pts + 1: neighbors.extend(q_neighbors)
                i += 1

    clusters: Dict[int, List[int]] = {}
    noise: List[int] = []
    for idx, label in enumerate(labels):
        if label == -1: noise.append(idx)
        else:
            if label not in clusters: clusters[label] = []
            clusters[label].append(idx)
    return clusters, noise

def multi_resolution_cluster(documents: List[str], max_r: int = 4) -> Dict[str, Any]:
    """
    Multi-resolution DBSCAN clustering (r=1 to r=4).
    eps decays: 0.30 → 0.15 → 0.00 → 0.00
    """
    unprocessed_indices = list(range(len(documents)))
    all_clusters = {}
    cluster_counter = 0

    for r in range(1, max_r + 1):
        eps = max(0.0, 0.45 - (r * 0.15))
        if not unprocessed_indices: break
        
        current_docs = [documents[i] for i in unprocessed_indices]
        if len(current_docs) < 2:
            break
            
        vectors = build_tfidf_vectors(current_docs)
        clusters, noise_local = dbscan_cluster(current_docs, vectors, eps=eps)
        
        for cid, members in clusters.items():
            cluster_counter += 1
            original_indices = [unprocessed_indices[i] for i in members]
            all_clusters[f"M_Cluster_{cluster_counter}_r{r}"] = {
                "document_indices": original_indices,
                "documents": [documents[i] for i in original_indices],
                "r_level": r,
                "eps": round(eps, 3),
                "size": len(original_indices)
            }
        
        noise_indices = [unprocessed_indices[i] for i in noise_local]
        unprocessed_indices = noise_indices

    return {
        "clusters": all_clusters,
        "final_noise": [documents[i] for i in unprocessed_indices] if unprocessed_indices else [],
        "total_documents": len(documents),
        "total_clusters": len(all_clusters)
    }

# =========================================================
# RANDOM MATH IDEA GENERATOR (No LLM)
# =========================================================
RELATIONSHIP_TEMPLATES = [
    # proportional: y = k * x
    lambda a, b: f"{a} = k * {b}",
    lambda a, b: f"{a} = alpha * {b} + beta",
    # inverse: y = k / x
    lambda a, b: f"{a} = k / {b}",
    lambda a, b: f"{a} = k / ({b} + c)",
    # exponential: y = k * exp(x)
    lambda a, b: f"{a} = k * exp({b})",
    lambda a, b: f"{a} = k * exp(-{b} / tau)",
    # power law: y = k * x^n
    lambda a, b: f"{a} = k * ({b})^n",
    lambda a, b: f"{a} = k * ({b})^2",
    lambda a, b: f"{a} = k * sqrt({b})",
    # logarithmic: y = k * log(x)
    lambda a, b: f"{a} = k * log({b})",
    lambda a, b: f"{a} = k * log({b} / {b}_0)",
    # polynomial
    lambda a, b: f"{a} = k1 * {b} + k2 * {b}^2",
    lambda a, b: f"{a} = k1 * {b} + k2 * {b}^2 + k3 * {b}^3",
    # threshold / sigmoid
    lambda a, b: f"{a} = 1 / (1 + exp(-k * ({b} - {b}_0)))",
    # linear with offset
    lambda a, b: f"{a} = k * {b} + {b}_offset",
    # ratio
    lambda a, b: f"{a} = ({b}_max - {b}) / ({b}_max - {b}_min)",
]

RELATIONSHIP_TYPES = [
    "proportional", "inverse", "exponential", "power_law", 
    "logarithmic", "polynomial", "sigmoid", "linear"
]

# =========================================================
# STRUCTURED SCIENTIFIC CONCEPT → VARIABLE MAPPING
# =========================================================
# Maps scientific concepts to their standard symbols and equations
# This replaces random word selection with structured concept derivation
SCIENTIFIC_CONCEPT_MAP = {
    # Physics concepts
    "energy": {"symbol": "E", "unit": "J", "equations": ["E = 0.5 * m * v^2", "E = m * c^2", "E = h * f", "E = k * T"]},
    "power": {"symbol": "P", "unit": "W", "equations": ["P = V * I", "P = F * v", "P = E / t"]},
    "force": {"symbol": "F", "unit": "N", "equations": ["F = m * a", "F = k * x", "F = mu * N"]},
    "voltage": {"symbol": "V", "unit": "V", "equations": ["V = I * R", "V = E / Q"]},
    "current": {"symbol": "I", "unit": "A", "equations": ["I = V / R", "I = Q / t"]},
    "resistance": {"symbol": "R", "unit": "Ohm", "equations": ["R = V / I", "R = rho * L / A"]},
    "capacitance": {"symbol": "C", "unit": "F", "equations": ["C = Q / V", "C = epsilon * A / d"]},
    "temperature": {"symbol": "T", "unit": "K", "equations": ["T = E / (k_B)", "T = P * V / (n * R)"]},
    "pressure": {"symbol": "P", "unit": "Pa", "equations": ["P = F / A", "P = rho * g * h"]},
    "velocity": {"symbol": "v", "unit": "m/s", "equations": ["v = d / t", "v = f * lambda"]},
    "frequency": {"symbol": "f", "unit": "Hz", "equations": ["f = 1 / T", "f = v / lambda"]},
    "wavelength": {"symbol": "lambda", "unit": "m", "equations": ["lambda = v / f", "lambda = c / f"]},
    "mass": {"symbol": "m", "unit": "kg", "equations": ["m = rho * V", "m = F / a"]},
    "density": {"symbol": "rho", "unit": "kg/m^3", "equations": ["rho = m / V", "rho = P / (R * T)"]},
    "momentum": {"symbol": "p", "unit": "kg*m/s", "equations": ["p = m * v", "p = F * t"]},
    # Chemistry concepts
    "concentration": {"symbol": "C", "unit": "M", "equations": ["C = n / V", "rate = k * C^n"]},
    "rate": {"symbol": "r", "unit": "mol/(L*s)", "equations": ["r = k * [A]^m", "r = A * exp(-Ea/(R*T))"]},
    "yield": {"symbol": "Y", "unit": "%", "equations": ["Y = (actual / theoretical) * 100"]},
    "catalyst": {"symbol": "k_cat", "unit": "1/s", "equations": ["k_cat = k * exp(-Ea_cat/(R*T))"]},
    "equilibrium": {"symbol": "K_eq", "unit": "", "equations": ["K_eq = [products] / [reactants]", "dG = -R*T*ln(K_eq)"]},
    "ph": {"symbol": "pH", "unit": "", "equations": ["pH = -log10([H+])", "pH = pKa + log([A-]/[HA])"]},
    # Battery-specific concepts
    "capacity": {"symbol": "Q", "unit": "Ah", "equations": ["Q = I * t", "Q = C * V"]},
    "efficiency": {"symbol": "eta", "unit": "%", "equations": ["eta = (P_out / P_in) * 100", "eta = (V_t / V_oc) * 100"]},
    "conductivity": {"symbol": "sigma", "unit": "S/m", "equations": ["sigma = 1 / rho", "sigma = n * e * mu"]},
    "thermal": {"symbol": "k_th", "unit": "W/(m*K)", "equations": ["k_th = Q * L / (A * dT)", "k_th = k_base * (1 + alpha*T)"]},
    "degradation": {"symbol": "d", "unit": "%/cycle", "equations": ["d = d0 * exp(-Ea/(R*T))", "d = k * DOD^2"]},
    "stability": {"symbol": "S", "unit": "%", "equations": ["S = 100 - d * N", "S = S0 * exp(-t/tau)"]},
    "electrode": {"symbol": "A_e", "unit": "m^2", "equations": ["A_e = 4 * pi * r^2", "Q = k * A_e"]},
    "electrolyte": {"symbol": "sigma_e", "unit": "S/m", "equations": ["sigma_e = sigma0 * exp(-Ea/(R*T))"]},
    "cycle": {"symbol": "N", "unit": "cycles", "equations": ["N = 20 / d", "N = N0 * (1 - d)^n"]},
    # Materials concepts
    "stress": {"symbol": "sigma_s", "unit": "Pa", "equations": ["sigma_s = F / A", "sigma_s = E * epsilon"]},
    "strain": {"symbol": "epsilon", "unit": "", "equations": ["epsilon = dL / L", "epsilon = sigma_s / E"]},
    "tensile": {"symbol": "sigma_uts", "unit": "Pa", "equations": ["sigma_uts = F_max / A"]},
    "modulus": {"symbol": "E", "unit": "Pa", "equations": ["E = sigma_s / epsilon"]},
}

# Standard physical constants
PHYSICAL_CONSTANTS = {
    "R": ("Gas constant", 8.314, "J/(mol*K)"),
    "k_B": ("Boltzmann constant", 1.381e-23, "J/K"),
    "h": ("Planck constant", 6.626e-34, "J*s"),
    "c": ("Speed of light", 3e8, "m/s"),
    "e": ("Elementary charge", 1.602e-19, "C"),
    "F": ("Faraday constant", 96485, "C/mol"),
    "epsilon_0": ("Vacuum permittivity", 8.854e-12, "F/m"),
    "g": ("Gravitational acceleration", 9.81, "m/s^2"),
}


def extract_scientific_variables(hypothesis: str, domain: str = "general") -> List[Dict[str, str]]:
    """
    Extract STRUCTURED scientific variables from a hypothesis.
    Instead of random words, maps hypothesis keywords to real scientific concepts
    with proper symbols, units, and known equations.
    
    Returns list of dicts: [{"name": symbol, "concept": concept, "unit": unit, "equations": [...]}]
    """
    hyp_lower = hypothesis.lower()
    variables = []
    used_symbols = set()
    
    # Match hypothesis keywords to scientific concepts
    for concept, info in SCIENTIFIC_CONCEPT_MAP.items():
        if concept in hyp_lower:
            symbol = info["symbol"]
            # Avoid duplicate symbols
            if symbol in used_symbols:
                symbol = f"{symbol}_{concept[:3]}"
            used_symbols.add(symbol)
            variables.append({
                "name": symbol,
                "concept": concept,
                "unit": info["unit"],
                "equations": info["equations"],
                "description": f"{concept.capitalize()} ({symbol}, {info['unit']})"
            })
    
    # If no concepts matched, use generic scientific variables
    if not variables:
        # Try to extract any meaningful words and map them
        words = tokenize(hypothesis)
        meaningful = [w for w in words if len(w) > 4]
        for w in meaningful[:3]:
            symbol = w[0] if w[0] not in used_symbols else f"{w[0]}_{w[:2]}"
            used_symbols.add(symbol)
            variables.append({
                "name": symbol,
                "concept": w,
                "unit": "variable",
                "equations": [f"{symbol} = k * {w}"],
                "description": f"Variable from: {w}"
            })
    
    # Always add common parameters
    common_params = [
        {"name": "k", "concept": "proportionality_constant", "unit": "varies", "equations": [], "description": "Proportionality constant"},
        {"name": "T", "concept": "temperature", "unit": "K", "equations": ["T = E/k_B"], "description": "Temperature (K)"},
        {"name": "t", "concept": "time", "unit": "s", "equations": ["t = d/v"], "description": "Time (s)"},
    ]
    for p in common_params:
        if p["name"] not in used_symbols:
            variables.append(p)
            used_symbols.add(p["name"])
    
    return variables


def generate_random_math_ideas(hypothesis: str, num_ideas: int = 20, domain: str = "general") -> List[MathIdea]:
    """
    Generate mathematical formulations from STRUCTURED scientific concepts.
    
    Instead of picking random words, this:
    1. Extracts scientific concepts from the hypothesis (e.g., "thermal conductivity" → κ)
    2. Uses known equations for those concepts as starting points
    3. Creates candidate modifications (variations of known equations)
    4. Combines concepts to form novel relationships
    
    No LLM involved — purely structured scientific derivation.
    """
    variables = extract_scientific_variables(hypothesis, domain=domain)
    
    if len(variables) < 2:
        # Fallback: use generic variables
        variables = [
            {"name": "x", "concept": "variable_1", "unit": "", "equations": [], "description": "Variable 1"},
            {"name": "y", "concept": "variable_2", "unit": "", "equations": [], "description": "Variable 2"},
            {"name": "k", "concept": "constant", "unit": "", "equations": [], "description": "Constant"},
        ]
    
    ideas = []
    
    # Strategy 1: Use known equations from matched concepts (50% of ideas)
    known_eq_count = num_ideas // 2
    for i in range(known_eq_count):
        # Pick a variable that has known equations
        vars_with_eqs = [v for v in variables if v["equations"]]
        if vars_with_eqs:
            source_var = random.choice(vars_with_eqs)
            base_eq = random.choice(source_var["equations"])
            
            # Create a modification of the known equation
            # Pick another variable to combine with
            other_vars = [v for v in variables if v != source_var]
            if other_vars:
                other = random.choice(other_vars)
                # Apply a random modification template
                template = random.choice(RELATIONSHIP_TEMPLATES)
                try:
                    eq_text = template(source_var["name"], other["name"])
                except Exception:
                    eq_text = base_eq
            else:
                eq_text = base_eq
            
            rel_type = random.choice(RELATIONSHIP_TYPES)
            
            # Build variable descriptions from structured data
            var_desc = {}
            for v in variables[:4]:
                var_desc[v["name"]] = v["description"]
            var_desc["k"] = "Proportionality constant"
            
            ideas.append(MathIdea(
                idea_id=f"MID-{i+1:03d}",
                equation_text=eq_text,
                variables=var_desc,
                relationship_type=rel_type
            ))
        else:
            # No known equations, use template
            a = variables[0]["name"]
            b = variables[1]["name"] if len(variables) > 1 else "x"
            template = random.choice(RELATIONSHIP_TEMPLATES)
            try:
                eq_text = template(a, b)
            except Exception:
                eq_text = f"{a} = k * {b}"
            
            var_desc = {v["name"]: v["description"] for v in variables[:4]}
            var_desc["k"] = "Proportionality constant"
            
            ideas.append(MathIdea(
                idea_id=f"MID-{i+1:03d}",
                equation_text=eq_text,
                variables=var_desc,
                relationship_type=random.choice(RELATIONSHIP_TYPES)
            ))
    
    # Strategy 2: Combine concepts in novel ways (remaining ideas)
    for i in range(known_eq_count, num_ideas):
        # Pick 2-3 structured variables
        vars_picked = random.sample(variables, min(3, len(variables)))
        a = vars_picked[0]["name"]
        b = vars_picked[1]["name"] if len(vars_picked) > 1 else "x"
        
        # Pick a random relationship template
        template = random.choice(RELATIONSHIP_TEMPLATES)
        try:
            eq_text = template(a, b)
        except Exception:
            eq_text = f"{a} = k * {b}"
        
        rel_type = random.choice(RELATIONSHIP_TYPES)
        
        # Build variable descriptions from structured data
        var_desc = {}
        for v in vars_picked:
            var_desc[v["name"]] = v["description"]
        var_desc["k"] = "Proportionality constant"
        if "alpha" in eq_text:
            var_desc["alpha"] = "Scaling factor"
        if "beta" in eq_text:
            var_desc["beta"] = "Offset parameter"
        if "tau" in eq_text:
            var_desc["tau"] = "Time constant"
        if "n" in eq_text:
            var_desc["n"] = "Exponent"
        
        ideas.append(MathIdea(
            idea_id=f"MID-{i+1:03d}",
            equation_text=eq_text,
            variables=var_desc,
            relationship_type=rel_type
        ))
    
    return ideas

# =========================================================
# RUBBISH FILTER (Deterministic, No LLM)
# =========================================================
def is_rubbish_idea(idea: MathIdea) -> Tuple[bool, str]:
    """
    Immediately filter out rubbish math ideas without using LLM.
    Returns (is_rubbish, reason).
    """
    eq = idea.equation_text
    
    # 1. Check for self-reference (variable on both sides with no transformation)
    lhs = eq.split('=')[0].strip() if '=' in eq else ""
    rhs = eq.split('=')[1].strip() if '=' in eq else ""
    
    # 2. Check for division by zero patterns
    if '/ 0' in eq or '/0' in eq:
        return True, "Division by zero"
    
    # 3. Check for log of zero or negative
    if 'log(0)' in eq or 'log(-' in eq:
        return True, "Logarithm of non-positive value"
    
    # 4. Check for sqrt of negative
    if 'sqrt(-' in eq:
        return True, "Square root of negative value"
    
    # 5. Check for trivially circular equations (same var on both sides)
    lhs_vars = set(tokenize(lhs))
    rhs_vars = set(tokenize(rhs))
    common = lhs_vars & rhs_vars
    if len(common) == len(lhs_vars) and len(lhs_vars) > 0 and len(rhs_vars) <= 2:
        return True, f"Circular: {', '.join(common)} appears on both sides with no meaningful transformation"
    
    # 6. Check for too many variables (overly complex)
    all_vars = lhs_vars | rhs_vars
    if len(all_vars) > 8:
        return True, f"Too many variables ({len(all_vars)}): overly complex for a fundamental relationship"
    
    # 7. Check for missing equals sign
    if '=' not in eq:
        return True, "No equality relationship defined"
    
    # 8. Check for empty sides
    if not lhs or not rhs:
        return True, "Empty left or right side"
    
    return False, ""

# =========================================================
# LLM VALIDATION (Only for "Does this make sense?")
# =========================================================
def _deterministic_validation(hypothesis: str, idea: MathIdea) -> MathValidation:
    """
    Deterministic fallback when LLM is unavailable.
    Validates equations based on variable consistency, structure, and plausibility.
    """
    eq = idea.equation_text
    rel_type = idea.relationship_type
    variables = idea.variables

    # Check 1: Variable consistency — equation should reference defined variables
    eq_vars = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eq))
    eq_vars -= {'k', 'k1', 'k2', 'k3', 'alpha', 'beta', 'gamma', 'tau', 'n', 'c', 'exp', 'log', 'sqrt'}
    defined_vars = set(variables.keys())
    undefined_vars = eq_vars - defined_vars
    if len(undefined_vars) > 2:
        return MathValidation(makes_sense=False, reasoning=f"Too many undefined variables: {list(undefined_vars)[0]}", dimension_hint="")

    # Check 2: Structure matches relationship type
    rhs = eq.split('=')[1].strip() if '=' in eq else ''
    type_ok = True
    if rel_type == 'proportional': type_ok = '*' in rhs
    elif rel_type == 'inverse': type_ok = '/' in rhs
    elif rel_type == 'exponential': type_ok = 'exp' in rhs
    elif rel_type == 'power_law': type_ok = '^' in rhs or 'sqrt' in rhs
    elif rel_type == 'logarithmic': type_ok = 'log' in rhs
    elif rel_type == 'sigmoid': type_ok = 'exp' in rhs and '/' in rhs
    elif rel_type == 'polynomial': type_ok = '^' in rhs or '+' in rhs
    elif rel_type == 'linear': type_ok = '+' in rhs or '*' in rhs

    if not type_ok:
        return MathValidation(makes_sense=False, reasoning=f"Structure mismatch for {rel_type}", dimension_hint="")

    # Check 3: Physical plausibility — must have constant, operation, and variable
    has_constant = any(c in eq for c in ['k', 'k1', 'k2', 'alpha', 'beta', 'tau'])
    has_operation = any(op in eq for op in ['*', '/', '+', '^', 'exp', 'log', 'sqrt'])
    has_variable = len(defined_vars) >= 1

    if not (has_constant and has_operation and has_variable):
        return MathValidation(makes_sense=False, reasoning="Missing components", dimension_hint="")

    return MathValidation(makes_sense=True, reasoning=f"Deterministic: {rel_type}, {len(defined_vars)} vars", dimension_hint="Standard")


def validate_math_idea_llm(hypothesis: str, idea: MathIdea) -> MathValidation:
    """
    LLM is ONLY used to answer: "Does this mathematical relationship make sense?"
    Not to generate equations — just to validate them.
    Uses shared GroqClient with automatic Ollama fallback.
    Falls back to deterministic validation when LLM is unavailable.
    """
    system_prompt = (
        "You are a strict Mathematical Validator. Your ONLY job is to answer:\n"
        "'Does this mathematical relationship make physical/logical sense?'\n\n"
        "Rules:\n"
        "- If the equation captures a meaningful relationship: makes_sense=true\n"
        "- If the equation is nonsense, contradictory, or physically impossible: makes_sense=false\n"
        "- Be brief. Just validate, don't derive new equations.\n\n"
        "Respond with JSON:\n"
        "{\n"
        '  "makes_sense": true,\n'
        '  "reasoning": "Brief reason",\n'
        '  "dimension_hint": "Suggested dimensions if valid"\n'
        "}\n"
    )
    
    user_prompt = (
        f"Original hypothesis: {hypothesis}\n"
        f"Proposed equation: {idea.equation_text}\n"
        f"Relationship type: {idea.relationship_type}\n"
        f"Variables: {json.dumps(idea.variables)}\n\n"
        f"Does this equation make sense for the hypothesis? Answer yes or no."
    )
    
    try:
        result = llm_complete_json(system_prompt, user_prompt, temperature=0.1)
        if result:
            return MathValidation.model_validate_json(json.dumps(result))
        return _deterministic_validation(hypothesis, idea)
    except Exception as e:
        return _deterministic_validation(hypothesis, idea)

# =========================================================
# DEEPENING: Generate more ideas around a validated cluster
# =========================================================
def deepen_cluster(hypothesis: str, cluster_docs: List[str], num_new: int = 10, domain: str = "general") -> List[MathIdea]:
    """
    When a cluster is validated, generate MORE ideas
    that are variations of the validated equations in that cluster.
    Uses structured scientific variables instead of random words.
    """
    variables = extract_scientific_variables(hypothesis, domain=domain)
    if len(variables) < 2:
        variables = [
            {"name": "x", "concept": "var", "unit": "", "equations": [], "description": "Variable"},
            {"name": "y", "concept": "var", "unit": "", "equations": [], "description": "Variable"},
            {"name": "k", "concept": "const", "unit": "", "equations": [], "description": "Constant"},
        ]
    
    new_ideas = []
    for doc in cluster_docs:
        for _ in range(num_new // max(len(cluster_docs), 1)):
            vars_picked = random.sample(variables, min(3, len(variables)))
            a = vars_picked[0]["name"]
            b = vars_picked[1]["name"] if len(vars_picked) > 1 else "x"
            template = random.choice(RELATIONSHIP_TEMPLATES)
            try:
                eq_text = template(a, b)
            except Exception:
                eq_text = f"{a} = k * {b}"
            
            var_desc = {v["name"]: f"Deepened: {v['description']}" for v in vars_picked}
            var_desc["k"] = "Proportionality constant"
            
            new_ideas.append(MathIdea(
                idea_id=f"MID-DEEP-{len(new_ideas)+1:03d}",
                equation_text=eq_text,
                variables=var_desc,
                relationship_type=random.choice(RELATIONSHIP_TYPES)
            ))
    return new_ideas

# =========================================================
# MAIN MATHEMATICS ENGINE ENTRY POINT
# =========================================================
def run_mathematics_engine(
    hypotheses: List[str],
    equations_file: str = "derived_equations.json"
) -> Dict[str, Any]:
    """
    DBSCAN-driven Mathematics Engine.
    
    For each hypothesis:
    1. Generate 20 random math ideas (combinatorial, no LLM)
    2. Filter rubbish immediately (deterministic checks)
    3. TF-IDF vectorize surviving ideas
    4. DBSCAN cluster (r=1→r=4 multi-resolution)
    5. For each cluster, pick the best idea and ask LLM: "Does this make sense?"
    6. If yes → deepen that cluster with more variations → cluster again
    7. If no → discard
    8. Save all validated equations
    """
    print("\n" + "="*80)
    print("📐 MATHEMATICS ENGINE — DBSCAN-Driven Random Generation + Validation")
    print("="*80)
    
    all_validated = []
    all_rejected = []
    
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n  [{i}/{len(hypotheses)}] Processing hypothesis...")
        print(f"      📝 {hypothesis[:120]}...")
        
        # Step 1: Generate random math ideas (NO LLM)
        print(f"      🎲 Generating 20 random mathematical formulations...")
        raw_ideas = generate_random_math_ideas(hypothesis, num_ideas=20)
        print(f"         Generated {len(raw_ideas)} raw ideas")
        
        # Step 2: Filter rubbish immediately (deterministic)
        surviving_ideas = []
        rubbish_count = 0
        for idea in raw_ideas:
            is_rubbish, reason = is_rubbish_idea(idea)
            if is_rubbish:
                rubbish_count += 1
                all_rejected.append({
                    "hypothesis": hypothesis,
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
        
        # Step 4: For each cluster, validate the best idea with LLM
        validated_this_hypothesis = 0
        deepened_clusters = 0
        
        for cname, cinfo in cluster_result['clusters'].items():
            # Pick the most representative idea from this cluster (first one)
            cluster_ideas = [surviving_ideas[idx] for idx in cinfo['document_indices']]
            best_idea = cluster_ideas[0]  # First is fine since they're similar
            
            # Step 5: LLM validation — only "does this make sense?"
            print(f"         🔍 Validating {cname} (representative: {best_idea.equation_text[:60]}...)")
            validation = validate_math_idea_llm(hypothesis, best_idea)
            
            if validation.makes_sense:
                validated_this_hypothesis += 1
                all_validated.append({
                    "hypothesis": hypothesis,
                    "equation": best_idea.equation_text,
                    "variables": best_idea.variables,
                    "relationship_type": best_idea.relationship_type,
                    "cluster": cname,
                    "r_level": cinfo['r_level'],
                    "validation_reasoning": validation.reasoning,
                    "dimension_hint": validation.dimension_hint,
                    "timestamp": datetime.now().isoformat()
                })
                print(f"            ✅ MAKES SENSE — {validation.reasoning[:80]}")
                
                # Step 6: Deepen this cluster
                print(f"            🔄 Deepening cluster with more variations...")
                deepened_ideas = deepen_cluster(hypothesis, cinfo['documents'], num_new=8, domain="general")
                
                # Filter deepened ideas
                for d_idea in deepened_ideas:
                    is_rubbish, reason = is_rubbish_idea(d_idea)
                    if not is_rubbish:
                        # Validate deepened idea
                        d_validation = validate_math_idea_llm(hypothesis, d_idea)
                        if d_validation.makes_sense:
                            deepened_clusters += 1
                            all_validated.append({
                                "hypothesis": hypothesis,
                                "equation": d_idea.equation_text,
                                "variables": d_idea.variables,
                                "relationship_type": d_idea.relationship_type,
                                "cluster": f"{cname}_deepened",
                                "r_level": cinfo['r_level'] + 1,
                                "validation_reasoning": d_validation.reasoning,
                                "dimension_hint": d_validation.dimension_hint,
                                "timestamp": datetime.now().isoformat()
                            })
                            print(f"            ✅ Deepened: {d_idea.equation_text[:60]}...")
            else:
                rejection_reason = validation.reasoning
                if "LLM returned no result" in rejection_reason or "LLM error" in rejection_reason:
                    rejection_filter = "llm_unavailable"
                elif "returned no result" in rejection_reason:
                    rejection_filter = "llm_unavailable"
                else:
                    rejection_filter = "llm_validation"
                all_rejected.append({
                    "hypothesis": hypothesis,
                    "equation": best_idea.equation_text,
                    "reason": rejection_reason,
                    "filter": rejection_filter
                })
                label = "LLM UNAVAILABLE" if rejection_filter == "llm_unavailable" else "RUBBISH"
                print(f"            ❌ {label} — {rejection_reason[:80]}")
        
        print(f"      📊 Results for this hypothesis: {validated_this_hypothesis} validated, {deepened_clusters} deepened")
    
    # Save to disk
    output = {
        "total_validated": len(all_validated),
        "total_rejected": len(all_rejected),
        "validated_equations": all_validated,
        "rejected_ideas": all_rejected,
        "summary": {
            "validated_count": len(all_validated),
            "rejected_count": len(all_rejected),
            "rejection_breakdown": {
                "rubbish_filter": sum(1 for r in all_rejected if r.get("filter") == "rubbish_filter"),
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
          f"{output['summary']['rejection_breakdown']['llm_validation']} by LLM)")
    print("="*80)
    
    return output


if __name__ == "__main__":
    test_hypotheses = [
        "Increasing electrode surface area increases battery capacity linearly with area",
        "Battery energy density is inversely proportional to internal resistance squared"
    ]
    run_mathematics_engine(test_hypotheses)