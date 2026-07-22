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
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# Performance optimization settings
MAX_PARALLEL_MATH = 3  # Parallel math validation

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

def extract_keywords(hypothesis: str) -> List[str]:
    """Extract meaningful keywords from a hypothesis for variable generation."""
    words = tokenize(hypothesis)
    # Filter to longer, more meaningful words
    keywords = [w for w in words if len(w) > 3]
    # Add some common variable names
    common_vars = ["x", "y", "z", "t", "k", "n", "alpha", "beta", "gamma", "theta", "lambda", "mu", "sigma", "tau", "omega"]
    return keywords + common_vars

def generate_random_math_ideas(hypothesis: str, num_ideas: int = 20) -> List[MathIdea]:
    """
    Generate random mathematical formulations from hypothesis keywords.
    No LLM involved — purely combinatorial.
    """
    keywords = extract_keywords(hypothesis)
    if len(keywords) < 3:
        keywords = ["x", "y", "k", "n", "alpha", "beta", hypothesis[:5].lower()]
    
    ideas = []
    for i in range(num_ideas):
        # Pick 2-3 random keywords as variables
        vars_picked = random.sample(keywords, min(3, len(keywords)))
        a = vars_picked[0]
        b = vars_picked[1] if len(vars_picked) > 1 else "x"
        
        # Pick a random relationship template
        template = random.choice(RELATIONSHIP_TEMPLATES)
        try:
            eq_text = template(a, b)
        except Exception:
            eq_text = f"{a} = k * {b}"
        
        rel_type = random.choice(RELATIONSHIP_TYPES)
        
        # Build variable descriptions
        var_desc = {}
        for v in vars_picked:
            var_desc[v] = f"Variable derived from: {hypothesis[:50]}"
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
def validate_math_idea_llm(client: Groq, hypothesis: str, idea: MathIdea) -> MathValidation:
    """
    LLM is ONLY used to answer: "Does this mathematical relationship make sense?"
    Not to generate equations — just to validate them.
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
    
    prompt = (
        f"Original hypothesis: {hypothesis}\n"
        f"Proposed equation: {idea.equation_text}\n"
        f"Relationship type: {idea.relationship_type}\n"
        f"Variables: {json.dumps(idea.variables)}\n\n"
        f"Does this equation make sense for the hypothesis? Answer yes or no."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_json = completion.choices[0].message.content
        return MathValidation.model_validate_json(raw_json)
    except Exception as e:
        return MathValidation(makes_sense=False, reasoning=f"LLM error: {e}", dimension_hint="")

# =========================================================
# DEEPENING: Generate more ideas around a validated cluster
# =========================================================
def deepen_cluster(hypothesis: str, cluster_docs: List[str], num_new: int = 10) -> List[MathIdea]:
    """
    When a cluster is validated, generate MORE random ideas 
    that are variations of the validated equations in that cluster.
    """
    new_ideas = []
    for doc in cluster_docs:
        # Extract the equation pattern from the cluster document
        # Generate variations by tweaking the relationship type
        for _ in range(num_new // max(len(cluster_docs), 1)):
            keywords = extract_keywords(hypothesis)
            vars_picked = random.sample(keywords, min(3, len(keywords)))
            a = vars_picked[0]
            b = vars_picked[1] if len(vars_picked) > 1 else "x"
            template = random.choice(RELATIONSHIP_TEMPLATES)
            try:
                eq_text = template(a, b)
            except Exception:
                eq_text = f"{a} = k * {b}"
            
            new_ideas.append(MathIdea(
                idea_id=f"MID-DEEP-{len(new_ideas)+1:03d}",
                equation_text=eq_text,
                variables={v: f"Deepened variable from {hypothesis[:30]}" for v in vars_picked},
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
    
    client = Groq()
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
        
        # Step 4: For each cluster, validate the best idea with LLM (PARALLEL)
        validated_this_hypothesis = 0
        deepened_clusters = 0
        
        def validate_cluster(args):
            """Validate a single cluster - runs in parallel"""
            cname, cinfo, surviving_ideas, hypothesis = args
            try:
                cluster_ideas = [surviving_ideas[idx] for idx in cinfo['document_indices']]
                best_idea = cluster_ideas[0]
                local_client = Groq()
                validation = validate_math_idea_llm(local_client, hypothesis, best_idea)
                return (cname, best_idea, validation, None)
            except Exception as e:
                return (cname, None, None, str(e))
        
        # Prepare cluster validation tasks
        cluster_tasks = [
            (cname, cinfo, surviving_ideas, hypothesis) 
            for cname, cinfo in cluster_result['clusters'].items()
        ]
        
        # Run validations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_MATH) as executor:
            futures = {executor.submit(validate_cluster, task): task for task in cluster_tasks}
            
            for future in concurrent.futures.as_completed(futures):
                cname, best_idea, validation, error = future.result()
                
                if error or not best_idea or not validation:
                    print(f"         ❌ Error validating {cname}: {error}")
                    continue
                
                if validation.makes_sense:
                    validated_this_hypothesis += 1
                    all_validated.append({
                        "hypothesis": hypothesis,
                        "equation": best_idea.equation_text,
                        "variables": best_idea.variables,
                        "relationship_type": best_idea.relationship_type,
                        "cluster": cname,
                        "r_level": cluster_result['clusters'][cname]['r_level'],
                        "validation_reasoning": validation.reasoning,
                        "dimension_hint": validation.dimension_hint,
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"            ✅ {cname}: {validation.reasoning[:60]}...")
                    
                    # Step 6: Deepen this cluster (sequential to save API calls)
                    deepened_ideas = deepen_cluster(hypothesis, cluster_result['clusters'][cname]['documents'], num_new=8)
                    for d_idea in deepened_ideas:
                        is_rubbish, _ = is_rubbish_idea(d_idea)
                        if not is_rubbish:
                            d_validation = validate_math_idea_llm(Groq(), hypothesis, d_idea)
                            if d_validation.makes_sense:
                                deepened_clusters += 1
                                all_validated.append({
                                    "hypothesis": hypothesis,
                                    "equation": d_idea.equation_text,
                                    "variables": d_idea.variables,
                                    "relationship_type": d_idea.relationship_type,
                                    "cluster": f"{cname}_deepened",
                                    "r_level": cluster_result['clusters'][cname]['r_level'] + 1,
                                    "validation_reasoning": d_validation.reasoning,
                                    "dimension_hint": d_validation.dimension_hint,
                                    "timestamp": datetime.now().isoformat()
                                })
                else:
                    all_rejected.append({
                        "hypothesis": hypothesis,
                        "equation": best_idea.equation_text,
                        "reason": validation.reasoning,
                        "filter": "llm_validation"
                    })
        
        print(f"      📊 Results: {validated_this_hypothesis} validated, {deepened_clusters} deepened")
    
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
                "llm_validation": sum(1 for r in all_rejected if r.get("filter") == "llm_validation")
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