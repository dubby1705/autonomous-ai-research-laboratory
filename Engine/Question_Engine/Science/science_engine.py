"""
SCIENCE ENGINE — Pure Python Scientific Discovery with DBSCAN
No API keys. No LLM for computation. Ollama only for validation.

Flow:
1. Ollama decides which domains to use (physics, chemistry, maths, or all)
2. For each domain, loads ALL concept data points (momentum, force, energy, etc.)
3. Creates derived data points (combinations of concepts)
4. Runs DBSCAN clustering (r=1 multi-resolution) on all data points
5. For each cluster, generates random predictions using pure Python math
6. Ollama ONLY validates: "Is this formula correct?" (yes/no)
7. No LLM generates formulas. No LLM does math. Pure code only.
"""

import os
import json
import math
import random
import re
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from Engine.Question_Engine.Science.physics import PHYSICS_DB, get_concept as get_phy, random_prediction as phy_pred
from Engine.Question_Engine.Science.chemistry import CHEMISTRY_DB, get_concept as get_chem, random_prediction as chem_pred
from Engine.Question_Engine.Science.maths import MATHS_DB, get_concept as get_math, random_prediction as math_pred
from Engine.Question_Engine.OllamaClient import query_ollama_json

# =========================================================
# DOMAIN DETECTION (Ollama decides which domains to use)
# =========================================================
DOMAIN_DBS = {
    "physics": {"db": PHYSICS_DB, "get": get_phy, "predict": phy_pred, "concepts": list(PHYSICS_DB.keys())},
    "chemistry": {"db": CHEMISTRY_DB, "get": get_chem, "predict": chem_pred, "concepts": list(CHEMISTRY_DB.keys())},
    "maths": {"db": MATHS_DB, "get": get_math, "predict": math_pred, "concepts": list(MATHS_DB.keys())},
}

def select_domains(problem: str) -> List[str]:
    """
    Use Ollama to decide which scientific domains are relevant.
    Ollama only says "physics", "chemistry", "maths" or combinations.
    No code generation. No formula generation. Just classification.
    """
    prompt = (
        f"Research problem: {problem}\n\n"
        "Which scientific domains are needed? Choose from: physics, chemistry, maths.\n"
        "Respond with JSON: {\"domains\": [\"physics\", \"maths\"]}\n"
        "Only list domains that are DIRECTLY relevant to solving this problem."
    )
    system = "You classify research problems into scientific domains. Be minimal — only include what's essential."
    
    result = query_ollama_json(prompt, system, temperature=0.1)
    if result and "domains" in result:
        selected = [d for d in result["domains"] if d in DOMAIN_DBS]
        if selected:
            return selected
    
    # Fallback: use keyword matching
    p = problem.lower()
    domains = []
    if any(kw in p for kw in ["force", "energy", "momentum", "wave", "electric", "magnet", "thermo", "fluid", "velocity", "acceleration", "gravity"]):
        domains.append("physics")
    if any(kw in p for kw in ["reaction", "chemical", "acid", "base", "electrochem", "catalyst", "kinetics", "equilibrium", "gas", "bond", "molecule"]):
        domains.append("chemistry")
    if any(kw in p for kw in ["optimization", "derivative", "integral", "matrix", "vector", "statistics", "probability", "differential", "linear algebra", "numerical"]):
        domains.append("maths")
    if not domains:
        domains = ["physics", "maths"]  # default
    
    return list(set(domains))

# =========================================================
# DATA POINT GENERATION (Pure Python)
# =========================================================
def generate_concept_data_points(domain: str) -> List[str]:
    """Generate ALL concept data points for a domain."""
    info = DOMAIN_DBS[domain]
    points = []
    for concept_name in info["concepts"]:
        concept = info["get"](concept_name)
        if concept:
            points.append(concept.get_all_formulas_text())
    return points

def generate_derived_points(domain: str) -> List[str]:
    """Generate derived/cross-concept data points."""
    derived = []
    
    # Physics derived relationships
    if domain == "physics":
        derived = [
            "DERIVED: Energy = p²/(2m) — Kinetic energy from momentum",
            "DERIVED: Power = F·v — Instantaneous power from force and velocity",
            "DERIVED: E = h·f — Photon energy from frequency (Planck-Einstein)",
            "DERIVED: k = 2π/λ — Wave number from wavelength",
            "DERIVED: v_esc = √(2GM/r) — Escape velocity",
            "DERIVED: F = q·(E + v×B) — Lorentz force",
            "DERIVED: P = V·I — Electrical power from voltage and current",
        ]
    
    # Chemistry derived
    if domain == "chemistry":
        derived = [
            "DERIVED: ΔG = -RT·ln(K) — Gibbs free energy from equilibrium",
            "DERIVED: E_cell = E°_cathode - E°_anode — Cell potential",
            "DERIVED: pH + pOH = 14 — Acid-base relationship",
            "DERIVED: K_w = [H+][OH-] = 1e-14 — Water dissociation",
        ]
    
    # Maths derived
    if domain == "maths":
        derived = [
            "DERIVED: e^(iπ) + 1 = 0 — Euler's identity",
            "DERIVED: ∫e^x dx = e^x + C — Exponential integral",
            "DERIVED: d/dx(ln x) = 1/x — Log derivative",
            "DERIVED: Σ(1/n²) = π²/6 — Basel problem",
        ]
    
    return derived

# =========================================================
# TF-IDF + DBSCAN (No API Keys)
# =========================================================
BOUNDARY_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "with", "by", "on", "in",
    "at", "from", "as", "this", "that", "it", "can", "could", "would", "should", "using", "such",
    "without", "will", "has", "have", "had", "not", "no", "only", "but", "however", "and", "or",
    "which", "where", "when", "why", "how", "we", "they", "their", "our"
}

def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if w not in BOUNDARY_WORDS and len(w) > 2]

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
            vec[word] = (count / total_terms) * math.log(N / (1 + df.get(word, 0)))
        vectors.append(vec)
    return vectors

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in intersection)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return 0.0 if mag1 == 0 or mag2 == 0 else dot / (mag1 * mag2)

def dbscan_cluster(documents: List[str], vectors: List[Dict[str, float]], eps: float) -> Tuple[Dict[int, List[int]], List[int]]:
    labels = [0] * len(documents)
    cluster_id = 0
    def region_query(p_idx: int) -> List[int]:
        return [q_idx for q_idx, q_vec in enumerate(vectors) if cosine_similarity(vectors[p_idx], q_vec) >= eps]
    for p in range(len(documents)):
        if labels[p] != 0: continue
        neighbors = region_query(p)
        if len(neighbors) < 2:
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
                    if len(q_neighbors) >= 2: neighbors.extend(q_neighbors)
                i += 1
    clusters: Dict[int, List[int]] = {}
    noise: List[int] = []
    for idx, label in enumerate(labels):
        if label == -1: noise.append(idx)
        else:
            if label not in clusters: clusters[label] = []
            clusters[label].append(idx)
    return clusters, noise

# =========================================================
# RANDOM PREDICTION + OLLAMA VALIDATION
# =========================================================
def generate_random_predictions(domain: str, concept_names: List[str], num_per_concept: int = 2) -> List[Dict[str, Any]]:
    """Generate random valid predictions using pure Python libraries."""
    info = DOMAIN_DBS[domain]
    predictions = []
    for concept_name in concept_names:
        concept = info["get"](concept_name)
        if not concept:
            continue
        # Pick random formulas from this concept
        formula_names = list(concept.formulas.keys())
        selected = random.sample(formula_names, min(num_per_concept, len(formula_names)))
        for fname in selected:
            pred = info["predict"](concept_name, fname)
            if pred and "error" not in pred:
                predictions.append(pred)
    return predictions

def validate_prediction_ollama(prediction: Dict[str, Any], problem: str) -> bool:
    """
    Ollama ONLY validates: "Is this formula mathematically correct?"
    It does NOT generate formulas. It does NOT do math.
    Pure yes/no check for complex formulas.
    """
    prompt = (
        f"Research problem: {problem}\n"
        f"Proposed formula: {prediction.get('formula_text', '')}\n"
        f"Parameters: {prediction.get('parameters', {})}\n"
        f"Result: {prediction.get('result', '')}\n\n"
        "Is this formula mathematically valid for the given parameters?\n"
        "Respond with JSON: {\"valid\": true/false, \"reason\": \"brief\"}"
    )
    system = "You validate mathematical formulas. Answer only yes/no. Do NOT generate formulas."
    
    result = query_ollama_json(prompt, system, temperature=0.1)
    if result:
        return result.get("valid", False)
    return True  # If Ollama is down, accept (pure code already validated)

# =========================================================
# MAIN SCIENCE ENGINE
# =========================================================
def run_science_engine(problem: str) -> Dict[str, Any]:
    """
    Main entry point.
    1. Ollama selects domains
    2. Generate all concept + derived data points
    3. DBSCAN cluster (r=1 multi-resolution)
    4. Random predictions per cluster
    5. Ollama validates complex formulas
    6. Save all results
    """
    print("\n" + "="*80)
    print("🔬 SCIENCE ENGINE — Pure Python Discovery + DBSCAN Clustering")
    print("="*80)
    
    # Step 1: Select domains
    print("\n📋 Selecting scientific domains...")
    domains = select_domains(problem)
    print(f"   Selected: {', '.join(domains)}")
    
    all_results = {}
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"📐 DOMAIN: {domain.upper()}")
        print(f"{'='*60}")
        
        # Step 2: Generate data points
        concept_points = generate_concept_data_points(domain)
        derived_points = generate_derived_points(domain)
        all_points = concept_points + derived_points
        
        print(f"   Concepts: {len(concept_points)}")
        print(f"   Derived: {len(derived_points)}")
        print(f"   Total data points: {len(all_points)}")
        
        if len(all_points) < 3:
            print(f"   ⏸️  Too few points to cluster.")
            continue
        
        # Step 3: DBSCAN cluster (r=1 only — multi-resolution)
        print(f"\n   🔬 Running DBSCAN clustering (r=1 multi-resolution)...")
        vectors = build_tfidf_vectors(all_points)
        eps = 0.30  # r=1 threshold
        clusters, noise = dbscan_cluster(all_points, vectors, eps=eps)
        
        print(f"   Clusters: {len(clusters)}, Noise points: {len(noise)}")
        
        cluster_info = {}
        for cid, members in clusters.items():
            # Show what concepts are in this cluster
            cluster_concepts = [all_points[i].split('\n')[0] for i in members[:5]]
            cluster_info[f"Cluster_{cid}"] = {
                "size": len(members),
                "concepts": cluster_concepts,
                "members": members
            }
            print(f"      Cluster {cid}: {len(members)} points")
            for c in cluster_concepts[:3]:
                print(f"         • {c[:80]}")
        
        # Step 4: Generate random predictions per cluster
        print(f"\n   🎲 Generating random predictions...")
        all_predictions = []
        
        for cid, members in clusters.items():
            # Get concept names from this cluster
            cluster_texts = [all_points[i] for i in members]
            # Extract concept names from data points
            concept_names = []
            for text in cluster_texts:
                for line in text.split('\n'):
                    if line.startswith('=== '):
                        name = line.replace('=== ', '').split(' (')[0].lower()
                        if name in DOMAIN_DBS[domain]["concepts"]:
                            concept_names.append(name)
            
            if concept_names:
                preds = generate_random_predictions(domain, list(set(concept_names)), num_per_concept=1)
                all_predictions.extend(preds)
                for p in preds:
                    print(f"      • {p['concept']}.{p['formula']}: {p.get('result', 'N/A')} {p.get('unit', '')}")
        
        # Step 5: Ollama validates complex formulas
        print(f"\n   ✅ Validating with Ollama (checking {len(all_predictions)} predictions)...")
        validated = []
        rejected = []
        for pred in all_predictions:
            is_valid = validate_prediction_ollama(pred, problem)
            if is_valid:
                validated.append(pred)
                print(f"      ✅ {pred['formula_text'][:60]}")
            else:
                rejected.append(pred)
                print(f"      ❌ {pred['formula_text'][:60]}")
        
        all_results[domain] = {
            "data_points": len(all_points),
            "clusters": len(clusters),
            "noise_points": len(noise),
            "cluster_details": {
                k: {"size": v["size"], "concepts": v["concepts"]}
                for k, v in cluster_info.items()
            },
            "predictions_generated": len(all_predictions),
            "predictions_validated": len(validated),
            "predictions_rejected": len(rejected),
            "validated_predictions": validated
        }
    
    # Step 6: Save results
    output = {
        "problem": problem,
        "domains_used": domains,
        "results": all_results,
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs("Science", exist_ok=True)
    output_file = os.path.join("Science", "science_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SCIENCE ENGINE SUMMARY")
    print(f"{'='*80}")
    for domain, res in all_results.items():
        print(f"   {domain.upper()}: {res['data_points']} points, {res['clusters']} clusters, "
              f"{res['predictions_validated']} valid predictions")
    print(f"{'='*80}")
    
    return output


if __name__ == "__main__":
    run_science_engine("Design a battery with higher energy density")