import os
import json
import random
import math
import re
import concurrent.futures
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set
import threading # Added for thread-safe printing

# Lock to prevent overlapping prints in the console when multi-threading
print_lock = threading.Lock()

# =========================================================
# LAYER 1: SCIENTIFIC CONCEPT EXTRACTION (ONTOLOGY)
# =========================================================
BOUNDARY_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "with", "by", "on", "in", 
    "at", "from", "as", "this", "that", "it", "can", "could", "would", "should", "using", "such", 
    "without", "will", "has", "have", "had", "not", "no", "only", "but", "however", "and", "or", 
    "which", "where", "when", "why", "how", "we", "they", "their", "our", "show", "shows", 
    "demonstrate", "compare", "search", "find", "found", "use", "used", "requires", "assumes",
    "must", "always", "during", "between", "through", "under", "over", "into"
}

def extract_scientific_concepts(text: str) -> List[str]:
    words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
    concepts = []
    current_concept = []
    
    for word in words:
        if word in BOUNDARY_WORDS or len(word) < 3:
            if current_concept:
                concept_str = " ".join(current_concept)
                if len(current_concept) > 1 or (len(current_concept) == 1 and len(concept_str) > 4):
                    concepts.append(concept_str)
                current_concept = []
        else:
            current_concept.append(word)
            
    if current_concept:
        concepts.append(" ".join(current_concept))
        
    return list(set([c.title() for c in concepts]))

# =========================================================
# LAYER 2: MATHEMATICAL VECTORIZATION (For Clustering)
# =========================================================
def tokenize_for_vectors(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if w not in BOUNDARY_WORDS]

def build_tfidf_vectors(documents: List[str]) -> List[Dict[str, float]]:
    doc_tokens = [tokenize_for_vectors(doc) for doc in documents]
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
# LAYER 3: CORE DBSCAN CLUSTERING ENGINE
# =========================================================
def dbscan_text_cluster(documents: List[str], vectors: List[Dict[str, float]], eps: float, min_pts: int = 1) -> Tuple[List[Dict[str, Any]], List[str]]:
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

    clusters, noise = {}, []
    for idx, label in enumerate(labels):
        if label == -1: noise.append(documents[idx])
        else:
            if label not in clusters: clusters[label] = []
            clusters[label].append(documents[idx])
    return [{"cluster_name": f"Node_{cid}", "items": items} for cid, items in clusters.items()], noise

# =========================================================
# LAYER 4: SCIENTIFIC KNOWLEDGE GRAPH & VALIDATION
# =========================================================
RELATIONSHIP_TYPES = [
    "Dependency", "Cause", "Effect", "Improvement", "Limitation", 
    "Trade-off", "Extension", "Generalization", "Specialization", 
    "Inspiration", "Analogy", "Contradiction", "Complementary"
]
WEIGHTS_FILE = "doscan_exploration_weights.json"

def load_exploration_weights() -> Dict[str, float]:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {rel: 1.0 for rel in RELATIONSHIP_TYPES}

def save_exploration_weights(weights: Dict[str, float]):
    with open(WEIGHTS_FILE, "w") as f: json.dump(weights, f, indent=4)

global_telemetry = {
    "concepts_extracted": set(), "kg_edges_generated": 0, "candidates_rejected": 0,
    "hypotheses_deferred": 0, "accepted_insights": 0
}

class ScientificOntologyReasoner:
    def __init__(self, full_kb: List[str]):
        self.full_kb = full_kb
        self.strategy_weights = load_exploration_weights()
        self.conflict_markers = {"limit", "fail", "cost", "vs", "versus", "conflict", "trade-off", "sacrifice", "but", "however", "degrade"}
        self.support_markers = {"improve", "cause", "lead", "allow", "depend", "require", "increase", "show", "demonstrate", "yield"}

    def _mine_empirical_evidence(self, c1: str, c2: str) -> Dict[str, List[str]]:
        support, conflict = [], []
        c1_low, c2_low = c1.lower(), c2.lower()
        
        for kb_item in self.full_kb:
            kb_low = kb_item.lower()
            if c1_low in kb_low and c2_low in kb_low:
                if any(m in kb_low for m in self.conflict_markers):
                    if len(conflict) < 2: conflict.append(kb_item)
                elif any(m in kb_low for m in self.support_markers):
                    if len(support) < 2: support.append(kb_item)
                else:
                    if len(support) < 2: support.append(kb_item)
                    
        return {"support": support, "conflict": conflict}

    def evaluate_scientific_candidate(self, c1: str, c2: str, rel_type: str, cluster_name: str) -> Optional[Dict[str, Any]]:
        evidence = self._mine_empirical_evidence(c1, c2)
        
        # LOGGING 1: Deferred Hypotheses (No evidence)
        if not evidence["support"] and not evidence["conflict"]:
            global_telemetry["hypotheses_deferred"] += 1
            with print_lock:
                print(f" ⏸️ [DEFERRED - NO EVIDENCE] {rel_type.upper()}: {c1} ↔ {c2}")
            return None 

        evidence_score = min(1.0, (len(evidence["support"]) * 0.4) + (len(evidence["conflict"]) * 0.3))
        novelty = min(1.0, 0.5 + (0.5 if len(evidence["conflict"]) > len(evidence["support"]) else 0.2))
        confidence = min(1.0, len(evidence["support"]) * 0.5)
        
        strategy_multiplier = self.strategy_weights.get(rel_type, 1.0)
        composite_score = ((evidence_score * 0.4) + (novelty * 0.3) + (confidence * 0.3)) * strategy_multiplier

        # LOGGING 2: Unexpected Conflicts (Conflict outweighs support)
        if len(evidence["conflict"]) > len(evidence["support"]):
            with print_lock:
                print(f" ⚠️ [UNEXPECTED CONFLICT] {rel_type.upper()}: {c1} ↔ {c2} (Conflicts: {len(evidence['conflict'])}, Support: {len(evidence['support'])})")

        # LOGGING 3: Rejected Candidates (Score too low despite some evidence)
        if composite_score < 0.45:
            global_telemetry["candidates_rejected"] += 1
            with print_lock:
                print(f" ❌ [REJECTED - LOW SCORE] {rel_type.upper()}: {c1} ↔ {c2} | Score: {composite_score:.3f}")
            return None
            
        global_telemetry["accepted_insights"] += 1

        templates = {
            "Trade-off": f"Observed a critical tension where optimizing '{c1}' inversely degrades '{c2}'.",
            "Dependency": f"Identified a foundational requirement: '{c1}' execution is strictly bounded by '{c2}'.",
            "Contradiction": f"Found empirical conflict: Theoretical models of '{c1}' fail under '{c2}' conditions.",
            "Improvement": f"Integration of '{c1}' demonstrably stabilizes the variance within '{c2}'.",
            "Analogy": f"Structural behaviors in '{c1}' provide a novel pathway to resolve bottlenecks in '{c2}'."
        }
        statement = templates.get(rel_type, f"Identified a significant '{rel_type}' interaction mapping '{c1}' to '{c2}'.")

        return {
            "insight_title": f"[{rel_type.upper()}] {c1} ↔ {c2}",
            "research_note": {
                "what_was_discovered": statement,
                "why_it_matters": f"Challenges existing models that treat '{c1}' and '{c2}' as isolated variables.",
                "assumptions_challenged": f"The assumption that '{c2}' can scale linearly independent of '{c1}'.",
                "potential_implications": f"Could require fundamental re-architecting of systems relying on '{c1}'.",
                "possible_weaknesses": "Evidence relies on contextual co-occurrence; causal mechanism requires isolation testing."
            },
            "insight_type": rel_type,
            "composite_score": round(composite_score, 3),
            "source_cluster": cluster_name,
            "evidence": {
                "supporting_evidence": evidence["support"],
                "contradicting_evidence": evidence["conflict"]
            },
            "scores": {
                "evidence_support": round(evidence_score, 2), "novelty": round(novelty, 2),
                "confidence": round(confidence, 2), "expected_impact": round(composite_score + 0.1, 2)
            },
            "next_steps": {
                "research_questions": [f"What precise parameter thresholds in '{c1}' trigger cascade failures in '{c2}'?"],
                "testable_hypotheses": [f"If '{c1}' is constrained dynamically, then '{c2}' stability will increase by >15%."],
                "possible_experiments": [f"Isolate '{c1}' in a control simulation while applying extreme variance loads to '{c2}'."],
                "predicted_outcomes": [f"Non-linear behavioral shift in '{c2}' confirming the {rel_type.lower()}."],
                "failure_conditions": [f"If '{c2}' remains unaffected during '{c1}' perturbation, the hypothesis is nullified."],
                "future_directions": [f"Investigate mathematical formulations unifying '{c1}' and '{c2}'."]
            }
        }

    def explore_concept_space(self, cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = cluster["items"]
        cluster_concepts = []
        for item in items:
            cluster_concepts.extend(extract_scientific_concepts(item))
        cluster_concepts = list(set(cluster_concepts))
        
        for c in cluster_concepts: global_telemetry["concepts_extracted"].add(c)

        global_concepts = []
        for i in range(min(15, len(self.full_kb))):
            global_concepts.extend(extract_scientific_concepts(random.choice(self.full_kb)))
        global_concepts = list(set(global_concepts))

        accepted_insights = []
        
        for c1 in cluster_concepts[:10]: 
            candidate_partners = list(set(cluster_concepts + random.sample(global_concepts, min(len(global_concepts), 8))))
            for c2 in candidate_partners:
                if c1 == c2: continue
                
                selected_rels = random.sample(RELATIONSHIP_TYPES, 3)
                for rel in selected_rels:
                    global_telemetry["kg_edges_generated"] += 1
                    insight = self.evaluate_scientific_candidate(c1, c2, rel, cluster["cluster_name"])
                    if insight:
                        accepted_insights.append(insight)

        accepted_insights.sort(key=lambda x: x["composite_score"], reverse=True)
        diverse_insights = []
        seen_pairs = set()
        for cand in accepted_insights:
            pair = tuple(sorted([cand["insight_title"]]))
            if pair not in seen_pairs and len(diverse_insights) < 4:
                seen_pairs.add(pair)
                diverse_insights.append(cand)
                
        return diverse_insights

# =========================================================
# LAYER 5: PIPELINE INTEGRATION
# =========================================================
def load_knowledge_base(filename: str = "deep_research_knowledge_base.json") -> List[str]:
    if not os.path.exists(filename): return []
    with open(filename, "r", encoding="utf-8") as f: data = json.load(f)
    flat = []
    for key, value in data.items():
        if isinstance(value, list): flat.extend([str(i).strip() for i in value])
        elif isinstance(value, dict): flat.extend([f"{k}: {v}" for k, v in value.items()])
        elif isinstance(value, str): flat.append(value.strip())
    return list(set(flat))

global_kb_cache = []

def parallel_reasoning_processing(cluster: Dict[str, Any]) -> Dict[str, Any]:
    if len(cluster["items"]) < 2:
        return {
            "cluster_name": cluster["cluster_name"], "items": cluster["items"],
            "breakthrough_found": False, "novel_predictions": [], "randomized_lateral_ideas": [], "exploratory_insights": []
        }
        
    reasoner = ScientificOntologyReasoner(global_kb_cache)
    insights = reasoner.explore_concept_space(cluster)
    
    predictions = [i["next_steps"]["testable_hypotheses"][0] for i in insights]
    ideas = [i["research_note"]["what_was_discovered"] for i in insights]
    
    return {
        "cluster_name": cluster["cluster_name"],
        "items": cluster["items"],
        "breakthrough_found": len(insights) > 0,
        "novel_predictions": predictions,
        "randomized_lateral_ideas": ideas,
        "exploratory_insights": insights
    }

# =========================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =========================================================
MAX_PARALLEL_CLUSTERS = 4  # Parallel cluster processing
ENABLE_VECTOR_CACHE = True  # Cache TF-IDF vectors

# Simple vector cache to avoid recomputation
_vector_cache = {}

def _get_cached_vectors(documents: List[str], force_rebuild: bool = False) -> List[Dict[str, float]]:
    """Cache TF-IDF vectors to avoid recomputation"""
    if ENABLE_VECTOR_CACHE and not force_rebuild:
        # Create a simple hash of the documents
        doc_hash = hash(tuple(sorted(documents)))
        if doc_hash in _vector_cache:
            return _vector_cache[doc_hash]
    
    vectors = build_tfidf_vectors(documents)
    
    if ENABLE_VECTOR_CACHE:
        doc_hash = hash(tuple(sorted(documents)))
        _vector_cache[doc_hash] = vectors
    
    return vectors

# =========================================================
# LAYER 6: MAIN EXECUTION ENGINE
# =========================================================
def run_doscan_algorithm(max_r: int = 4):
    global global_kb_cache, global_telemetry
    print("\n" + "="*80)
    print("🔬 INITIATING DOSCAN: FAILED / UNEXPECTED PREDICTION LOGGING MODE")
    print("="*80)
    
    global_kb_cache = load_knowledge_base()
    unprocessed_items = global_kb_cache.copy()
    if not unprocessed_items:
        print("❌ Knowledge base empty. Run Phase 2 first.")
        return

    _doscan_cluster_loop(unprocessed_items, max_r=max_r, phase_label="PRIMARY")

def run_doscan_deepening(injected_hypotheses: List[str], max_r: int = 4):
    """
    Recursively deepens research on approved hypotheses from the Hypothesis Engine.
    Takes refined hypotheses, injects them into the knowledge base, and runs
    DOSCAN's full r=1→r=4 clustering cycle to discover deeper relationships.
    """
    global global_kb_cache, global_telemetry
    
    if not injected_hypotheses:
        print("\n⏸️  [DEEPENING] No approved hypotheses to deepen. Skipping feedback loop.")
        return
    
    print("\n" + "="*80)
    print("🔄 DOSCAN DEEPENING CYCLE — Recursive Hypothesis-Driven Clustering")
    print("="*80)
    
    # Step 1: Reload updated knowledge base
    global_kb_cache = load_knowledge_base()
    
    # Step 2: Tag and inject the approved hypotheses as fresh research material
    tagged_hypotheses = [f"[APPROVED_HYPOTHESIS] {h}" for h in injected_hypotheses]
    
    # Step 3: Append them to the knowledge base cache for clustering context
    global_kb_cache.extend(tagged_hypotheses)
    
    # Step 4: Also inject them as a separate unprocessed cluster input
    # The hypotheses themselves become the primary items to cluster
    deep_items = tagged_hypotheses.copy()
    
    print(f"\n📥 Injected {len(deep_items)} approved hypotheses into DOSCAN deepening pipeline...")
    print(f"📚 Knowledge base expanded to {len(global_kb_cache)} total items.\n")
    
    # Step 5: Run the full multi-resolution clustering loop (r=1 to r=4) on the hypotheses
    _doscan_cluster_loop(deep_items, max_r=max_r, phase_label="DEEPEN")
    
    print(f"\n🔁 [DEEPENING CYCLE COMPLETE] Hypotheses have been clustered and deepened.")
    print("="*80)

def _doscan_cluster_loop(unprocessed_items: List[str], max_r: int = 4, phase_label: str = "PRIMARY"):
    """
    Internal shared clustering loop used by both the primary DOSCAN run
    and the deepening feedback cycle.
    """
    global global_kb_cache, global_telemetry
    
    # Reset telemetry counters for this cycle
    global_telemetry = {
        "concepts_extracted": set(), "kg_edges_generated": 0, "candidates_rejected": 0,
        "hypotheses_deferred": 0, "accepted_insights": 0
    }
    
    # Load existing breakthroughs so we don't overwrite them
    existing_breakthroughs = []
    if os.path.exists("doscan_breakthroughs.json"):
        try:
            with open("doscan_breakthroughs.json", "r", encoding="utf-8") as f:
                existing_breakthroughs = json.load(f)
        except Exception:
            existing_breakthroughs = []

    r = 1
    final_breakthroughs = list(existing_breakthroughs)
    strategy_weights = load_exploration_weights()

    while r <= max_r and unprocessed_items:
        eps = max(0.0, 0.45 - (r * 0.15))
        print(f"\n⚡ [{phase_label}] Layer (r = {r} | Expansion Limit: {eps*100:.1f}%) — Items: {len(unprocessed_items)}\n")
        
        vectors = build_tfidf_vectors(unprocessed_items)
        clusters, noise = dbscan_text_cluster(unprocessed_items, vectors, eps=eps)
        failed_items_for_next_r = noise.copy()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_CLUSTERS) as executor:
            futures = [executor.submit(parallel_reasoning_processing, c) for c in clusters]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["breakthrough_found"]:
                    final_breakthroughs.append({
                        "cluster": result["cluster_name"],
                        "phase": phase_label,
                        "r_level": r,
                        "base_elements": result["items"],
                        "predictions": result["novel_predictions"],
                        "random_thoughts": result["randomized_lateral_ideas"],
                        "scientific_research_insights": result["exploratory_insights"]
                    })
                else:
                    failed_items_for_next_r.extend(result["items"])

        if failed_items_for_next_r:
            unprocessed_items = failed_items_for_next_r
            r += 1
        else:
            print(f"\n🎉 [{phase_label}] Full scientific space graph convergence completed.")
            break

    # Update strategy weights using insights from this cycle
    for entry in final_breakthroughs:
        for insight in entry.get("scientific_research_insights", []):
            i_type = insight.get("insight_type")
            if i_type and "composite_score" in insight:
                if insight["composite_score"] > 0.65:
                    strategy_weights[i_type] = min(2.5, strategy_weights.get(i_type, 1.0) + 0.15) 
                else:
                    strategy_weights[i_type] = max(0.4, strategy_weights.get(i_type, 1.0) - 0.08)
    save_exploration_weights(strategy_weights)

    print("\n" + "="*80)
    print(f"📊 [{phase_label}] DOSCAN ONTOLOGY SUMMARY LOG")
    print("="*80)
    print(f"  🔹 Pure Scientific Concepts Extracted: {len(global_telemetry['concepts_extracted'])}")
    print(f"  🔹 Knowledge Graph Edges Generated:    {global_telemetry['kg_edges_generated']}")
    print(f"  🔹 Relationships Auto-Rejected:        {global_telemetry['candidates_rejected']}")
    print(f"  🔹 Hypotheses Deferred (No Evidence):  {global_telemetry['hypotheses_deferred']}")
    print(f"  🔹 Final Accepted Insights:            {global_telemetry['accepted_insights']}")
    print("="*80)

    if final_breakthroughs:
        with open("doscan_breakthroughs.json", "w", encoding="utf-8") as f:
            json.dump(final_breakthroughs, f, indent=4)
        print(f"[DOSCAN {phase_label}] Scientific insights merged into 'doscan_breakthroughs.json'.\n")

if __name__ == "__main__":
    run_doscan_algorithm()
