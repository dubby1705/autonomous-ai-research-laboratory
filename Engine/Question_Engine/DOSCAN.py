import os
import json
import random
import math
import re
import concurrent.futures
from collections import Counter
from typing import List, Dict, Any, Tuple

# =========================================================
# PRO LAYER 1: MATHEMATICAL VECTORIZATION (TF-IDF & COSINE)
# =========================================================
def tokenize(text: str) -> List[str]:
    """Extracts alphanumeric words, ignoring case and basic punctuation."""
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

def build_tfidf_vectors(documents: List[str]) -> List[Dict[str, float]]:
    """Converts a list of text strings into TF-IDF mathematical vectors."""
    doc_tokens = [tokenize(doc) for doc in documents]
    N = len(documents)
    
    # Calculate Document Frequency (DF)
    df = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
        
    vectors = []
    for tokens in doc_tokens:
        vec = {}
        tf = Counter(tokens)
        total_terms = len(tokens) if tokens else 1
        
        for word, count in tf.items():
            # Term Frequency * Inverse Document Frequency
            term_freq = count / total_terms
            inv_doc_freq = math.log(N / (1 + df[word])) 
            vec[word] = term_freq * inv_doc_freq
        vectors.append(vec)
        
    return vectors

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Calculates the exact angular similarity between two document vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[w] * vec2[w] for w in intersection)
    
    mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

# =========================================================
# PRO LAYER 2: TRUE DBSCAN ALGORITHM
# =========================================================
def dbscan_text_cluster(documents: List[str], vectors: List[Dict[str, float]], eps: float, min_pts: int = 1) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Groups highly similar items and isolates 'noise' to be processed at a higher radius.
    """
    labels = [0] * len(documents) # 0 = undefined, -1 = noise, >0 = cluster ID
    cluster_id = 0
    
    def region_query(p_idx: int) -> List[int]:
        neighbors = []
        for q_idx, q_vec in enumerate(vectors):
            # eps here is similarity threshold. If sim > eps, they are neighbors.
            if cosine_similarity(vectors[p_idx], q_vec) >= eps:
                neighbors.append(q_idx)
        return neighbors

    for p in range(len(documents)):
        if labels[p] != 0:
            continue # Already processed
            
        neighbors = region_query(p)
        
        # If not enough density, mark as noise (to be expanded in next r-level)
        if len(neighbors) < min_pts + 1: # +1 includes itself
            labels[p] = -1
        else:
            cluster_id += 1
            labels[p] = cluster_id
            
            # Expand cluster
            i = 0
            while i < len(neighbors):
                q = neighbors[i]
                if labels[q] == -1:
                    labels[q] = cluster_id # Upgrade from noise to border point
                elif labels[q] == 0:
                    labels[q] = cluster_id
                    q_neighbors = region_query(q)
                    if len(q_neighbors) >= min_pts + 1:
                        neighbors.extend(q_neighbors)
                i += 1

    # Format output
    clusters = {}
    noise = []
    
    for idx, label in enumerate(labels):
        if label == -1:
            noise.append(documents[idx])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(documents[idx])
            
    formatted_clusters = [
        {"cluster_name": f"DBSCAN_Node_{cid}", "items": items}
        for cid, items in clusters.items()
    ]
    
    return formatted_clusters, noise

# =========================================================
# PRO LAYER 3: KNOWLEDGE INGESTION & LATERAL SYNTHESIS
# =========================================================
def load_knowledge_base(filename: str = "deep_research_knowledge_base.json") -> List[str]:
    """Extracts raw strings from Phase 2 knowledge JSON."""
    if not os.path.exists(filename):
        print(f"❌ '{filename}' not found. Ensure Phase 2 ran successfully.")
        return []
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    flat_concepts = []
    for key, value in data.items():
        if isinstance(value, list):
            flat_concepts.extend([str(item).strip() for item in value])
        elif isinstance(value, dict):
            for k, v in value.items():
                flat_concepts.append(f"{k}: {v}")
        elif isinstance(value, str):
            flat_concepts.append(value.strip())
            
    return list(set(flat_concepts))

def mathematical_lateral_thinking(cluster: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesizes new combinations from clustered nodes."""
    items = cluster["items"]
    
    # 🐛 FIX: Always return the "items" key, even on failure, to prevent KeyError.
    if len(items) < 2:
        return {
            "cluster_name": cluster["cluster_name"],
            "items": items, 
            "breakthrough_found": False,
            "novel_predictions": [],
            "randomized_lateral_ideas": []
        }
        
    novel_predictions = []
    randomized_ideas = []
    
    # Extract dominant keywords from this specific cluster using basic frequency
    all_text = " ".join(items)
    tokens = [t for t in tokenize(all_text) if len(t) > 3]
    top_keywords = [word for word, count in Counter(tokens).most_common(4)]
    
    if len(top_keywords) >= 2:
        synthesis = f"Extrapolated Matrix: High correlation between [{top_keywords[0].upper()}] systems and [{top_keywords[1].upper()}] frameworks."
        novel_predictions.append(synthesis)

    shuffled_pool = list(items)
    random.shuffle(shuffled_pool)
    
    for i in range(0, len(shuffled_pool), 2):
        if i + 1 < len(shuffled_pool):
            wild_idea = f"SYNTHESIS: Injecting parameters of ({shuffled_pool[i][:50]}...) into the operational constraints of ({shuffled_pool[i+1][:50]}...)"
            randomized_ideas.append(wild_idea)

    return {
        "cluster_name": cluster["cluster_name"],
        "items": items,
        "breakthrough_found": True,
        "novel_predictions": novel_predictions,
        "randomized_lateral_ideas": randomized_ideas
    }

# =========================================================
# MAIN EXECUTION: COGNITIVE BREATHING LOOP
# =========================================================
def run_doscan_algorithm(max_r: int = 4):
    print("\n" + "="*70)
    print("🌀 RUNNING PRO DBSCAN/DOSCAN ENGINE (TF-IDF + COSINE VECTORS)")
    print("="*70)
    
    unprocessed_concepts = load_knowledge_base()
    if not unprocessed_concepts:
        return

    r = 1
    final_breakthroughs = []

    while r <= max_r and unprocessed_concepts:
        # Convert distance metric (r) into Cosine Similarity Epsilon (eps)
        # r=1: Must be 30% similar. r=2: 15% similar. r=3: 5% similar. r=4: 0% (Force combine)
        eps = max(0.0, 0.45 - (r * 0.15)) 
        
        print(f"\n⚡ Ingesting Layer (r = {r} | Min Similarity: {eps*100:.1f}%) — Data Pool: {len(unprocessed_concepts)} items")
        
        vectors = build_tfidf_vectors(unprocessed_concepts)
        clusters, noise = dbscan_text_cluster(unprocessed_concepts, vectors, eps=eps)
        
        print(f"Generated {len(clusters)} dense clusters. {len(noise)} items rejected as noise.")
        
        failed_concepts_for_next_r = noise.copy() # Noise gets pushed directly to next tier

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(mathematical_lateral_thinking, c) for c in clusters]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                if result["breakthrough_found"]:
                    print(f"  ✅ Node [{result['cluster_name']}] Succeeded -> {len(result['randomized_lateral_ideas'])} new insights.")
                    final_breakthroughs.append({
                        "cluster": result["cluster_name"],
                        "r_level": r,
                        "similarity_threshold": f"{eps*100:.1f}%",
                        "base_elements": result["items"],
                        "predictions": result["novel_predictions"],
                        "random_thoughts": result["randomized_lateral_ideas"]
                    })
                else:
                    print(f"  ❌ Node [{result['cluster_name']}] collapsed (insufficient data). Demoting items to noise.")
                    failed_concepts_for_next_r.extend(result["items"])

        # Cognitive Breathing Loop
        if failed_concepts_for_next_r:
            unprocessed_concepts = failed_concepts_for_next_r
            r += 1
        else:
            print("\n🎉 Matrix convergence achieved! All data paths successfully categorized.")
            break

    if final_breakthroughs:
        with open("doscan_breakthroughs.json", "w", encoding="utf-8") as f:
            json.dump(final_breakthroughs, f, indent=4)
        print(f"\n[DOSCAN Complete] Breakthrough data safely written to 'doscan_breakthroughs.json'.")

if __name__ == "__main__":
    run_doscan_algorithm()