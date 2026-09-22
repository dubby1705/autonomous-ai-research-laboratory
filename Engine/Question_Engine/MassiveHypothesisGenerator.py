"""
MassiveHypothesisGenerator.py — Massive Hypothesis Generation Engine
=====================================================================
Generates hundreds or thousands of diverse research hypotheses from
the knowledge graph using multiple clustering strategies:

  1. Graph partitioning (community detection)
  2. Semantic similarity clustering (TF-IDF + DBSCAN)
  3. Topic clustering (LDA-style topic modeling)
  4. Hypothesis diversification (evolutionary mutation)

Each cluster produces one or more candidate ideas, maximizing diversity
and avoiding duplicates.

This module does NOT keep all ideas in memory — it processes in batches
and streams results to disk.
"""

import os
import json
import math
import random
import re
import hashlib
import time
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CONSTANTS
# =========================================================
BOUNDARY_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "with",
    "by", "on", "in", "at", "from", "as", "this", "that", "it", "can", "could",
    "would", "should", "using", "such", "without", "will", "has", "have", "had",
    "not", "no", "only", "but", "however", "and", "or", "which", "where", "when",
    "why", "how", "we", "they", "their", "our", "show", "shows", "demonstrate",
    "compare", "search", "find", "found", "use", "used", "requires", "assumes",
    "must", "always", "during", "between", "through", "under", "over", "into",
}

# Default target cluster counts
DEFAULT_TARGET_CLUSTERS = 100
DEFAULT_MAX_CLUSTERS = 5000
DEFAULT_BATCH_SIZE = 200

# Diversity strategies
STRATEGY_GRAPH_PARTITION = "graph_partition"
STRATEGY_SEMANTIC = "semantic_similarity"
STRATEGY_TOPIC = "topic_clustering"
STRATEGY_DIVERSIFICATION = "hypothesis_diversification"

ALL_STRATEGIES = [
    STRATEGY_GRAPH_PARTITION,
    STRATEGY_SEMANTIC,
    STRATEGY_TOPIC,
    STRATEGY_DIVERSIFICATION,
]


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def _tokenize(text: str) -> List[str]:
    """Tokenize text into meaningful words."""
    return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if w not in BOUNDARY_WORDS]


def _build_tfidf_vectors(documents: List[str]) -> List[Dict[str, float]]:
    """Build TF-IDF vectors for a list of documents."""
    doc_tokens = [_tokenize(doc) for doc in documents]
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


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[w] * vec2[w] for w in intersection)
    mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
    return 0.0 if mag1 == 0 or mag2 == 0 else dot_product / (mag1 * mag2)


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    set1 = set(_tokenize(text1))
    set2 = set(_tokenize(text2))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _deduplicate(items: List[str], threshold: float = 0.85) -> List[str]:
    """Remove near-duplicate items using Jaccard similarity."""
    unique = []
    for item in items:
        is_dup = False
        for existing in unique:
            if _jaccard_similarity(item, existing) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
    return unique


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    slug = slug.strip('_')
    if len(slug) > max_len:
        slug = slug[:max_len]
    if not slug:
        slug = "cluster"
    return slug


def _hash_text(text: str) -> str:
    """Create a stable hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# =========================================================
# STRATEGY 1: GRAPH PARTITIONING
# =========================================================
def _build_concept_graph(kb_items: List[str]) -> Dict[str, Set[str]]:
    """Build a concept co-occurrence graph from knowledge base items."""
    graph: Dict[str, Set[str]] = defaultdict(set)
    for item in kb_items:
        concepts = _tokenize(item)
        # Connect consecutive concepts
        for i in range(len(concepts)):
            for j in range(i + 1, min(i + 4, len(concepts))):
                graph[concepts[i]].add(concepts[j])
                graph[concepts[j]].add(concepts[i])
    return graph


def _graph_partition_clusters(kb_items: List[str], target_clusters: int) -> List[List[str]]:
    """
    Partition knowledge base items into clusters using graph-based
    community detection (label propagation).
    """
    if len(kb_items) <= target_clusters:
        return [[item] for item in kb_items]

    # Build concept graph
    graph = _build_concept_graph(kb_items)

    # Build item-concept mapping
    item_concepts = []
    for item in kb_items:
        item_concepts.append(set(_tokenize(item)))

    # Label propagation on items
    labels = list(range(len(kb_items)))
    item_graph: Dict[int, Set[int]] = defaultdict(set)

    # Build item similarity graph (top-k neighbors)
    for i in range(len(kb_items)):
        for j in range(i + 1, len(kb_items)):
            sim = _jaccard_similarity(kb_items[i], kb_items[j])
            if sim > 0.15:
                item_graph[i].add(j)
                item_graph[j].add(i)

    # Label propagation
    for _ in range(10):
        new_labels = labels.copy()
        for node in range(len(kb_items)):
            neighbors = item_graph.get(node, set())
            if not neighbors:
                continue
            label_counts = Counter(labels[n] for n in neighbors)
            new_labels[node] = label_counts.most_common(1)[0][0]
        labels = new_labels

    # Group by label
    clusters_dict: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_dict[label].append(idx)

    # Merge small clusters
    clusters = [clusters_dict[k] for k in sorted(clusters_dict.keys())]
    merged = []
    current = []
    for cluster in clusters:
        if len(current) + len(cluster) <= max(2, len(kb_items) // target_clusters):
            current.extend(cluster)
        else:
            if current:
                merged.append(current)
            current = list(cluster)
    if current:
        merged.append(current)

    # If still too many clusters, merge smallest ones
    while len(merged) > target_clusters:
        merged.sort(key=len)
        smallest = merged.pop(0)
        merged[0].extend(smallest)

    return [[kb_items[i] for i in cluster] for cluster in merged]


# =========================================================
# STRATEGY 2: SEMANTIC SIMILARITY (DBSCAN)
# =========================================================
def _dbscan_cluster(documents: List[str], vectors: List[Dict[str, float]],
                    eps: float, min_pts: int = 1) -> Tuple[List[List[int]], List[int]]:
    """DBSCAN clustering on TF-IDF vectors."""
    labels = [0] * len(documents)
    cluster_id = 0

    def region_query(p_idx: int) -> List[int]:
        return [q_idx for q_idx, q_vec in enumerate(vectors)
                if _cosine_similarity(vectors[p_idx], q_vec) >= eps]

    for p in range(len(documents)):
        if labels[p] != 0:
            continue
        neighbors = region_query(p)
        if len(neighbors) < min_pts + 1:
            labels[p] = -1
        else:
            cluster_id += 1
            labels[p] = cluster_id
            i = 0
            while i < len(neighbors):
                q = neighbors[i]
                if labels[q] == -1:
                    labels[q] = cluster_id
                elif labels[q] == 0:
                    labels[q] = cluster_id
                    q_neighbors = region_query(q)
                    if len(q_neighbors) >= min_pts + 1:
                        neighbors.extend(q_neighbors)
                i += 1

    clusters: Dict[int, List[int]] = defaultdict(list)
    noise: List[int] = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise.append(idx)
        else:
            clusters[label].append(idx)

    return list(clusters.values()), noise


def _semantic_clusters(kb_items: List[str], target_clusters: int) -> List[List[str]]:
    """Cluster knowledge base items using multi-resolution DBSCAN."""
    if len(kb_items) <= target_clusters:
        return [[item] for item in kb_items]

    vectors = _build_tfidf_vectors(kb_items)
    all_clusters: List[List[int]] = []
    unprocessed = list(range(len(kb_items)))

    # Multi-resolution: start coarse, get finer
    eps_values = [0.45, 0.35, 0.25, 0.15, 0.05]
    for eps in eps_values:
        if not unprocessed:
            break
        current_docs = [kb_items[i] for i in unprocessed]
        current_vectors = [vectors[i] for i in unprocessed]
        clusters, noise = _dbscan_cluster(current_docs, current_vectors, eps=eps)

        for cluster in clusters:
            original_indices = [unprocessed[i] for i in cluster]
            all_clusters.append(original_indices)

        unprocessed = [unprocessed[i] for i in noise]
        if len(all_clusters) >= target_clusters:
            break

    # Add remaining noise as singleton clusters
    for idx in unprocessed:
        all_clusters.append([idx])

    # Merge or split to reach target
    while len(all_clusters) > target_clusters:
        all_clusters.sort(key=len)
        smallest = all_clusters.pop(0)
        all_clusters[0].extend(smallest)

    return [[kb_items[i] for i in cluster] for cluster in all_clusters]


# =========================================================
# STRATEGY 3: TOPIC CLUSTERING
# =========================================================
def _topic_clusters(kb_items: List[str], target_clusters: int) -> List[List[str]]:
    """
    Cluster knowledge base items using topic modeling.
    Uses a simplified LDA-style approach with TF-IDF topic assignment.
    """
    if len(kb_items) <= target_clusters:
        return [[item] for item in kb_items]

    vectors = _build_tfidf_vectors(kb_items)

    # Build vocabulary
    vocab = set()
    for vec in vectors:
        vocab.update(vec.keys())
    vocab = sorted(vocab)

    # Simplified topic modeling: assign each document to its dominant topic
    # using k-means-like clustering on TF-IDF vectors
    k = min(target_clusters, len(kb_items))

    # Initialize centroids with k-means++
    centroids = []
    first_idx = random.randrange(len(kb_items))
    centroids.append(vectors[first_idx])

    for _ in range(1, k):
        # Compute distance to nearest centroid for each point
        distances = []
        for vec in vectors:
            min_dist = min(1.0 - _cosine_similarity(vec, c) for c in centroids)
            distances.append(min_dist)
        total = sum(distances)
        if total <= 0:
            break
        # Weighted random selection
        r = random.random() * total
        cumulative = 0
        chosen = 0
        for i, d in enumerate(distances):
            cumulative += d
            if cumulative >= r:
                chosen = i
                break
        centroids.append(vectors[chosen])

    # Assign documents to nearest centroid
    assignments: Dict[int, List[int]] = defaultdict(list)
    for idx, vec in enumerate(vectors):
        best_c = 0
        best_sim = -1
        for c_idx, centroid in enumerate(centroids):
            sim = _cosine_similarity(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_c = c_idx
        assignments[best_c].append(idx)

    clusters = list(assignments.values())

    # Merge small clusters
    merged = []
    current = []
    for cluster in clusters:
        if len(current) + len(cluster) <= max(2, len(kb_items) // target_clusters):
            current.extend(cluster)
        else:
            if current:
                merged.append(current)
            current = list(cluster)
    if current:
        merged.append(current)

    while len(merged) > target_clusters:
        merged.sort(key=len)
        smallest = merged.pop(0)
        merged[0].extend(smallest)

    return [[kb_items[i] for i in cluster] for cluster in merged]


# =========================================================
# STRATEGY 4: HYPOTHESIS DIVERSIFICATION
# =========================================================
DIVERSIFICATION_TEMPLATES = [
    "Novel approach using {concept} to achieve {goal}",
    "Hybrid strategy combining {concept1} and {concept2} for {goal}",
    "Bio-inspired design based on {concept} principles for {goal}",
    "Multi-scale optimization of {concept} targeting {goal}",
    "Adaptive control strategy leveraging {concept} for {goal}",
    "Quantum-enhanced {concept} methodology for {goal}",
    "Gradient-based optimization of {concept} to improve {goal}",
    "Reinforcement learning approach for {concept} optimization in {goal}",
    "Meta-heuristic search over {concept} space for {goal}",
    "Information-theoretic framework for {concept} in {goal}",
    "Topology-optimized {concept} design for {goal}",
    "Self-healing {concept} architecture for {goal}",
    "Fractal-based {concept} patterning for {goal}",
    "Entropy-driven {concept} refinement for {goal}",
    "Coupled multi-physics {concept} model for {goal}",
    "Hierarchical {concept} decomposition for {goal}",
    "Stochastic {concept} exploration for {goal}",
    "Adversarial {concept} validation for {goal}",
    "Transfer learning across {concept} domains for {goal}",
    "Generative {concept} synthesis for {goal}",
]


def _diversification_clusters(kb_items: List[str], target_clusters: int,
                              problem_statement: str) -> List[List[str]]:
    """
    Generate diverse hypothesis clusters by combining concepts from
    the knowledge base in novel ways.
    """
    # Extract key concepts from knowledge base
    all_concepts = []
    for item in kb_items:
        all_concepts.extend(_tokenize(item))
    concept_counts = Counter(all_concepts)
    top_concepts = [c for c, _ in concept_counts.most_common(50) if len(c) > 3]

    if len(top_concepts) < 2:
        top_concepts = ["optimization", "design", "performance", "efficiency",
                        "stability", "materials", "control", "integration"]

    # Extract goal from problem statement
    goal_words = _tokenize(problem_statement)
    goal = " ".join(goal_words[:8]) if goal_words else "optimal performance"

    clusters = []
    seen_combinations = set()

    # Generate diverse combinations
    for i in range(target_clusters):
        # Pick 1-2 concepts
        if random.random() < 0.5:
            concept = random.choice(top_concepts)
            concept2 = random.choice(top_concepts)
            template = random.choice(DIVERSIFICATION_TEMPLATES)
            # Provide all possible placeholders to avoid KeyError
            hypothesis = template.format(concept=concept, concept1=concept, concept2=concept2, goal=goal)
        else:
            concept = random.choice(top_concepts)
            template = random.choice(DIVERSIFICATION_TEMPLATES)
            # Provide all possible placeholders to avoid KeyError
            hypothesis = template.format(concept=concept, concept1=concept, concept2=concept, goal=goal)

        combo_key = _hash_text(hypothesis)
        if combo_key in seen_combinations:
            continue
        seen_combinations.add(combo_key)

        # Create a cluster with this hypothesis and related KB items
        related = [item for item in kb_items if any(c in item.lower() for c in _tokenize(hypothesis)[:5])]
        cluster_items = [hypothesis] + related[:3]
        clusters.append(cluster_items)

    return clusters


# =========================================================
# MAIN CLUSTER ENGINE
# =========================================================
def generate_hypothesis_clusters(
    kb_items: List[str],
    problem_statement: str,
    target_clusters: int = DEFAULT_TARGET_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    strategies: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_file: str = "hypothesis_clusters.json",
) -> Dict[str, Any]:
    """
    Generate a massive number of diverse hypothesis clusters.

    Args:
        kb_items: Knowledge base items (strings)
        problem_statement: The research problem
        target_clusters: Number of clusters to generate
        max_clusters: Maximum allowed clusters
        strategies: Which clustering strategies to use
        batch_size: Batch size for processing
        output_file: Where to save results

    Returns:
        Summary dict with cluster statistics
    """
    if not kb_items:
        return {"error": "Empty knowledge base", "clusters": 0}

    target_clusters = min(target_clusters, max_clusters)
    if strategies is None:
        strategies = ALL_STRATEGIES

    print("\n" + "=" * 80)
    print("  MASSIVE HYPOTHESIS GENERATION ENGINE")
    print("=" * 80)
    print(f"  Knowledge items: {len(kb_items)}")
    print(f"  Target clusters: {target_clusters}")
    print(f"  Strategies: {', '.join(strategies)}")
    print("=" * 80)

    all_clusters: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    strategy_counts: Dict[str, int] = {}

    # Run each strategy
    for strategy in strategies:
        print(f"\n  >> Running strategy: {strategy}")

        if strategy == STRATEGY_GRAPH_PARTITION:
            clusters = _graph_partition_clusters(kb_items, target_clusters // len(strategies))
        elif strategy == STRATEGY_SEMANTIC:
            clusters = _semantic_clusters(kb_items, target_clusters // len(strategies))
        elif strategy == STRATEGY_TOPIC:
            clusters = _topic_clusters(kb_items, target_clusters // len(strategies))
        elif strategy == STRATEGY_DIVERSIFICATION:
            clusters = _diversification_clusters(kb_items, target_clusters // len(strategies), problem_statement)
        else:
            continue

        # Process clusters in batches
        for batch_start in range(0, len(clusters), batch_size):
            batch = clusters[batch_start:batch_start + batch_size]
            for cluster_items in batch:
                if not cluster_items:
                    continue

                # Create cluster representation
                cluster_text = " | ".join(cluster_items[:5])
                cluster_hash = _hash_text(cluster_text)

                if cluster_hash in seen_hashes:
                    continue
                seen_hashes.add(cluster_hash)

                cluster_entry = {
                    "cluster_id": f"HC_{len(all_clusters) + 1:05d}",
                    "strategy": strategy,
                    "items": cluster_items[:10],
                    "item_count": len(cluster_items),
                    "representative_text": cluster_text[:500],
                    "created_at": time.time(),
                }
                all_clusters.append(cluster_entry)
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                # Stop if we've reached the target
                if len(all_clusters) >= target_clusters:
                    break
            if len(all_clusters) >= target_clusters:
                break

        print(f"    -> {strategy_counts.get(strategy, 0)} clusters generated")

    # Deduplicate final clusters
    print(f"\n  Deduplicating {len(all_clusters)} clusters...")
    deduped = []
    seen = set()
    for cluster in all_clusters:
        h = _hash_text(cluster["representative_text"][:200])
        if h not in seen:
            seen.add(h)
            deduped.append(cluster)

    # Save to disk
    result = {
        "problem_statement": problem_statement,
        "target_clusters": target_clusters,
        "total_clusters": len(deduped),
        "strategy_counts": strategy_counts,
        "clusters": deduped,
        "generated_at": time.time(),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"\n  [OK] Generated {len(deduped)} unique hypothesis clusters")
    print(f"  [FILE] Saved to: {output_file}")
    print("=" * 80)

    return result


# =========================================================
# BATCHED STREAMING GENERATOR (for very large scale)
# =========================================================
def stream_hypothesis_clusters(
    kb_items: List[str],
    problem_statement: str,
    target_clusters: int = 1000,
    batch_size: int = 100,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream hypothesis clusters in batches without keeping all in memory.
    Yields cluster dicts one at a time.
    """
    if not kb_items:
        return

    # Use diversification strategy for massive scale
    all_concepts = []
    for item in kb_items:
        all_concepts.extend(_tokenize(item))
    concept_counts = Counter(all_concepts)
    top_concepts = [c for c, _ in concept_counts.most_common(100) if len(c) > 3]

    if len(top_concepts) < 2:
        top_concepts = ["optimization", "design", "performance", "efficiency",
                        "stability", "materials", "control", "integration"]

    goal_words = _tokenize(problem_statement)
    goal = " ".join(goal_words[:8]) if goal_words else "optimal performance"

    seen = set()
    generated = 0

    while generated < target_clusters:
        batch = []
        for _ in range(batch_size):
            if generated >= target_clusters:
                break

            concept = random.choice(top_concepts)
            concept2 = random.choice(top_concepts)
            template = random.choice(DIVERSIFICATION_TEMPLATES)

            if random.random() < 0.5:
                hypothesis = template.format(concept=concept, concept1=concept, concept2=concept2, goal=goal)
            else:
                hypothesis = template.format(concept=concept, concept1=concept, concept2=concept, goal=goal)

            h = _hash_text(hypothesis)
            if h in seen:
                continue
            seen.add(h)

            related = [item for item in kb_items if any(c in item.lower() for c in _tokenize(hypothesis)[:5])]

            cluster_entry = {
                "cluster_id": f"STREAM_{generated + 1:06d}",
                "strategy": STRATEGY_DIVERSIFICATION,
                "items": [hypothesis] + related[:3],
                "item_count": 1 + len(related[:3]),
                "representative_text": hypothesis,
                "created_at": time.time(),
            }
            batch.append(cluster_entry)
            generated += 1

        for entry in batch:
            yield entry


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Test with sample data
    test_kb = [
        "Graphene has exceptional thermal conductivity",
        "Silicon anodes offer 10x capacity but degrade rapidly",
        "Solid-state electrolytes improve safety",
        "Nano-porous structures increase surface area",
        "Carbon nanotubes have high tensile strength",
        "Lithium metal anodes have high energy density",
        "Polymer electrolytes are flexible but low conductivity",
        "Ceramic electrolytes are rigid but high conductivity",
        "Composite electrodes balance capacity and stability",
        "Thermal management is critical for battery safety",
    ]
    result = generate_hypothesis_clusters(
        test_kb,
        "Design a better battery with higher energy density",
        target_clusters=20,
    )
    print(f"\nGenerated {result['total_clusters']} clusters")