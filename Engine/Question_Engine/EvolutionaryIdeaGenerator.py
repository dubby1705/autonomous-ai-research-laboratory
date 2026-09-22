"""
EvolutionaryIdeaGenerator.py — Evolutionary Idea Generation Engine
===================================================================
Each hypothesis cluster generates one or more candidate ideas through
evolutionary computation:

  1. Selection: Choose the most promising clusters
  2. Crossover: Combine ideas from different clusters
  3. Mutation: Introduce random variations
  4. Fitness: Score ideas based on domain-specific criteria

The objective is not one answer — it is exploring an enormous design space.
"""

import os
import json
import math
import random
import re
import time
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_POPULATION_SIZE = 1000
DEFAULT_GENERATIONS = 5
DEFAULT_MUTATION_RATE = 0.3
DEFAULT_CROSSOVER_RATE = 0.5
DEFAULT_ELITISM = 0.1

# Idea templates for evolutionary generation
IDEA_TEMPLATES = [
    "Implement {concept} with {technique} to achieve {goal}",
    "Design a {concept}-based system using {technique} for {goal}",
    "Develop {concept} enhanced by {technique} targeting {goal}",
    "Create a hybrid {concept}/{technique} approach for {goal}",
    "Optimize {concept} through {technique} to improve {goal}",
    "Engineer {concept} with {technique} integration for {goal}",
    "Build an adaptive {concept} system leveraging {technique} for {goal}",
    "Apply {technique} to {concept} for enhanced {goal}",
    "Integrate {concept} and {technique} in a unified framework for {goal}",
    "Novel {concept} architecture powered by {technique} for {goal}",
]

TECHNIQUES = [
    "machine learning", "deep neural networks", "reinforcement learning",
    "genetic algorithms", "simulated annealing", "particle swarm optimization",
    "Bayesian optimization", "gradient descent", "transfer learning",
    "multi-objective optimization", "topology optimization", "meta-heuristics",
    "surrogate modeling", "Monte Carlo simulation", "finite element analysis",
    "computational fluid dynamics", "quantum computing", "edge computing",
    "federated learning", "attention mechanisms", "graph neural networks",
    "physics-informed neural networks", "digital twins", "model predictive control",
    "adaptive filtering", "signal processing", "information theory",
    "thermodynamic optimization", "electrochemical modeling", "multi-physics simulation",
]

# Domain-specific validation keywords
DOMAIN_VALIDATION_KEYWORDS = {
    "gpu": ["thermal", "memory", "bandwidth", "power", "cost", "manufacturability", "throughput", "latency"],
    "battery": ["energy density", "stability", "cycle life", "safety", "capacity", "charging", "degradation"],
    "floating": ["lift", "buoyancy", "stability", "energy", "battery", "thermal", "safety", "manufacturing", "cost"],
    "medicine": ["toxicity", "safety", "bioavailability", "efficacy", "selectivity", "half-life"],
    "cpu": ["performance", "power", "area", "thermal", "cost", "manufacturability", "ipc", "latency"],
    "material": ["strength", "stiffness", "toughness", "corrosion", "thermal", "cost", "manufacturability"],
    "robotics": ["precision", "repeatability", "payload", "response", "energy", "safety", "cost"],
    "aerospace": ["lift", "drag", "thrust", "stability", "fuel", "structural", "safety", "cost"],
    "chemistry": ["yield", "selectivity", "rate", "stability", "safety", "cost", "purity"],
    "general": ["feasibility", "performance", "cost", "safety", "scalability", "reliability"],
}


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def _tokenize(text: str) -> List[str]:
    """Tokenize text into meaningful words."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "with",
        "by", "on", "in", "at", "from", "as", "this", "that", "it", "can", "could",
        "would", "should", "using", "such", "without", "will", "has", "have", "had",
        "not", "no", "only", "but", "however", "and", "or", "which", "where", "when",
        "why", "how", "we", "they", "their", "our", "show", "shows", "demonstrate",
        "compare", "search", "find", "found", "use", "used", "requires", "assumes",
        "must", "always", "during", "between", "through", "under", "over", "into",
    }
    return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if w not in stopwords]


def _hash_text(text: str) -> str:
    """Create a stable hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    set1 = set(_tokenize(text1))
    set2 = set(_tokenize(text2))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _extract_concepts(cluster_items: List[str]) -> List[str]:
    """Extract key concepts from cluster items."""
    all_tokens = []
    for item in cluster_items:
        all_tokens.extend(_tokenize(item))
    counts = Counter(all_tokens)
    return [c for c, _ in counts.most_common(20) if len(c) > 3]


def _detect_domain(problem: str) -> str:
    """Detect the research domain from the problem statement."""
    problem_lower = problem.lower()
    domain_scores = {}
    for domain, keywords in DOMAIN_VALIDATION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in problem_lower)
        if score > 0:
            domain_scores[domain] = score
    if not domain_scores:
        return "general"
    # Prefer more specific domains over general ones
    best_domain = max(domain_scores, key=domain_scores.get)
    # If "battery" is detected, prefer it over "floating" (which shares "energy" keyword)
    if "battery" in domain_scores and best_domain == "floating":
        return "battery"
    return best_domain


# =========================================================
# EVOLUTIONARY OPERATORS
# =========================================================
def _generate_initial_idea(cluster: Dict[str, Any], problem_statement: str) -> str:
    """Generate an initial candidate idea from a cluster."""
    items = cluster.get("items", [])
    concepts = _extract_concepts(items)

    if not concepts:
        concepts = ["optimization", "design", "performance"]

    concept = random.choice(concepts)
    technique = random.choice(TECHNIQUES)

    goal_words = _tokenize(problem_statement)
    goal = " ".join(goal_words[:8]) if goal_words else "optimal performance"

    template = random.choice(IDEA_TEMPLATES)
    # Provide all possible placeholders to avoid KeyError
    return template.format(concept=concept, concept1=concept, concept2=concept, technique=technique, goal=goal)


def _crossover(idea1: str, idea2: str) -> str:
    """Crossover: combine two ideas to create a new one."""
    words1 = _tokenize(idea1)
    words2 = _tokenize(idea2)

    if not words1 or not words2:
        return idea1

    # Take first half of idea1 and second half of idea2
    split1 = max(1, len(words1) // 2)
    split2 = max(1, len(words2) // 2)

    new_words = words1[:split1] + words2[split2:]
    return " ".join(new_words).capitalize()


def _mutate(idea: str, concepts: List[str], problem_statement: str) -> str:
    """Mutation: introduce random variations to an idea."""
    words = _tokenize(idea)
    if not words:
        return idea

    mutation_type = random.random()

    if mutation_type < 0.3 and concepts:
        # Replace a random word with a concept
        idx = random.randrange(len(words))
        words[idx] = random.choice(concepts)
    elif mutation_type < 0.5:
        # Add a technique
        technique = random.choice(TECHNIQUES)
        words.append(technique)
    elif mutation_type < 0.7:
        # Remove a random word
        if len(words) > 3:
            idx = random.randrange(len(words))
            words.pop(idx)
    elif mutation_type < 0.85:
        # Swap two words
        if len(words) > 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
    else:
        # Add goal context
        goal_words = _tokenize(problem_statement)
        if goal_words:
            words.extend(goal_words[:3])

    return " ".join(words).capitalize()


def _fitness(idea: str, domain: str, problem_statement: str) -> float:
    """
    Compute fitness score for an idea based on domain-specific criteria.
    Higher is better.
    """
    score = 0.0
    idea_lower = idea.lower()
    problem_lower = problem_statement.lower()

    # 1. Relevance to problem (0-3)
    problem_tokens = set(_tokenize(problem_statement))
    idea_tokens = set(_tokenize(idea))
    overlap = len(problem_tokens & idea_tokens)
    score += min(3.0, overlap * 0.5)

    # 2. Domain keyword coverage (0-3)
    domain_keywords = DOMAIN_VALIDATION_KEYWORDS.get(domain, DOMAIN_VALIDATION_KEYWORDS["general"])
    keyword_hits = sum(1 for kw in domain_keywords if kw in idea_lower)
    score += min(3.0, keyword_hits * 0.5)

    # 3. Length penalty (0-2): too short or too long is bad
    word_count = len(idea_tokens)
    if 8 <= word_count <= 25:
        score += 2.0
    elif 5 <= word_count <= 35:
        score += 1.0

    # 4. Novelty bonus (0-1): contains technique keywords
    technique_hits = sum(1 for t in TECHNIQUES if t in idea_lower)
    score += min(1.0, technique_hits * 0.25)

    # 5. Specificity bonus (0-1): contains specific numbers or materials
    if re.search(r'\d+', idea):
        score += 0.5
    if any(m in idea_lower for m in ["graphene", "silicon", "polymer", "ceramic", "composite",
                                      "nanoparticle", "nanotube", "electrolyte", "catalyst"]):
        score += 0.5

    return score


# =========================================================
# MAIN EVOLUTIONARY ENGINE
# =========================================================
def generate_candidate_ideas(
    clusters: List[Dict[str, Any]],
    problem_statement: str,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    elitism: float = DEFAULT_ELITISM,
    output_file: str = "candidate_ideas.json",
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Generate candidate ideas from hypothesis clusters using evolutionary computation.

    Args:
        clusters: List of hypothesis cluster dicts
        problem_statement: The research problem
        population_size: Number of ideas to generate
        generations: Number of evolutionary generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        elitism: Fraction of top ideas to preserve each generation
        output_file: Where to save results
        batch_size: Batch size for processing

    Returns:
        Summary dict with idea statistics
    """
    if not clusters:
        return {"error": "No clusters provided", "ideas": 0}

    domain = _detect_domain(problem_statement)
    print("\n" + "=" * 80)
    print("  EVOLUTIONARY IDEA GENERATION ENGINE")
    print("=" * 80)
    print(f"  Clusters: {len(clusters)}")
    print(f"  Target population: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Domain: {domain}")
    print("=" * 80)

    # Step 1: Generate initial population
    print("\n  >> Generating initial population...")
    population: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()

    for cluster in clusters:
        if len(population) >= population_size:
            break
        idea = _generate_initial_idea(cluster, problem_statement)
        h = _hash_text(idea)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        fitness = _fitness(idea, domain, problem_statement)
        population.append({
            "idea": idea,
            "fitness": fitness,
            "cluster_id": cluster.get("cluster_id", "unknown"),
            "generation": 0,
        })

    # Fill remaining population with mutations
    while len(population) < population_size:
        if not population:
            break
        parent = random.choice(population)
        concepts = _extract_concepts(parent.get("cluster_id", "").split())
        idea = _mutate(parent["idea"], concepts, problem_statement)
        h = _hash_text(idea)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        fitness = _fitness(idea, domain, problem_statement)
        population.append({
            "idea": idea,
            "fitness": fitness,
            "cluster_id": parent.get("cluster_id", "unknown"),
            "generation": 0,
        })

    print(f"    -> Initial population: {len(population)} ideas")

    # Step 2: Evolutionary loop
    for gen in range(1, generations + 1):
        print(f"\n  >> Generation {gen}/{generations}...")

        # Sort by fitness
        population.sort(key=lambda x: x["fitness"], reverse=True)

        # Elitism: preserve top ideas
        elite_count = max(1, int(len(population) * elitism))
        elites = population[:elite_count]

        # Generate offspring
        offspring = []
        target_offspring = len(population) - elite_count

        while len(offspring) < target_offspring:
            # Tournament selection
            tournament_size = 3
            parent1 = max(random.sample(population, min(tournament_size, len(population))),
                          key=lambda x: x["fitness"])
            parent2 = max(random.sample(population, min(tournament_size, len(population))),
                          key=lambda x: x["fitness"])

            # Crossover
            if random.random() < crossover_rate:
                child_idea = _crossover(parent1["idea"], parent2["idea"])
            else:
                child_idea = parent1["idea"]

            # Mutation
            if random.random() < mutation_rate:
                concepts = _extract_concepts([parent1["idea"], parent2["idea"]])
                child_idea = _mutate(child_idea, concepts, problem_statement)

            h = _hash_text(child_idea)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            fitness = _fitness(child_idea, domain, problem_statement)
            offspring.append({
                "idea": child_idea,
                "fitness": fitness,
                "cluster_id": parent1.get("cluster_id", "unknown"),
                "generation": gen,
            })

        # New population = elites + offspring
        population = elites + offspring

        # Report
        avg_fitness = sum(x["fitness"] for x in population) / len(population)
        best_fitness = max(x["fitness"] for x in population)
        print(f"    -> Population: {len(population)}, Avg fitness: {avg_fitness:.2f}, Best: {best_fitness:.2f}")

    # Step 3: Final selection
    population.sort(key=lambda x: x["fitness"], reverse=True)

    # Deduplicate final ideas
    final_ideas = []
    seen = set()
    for idea_entry in population:
        h = _hash_text(idea_entry["idea"])
        if h in seen:
            continue
        seen.add(h)
        final_ideas.append(idea_entry)

    # Step 4: Save to disk
    result = {
        "problem_statement": problem_statement,
        "domain": domain,
        "total_ideas": len(final_ideas),
        "generations": generations,
        "population_size": population_size,
        "ideas": final_ideas,
        "generated_at": time.time(),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\n" + "=" * 80)
    print(f"  [OK] Generated {len(final_ideas)} candidate ideas")
    print(f"  [FILE] Saved to: {output_file}")
    print("=" * 80)

    return result


# =========================================================
# BATCHED STREAMING GENERATOR
# =========================================================
def stream_candidate_ideas(
    clusters: List[Dict[str, Any]],
    problem_statement: str,
    target_ideas: int = 1000,
    batch_size: int = 100,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream candidate ideas in batches without keeping all in memory.
    Yields idea dicts one at a time.
    """
    if not clusters:
        return

    domain = _detect_domain(problem_statement)
    seen = set()
    generated = 0

    while generated < target_ideas:
        batch = []
        for _ in range(batch_size):
            if generated >= target_ideas:
                break

            cluster = random.choice(clusters)
            idea = _generate_initial_idea(cluster, problem_statement)

            # Apply random mutations for diversity
            for _ in range(random.randint(0, 3)):
                concepts = _extract_concepts(cluster.get("items", []))
                idea = _mutate(idea, concepts, problem_statement)

            h = _hash_text(idea)
            if h in seen:
                continue
            seen.add(h)

            fitness = _fitness(idea, domain, problem_statement)
            batch.append({
                "idea": idea,
                "fitness": fitness,
                "cluster_id": cluster.get("cluster_id", "unknown"),
                "generation": 0,
            })
            generated += 1

        for entry in batch:
            yield entry


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # Test with sample clusters
    test_clusters = [
        {
            "cluster_id": "HC_00001",
            "items": ["Graphene has exceptional thermal conductivity", "Nano-porous structures increase surface area"],
        },
        {
            "cluster_id": "HC_00002",
            "items": ["Silicon anodes offer 10x capacity but degrade rapidly", "Composite electrodes balance capacity"],
        },
        {
            "cluster_id": "HC_00003",
            "items": ["Solid-state electrolytes improve safety", "Ceramic electrolytes are rigid but high conductivity"],
        },
    ]
    result = generate_candidate_ideas(
        test_clusters,
        "Design a better battery with higher energy density",
        population_size=50,
        generations=3,
    )
    print(f"\nGenerated {result['total_ideas']} ideas")