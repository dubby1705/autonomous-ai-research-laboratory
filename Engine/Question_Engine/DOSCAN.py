import os
import json
import time
import concurrent.futures
from typing import List, Dict, Any
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------
# PYDANTIC SCHEMAS FOR STRUCTURED AI OUTPUT
# ---------------------------------------------------------
class ConceptCluster(BaseModel):
    cluster_name: str
    items: List[str]

class ClusteringResult(BaseModel):
    clusters: List[ConceptCluster]

class ClusterThought(BaseModel):
    breakthrough_found: bool = Field(description="True if a novel, exciting idea was generated. False if the cluster is a dead end.")
    novel_predictions: List[str] = Field(description="Logical extrapolations based on the items in this cluster.")
    randomized_lateral_ideas: List[str] = Field(description="Wild, out-of-the-box randomized thoughts combining these elements.")

# ---------------------------------------------------------
# CORE DOSCAN LOGIC
# ---------------------------------------------------------
def load_knowledge_base(filename: str = "deep_research_knowledge_base.json") -> List[str]:
    """Flattens the entire JSON knowledge base into a single list of raw concepts."""
    if not os.path.exists(filename):
        print(f"❌ Could not find {filename}. Run Phase 1 & 2 first.")
        return []
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    flat_concepts = []
    # Extract strings from all lists and dictionaries in the JSON
    for key, value in data.items():
        if isinstance(value, list):
            flat_concepts.extend(value)
        elif isinstance(value, dict):
            flat_concepts.extend([f"{k}: {v}" for k, v in value.items()])
        elif isinstance(value, str):
            flat_concepts.append(value)
            
    return flat_concepts

def semantic_cluster(client: Groq, concepts: List[str], r: int) -> List[ConceptCluster]:
    """Clusters concepts based on conceptual distance 'r' without using predefined keys."""
    print(f"\n[DOSCAN] 🌀 Running Semantic Clustering at r={r}...")
    
    if r == 1:
        distance_logic = "r=1: Create VERY TIGHT clusters. Group only items that are physically or directly related (e.g., car body, wheel, window)."
    elif r == 2:
        distance_logic = "r=2: Create MODERATE clusters. Group sub-systems and adjacent concepts together."
    else:
        distance_logic = f"r={r}: Create BROAD, wild, cross-disciplinary clusters. Group seemingly unrelated things to force new connections."

    system_prompt = (
        "You are a semantic clustering algorithm. Group the provided concepts based on the requested 'r' distance rule.\n\n"
        "You MUST respond purely with a valid JSON object matching exactly:\n"
        "{\n"
        '  "clusters": [\n'
        '    {"cluster_name": "string", "items": ["string", "string"]}\n'
        '  ]\n'
        "}\n"
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Distance Rule: {distance_logic}\n\nConcepts to cluster:\n{json.dumps(concepts)}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_json = completion.choices[0].message.content
        result = ClusteringResult.model_validate_json(raw_json)
        return result.clusters
    except Exception as e:
        print(f"❌ Clustering failed at r={r}: {e}")
        return []

def think_in_cluster(client: Groq, cluster: ConceptCluster) -> tuple:
    """The Prediction & Randomization engine. Runs on a single cluster."""
    system_prompt = (
        "You are an AI Lateral Thinking Engine. Analyze the items in this cluster. "
        "Generate logical predictions, and then generate highly randomized, lateral 'what-if' thoughts. "
        "If the items are too mundane to yield a genuine breakthrough, set 'breakthrough_found' to false.\n\n"
        "You MUST respond purely with a valid JSON object matching exactly:\n"
        "{\n"
        '  "breakthrough_found": boolean,\n'
        '  "novel_predictions": ["string"],\n'
        '  "randomized_lateral_ideas": ["string"]\n'
        "}\n"
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Cluster Name: {cluster.cluster_name}\nItems: {json.dumps(cluster.items)}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.8, # High temperature for randomized, creative thoughts
        )
        raw_json = completion.choices[0].message.content
        thought = ClusterThought.model_validate_json(raw_json)
        return (cluster, thought)
    except Exception as e:
        return (cluster, None)

def run_doscan_algorithm(max_r: int = 3):
    client = Groq()
    unprocessed_concepts = load_knowledge_base()
    
    if not unprocessed_concepts:
        return

    r = 1
    final_breakthroughs = []

    while r <= max_r and unprocessed_concepts:
        print(f"\n{'='*60}")
        print(f"🧠 INITIATING DOSCAN LAYER (r = {r})")
        print(f"{'='*60}")
        
        clusters = semantic_cluster(client, unprocessed_concepts, r)
        if not clusters:
            print("No clusters formed. Expanding r...")
            r += 1
            continue
            
        print(f"Formed {len(clusters)} clusters. Thinking simultaneously...")
        
        failed_concepts_for_next_r = []

        # Run predictions on all clusters AT THE SAME TIME using ThreadPool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all cluster tasks to the thread pool
            future_to_cluster = {executor.submit(think_in_cluster, client, c): c for c in clusters}
            
            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster, thought = future.result()
                
                if thought and thought.breakthrough_found:
                    print(f"\n✅ BREAKTHROUGH in [{cluster.cluster_name}]:")
                    for idea in thought.randomized_lateral_ideas:
                        print(f"   💡 {idea}")
                    final_breakthroughs.append({
                        "cluster": cluster.cluster_name,
                        "r_level": r,
                        "predictions": thought.novel_predictions,
                        "random_thoughts": thought.randomized_lateral_ideas
                    })
                else:
                    print(f"\n❌ Dead end in [{cluster.cluster_name}]. Tossing items to higher r-level.")
                    # If nothing is found, send these items back to the pool for the next r-level
                    failed_concepts_for_next_r.extend(cluster.items)

        # Update the concept pool for the next iteration
        if failed_concepts_for_next_r:
            unprocessed_concepts = failed_concepts_for_next_r
            r += 1
        else:
            print("\n🎉 All concepts successfully processed into breakthroughs!")
            break
            
    # Save the DOSCAN breakthroughs
    if final_breakthroughs:
        with open("doscan_breakthroughs.json", "w", encoding="utf-8") as f:
            json.dump(final_breakthroughs, f, indent=4)
        print("\n[DOSCAN] ✅ Simultaneous parallel thinking complete. Results saved to 'doscan_breakthroughs.json'.")

if __name__ == "__main__":
    # Ensure your API key is exported in your terminal before running this directly
    run_doscan_algorithm()