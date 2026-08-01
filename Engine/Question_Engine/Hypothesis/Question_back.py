import os
import json
import math
import random
import re
import concurrent.futures
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# =========================================================
# SOCRATIC QUESTION SCHEMAS
# =========================================================
SOCRATIC_QUESTIONS = [
    "Why?",
    "What if?",
    "Does this always hold?",
    "Can I improve it?",
    "What would prove this wrong?"
]

class SocraticResponse(BaseModel):
    question: str = Field(description="The Socratic question being answered.")
    answer: str = Field(description="A deep, lateral, randomized scientific answer to the question.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 on how valid this answer is.")
    key_insight: str = Field(description="One-sentence distilled insight from this line of questioning.")

class FinalSolution(BaseModel):
    research_title: str = Field(description="Title of the final research solution.")
    executive_summary: str = Field(description="Concise summary of the total research finding.")
    core_mechanism: str = Field(description="The fundamental mechanism or principle discovered.")
    key_evidence: List[str] = Field(description="List of key evidence supporting the solution.")
    falsification_criteria: List[str] = Field(description="What would prove this solution wrong.")
    open_questions: List[str] = Field(description="Remaining open questions for future research.")
    proposed_experiment: str = Field(description="A single concrete experiment to validate this solution.")

# =========================================================
# LAYER 1: TF-IDF VECTORIZATION (Shared with DOSCAN)
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
# LAYER 2: DBSCAN CLUSTERING (Multi-Resolution)
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
    noise_history = {}
    cluster_counter = 0

    for r in range(1, max_r + 1):
        eps = max(0.0, 0.45 - (r * 0.15))
        if not unprocessed_indices: break
        
        current_docs = [documents[i] for i in unprocessed_indices]
        if len(current_docs) < 2:
            noise_history[r] = unprocessed_indices.copy()
            break
            
        vectors = build_tfidf_vectors(current_docs)
        clusters, noise_local = dbscan_cluster(current_docs, vectors, eps=eps)
        
        # Map cluster indices back to original document indices
        for cid, members in clusters.items():
            cluster_counter += 1
            original_indices = [unprocessed_indices[i] for i in members]
            all_clusters[f"Q_Cluster_{cluster_counter}_r{r}"] = {
                "document_indices": original_indices,
                "documents": [documents[i] for i in original_indices],
                "r_level": r,
                "eps": round(eps, 3),
                "size": len(original_indices)
            }
        
        # Noise becomes the input for next round (with original indices)
        noise_indices = [unprocessed_indices[i] for i in noise_local]
        noise_history[r] = noise_indices
        unprocessed_indices = noise_indices

    return {
        "clusters": all_clusters,
        "final_noise": [documents[i] for i in unprocessed_indices] if unprocessed_indices else [],
        "total_documents": len(documents),
        "total_clusters": len(all_clusters)
    }

# =========================================================
# LAYER 3: SOCRATIC QUESTIONING ENGINE (Iterative Refinement)
# =========================================================
def ask_socratic_questions(client: Groq, hypothesis: str) -> List[Dict[str, Any]]:
    """
    For a single hypothesis, generate answers to all 5 Socratic questions
    using the LLM. Returns a list of dicts with question, answer, confidence, key_insight.
    """
    system_prompt = (
        "You are a Socratic Scientific Challenger. Your purpose is to stress-test hypotheses "
        "by asking and answering deep probing questions. Be creative, lateral, and rigorous.\n\n"
        "You MUST respond purely with a valid JSON object matching this schema:\n"
        "{\n"
        '  "responses": [\n'
        "    {\n"
        '      "question": "The Socratic question",\n'
        '      "answer": "Deep scientific answer",\n'
        '      "confidence": 0.0-1.0,\n'
        '      "key_insight": "One-sentence insight"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Provide exactly 5 responses, one for each question."
    )
    
    # Randomly shuffle the question order for lateral thinking diversity
    shuffled_questions = SOCRATIC_QUESTIONS.copy()
    random.shuffle(shuffled_questions)
    
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(shuffled_questions)])
    
    prompt = (
        f"Hypothesis to challenge: {hypothesis}\n\n"
        f"Answer each of these Socratic questions with deep scientific reasoning:\n{questions_text}\n\n"
        "Be random, lateral, and think outside the box. Include counter-arguments."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.9,  # High temperature for randomness
        )
        raw_json = completion.choices[0].message.content
        data = json.loads(raw_json)
        responses = data.get("responses", [])
        
        # Sort back to original question order for consistency
        question_order = {q: i for i, q in enumerate(SOCRATIC_QUESTIONS)}
        responses.sort(key=lambda r: question_order.get(r.get("question", ""), 99))
        
        return responses
    except Exception as e:
        print(f"   ❌ Error in Socratic questioning: {e}")
        return []


def refine_hypothesis_socratic(client: Groq, hypothesis: str, 
                                max_iterations: int = 3) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Iteratively refines a hypothesis using Socratic questioning.
    
    For each iteration:
    1. Ask 5 Socratic questions about the current hypothesis
    2. Use the answers to produce a refined hypothesis
    3. If the refined hypothesis is meaningfully different, repeat
    4. Stop when convergence (hypothesis stops changing) or max_iterations reached
    
    Returns:
        (final_hypothesis, all_responses, iterations_used)
    """
    current_hypothesis = hypothesis
    all_responses = []
    
    for iteration in range(max_iterations):
        print(f"\n      🔄 Socratic refinement iteration {iteration + 1}/{max_iterations}")
        
        # Step 1: Ask questions about current hypothesis
        responses = ask_socratic_questions(client, current_hypothesis)
        if not responses:
            break
        
        all_responses.extend(responses)
        
        # Step 2: Use LLM to synthesize a refined hypothesis from the answers
        refine_prompt = (
            "You are a Hypothesis Refinement Engine. Given a hypothesis and its Socratic "
            "question-answer pairs, produce a STRICTLY BETTER version of the hypothesis.\n\n"
            "Rules:\n"
            "- Address the weaknesses revealed by the questions\n"
            "- Incorporate the key insights from the answers\n"
            "- Make the hypothesis more specific, testable, and falsifiable\n"
            "- If the hypothesis cannot be improved, return it unchanged\n\n"
            "You MUST respond with a valid JSON object:\n"
            "{\n"
            '  "refined_hypothesis": "The improved hypothesis",\n'
            '  "changes_made": "What changed and why",\n'
            '  "improvement_score": 0.0-1.0\n'
            "}\n"
        )
        
        qa_summary = "\n".join([
            f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}\nInsight: {r.get('key_insight', '')}"
            for r in responses
        ])
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": refine_prompt},
                    {"role": "user", "content": 
                        f"Original hypothesis: {current_hypothesis}\n\n"
                        f"Socratic analysis:\n{qa_summary}\n\n"
                        f"Produce a refined hypothesis."}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            refine_data = json.loads(completion.choices[0].message.content)
            refined = refine_data.get("refined_hypothesis", current_hypothesis)
            improvement = refine_data.get("improvement_score", 0.0)
            
            # Step 3: Check for convergence
            # Simple string similarity: if they're very similar, stop
            words_old = set(current_hypothesis.lower().split())
            words_new = set(refined.lower().split())
            if words_old and words_new:
                jaccard = len(words_old & words_new) / len(words_old | words_new)
            else:
                jaccard = 0.0
            
            print(f"      📊 Improvement score: {improvement:.2f}, Similarity to previous: {jaccard:.2f}")
            
            if jaccard > 0.85 or improvement < 0.1:
                print(f"      ✅ Hypothesis converged after {iteration + 1} iterations.")
                break
            
            current_hypothesis = refined
            
        except Exception as e:
            print(f"      ⚠️ Refinement error: {e}")
            break
    
    return current_hypothesis, all_responses, iteration + 1

def generate_final_solution(client: Groq, all_socratic_results: List[Dict[str, Any]], original_problem: str) -> Dict[str, Any]:
    """
    Synthesizes all Socratic answers + cluster analysis into a final research solution
    using the LLM.
    """
    system_prompt = (
        "You are a Nobel-level Research Synthesizer. Given a research problem and a set of "
        "Socratic question-answer pairs from deep probing, synthesize a final research solution.\n\n"
        "You MUST respond purely with a valid JSON object matching this schema:\n"
        "{\n"
        '  "research_title": "string",\n'
        '  "executive_summary": "string",\n'
        '  "core_mechanism": "string",\n'
        '  "key_evidence": ["string"],\n'
        '  "falsification_criteria": ["string"],\n'
        '  "open_questions": ["string"],\n'
        '  "proposed_experiment": "string"\n'
        "}\n"
    )
    
    # Build a compact summary of all Q&A
    qa_summary = ""
    for item in all_socratic_results:
        qa_summary += f"Q: {item['question']}\nA: {item['answer']}\nInsight: {item.get('key_insight', '')}\n\n"
    
    prompt = (
        f"Original Research Problem: {original_problem}\n\n"
        f"Socratic Deep Probe Results:\n{qa_summary}\n\n"
        "Synthesize a definitive final research solution. Be bold but rigorous."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4,  # Moderate temperature for synthesis
        )
        raw_json = completion.choices[0].message.content
        return FinalSolution.model_validate_json(raw_json).model_dump()
    except Exception as e:
        print(f"❌ Error in Final Solution synthesis: {e}")
        return {}

# =========================================================
# LAYER 4: MAIN QUESTIONING ENGINE
# =========================================================
def _generate_fallback_socratic_responses(hypothesis: str) -> List[Dict[str, Any]]:
    """Generate deterministic Socratic responses when LLM is unavailable."""
    responses = []
    for question in SOCRATIC_QUESTIONS:
        if question == "Why?":
            answer = f"This approach addresses the core challenge by leveraging established scientific principles and novel optimization strategies for: {hypothesis[:100]}"
            insight = "The hypothesis is grounded in fundamental scientific principles"
        elif question == "What if?":
            answer = f"Alternative approaches could yield different trade-offs, but this hypothesis offers a balanced solution with measurable outcomes"
            insight = "Multiple approaches exist; this one balances performance and feasibility"
        elif question == "Does this always hold?":
            answer = "The hypothesis holds under standard operating conditions but may require validation under extreme or edge cases"
            insight = "Boundary conditions need empirical validation"
        elif question == "Can I improve it?":
            answer = "Further refinement through iterative testing and parameter optimization could enhance the proposed approach"
            insight = "Iterative improvement through experimental feedback is possible"
        else:  # "What would prove this wrong?"
            answer = "Failure to achieve measurable improvement over baseline metrics would falsify this hypothesis"
            insight = "Clear falsification criteria: no improvement over baseline"
        
        responses.append({
            "question": question,
            "answer": answer,
            "confidence": 0.6,
            "key_insight": insight,
        })
    return responses


def _generate_fallback_final_solution(all_socratic_results: List[Dict[str, Any]], original_problem: str) -> Dict[str, Any]:
    """Generate a deterministic final solution when LLM is unavailable."""
    insights = [r.get("key_insight", "") for r in all_socratic_results[:5]]
    return {
        "research_title": f"Research Solution for: {original_problem[:80]}",
        "executive_summary": f"This research proposes a novel approach to address: {original_problem}. Based on Socratic analysis, the approach is grounded in established scientific principles and offers measurable improvement over current baselines.",
        "core_mechanism": "The proposed approach leverages domain-specific optimization strategies combined with established scientific principles to achieve measurable performance improvements.",
        "key_evidence": insights if insights else ["Socratic analysis supports the proposed approach"],
        "falsification_criteria": [
            "Failure to achieve measurable improvement over baseline metrics",
            "Inconsistency with established physical or chemical laws",
            "Inability to reproduce results under controlled conditions",
        ],
        "open_questions": [
            "What are the optimal parameters for maximum performance?",
            "How does the approach scale under different operating conditions?",
            "What are the long-term reliability implications?",
        ],
        "proposed_experiment": "Construct a prototype and measure performance metrics against the current state-of-the-art baseline under controlled conditions.",
    }


def run_questioning_engine(original_problem: str) -> Dict[str, Any]:
    """
    Main entry point for the Questioning Engine.
    1. Loads approved hypotheses from verified_hypotheses.json
    2. For each, asks 5 Socratic questions via LLM
    3. Collects all answers, clusters them with multi-resolution DBSCAN
    4. Generates a final research solution
    5. Saves everything to disk
    """
    print("\n" + "="*80)
    print("  PHASE 5: SOCRATIC QUESTIONING ENGINE — DEEP PROBING & FINAL SOLUTION")
    print("="*80)
    
    # Check if Groq is available
    _groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    client = None
    if _groq_key:
        try:
            client = Groq()
        except Exception as e:
            print(f"   Groq client init failed: {e}")
            client = None
    
    if client is None:
        print("   LLM unavailable. Using deterministic fallback for Socratic questioning...")
    
    # Step 1: Load verified hypotheses
    approved_hypotheses = []
    if os.path.exists("verified_hypotheses.json"):
        with open("verified_hypotheses.json", "r", encoding="utf-8") as f:
            all_verified = json.load(f)
            for item in all_verified:
                if item.get("makes_sense") and item.get("relevance_score", 0) >= 7:
                    refined = item.get("refined_hypothesis", item.get("original_prediction", ""))
                    approved_hypotheses.append(refined)
    
    if not approved_hypotheses:
        print("\n⚠️ No approved hypotheses found. Using the original problem as seed.\n")
        approved_hypotheses = [f"Core problem: {original_problem}"]
    
    print(f"\n📋 Loaded {len(approved_hypotheses)} approved hypotheses for deep questioning.\n")
    
    # Step 2: Iterative Socratic refinement for each hypothesis
    all_socratic_responses = []
    socratic_log = []
    refined_hypotheses = []
    
    for i, hypothesis in enumerate(approved_hypotheses, 1):
        print(f"\n  Deep Probing Hypothesis #{i}:")
        print(f"   {hypothesis[:120]}...")
        
        if client is not None:
            # Use iterative refinement: question -> refine -> question again -> converge
            final_hypothesis, responses, iterations_used = refine_hypothesis_socratic(
                client, hypothesis, max_iterations=3
            )
        else:
            # Fallback: generate deterministic Socratic responses
            responses = _generate_fallback_socratic_responses(hypothesis)
            final_hypothesis = hypothesis
            iterations_used = 0
            print(f"   [FALLBACK] Generated {len(responses)} deterministic Socratic responses")
        
        if responses:
            for j, resp in enumerate(responses, 1):
                question = resp.get("question", "Unknown")
                answer = resp.get("answer", "")
                conf = resp.get("confidence", 0.0)
                print(f"   [{j}/5] {question}")
                print(f"       ↳ Confidence: {conf:.2f} | Insight: {resp.get('key_insight', '')[:80]}...")
                
                all_socratic_responses.append({
                    "source_hypothesis": hypothesis,
                    "refined_hypothesis": final_hypothesis,
                    "question": question,
                    "answer": answer,
                    "confidence": conf,
                    "key_insight": resp.get("key_insight", ""),
                    "hypothesis_index": i,
                    "refinement_iterations": iterations_used
                })
                socratic_log.append(answer)
        
        refined_hypotheses.append(final_hypothesis)
        if final_hypothesis != hypothesis:
            print(f"\n   📈 Hypothesis refined after {iterations_used} iterations:")
            print(f"      Before: {hypothesis[:100]}...")
            print(f"      After:  {final_hypothesis[:100]}...")
    
    if not socratic_log:
        print("   No Socratic responses generated. Aborting.")
        return {"error": "No responses", "approved_hypotheses_questioned": approved_hypotheses}
    
    print(f"\n📊 Collected {len(socratic_log)} Socratic answer passages.")
    
    # Step 3: Multi-resolution DBSCAN clustering on the answer documents
    print("\n⚡ Running multi-resolution DBSCAN clustering on Socratic answers...")
    cluster_result = multi_resolution_cluster(socratic_log, max_r=4)
    
    print(f"   Total documents: {cluster_result['total_documents']}")
    print(f"   Clusters formed: {cluster_result['total_clusters']}")
    for cname, cinfo in cluster_result['clusters'].items():
        print(f"      {cname}: {cinfo['size']} items (r={cinfo['r_level']}, eps={cinfo['eps']})")
    if cluster_result['final_noise']:
        print(f"   Noise (unclustered): {len(cluster_result['final_noise'])} items")
    
    # Step 4: Show random samples from each cluster
    print("\n🎲 Random lateral insights from each cluster:")
    for cname, cinfo in cluster_result['clusters'].items():
        sample = random.choice(cinfo['documents'])
        print(f"   [{cname}] → {sample[:150]}...")
    
    # Step 5: Generate the final research solution
    print("\n  Synthesizing final research solution from all deep probes...")
    if client is not None:
        final_solution = generate_final_solution(client, all_socratic_responses, original_problem)
    else:
        final_solution = _generate_fallback_final_solution(all_socratic_responses, original_problem)
        print("   [FALLBACK] Generated deterministic final solution")
    
    # Step 6: Package the complete result
    final_output = {
        "original_problem": original_problem,
        "approved_hypotheses_questioned": approved_hypotheses,
        "socratic_probing": {
            "total_questions_asked": len(all_socratic_responses),
            "questions_per_hypothesis": 5,
            "responses": all_socratic_responses
        },
        "clustering_analysis": {
            "algorithm": "DBSCAN (multi-resolution r=1→4)",
            "eps_decay": [max(0.0, 0.45 - (r * 0.15)) for r in range(1, 5)],
            "total_clusters": cluster_result['total_clusters'],
            "cluster_details": {
                cname: {
                    "size": cinfo["size"],
                    "r_level": cinfo["r_level"],
                    "eps": cinfo["eps"],
                    "sample_documents": random.sample(cinfo["documents"], min(2, cinfo["size"]))
                }
                for cname, cinfo in cluster_result['clusters'].items()
            },
            "noise_items": cluster_result['final_noise']
        },
        "final_research_solution": final_solution
    }
    
    # Step 7: Save to disk
    output_file = "final_research_solution.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print("\n" + "="*80)
    print("🏁 SOCRATIC QUESTIONING ENGINE COMPLETE")
    print("="*80)
    print(f"   ❓ Total Socratic questions asked: {len(all_socratic_responses)}")
    print(f"   🔬 DBSCAN clusters formed: {cluster_result['total_clusters']}")
    print(f"   📄 Final solution saved to: '{output_file}'")
    print("="*80)
    
    return final_output

def print_final_solution_report(solution: Dict[str, Any]):
    """Pretty-prints the final research solution to console."""
    if not solution or "final_research_solution" not in solution:
        print("\n❌ No final solution to display.")
        return
    
    fs = solution["final_research_solution"]
    print("\n" + "="*80)
    print("🏆 FINAL RESEARCH SOLUTION")
    print("="*80)
    print(f"\n📌 TITLE: {fs.get('research_title', 'N/A')}")
    print(f"\n📝 EXECUTIVE SUMMARY:")
    print(f"   {fs.get('executive_summary', 'N/A')}")
    print(f"\n⚙️  CORE MECHANISM:")
    print(f"   {fs.get('core_mechanism', 'N/A')}")
    print(f"\n🔑 KEY EVIDENCE:")
    for ev in fs.get('key_evidence', []):
        print(f"   • {ev}")
    print(f"\n❌ FALSIFICATION CRITERIA (What would prove this wrong):")
    for crit in fs.get('falsification_criteria', []):
        print(f"   • {crit}")
    print(f"\n❓ OPEN QUESTIONS:")
    for q in fs.get('open_questions', []):
        print(f"   • {q}")
    print(f"\n🧪 PROPOSED EXPERIMENT:")
    print(f"   {fs.get('proposed_experiment', 'N/A')}")
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test with a sample problem if run standalone
    test_problem = "Designing an optimal battery with high energy density and safety"
    result = run_questioning_engine(test_problem)
    print_final_solution_report(result)