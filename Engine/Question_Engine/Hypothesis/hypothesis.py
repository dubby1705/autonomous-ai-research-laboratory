import os
import json
import concurrent.futures
from typing import List, Dict
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# Performance optimization settings
MAX_PARALLEL_VERIFICATIONS = 3  # Parallel verification calls

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class GeneratedPredictions(BaseModel):
    predictions: List[str] = Field(description="A list of highly lateral, randomized, and creative scientific hypotheses.")

class VerificationResult(BaseModel):
    classification: str = Field(description="One of: Excellent, Plausible, Speculative, Weak, Contradicted")
    novelty: int = Field(description="Score 1-10: How new/unexpected is the idea?")
    feasibility: int = Field(description="Score 1-10: Can this be realistically tested with current technology?")
    evidence: int = Field(description="Score 1-10: How much existing evidence supports this?")
    consistency: int = Field(description="Score 1-10: Is it logically consistent with known laws?")
    testability: int = Field(description="Score 1-10: Can we design a clear experiment to test it?")
    risk: int = Field(description="Score 1-10: How likely is catastrophic failure or fatal flaw (higher = riskier)?")
    final_score: float = Field(description="Weighted average (0.0-10.0): novelty*0.15 + evidence*0.25 + feasibility*0.15 + consistency*0.20 + testability*0.15 + (10-risk)*0.10")
    reasoning: str = Field(description="Detailed analysis of each dimension and the overall judgment.")
    refined_hypothesis: str = Field(description="A polished version of the hypothesis incorporating identified weaknesses.")
    primary_weakness: str = Field(description="The single biggest weakness of this hypothesis.")

# ==========================================
# HYPOTHESIS ENGINE LOGIC
# ==========================================
def generate_random_predictions(client: Groq, problem_statement: str, kb_context: str) -> List[str]:
    print("   [Generator] Brainstorming randomized lateral predictions...")
    
    system_prompt = (
        "You are the Random Idea Generator of an Autonomous AI Research Lab. "
        "Your goal is to generate 5 wild, lateral, and highly creative scientific predictions "
        "or hypotheses based on the problem statement. Don't worry about perfect accuracy here; "
        "focus on out-of-the-box thinking.\n\n"
        "You MUST respond purely with a valid JSON object matching this schema:\n"
        "{\n"
        '  "predictions": ["string", "string", ...]\n'
        "}\n"
    )
    
    prompt = f"Problem: {problem_statement}\nContext: {kb_context}\nGenerate 5 random hypotheses."

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.8, # Higher temperature for randomness
        )
        raw_json = completion.choices[0].message.content
        data = GeneratedPredictions.model_validate_json(raw_json)
        return data.predictions
    except Exception as e:
        print(f"❌ Error in Generator: {e}")
        return []

def verify_prediction(client: Groq, problem_statement: str, prediction: str) -> dict:
    system_prompt = (
        "You are a rigorous Scientific Reviewer. Evaluate hypotheses across 6 dimensions.\n\n"
        "CLASSIFICATION (choose one):\n"
        "- Excellent: Strong evidence, clear mechanism, testable, consistent with known laws\n"
        "- Plausible: Reasonable idea, some evidence, testable but may have challenges\n"
        "- Speculative: Interesting but lacks evidence or clear mechanism; worth exploring\n"
        "- Weak: Significant flaws, unclear mechanism, poor evidence, or untestable\n"
        "- Contradicted: Conflicts with established laws or strong contradictory evidence\n\n"
        "SCORING (1-10 for each dimension):\n"
        "- novelty: How new/unexpected is the idea? (1=well-known, 10=truly novel)\n"
        "- feasibility: Can this be realistically tested with current technology? (1=impossible, 10=trivial)\n"
        "- evidence: How much existing evidence supports this? (1=none, 10=strong empirical support)\n"
        "- consistency: Is it logically consistent with known laws? (1=contradicts known laws, 10=perfectly consistent)\n"
        "- testability: Can we design a clear experiment? (1=untestable, 10=clear experiment exists)\n"
        "- risk: How likely is catastrophic failure or fatal flaw? (1=very safe, 10=extremely risky)\n\n"
        "final_score = novelty*0.15 + evidence*0.25 + feasibility*0.15 + consistency*0.20 + testability*0.15 + (10-risk)*0.10\n\n"
        "IMPORTANT: 'Hard' is NOT the same as 'Impossible'. A hypothesis can be difficult but still Plausible.\n"
        "Active research areas (quantum computing, fusion, etc.) are NOT 'Contradicted' just because they're hard.\n"
        "Be fair: distinguish between 'this is hard' and 'this violates physics'.\n\n"
        "You MUST respond with a valid JSON object:\n"
        "{\n"
        '  "classification": "Plausible",\n'
        '  "novelty": 7,\n'
        '  "feasibility": 5,\n'
        '  "evidence": 4,\n'
        '  "consistency": 8,\n'
        '  "testability": 6,\n'
        '  "risk": 5,\n'
        '  "final_score": 6.0,\n'
        '  "reasoning": "Detailed analysis...",\n'
        '  "refined_hypothesis": "Improved version...",\n'
        '  "primary_weakness": "The single biggest weakness"\n'
        "}\n"
    )

    prompt = f"Core Problem: {problem_statement}\n\nHypothesis to Evaluate: {prediction}\n\nScore across all 6 dimensions. Be fair but rigorous."

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1, # Low temperature for strict logic
        )
        raw_json = completion.choices[0].message.content
        data = VerificationResult.model_validate_json(raw_json)
        return data.model_dump()
    except Exception as e:
        print(f"❌ Error in Verifier: {e}")
        return None

def run_hypothesis_engine(problem_statement: str) -> List[str]:
    """
    Returns a list of approved refined hypotheses (classification in Excellent,Plausible,Speculative).
    These are fed back into DOSCAN for deeper recursive clustering.
    """
    print("\n" + "="*80)
    print("🧠 PHASE 4: HYPOTHESIS ENGINE (6-DIMENSION SCORING & CLASSIFICATION)")
    print("="*80)
    
    client = Groq()
    
    # Try to load context from Phase 2
    kb_context = "No prior knowledge base found."
    if os.path.exists("deep_research_knowledge_base.json"):
        with open("deep_research_knowledge_base.json", "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            kb_context = kb_data.get("core_research_thesis", "")

    # Step 1: Generate
    raw_predictions = generate_random_predictions(client, problem_statement, kb_context)
    if not raw_predictions:
        print("❌ Failed to generate hypotheses.")
        return []

    # Step 2: Verify (PARALLEL OPTIMIZATION)
    verified_results = []
    approved_refined_hypotheses = []
    summary_stats = {"Excellent": 0, "Plausible": 0, "Speculative": 0, "Weak": 0, "Contradicted": 0}
    
    print("\n   [Verifier] Evaluating predictions across 6 dimensions (PARALLEL)...")
    
    # Create a thread pool for parallel verification
    def verify_single_prediction(pred):
        """Wrapper for parallel verification"""
        try:
            return verify_prediction(Groq(), problem_statement, pred)
        except Exception as e:
            print(f"      ⚠️ Verification error: {e}")
            return None
    
    # Run verifications in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_VERIFICATIONS) as executor:
        futures = {executor.submit(verify_single_prediction, pred): pred for pred in raw_predictions}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            pred = futures[future]
            try:
                verification = future.result()
                if verification:
                    verification["original_prediction"] = pred
                    verified_results.append(verification)
                    
                    cls = verification.get("classification", "Weak")
                    score = verification.get("final_score", 0.0)
                    summary_stats[cls] = summary_stats.get(cls, 0) + 1
                    
                    # Console output
                    status_icons = {
                        "Excellent": "🏆", "Plausible": "✅", "Speculative": "🔮",
                        "Weak": "⚠️", "Contradicted": "❌"
                    }
                    icon = status_icons.get(cls, "❓")
                    print(f"\n   [{i}/{len(raw_predictions)}] {icon} {cls} (Score: {score:.1f}/10)")
                    print(f"      {pred[:80]}...")
                    
                    # Collect approved: Excellent, Plausible, and Speculative all move forward
                    if cls in ("Excellent", "Plausible", "Speculative"):
                        refined = verification.get("refined_hypothesis", pred)
                        approved_refined_hypotheses.append(refined)
            except Exception as e:
                print(f"\n   [{i}/{len(raw_predictions)}] ❌ Error: {e}")
    
    # Save to disk
    output_filename = "verified_hypotheses.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(verified_results, f, indent=4)
        
    total = len(raw_predictions)
    accepted = len(approved_refined_hypotheses)
    
    print("\n" + "="*80)
    print(f"� HYPOTHESIS VERIFICATION SUMMARY")
    print("="*80)
    print(f"   Generated: {total}")
    for cls, count in summary_stats.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"   {cls}: {count} ({pct:.0f}%)")
    print(f"   ─────────────────────────────")
    print(f"   Accepted (Ex+Pl+Sp): {accepted}/{total} ({accepted/total*100:.0f}%)")
    print(f"   Filtered (Weak+Cont): {total-accepted}/{total} ({(total-accepted)/total*100:.0f}%)")
    print("="*80)
    
    return approved_refined_hypotheses
