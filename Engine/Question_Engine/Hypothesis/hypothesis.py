import os
import json
from typing import List, Dict
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class GeneratedPredictions(BaseModel):
    predictions: List[str] = Field(description="A list of highly lateral, randomized, and creative scientific hypotheses.")

class VerificationResult(BaseModel):
    makes_sense: bool = Field(description="True if the prediction logically addresses the core problem, False if it is complete nonsense.")
    relevance_score: int = Field(description="Score from 1 to 10 evaluating the scientific viability of the prediction.")
    reasoning: str = Field(description="Deep logical analysis of WHY it makes sense or why it fails.")
    refined_hypothesis: str = Field(description="If it makes sense, a polished version of the hypothesis. If not, explain the fatal flaw.")

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
        "You are the strict Scientific Verifier of an Autonomous AI Research Lab. "
        "You will be given a core problem and a random prediction. Your job is to rigorously "
        "evaluate if this prediction makes ANY logical sense in solving or understanding the problem.\n\n"
        "You MUST respond purely with a valid JSON object matching this schema:\n"
        "{\n"
        '  "makes_sense": boolean,\n'
        '  "relevance_score": integer (1-10),\n'
        '  "reasoning": "string",\n'
        '  "refined_hypothesis": "string"\n'
        "}\n"
    )

    prompt = f"Core Problem: {problem_statement}\nRandom Prediction to Verify: {prediction}"

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

def run_hypothesis_engine(problem_statement: str):
    print("\n" + "="*80)
    print("🧠 PHASE 4: HYPOTHESIS ENGINE (RANDOM GENERATION & VERIFICATION)")
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
        return

    # Step 2: Verify
    verified_results = []
    print("\n   [Verifier] Cross-examining predictions against the core problem...")
    
    for i, pred in enumerate(raw_predictions, 1):
        print(f"\n   Testing Hypothesis #{i}...")
        verification = verify_prediction(client, problem_statement, pred)
        
        if verification:
            verification["original_prediction"] = pred
            verified_results.append(verification)
            
            # Console output formatting
            status = "✅ MAKES SENSE" if verification["makes_sense"] else "❌ NONSENSE"
            score = verification["relevance_score"]
            print(f"      Status: {status} (Score: {score}/10)")
            print(f"      Original: {pred}")
            print(f"      Reasoning: {verification['reasoning']}")

    # Save to disk
    output_filename = "verified_hypotheses.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(verified_results, f, indent=4)
        
    print("\n" + "="*80)
    print(f"💾 Verified hypotheses successfully saved to '{output_filename}'")
    print("="*80)