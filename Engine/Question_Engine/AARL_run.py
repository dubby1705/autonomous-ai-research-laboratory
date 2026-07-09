import os
import json

# ==========================================
# PASTE YOUR WORKING GROQ API KEY HERE ONCE
# ==========================================
GROQ_API_KEY = "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"
# ==========================================

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from Problem_analyser import analyze_research_problem, print_analysis_report
from Knowledge import build_knowledge_base
from DOSCAN import run_doscan_algorithm  # <-- NEW IMPORT

def main():
    print("="*80)
    print("🚀 AUTONOMOUS AI RESEARCH LABORATORY (AARL) — MACRO KNOWLEDGE PIPELINE")
    print("="*80)
    
    if GROQ_API_KEY.startswith("gsk_your_key_here"):
        print("❌ ERROR: Please configure your valid API Key inside AARL_run.py.")
        return

    user_problem = input("\nEnter your core technical research topic/problem: ")
    if not user_problem.strip():
        print("Error: Target statement is empty. Terminating.")
        return

    # --- PHASE 1: STRUCTURAL ANALYSIS ---
    print("\n>>> PHASE 1: Executing Matrix Decomposition...")
    analysis_result = analyze_research_problem(user_problem)
    if not analysis_result:
        return
    print_analysis_report(analysis_result)
    analysis_summary = (
        f"Domain: {analysis_result.main_domain}. Subs: {', '.join(analysis_result.sub_domains)}. "
        f"Formulas/Laws: {', '.join(analysis_result.governing_laws_and_equations)}. "
        f"Core Focus Items: {', '.join(analysis_result.problem_breakdown)}."
    )

    # --- PHASE 2: RIGOROUS KNOWLEDGE DEPLOYMENT ---
    print("\n>>> PHASE 2: Mapping Scientific Prior Art, Gaps, and Hypotheses...")
    knowledge_result = build_knowledge_base(user_problem, analysis_summary)
    if not knowledge_result:
        return
    print("\n[AARL Status] Knowledge Base built.")

    # --- PHASE 3: DOSCAN PARALLEL LATERAL THINKING ---
    print("\n>>> PHASE 3: Initializing DOSCAN Lateral Neural Expansion...")
    run_doscan_algorithm()
    
    print("\n================================================================================")
    print("🔬 RESEARCH CYCLE COMPLETE. ALL DATA ARTIFACTS SAVED TO DISK.")
    print("================================================================================")

if __name__ == "__main__":
    main()