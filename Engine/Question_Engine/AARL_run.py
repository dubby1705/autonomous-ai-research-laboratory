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
        print("Phase 1 collapsed. Halting pipeline.")
        return

    print_analysis_report(analysis_result)

    # Condense structural data to build a complete landscape for Phase 2
    analysis_summary = (
        f"Domain: {analysis_result.main_domain}. "
        f"Subs: {', '.join(analysis_result.sub_domains)}. "
        f"Formulas/Laws: {', '.join(analysis_result.governing_laws_and_equations)}. "
        f"Core Focus Items: {', '.join(analysis_result.problem_breakdown)}."
    )

    # --- PHASE 2: RIGOROUS KNOWLEDGE DEPLOYMENT ---
    print("\n>>> PHASE 2: Mapping Scientific Prior Art, Gaps, and Hypotheses...")
    knowledge_result = build_knowledge_base(user_problem, analysis_summary)

    if knowledge_result:
        print("\n" + "="*80)
        print("🧠 DEEP KNOWLEDGE INGESTION COMPLETE")
        print("="*80)
        print(f"\n🎯 CORE THESIS: {knowledge_result.get('core_research_thesis')}")
        
        print("\n🚨 CRITICAL SCIENTIFIC GAPS / UNKNOWNS DETECTED:")
        for unknown in knowledge_result.get("critical_unanswered_unknowns", []):
            print(f"  ⚡ {unknown}")
            
        print("\n🧪 ACTIONABLE HYPOTHESES & EXPERIMENTAL VERIFICATION FLAGS:")
        for hyp, metric in knowledge_result.get("proposed_testable_hypotheses", {}).items():
            print(f"  • Claim: {hyp}")
            print(f"    ↳ Validate via: {metric}")
            
        print("\n[AARL Pipeline State] Structured research data asset finalized and output to 'deep_research_knowledge_base.json'.")
        print("================================================================================")

if __name__ == "__main__":
    main()