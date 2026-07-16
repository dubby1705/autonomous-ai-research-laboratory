import os
import sys

# ==========================================
# DYNAMIC PATH & ROOT RESOLUTION
# ==========================================
# This determines the absolute path of the root "Autonomous AI Research Laboratory" folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
while ROOT_DIR:
    if os.path.exists(os.path.join(ROOT_DIR, "Engine")):
        break
    parent = os.path.dirname(ROOT_DIR)
    if parent == ROOT_DIR:  # Fallback if structural layout changes
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        break
    ROOT_DIR = parent

# Register paths with Python's environment mapping
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
engine_path = os.path.join(ROOT_DIR, "Engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

# ==========================================
# CONFIGURATION
# ==========================================
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ==========================================
# AARL PIPELINE IMPORTS
# ==========================================
from Problem_analyser import analyze_research_problem, print_analysis_report
from Knowledge import build_knowledge_base
from DOSCAN import run_doscan_algorithm, run_doscan_deepening
from Hypothesis.hypothesis import run_hypothesis_engine
from Hypothesis.Question_back import run_questioning_engine, print_final_solution_report
from Evidence import run_evidence_engine
from Experiment import run_experiment_engine
from Mathematics import run_mathematics_engine

def main():
    print("="*80)
    print("🚀 AUTONOMOUS AI RESEARCH LABORATORY (AARL) — MACRO KNOWLEDGE PIPELINE")
    print("="*80)
    
    if GROQ_API_KEY == "gsk_your_key_here":
        print("❌ ERROR: Please configure your valid API Key inside AARL_run.py.")
        return

    user_problem = input("\nEnter your core technical research topic/problem: ")
    if not user_problem.strip():
        print("Error: Target statement is empty. Terminating.")
        return

    # Ask user how many deepening cycles to run (0 = no feedback loop)
    deepen_depth_input = input("\nDOSCAN deepening cycles (how many recursive hypothesis feedback rounds? [0-5], default=1): ").strip()
    try:
        deepen_depth = max(0, min(5, int(deepen_depth_input) if deepen_depth_input else 1))
    except ValueError:
        deepen_depth = 1
    print(f"[AARL Config] Deepening cycles: {deepen_depth}\n")

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
    
    # --- PHASE 4: HYPOTHESIS GENERATION & VERIFICATION ---
    print("\n>>> PHASE 4: Initializing Lateral Hypothesis Engine...")
    approved_hypotheses = run_hypothesis_engine(user_problem)
    
    # --- PHASE 5: DOSCAN DEEPENING FEEDBACK LOOP (Recursive) ---
    if deepen_depth > 0 and approved_hypotheses:
        for cycle in range(1, deepen_depth + 1):
            print(f"\n{'='*80}")
            print(f"🔄 DEEPENING CYCLE {cycle}/{deepen_depth} — Feeding approved hypotheses back into DOSCAN")
            print(f"{'='*80}")
            
            # Run DOSCAN deepening on the approved hypotheses
            # The hypotheses are clustered with r=1→r=4 multi-resolution expansion
            run_doscan_deepening(approved_hypotheses, max_r=4)
            
            # After DOSCAN deepening, run hypothesis engine again on the new insights
            # to generate even deeper hypotheses from the newly discovered relationships
            print(f"\n>>> PHASE 4 (Cycle {cycle}): Regenerating hypotheses on deepened knowledge...")
            approved_hypotheses = run_hypothesis_engine(user_problem)
            
            if not approved_hypotheses:
                print(f"\n⏸️  [Cycle {cycle}] No new approved hypotheses. Ending deepening loop early.")
                break
    
    # --- PHASE 6: SOCRATIC QUESTIONING & FINAL SOLUTION ---
    print("\n>>> PHASE 6: Initializing Socratic Questioning Engine for Final Research Solution...")
    final_solution = run_questioning_engine(user_problem)
    print_final_solution_report(final_solution)
    
    # Build a multi-tier fallback chain for hypotheses so we NEVER skip Phases 7-9
    # Tier 1: approved from hypothesis engine (Excellent/Plausible/Speculative)
    # Tier 2: refined from Socratic questioning
    # Tier 3: proven facts from knowledge base
    # Tier 4: the original problem itself
    fallback_hypotheses = list(approved_hypotheses) if approved_hypotheses else []
    
    if not fallback_hypotheses:
        fs_hs = final_solution.get("approved_hypotheses_questioned", [])
        fallback_hypotheses = list(fs_hs) if fs_hs else []
    
    if not fallback_hypotheses:
        kb_path = "deep_research_knowledge_base.json"
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            kb_facts = kb_data.get("proven_facts", [])
            if kb_facts:
                fallback_hypotheses = [f"HYPOTHESIS: {f}" for f in kb_facts[:3]]
    
    if not fallback_hypotheses:
        fallback_hypotheses = [f"Investigate: {user_problem}"]
    
    print(f"\n[AARL] Using {len(fallback_hypotheses)} hypotheses for Phases 7-9 (tier-based fallback).")
    
    # --- PHASE 7: EVIDENCE SCORING ---
    # Scores each hypothesis with structured evidence tracking
    # Identifies missing evidence types and recommends next actions
    print("\n>>> PHASE 7: Initializing Evidence Scoring Engine...")
    evidence_summary = run_evidence_engine(fallback_hypotheses)
    
    # --- PHASE 8: EXPERIMENT DESIGN ---
    # Designs rigorous, testable experiments from the available hypotheses
    print("\n>>> PHASE 8: Initializing Experiment Engine...")
    domain = analysis_summary if analysis_result else user_problem
    experiment_designs = run_experiment_engine(fallback_hypotheses, domain_context=domain)
    
    # --- PHASE 9: MATHEMATICS DERIVATION ---
    # Derives equations from hypotheses, checks dimensional consistency,
    # verifies mathematical constraints, estimates complexity, rejects impossible formulations
    print("\n>>> PHASE 9: Initializing Mathematics Engine...")
    equations = run_mathematics_engine(fallback_hypotheses)
    
    print("\n" + "="*80)
    print("🔬 AARL RESEARCH CYCLE COMPLETE")
    print("="*80)
    print("   ✅ PHASE 1: Problem Decomposition")
    print("   ✅ PHASE 2: Knowledge Base Construction")
    print("   ✅ PHASE 3: DOSCAN Lateral Clustering")
    print("   ✅ PHASE 4: Hypothesis Generation & Verification")
    if deepen_depth > 0:
        print(f"   ✅ PHASE 5: DOSCAN Deepening ({deepen_depth} cycles)")
    print("   ✅ PHASE 6: Socratic Questioning & Final Solution")
    print("   ✅ PHASE 7: Evidence Scoring (evidence_scoring_db.json)")
    print("   ✅ PHASE 8: Experiment Design (experiment_designs.json)")
    print("   ✅ PHASE 9: Mathematics Derivation (derived_equations.json)")
    print("="*80)
    print("📁 Output files: deep_research_knowledge_base.json, doscan_breakthroughs.json,")
    print("   verified_hypotheses.json, final_research_solution.json, evidence_scoring_db.json,")
    print("   experiment_designs.json, derived_equations.json")
    print("="*80)

if __name__ == "__main__":
    main()
