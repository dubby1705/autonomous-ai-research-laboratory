import os
import sys
import json
import random as _random
import time

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
from Mathematics import run_mathematics_engine

def _load_json(path: str) -> dict:
    """Safely load a JSON file, returning empty dict on failure."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _count_items(data, key: str) -> int:
    """Count items in a JSON key, handling both lists and dicts."""
    val = data.get(key, {})
    if isinstance(val, list):
        return len(val)
    if isinstance(val, dict):
        return len(val)
    if isinstance(val, str):
        return 1
    return 0

def main():
    aarl_start = time.time()
    
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
    
    # ---- Stats tracking across all phases ----
    stats = {
        "problem": user_problem,
        "kb_nodes": 0,
        "relationships": 0,
        "doscan_clusters": 0,
        "hypotheses_generated": 0,
        "hypotheses_approved": 0,
        "hypotheses_rejected": 0,
        "top_confidence": 0.0,
        "deepening_cycles": 0
    }

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
    
    # --- PHASE 5: DEEPENING FEEDBACK LOOP (Knowledge Base + DBSCAN + LLM Validation) ---
    # For each approved hypothesis: mine knowledge base → TF-IDF → DBSCAN cluster (r=1→r=4)
    # → LLM only says "does this make sense?" → deepen if yes
    if deepen_depth > 0 and approved_hypotheses:
        for cycle in range(1, deepen_depth + 1):
            print(f"\n{'='*80}")
            print(f"🔄 DEEPENING CYCLE {cycle}/{deepen_depth} — KB + DBSCAN + LLM Validation")
            print(f"{'='*80}")
            
            deepened_hypotheses = []
            
            for hyp in approved_hypotheses:
                print(f"\n   📝 Deepening: {hyp[:100]}...")
                
                # Step 1: Load knowledge base content
                kb_items = []
                if os.path.exists("deep_research_knowledge_base.json"):
                    with open("deep_research_knowledge_base.json", "r", encoding="utf-8") as f:
                        kb_data = json.load(f)
                    for key, value in kb_data.items():
                        if isinstance(value, list):
                            kb_items.extend(value)
                        elif isinstance(value, dict):
                            for k, v in value.items():
                                kb_items.append(f"{k}: {v}")
                        elif isinstance(value, str):
                            kb_items.append(value)
                
                if not kb_items:
                    print(f"      ⏸️  No KB data. Skipping.")
                    continue
                
                # Step 2: Filter KB items relevant to the hypothesis (simple keyword overlap)
                hyp_words = set(hyp.lower().split())
                relevant_items = []
                for item in kb_items:
                    item_words = set(item.lower().split())
                    overlap = len(hyp_words & item_words)
                    if overlap >= 2:  # At least 2 words in common
                        relevant_items.append(item)
                
                if len(relevant_items) < 3:
                    # If not enough relevant items, add some random ones from KB
                    relevant_items.extend(_random.sample(kb_items, min(5, len(kb_items))))
                
                print(f"      📚 Found {len(relevant_items)} relevant KB items for deepening")
                
                # Step 3: TF-IDF + DBSCAN cluster (r=1→r=4) on the relevant KB items
                from DOSCAN import build_tfidf_vectors as doscan_tfidf, dbscan_text_cluster as doscan_dbscan
                
                current_items = list(set(relevant_items))  # Deduplicate
                
                for r in range(1, 5):  # r=1 to r=4
                    eps = max(0.0, 0.45 - (r * 0.15))
                    if len(current_items) < 2:
                        break
                    
                    vectors = doscan_tfidf(current_items)
                    clusters, noise = doscan_dbscan(current_items, vectors, eps=eps)
                    
                    print(f"      🔬 r={r} (eps={eps:.2f}): {len(clusters)} clusters, {len(noise)} noise items")
                    
                    # Extract insights from each cluster
                    for ci, cluster in enumerate(clusters, 1):
                        cluster_text = " | ".join(cluster["items"][:3])  # Top 3 items
                        print(f"         Cluster {ci}: '{cluster_text[:80]}...'")
                        
                        # Use LLM to validate if this cluster's content deepens the hypothesis
                        from groq import Groq
                        v_client = Groq()
                        try:
                            from groq import Groq as _G
                            
                            completion = v_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": "You validate if a knowledge cluster deepens a hypothesis. Answer ONLY with JSON: {\"deepens\": true/false, \"reason\": \"brief\", \"refined_hypothesis\": \"improved version if deepens\"}"},
                                    {"role": "user", "content": f"Hypothesis: {hyp}\n\nNew knowledge cluster: {cluster_text}\n\nDoes this knowledge deepen the hypothesis? If yes, produce a refined hypothesis."}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.1,
                            )
                            result = json.loads(completion.choices[0].message.content)
                            
                            if result.get("deepens", False):
                                refined = result.get("refined_hypothesis", hyp)
                                deepened_hypotheses.append(refined)
                                print(f"            ✅ Deepens: {refined[:100]}...")
                            else:
                                print(f"            ❌ Does not deepen: {result.get('reason', '')[:80]}")
                        except Exception as e:
                            print(f"            ⚠️ Validation error: {e}")
                    
                    # Noise becomes input for next round
                    current_items = noise
            
            # Replace approved hypotheses with deepened versions
            if deepened_hypotheses:
                approved_hypotheses = deepened_hypotheses
                print(f"\n   📊 Cycle {cycle} produced {len(deepened_hypotheses)} deepened hypotheses")
            else:
                print(f"\n   ⏸️  No hypotheses deepened in cycle {cycle}. Ending loop.")
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
    
    # --- PHASE 8: MATHEMATICS DERIVATION ---
    # Derives equations from hypotheses, checks dimensional consistency,
    # verifies mathematical constraints, estimates complexity, rejects impossible formulations
    print("\n>>> PHASE 8: Initializing Mathematics Engine...")
    equations = run_mathematics_engine(fallback_hypotheses)
    
    # ---- Gather all stats from output files ----
    aarl_elapsed = time.time() - aarl_start
    
    kb_data = _load_json("deep_research_knowledge_base.json")
    doscan_data = _load_json("doscan_breakthroughs.json")
    verified_data = _load_json("verified_hypotheses.json")
    equations_data = _load_json("derived_equations.json")
    evidence_data = _load_json("evidence_scoring_db.json")
    
    # Knowledge base: count total strings across all fields
    total_kb_items = 0
    for v in kb_data.values():
        if isinstance(v, list):
            total_kb_items += len(v)
        elif isinstance(v, dict):
            total_kb_items += len(v)
        elif isinstance(v, str):
            total_kb_items += 1
    stats["kb_nodes"] = total_kb_items
    
    # Relationships: from doscan telemetry concepts extracted
    stats["relationships"] = len(doscan_data) * 3 if isinstance(doscan_data, list) else 0
    if isinstance(doscan_data, list):
        total_rel = 0
        for entry in doscan_data:
            insights = entry.get("scientific_research_insights", [])
            total_rel += len(insights)
        stats["relationships"] = total_rel
    
    # DOSCAN clusters
    if isinstance(doscan_data, list):
        stats["doscan_clusters"] = len(doscan_data)
    
    # Hypotheses from verified_hypotheses.json
    if isinstance(verified_data, list):
        stats["hypotheses_generated"] = len(verified_data)
        stats["hypotheses_approved"] = sum(
            1 for v in verified_data 
            if v.get("classification") in ("Excellent", "Plausible", "Speculative")
        )
        stats["hypotheses_rejected"] = stats["hypotheses_generated"] - stats["hypotheses_approved"]
    
    # Top confidence from evidence scoring
    if isinstance(evidence_data, dict):
        scores = evidence_data.get("scores", {})
        if scores:
            confs = [s.get("confidence", 0) for s in scores.values() if isinstance(s, dict)]
            if confs:
                stats["top_confidence"] = max(confs) * 100
    
    # Deepening cycles used
    stats["deepening_cycles"] = deepen_depth if deepen_depth > 0 and approved_hypotheses else 0
    
    # Mathematics: count validated equations
    math_validated = len(equations_data.get("validated_equations", [])) if isinstance(equations_data, dict) else 0
    math_rejected = len(equations_data.get("rejected_ideas", [])) if isinstance(equations_data, dict) else 0
    
    # ---- Print professional summary ----
    mins = int(aarl_elapsed // 60)
    secs = int(aarl_elapsed % 60)
    if mins > 0:
        runtime_str = f"{mins}m {secs}s"
    else:
        runtime_str = f"{secs}s"
    
    print("\n" + "="*80)
    print("📊  A A R L   R E S E A R C H   S U M M A R Y")
    print("="*80)
    print(f"")
    print(f"  Research Problem:")
    print(f"    {stats['problem']}")
    print(f"")
    print(f"  Knowledge Graph")
    print(f"    Knowledge Nodes:         {stats['kb_nodes']:>8,}")
    print(f"    Relationships Found:     {stats['relationships']:>8,}")
    print(f"    DOSCAN Clusters:         {stats['doscan_clusters']:>8,}")
    print(f"")
    print(f"  Hypothesis Pipeline")
    print(f"    Hypotheses Generated:    {stats['hypotheses_generated']:>8,}")
    print(f"    Approved:                {stats['hypotheses_approved']:>8,}")
    print(f"    Rejected:                {stats['hypotheses_rejected']:>8,}")
    print(f"    Deepening Cycles:        {stats['deepening_cycles']:>8,}")
    print(f"")
    print(f"  Mathematics")
    print(f"    Validated Equations:     {math_validated:>8,}")
    print(f"    Rejected Equations:      {math_rejected:>8,}")
    print(f"")
    print(f"  Evidence & Confidence")
    print(f"    Top Hypothesis Confidence:  {stats['top_confidence']:.0f}%")
    print(f"")
    print(f"  Performance")
    print(f"    Total Runtime:           {runtime_str:>8}")
    print(f"")
    print("="*80)
    print("📁 Output Files Generated:")
    files = [
        ("Knowledge Base",     "deep_research_knowledge_base.json"),
        ("DOSCAN Insights",    "doscan_breakthroughs.json"),
        ("Verified Hypotheses","verified_hypotheses.json"),
        ("Final Solution",     "final_research_solution.json"),
        ("Evidence Scores",    "evidence_scoring_db.json"),
        ("Derived Equations",  "derived_equations.json"),
    ]
    for label, fname in files:
        exists = "✅" if os.path.exists(fname) else "❌"
        print(f"    {exists} {label:20s} → {fname}")
    print("="*80)
    print(f"  🔬 AARL Research Cycle Complete — {runtime_str}")
    print("="*80)

if __name__ == "__main__":
    main()
