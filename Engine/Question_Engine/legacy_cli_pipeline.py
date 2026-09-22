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
# Read the Groq API key from the environment first, then fall back to
# prompting the user interactively. A placeholder is used as the final
# fallback so the error is clear and actionable instead of a 401 crash.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = input("\nEnter your Groq API key (or press Enter to use Ollama local fallback): ").strip()
        if GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        else:
            print("[AARL] No API key provided — pipeline will use deterministic fallbacks.")
    except EOFError:
        # Non-interactive mode (e.g., piped input) — skip prompt and use fallbacks
        print("[AARL] Non-interactive mode — no API key provided. Using deterministic fallbacks.")
else:
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
from PhysicsMathematics import run_domain_aware_mathematics
from ResearchOutputGenerator import run_research_output_generator
from Research.compare import run_research_comparison
from Research.simulation_engine import run_simulation_comparison

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


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert a research problem into a safe directory name."""
    import re
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    slug = slug.strip('_')
    if len(slug) > max_len:
        slug = slug[:max_len]
    if not slug:
        slug = "research_problem"
    return slug


def _get_problem_output_dir(problem: str) -> str:
    """Get the problem-specific output directory for all research outputs."""
    problem_slug = _slugify(problem)
    output_dir = os.path.join(ROOT_DIR, "Research", problem_slug)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

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

    # Create problem-specific output directory for all research outputs
    problem_output_dir = _get_problem_output_dir(user_problem)
    print(f"\n📁 Research output directory: {problem_output_dir}")

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
    print("\n>>> PHASE 4: Initializing Lateral Hypothsis Engine...")
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
                        
                        # Use shared GroqClient with Ollama fallback
                        from GroqClient import llm_complete_json
                        result = llm_complete_json(
                            system_prompt="You validate if a knowledge cluster deepens a hypothesis. Answer ONLY with JSON: {\"deepens\": true/false, \"reason\": \"brief\", \"refined_hypothesis\": \"improved version if deepens\"}",
                            user_prompt=f"Hypothesis: {hyp}\n\nNew knowledge cluster: {cluster_text}\n\nDoes this knowledge deepen the hypothesis? If yes, produce a refined hypothesis.",
                            temperature=0.1
                        )
                        if result and result.get("deepens", False):
                            refined = result.get("refined_hypothesis") or hyp
                            # LLMs sometimes return nested objects/None for refined_hypothesis — normalize to str
                            if isinstance(refined, dict):
                                refined = refined.get("hypothesis") or refined.get("text") or json.dumps(refined, ensure_ascii=False)
                            elif not isinstance(refined, str):
                                refined = str(refined)
                            refined = refined.strip()
                            if not refined:
                                refined = hyp
                            deepened_hypotheses.append(refined)
                            print(f"            ✅ Deepens: {refined[:100]}...")
                        else:
                            reason = result.get("reason", "No improvement") if result else "LLM unavailable"
                            if not isinstance(reason, str):
                                reason = json.dumps(reason, ensure_ascii=False) if isinstance(reason, (dict, list)) else str(reason)
                            print(f"            ❌ Does not deepen: {reason[:80]}")
                    
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
    
    # Build fallback hypotheses for Phases 7-9
    # Only use ACTUALLY verified hypotheses — never the raw research problem
    # Tier 1: approved from hypothesis engine (Excellent/Plausible/Speculative)
    # Tier 2: refined from Socratic questioning
    verified_hypotheses = list(approved_hypotheses) if approved_hypotheses else []
    
    if not verified_hypotheses:
        fs_hs = final_solution.get("approved_hypotheses_questioned", [])
        verified_hypotheses = list(fs_hs) if fs_hs else []
    
    # Count genuine verified hypotheses (not fallback garbage)
    verified_count = len(verified_hypotheses)
    
    if verified_count > 0:
        fallback_hypotheses = verified_hypotheses
        print(f"\n[AARL] Using {verified_count} verified hypotheses for Phases 7-9.")
    else:
        fallback_hypotheses = []
        print(f"\n[AARL] ⚠️ No verified hypotheses generated. Phases 7-9 will be skipped.")
    
    # --- PHASE 7: EVIDENCE SCORING ---
    # Scores each hypothesis with structured evidence tracking
    # Identifies missing evidence types and recommends next actions
    print("\n>>> PHASE 7: Initializing Evidence Scoring Engine...")
    evidence_summary = run_evidence_engine(fallback_hypotheses, research_problem=user_problem)
    
    # --- DOMAIN DETECTION (Before Phase 8 & 9) ---
    # Analyze the research problem and knowledge graph to detect scientific domains
    # This determines which specialized math engines and simulators to use
    print("\n>>> DOMAIN DETECTION: Analyzing research problem for scientific domains...")
    from DomainDetector import detect_domains, detect_simulator, format_domain_report, get_math_engines
    
    # Load knowledge base for enriched domain detection
    kb_data = _load_json("deep_research_knowledge_base.json")
    detected_domains = detect_domains(user_problem, kb_data)
    print(format_domain_report(detected_domains))
    
    detected_simulator = detect_simulator(user_problem, kb_data)
    math_engines = get_math_engines(detected_domains)
    print(f"  -> Selected Simulator: {detected_simulator}")
    print(f"  -> Activated Math Engines: {math_engines}")
    
    # --- PHASE 8: DOMAIN-AWARE MATHEMATICS DERIVATION ---
    # Derives equations from hypotheses using specialized engines
    # (Physics, Chemistry, Quantum) based on detected domains
    print("\n>>> PHASE 8: Initializing Domain-Aware Mathematics Engine...")
    if math_engines:
        equations = run_domain_aware_mathematics(fallback_hypotheses, detected_domains)
    else:
        equations = run_mathematics_engine(fallback_hypotheses)
    
    # --- PHASE 9: DOMAIN-AWARE FULLY AUDITABLE SIMULATION ---
    # Uses the correct simulator based on detected domains.
    # Generates standalone Python simulation files that are fully auditable.
    # ONLY runs if we have genuinely verified hypotheses (>0 from hypothesis engine)
    if verified_count > 0:
        print("\n>>> PHASE 9: Initializing Domain-Aware Simulation Comparison...")
        best_hypothesis = fallback_hypotheses[0]
        if isinstance(best_hypothesis, dict):
            best_hypothesis = best_hypothesis.get("refined_hypothesis", str(best_hypothesis))
        
        # Route to the correct simulator based on domain detection
        research_dir = os.path.join(ROOT_DIR, "Research")
        simulators_dir = os.path.join(ROOT_DIR, "Research", "simulators")
        if research_dir not in sys.path:
            sys.path.insert(0, research_dir)
        if simulators_dir not in sys.path:
            sys.path.insert(0, simulators_dir)

        # Determine which simulator files actually exist.
        # Only use a domain-specific simulator if the actual file exists.
        # Otherwise fall back to the domain-agnostic generic simulator.
        # NOTE: We never fall back to the battery-based pipeline for
        # non-battery problems (this was the source of battery contamination).
        simulator_file = os.path.join(simulators_dir, detected_simulator)
        if not os.path.exists(simulator_file):
            print(f"  ⚠️ Domain-specific simulator '{detected_simulator}' not found.")
            detected_simulator = "general_simulator.py"

        print(f"  Using simulator: {detected_simulator}")
        comparison_report = None
        errors = []

        # Attempt 1: The selected simulator (domain-specific or generic)
        try:
            sim_name = detected_simulator.replace(".py", "")
            sim_module = __import__(sim_name)
            comparison_report = sim_module.run_simulation(str(best_hypothesis), user_problem)
            print(f"\n   ✅ Phase 9 complete — {detected_simulator} finished")
        except Exception as e:
            errors.append(f"{detected_simulator}: {e}")
            print(f"   ⚠️ Simulator '{detected_simulator}' failed: {e}")

        # Attempt 2: Generic domain-agnostic simulator (if not already tried)
        if comparison_report is None and detected_simulator != "general_simulator.py":
            print(f"   🔄 Falling back to generic domain-agnostic simulator...")
            try:
                sim_module = __import__("general_simulator")
                comparison_report = sim_module.run_simulation(str(best_hypothesis), user_problem)
                print(f"\n   ✅ Phase 9 complete — generic domain-agnostic simulation finished")
            except Exception as e:
                errors.append(f"general_simulator: {e}")
                print(f"   ⚠️ Generic simulator failed: {e}")

        # Attempt 3: LLM-powered domain comparison (never battery-specific)
        if comparison_report is None:
            print(f"   🔄 Falling back to LLM-powered domain comparison...")
            try:
                comparison_report = run_research_comparison(user_problem, str(best_hypothesis))
                print(f"\n   ✅ Phase 9 complete — LLM-powered comparison finished")
            except Exception as e:
                errors.append(f"LLM comparison: {e}")
                print(f"   ⚠️ LLM comparison also failed: {e}")

        if comparison_report is None:
            print(f"   ❌ Phase 9 failed after all attempts:")
            for err in errors:
                print(f"      - {err}")
            print(f"   The battery-based comparison pipeline is intentionally NOT used")
            print(f"   for non-battery research problems (battery contamination fix).")
    else:
        print("\n⏸️  >>> PHASE 9 SKIPPED — No verified hypotheses to simulate.")
        print("   The hypothesis engine produced 0 verified hypotheses.")
        print("   Simulation requires at least 1 verified hypothesis.")
    
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
    stats["math_validated"] = math_validated
    stats["math_rejected"] = math_rejected

    # ---- Copy all output files to problem-specific directory ----
    print("\n📁 Organizing research outputs into problem-specific directory...")
    import shutil
    output_files = [
        "deep_research_knowledge_base.json",
        "doscan_breakthroughs.json",
        "doscan_exploration_weights.json",
        "verified_hypotheses.json",
        "final_research_solution.json",
        "evidence_scoring_db.json",
        "derived_equations.json",
    ]
    for fname in output_files:
        src = os.path.join(ROOT_DIR, fname)
        if os.path.exists(src):
            dst = os.path.join(problem_output_dir, fname)
            try:
                shutil.copy2(src, dst)
                print(f"    ✅ {fname} → {problem_output_dir}")
            except Exception as e:
                print(f"    ⚠️ Could not copy {fname}: {e}")

    # Also copy simulation reports if they exist
    sim_files = ["comparison_report.json", "parameter_changes.json"]
    for fname in sim_files:
        src = os.path.join(ROOT_DIR, "Research", fname)
        if os.path.exists(src):
            dst = os.path.join(problem_output_dir, fname)
            try:
                shutil.copy2(src, dst)
                print(f"    ✅ {fname} → {problem_output_dir}")
            except Exception as e:
                print(f"    ⚠️ Could not copy {fname}: {e}")

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

    # --- PHASE 10: RESEARCH OUTPUT GENERATOR ---
    # Collects all existing AARL outputs and generates human-readable
    # text reports in Research/Research_Output/
    # Does NOT regenerate hypotheses, rerun simulations, or call the LLM.
    print("\n>>> PHASE 10: Generating Human-Readable Research Output Package...")
    try:
        output_dir = run_research_output_generator(
            root_dir=ROOT_DIR,
            user_problem=user_problem,
            runtime_str=runtime_str,
            stats=stats,
        )
        print(f"\n   ✅ Phase 10 complete — Research output package generated")
        print(f"   📁 Reports saved to: {output_dir}")
        report_files = [
            "01_Research_Summary.txt",
            "02_Final_Design_Report.txt",
            "03_Design_Architecture.txt",
            "04_Engineering_Calculations.txt",
            "05_Materials_and_Technologies.txt",
            "06_Simulation_Report.txt",
            "07_Evidence_and_Confidence.txt",
            "08_Hypotheses_Analysis.txt",
            "09_Knowledge_Graph_Summary.txt",
            "10_Limitations_and_Next_Steps.txt",
        ]
        for rf in report_files:
            exists = "✅" if os.path.exists(os.path.join(output_dir, rf)) else "❌"
            print(f"      {exists} {rf}")
    except Exception as e:
        print(f"   ⚠️ Phase 10 failed: {e}")
        print(f"   📝 JSON outputs remain available for manual report generation.")

    print("\n" + "="*80)
    print("  🎓 AARL COMPLETE — Research package ready for review")
    print("="*80)

if __name__ == "__main__":
    main()