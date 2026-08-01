import os
import json
import re
import sys
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from GroqClient import llm_complete_json, llm_complete

def _repair_json(raw: str) -> str:
    """
    Attempt to repair common JSON generation issues from LLMs:
    - Trailing unescaped newlines inside strings
    - Broken last array element
    - Missing closing brackets
    """
    # Remove any text before the first '{'
    first_brace = raw.find('{')
    if first_brace >= 0:
        raw = raw[first_brace:]
    
    # Remove any text after the last '}'
    last_brace = raw.rfind('}')
    if last_brace >= 0:
        raw = raw[:last_brace + 1]
    
    # Replace literal newlines inside strings with escaped newlines
    # First, find all string values and escape internal newlines
    lines = raw.split('\n')
    repaired_lines = []
    in_string = False
    for line in lines:
        # Count unescaped quotes to track string boundaries
        quote_count = line.count('"') - line.count('\\"')
        if quote_count % 2 != 0:
            in_string = not in_string
        if in_string:
            # This line is inside a string value, escape it
            repaired_lines.append(line.rstrip('\n\r'))
        else:
            repaired_lines.append(line)
    
    repaired = '\n'.join(repaired_lines)
    
    # Remove trailing comma before closing brace/bracket (common JSON error)
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    return repaired

class ResearchKnowledge(BaseModel):
    core_research_thesis: str = Field(description="The central objective and primary scientific assumption being investigated.")
    state_of_the_art_prior_art: List[str] = Field(description="Current leading solutions, industry standards, and academic baselines already achieved.")
    proven_facts: List[str] = Field(description="Irrefutable axioms, chemical, physical, or logical parameters established by empirical consensus.")
    critical_unanswered_unknowns: List[str] = Field(description="The precise technical gaps, unmapped variables, or physical limitations preventing a complete solution.")
    proposed_testable_hypotheses: Dict[str, str] = Field(description="Proposed actionable claims paired with their corresponding verification metric/test method.")
    failure_modes_and_risks: List[str] = Field(description="Expected technical degradation, safety concerns, environmental risks, or operational failures.")
    required_empirical_data_inputs: List[str] = Field(description="Granular telemetry, data streams, materials characterization, or academic papers required next.")


def _generate_fallback_knowledge_base(problem_statement: str, analysis_summary: str) -> dict:
    """Generate a deterministic knowledge base when no LLM is available.
    
    This produces a domain-aware knowledge base by detecting the research domain
    from the problem statement and generating domain-specific content.
    """
    # Try to detect domain using domain_metrics
    _sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Research", "simulators")
    if _sim_dir not in sys.path:
        sys.path.insert(0, _sim_dir)
    try:
        from domain_metrics import detect_domain, get_domain_description
        domain = detect_domain(problem_statement)
        domain_desc = get_domain_description(domain)
    except Exception:
        domain = "general"
        domain_desc = "General — domain-agnostic engineering analysis"
    
    # Domain-specific knowledge bases
    knowledge_bases = {
        "computer_architecture": {
            "core_research_thesis": "Designing a novel CPU architecture that maximizes performance while minimizing power consumption and area usage",
            "state_of_the_art_prior_art": [
                "Out-of-Order Execution",
                "Pipelining",
                "Superscalar Processing",
                "Cache Hierarchy",
                "Branch Prediction",
            ],
            "proven_facts": [
                "Moores Law dictates transistor density doubles approximately every two years",
                "Amdahls Law limits maximum theoretical speedup from parallel processing",
                "Pipelining increases throughput but may decrease latency",
                "Cache hit rate significantly impacts system performance",
            ],
            "critical_unanswered_unknowns": [
                "Optimal pipeline depth for modern workloads",
                "Best instruction set architecture for emerging applications",
                "Most effective branch prediction algorithm for modern CPUs",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Verification via simulation and emulation using workloads like SPEC CPU2017 and PARSEC",
                "hypothesis2": "Validation through FPGA prototyping and power measurement",
                "hypothesis3": "Evaluation using architectural simulators like Gem5 and McPAT",
            },
            "failure_modes_and_risks": [
                "Increased power consumption due to inefficient design",
                "Decreased performance from suboptimal pipeline or cache design",
                "Incompatibility with existing software due to novel instruction set architecture",
            ],
            "required_empirical_data_inputs": [
                "Workload characteristics like instruction mix and memory access patterns",
                "Power consumption and thermal data from existing CPUs",
                "Performance metrics like IPC and throughput from similar architectures",
            ],
        },
        "chemistry": {
            "core_research_thesis": f"Developing a chemical approach to: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Sulfuric acid (H2SO4) as benchmark strong acid",
                "Catalytic reaction optimization",
                "Electrochemical synthesis methods",
            ],
            "proven_facts": [
                "pH scale is logarithmic (pH = -log10[H+])",
                "Acid strength is measured by dissociation constant Ka",
                "Superacids have Hammett acidity function H0 < -12",
            ],
            "critical_unanswered_unknowns": [
                "Optimal catalyst composition for maximum yield",
                "Reaction kinetics under extreme conditions",
                "Thermal stability limits of novel compounds",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Measure pH and Ka using titration and conductivity",
                "hypothesis2": "Validate reaction rate via spectrophotometry",
                "hypothesis3": "Assess thermal stability via DSC and TGA",
            },
            "failure_modes_and_risks": [
                "Corrosion of equipment from strong acids",
                "Uncontrolled exothermic reactions",
                "Toxic byproduct formation",
            ],
            "required_empirical_data_inputs": [
                "pH measurements under standard conditions",
                "Reaction rate constants at various temperatures",
                "Yield and selectivity percentages",
            ],
        },
        "battery": {
            "core_research_thesis": f"Developing a battery technology for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Lithium-ion batteries with graphite anodes",
                "Solid-state electrolyte research",
                "Silicon anode development",
            ],
            "proven_facts": [
                "Energy density is limited by electrode material capacity",
                "Internal resistance affects power density and charging speed",
                "Cycle life degrades with depth of discharge",
            ],
            "critical_unanswered_unknowns": [
                "Optimal electrode material composition for maximum energy density",
                "Best electrolyte formulation for safety and performance",
                "Long-term degradation mechanisms at high C-rates",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Measure energy density via constant current discharge",
                "hypothesis2": "Assess cycle life through accelerated aging tests",
                "hypothesis3": "Evaluate thermal behavior via calorimetry",
            },
            "failure_modes_and_risks": [
                "Thermal runaway and fire risk",
                "Electrolyte decomposition and gas generation",
                "Mechanical degradation of electrode structure",
            ],
            "required_empirical_data_inputs": [
                "Energy density measurements (Wh/kg)",
                "Internal resistance via EIS",
                "Cycle life data at various C-rates",
            ],
        },
        "aerospace": {
            "core_research_thesis": f"Developing an aerospace solution for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Conventional airfoil designs",
                "Fixed winglet technology",
                "Turbofan propulsion systems",
            ],
            "proven_facts": [
                "Lift is proportional to lift coefficient and velocity squared",
                "Drag reduction improves fuel efficiency",
                "Structural margin must exceed safety factor of 1.5",
            ],
            "critical_unanswered_unknowns": [
                "Optimal airfoil geometry for specific flight regimes",
                "Trade-off between lift enhancement and drag increase",
                "Material selection for weight reduction vs structural integrity",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Validate lift and drag via wind tunnel testing",
                "hypothesis2": "Assess structural integrity through finite element analysis",
                "hypothesis3": "Evaluate fuel efficiency through flight simulation",
            },
            "failure_modes_and_risks": [
                "Structural failure under extreme loads",
                "Aerodynamic stall at low speeds",
                "Material fatigue from cyclic loading",
            ],
            "required_empirical_data_inputs": [
                "Wind tunnel force measurements",
                "Structural stress test data",
                "Fuel consumption metrics from test flights",
            ],
        },
        "robotics": {
            "core_research_thesis": f"Developing a robotic system for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Industrial robotic arms with 6 DOF",
                "PID control systems",
                "Optical and magnetic encoders",
            ],
            "proven_facts": [
                "Position accuracy is limited by encoder resolution",
                "Repeatability is affected by mechanical backlash",
                "Response time depends on control loop bandwidth",
            ],
            "critical_unanswered_unknowns": [
                "Optimal sensor fusion strategy for precision improvement",
                "Trade-off between payload capacity and speed",
                "Energy efficiency of different actuator types",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Measure position accuracy via laser tracker",
                "hypothesis2": "Assess repeatability through ISO 9283 tests",
                "hypothesis3": "Evaluate response time via step input tests",
            },
            "failure_modes_and_risks": [
                "Sensor drift and calibration errors",
                "Mechanical wear in joints and actuators",
                "Control instability under varying loads",
            ],
            "required_empirical_data_inputs": [
                "Position accuracy measurements",
                "Repeatability test data",
                "Power consumption during operation",
            ],
        },
        "civil_engineering": {
            "core_research_thesis": f"Developing a structural solution for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Reinforced concrete structures",
                "Steel frame construction",
                "Seismic design codes",
            ],
            "proven_facts": [
                "Load capacity depends on material strength and cross-section",
                "Deflection is proportional to load and span length",
                "Safety factor must account for material variability",
            ],
            "critical_unanswered_unknowns": [
                "Optimal material composition for load-bearing capacity",
                "Seismic resistance of novel structural designs",
                "Long-term durability under environmental exposure",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Validate load capacity through structural testing",
                "hypothesis2": "Assess seismic resistance via shake table tests",
                "hypothesis3": "Evaluate durability through accelerated aging",
            },
            "failure_modes_and_risks": [
                "Structural collapse under overload",
                "Corrosion of reinforcement steel",
                "Foundation settlement and cracking",
            ],
            "required_empirical_data_inputs": [
                "Load test measurements",
                "Material strength test data",
                "Seismic performance data",
            ],
        },
        "medicine": {
            "core_research_thesis": f"Developing a therapeutic approach for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Conventional drug delivery systems",
                "Oral and intravenous administration",
                "Liposomal encapsulation technology",
            ],
            "proven_facts": [
                "Bioavailability is affected by first-pass metabolism",
                "Toxicity is dose-dependent",
                "Drug efficacy requires sufficient plasma concentration",
            ],
            "critical_unanswered_unknowns": [
                "Optimal nanoparticle size for maximum bioavailability",
                "Target specificity vs off-target effects",
                "Controlled release kinetics for sustained therapeutic levels",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Measure bioavailability via pharmacokinetic studies",
                "hypothesis2": "Assess toxicity through cell viability assays",
                "hypothesis3": "Evaluate efficacy in animal models",
            },
            "failure_modes_and_risks": [
                "Adverse immune reactions",
                "Drug-drug interactions",
                "Variable patient response",
            ],
            "required_empirical_data_inputs": [
                "Pharmacokinetic profile data",
                "Toxicity assay results",
                "Clinical trial efficacy data",
            ],
        },
        "materials": {
            "core_research_thesis": f"Developing a material solution for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Carbon fiber composites",
                "Aluminum and titanium alloys",
                "Ceramic matrix composites",
            ],
            "proven_facts": [
                "Tensile strength depends on fiber alignment and matrix bonding",
                "Thermal conductivity varies with crystalline structure",
                "Fracture toughness is critical for structural applications",
            ],
            "critical_unanswered_unknowns": [
                "Optimal fiber-matrix interface for maximum strength",
                "Trade-off between strength and ductility",
                "Long-term corrosion resistance in service environments",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Measure tensile strength via universal testing machine",
                "hypothesis2": "Assess thermal conductivity via laser flash analysis",
                "hypothesis3": "Evaluate corrosion resistance via salt spray testing",
            },
            "failure_modes_and_risks": [
                "Delamination under cyclic loading",
                "Thermal degradation at high temperatures",
                "Corrosion in aggressive environments",
            ],
            "required_empirical_data_inputs": [
                "Tensile test data",
                "Thermal property measurements",
                "Corrosion rate data",
            ],
        },
        "machine_learning": {
            "core_research_thesis": f"Developing a machine learning solution for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Transformer architectures",
                "Knowledge distillation techniques",
                "Quantization-aware training",
            ],
            "proven_facts": [
                "Model accuracy depends on training data quality and quantity",
                "Inference latency is proportional to model size",
                "Overfitting reduces generalization performance",
            ],
            "critical_unanswered_unknowns": [
                "Optimal model architecture for specific tasks",
                "Trade-off between accuracy and inference speed",
                "Generalization to out-of-distribution data",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Validate accuracy on held-out test set",
                "hypothesis2": "Assess inference latency via benchmark suite",
                "hypothesis3": "Evaluate generalization via cross-validation",
            },
            "failure_modes_and_risks": [
                "Overfitting to training data",
                "Distribution shift in deployment",
                "Computational resource constraints",
            ],
            "required_empirical_data_inputs": [
                "Training and validation datasets",
                "Benchmark performance metrics",
                "Inference time measurements",
            ],
        },
    }
    
    # Select domain-specific or generic fallback
    if domain in knowledge_bases:
        kb = knowledge_bases[domain]
    else:
        # Generic fallback
        kb = {
            "core_research_thesis": f"Researching and developing solutions for: {problem_statement}",
            "state_of_the_art_prior_art": [
                "Current industry-standard approaches",
                "Established academic baselines",
                "Conventional design methodologies",
            ],
            "proven_facts": [
                "Fundamental physical and chemical principles apply",
                "Performance optimization requires trade-off analysis",
                "Empirical validation is essential for novel designs",
            ],
            "critical_unanswered_unknowns": [
                "Optimal design parameters for the specific application",
                "Interaction effects between multiple subsystems",
                "Long-term reliability under operational conditions",
            ],
            "proposed_testable_hypotheses": {
                "hypothesis1": "Validate through computational simulation and modeling",
                "hypothesis2": "Assess via prototype testing and measurement",
                "hypothesis3": "Evaluate using standardized benchmark tests",
            },
            "failure_modes_and_risks": [
                "Performance degradation under extreme conditions",
                "Manufacturing and scalability challenges",
                "Cost and resource constraints",
            ],
            "required_empirical_data_inputs": [
                "Performance metrics from similar designs",
                "Material properties and specifications",
                "Operational environment characteristics",
            ],
        }
    
    kb["domain"] = domain
    kb["domain_description"] = domain_desc
    
    # Save to disk
    output_filename = "deep_research_knowledge_base.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=4)
    
    print(f"[Knowledge Engine] Fallback knowledge base written to '{output_filename}' (domain: {domain})")
    return kb


def build_knowledge_base(problem_statement: str, analysis_summary: str) -> dict:
    print("\n[Knowledge Engine] Initializing deep academic extraction pipeline...")

    system_prompt = (
        "You are the Core Knowledge and Epistemology Engine of an Autonomous Research Lab. "
        "Your duty is to generate exhaustive, highly specific domain knowledge while aggressively identifying gaps "
        "and defining precise experimental validation hooks.\n\n"
        "You MUST respond with a valid JSON object matching EXACTLY this schema. "
        "CRITICAL: Do NOT use colons inside dictionary keys (like 'Hypothesis 1: ...'). "
        "Use simple keys instead. CRITICAL: All string values must be on a single line — no line breaks inside strings.\n\n"
        "{\n"
        '  "core_research_thesis": "string",\n'
        '  "state_of_the_art_prior_art": ["string"],\n'
        '  "proven_facts": ["string"],\n'
        '  "critical_unanswered_unknowns": ["string"],\n'
        '  "proposed_testable_hypotheses": {"simple key": "verification method"},\n'
        '  "failure_modes_and_risks": ["string"],\n'
        '  "required_empirical_data_inputs": ["string"]\n'
        "}\n"
        "No wrapper objects, no markdown blocks, no extra text."
    )

    prompt = (
        f"Problem Scope: {problem_statement}\n\n"
        f"Domain Structural Context: {analysis_summary}\n\n"
        "Execute deep extraction of parameters, unknowns, hypotheses, and vulnerabilities."
    )

    evidence_debug = os.environ.get("AARL_EVIDENCE_DEBUG", "").lower() in ("1", "true", "yes")
    if evidence_debug:
        print("\n[EVIDENCE DEBUG] === Knowledge base LLM (feeds Phase 7 literature evidence) ===")
        print("[EVIDENCE DEBUG] research_problem:", problem_statement)
        print("[EVIDENCE DEBUG] system_prompt:", system_prompt)
        print("[EVIDENCE DEBUG] user_prompt:", prompt)

    try:
        if evidence_debug:
            raw_llm = llm_complete(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.25,
                response_format={"type": "json_object"},
            )
            print("[EVIDENCE DEBUG] raw_llm_response (before JSON parse / disk write):")
            print(raw_llm or "(empty)")
            result = json.loads(raw_llm) if raw_llm else None
        else:
            result = llm_complete_json(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.25
            )
        if not result:
            print("   LLM unavailable. Using deterministic fallback knowledge base...")
            return _generate_fallback_knowledge_base(problem_statement, analysis_summary)
        
        raw_json = json.dumps(result)
        # Attempt to repair common JSON formatting issues from LLM output
        repaired_json = _repair_json(raw_json)
        if repaired_json != raw_json:
            print("   [JSON Repair] Applied fixes to LLM output")
        knowledge_data = ResearchKnowledge.model_validate_json(repaired_json)
        
        knowledge_dict = knowledge_data.model_dump()
        output_filename = "deep_research_knowledge_base.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(knowledge_dict, f, indent=4)
            
        print(f"[Knowledge Engine] Master dataset successfully written to '{output_filename}'")
        return knowledge_dict

    except ValidationError as ve:
        print(f"\n   Knowledge validation error. Using fallback: {ve}")
        return _generate_fallback_knowledge_base(problem_statement, analysis_summary)
    except Exception as e:
        print(f"\n   Exception in Knowledge.py: {e}. Using fallback...")
        return _generate_fallback_knowledge_base(problem_statement, analysis_summary)