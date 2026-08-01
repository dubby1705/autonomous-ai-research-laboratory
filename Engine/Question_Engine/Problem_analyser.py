import os
import json
import sys
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError
from GroqClient import llm_complete_json

class DomainAnalysis(BaseModel):
    main_domain: str = Field(description="Primary overarching academic or technical field.")
    sub_domains: List[str] = Field(description="Highly specific technical sub-fields requiring literature review.")
    problem_breakdown: List[str] = Field(description="Deconstruction of variables, technical requirements, and target metrics.")
    governing_laws_and_equations: List[str] = Field(description="Scientific laws, mathematical theorems, or foundational equations that rule this problem.")
    key_research_questions: List[str] = Field(description="Deep scientific questions that must be answered to innovate or solve the issue.")
    suggested_methodologies: List[str] = Field(description="Advanced experimental setups, computational frameworks, simulation models, or algorithms needed.")
    interdisciplinary_links: Dict[str, str] = Field(description="Mapping of secondary fields to what specific insight or cross-over data they provide.")


def _generate_fallback_analysis(problem_statement: str) -> DomainAnalysis:
    """Generate a deterministic domain analysis when no LLM is available."""
    # Try to detect domain
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
    
    # Domain-specific analyses
    analyses = {
        "computer_architecture": DomainAnalysis(
            main_domain="Computer Architecture and Computer Engineering",
            sub_domains=[
                "Instruction Set Architecture Design",
                "Microarchitecture and Pipeline Optimization",
                "Cache Hierarchy and Memory Systems",
                "Power-Efficient Computing",
                "Performance Analysis and Benchmarking",
            ],
            problem_breakdown=[
                "Instruction throughput (IPC) optimization",
                "Power consumption reduction under TDP constraints",
                "Die area minimization for cost efficiency",
                "Cache hit rate improvement for memory latency reduction",
                "Branch prediction accuracy enhancement",
                "Pipeline stall reduction through hazard detection",
            ],
            governing_laws_and_equations=[
                "Amdahls Law: Speedup = 1 / (1 - P + P/N)",
                "Moores Law: Transistor density doubles approximately every 2 years",
                "Power = C * V^2 * f (dynamic power consumption)",
                "IPC = Instructions Executed / Clock Cycles",
                "Cache hit rate = Hits / (Hits + Misses)",
            ],
            key_research_questions=[
                "What is the optimal pipeline depth for modern workload characteristics?",
                "Which instruction set architecture maximizes IPC for target applications?",
                "How can branch prediction algorithms be improved for better accuracy?",
                "What cache hierarchy configuration minimizes memory latency?",
                "How can power consumption be reduced without sacrificing performance?",
            ],
            suggested_methodologies=[
                "Architectural simulation using Gem5 and McPAT",
                "FPGA prototyping for hardware validation",
                "Benchmark testing with SPEC CPU2017 and PARSEC workloads",
                "Power analysis using RTL-level power estimation tools",
                "Performance counter-based profiling and analysis",
            ],
            interdisciplinary_links={
                "Semiconductor Physics": "Transistor scaling and leakage current modeling",
                "Thermodynamics": "Thermal design and cooling requirements analysis",
                "Signal Processing": "SIMD and vector processing optimization",
                "Software Engineering": "Compiler optimization and ISA co-design",
            },
        ),
        "chemistry": DomainAnalysis(
            main_domain="Chemistry and Chemical Engineering",
            sub_domains=[
                "Acid-Base Chemistry",
                "Reaction Kinetics",
                "Catalysis",
                "Thermodynamics",
                "Synthesis and Purification",
            ],
            problem_breakdown=[
                "Acid strength optimization (pH and Ka)",
                "Reaction rate enhancement",
                "Yield and selectivity improvement",
                "Thermal stability assessment",
                "Catalyst efficiency optimization",
            ],
            governing_laws_and_equations=[
                "pH = -log10[H+]",
                "Ka = [H+][A-] / [HA]",
                "Arrhenius equation: k = A * exp(-Ea/RT)",
                "Gibbs free energy: G = H - TS",
                "Le Chateliers principle for equilibrium",
            ],
            key_research_questions=[
                "What molecular structure maximizes acid strength?",
                "Which catalyst provides the highest turnover frequency?",
                "How can activation energy be reduced?",
                "What conditions maximize yield and selectivity?",
                "How does temperature affect reaction kinetics?",
            ],
            suggested_methodologies=[
                "Spectrophotometric analysis for reaction monitoring",
                "Titration for pH and concentration measurement",
                "DSC and TGA for thermal stability analysis",
                "Chromatography for purity assessment",
                "Electrochemical measurement for Ka determination",
            ],
            interdisciplinary_links={
                "Thermodynamics": "Energy barriers and equilibrium constants",
                "Materials Science": "Catalyst material properties and stability",
                "Chemical Engineering": "Process optimization and scale-up",
            },
        ),
        "battery": DomainAnalysis(
            main_domain="Electrochemistry and Energy Storage",
            sub_domains=[
                "Battery Cell Design",
                "Electrode Materials",
                "Electrolyte Chemistry",
                "Energy Density Optimization",
                "Cycle Life Analysis",
            ],
            problem_breakdown=[
                "Energy density maximization (Wh/kg)",
                "Internal resistance minimization",
                "Cycle life extension",
                "Charging efficiency improvement",
                "Thermal management optimization",
            ],
            governing_laws_and_equations=[
                "Nernst equation: E = E0 - (RT/nF) * ln(Q)",
                "Butler-Volmer equation for electrode kinetics",
                "Ficks laws of diffusion",
                "Ohms law: V = IR for internal resistance",
                "Capacity: Q = I * t",
            ],
            key_research_questions=[
                "What electrode material composition maximizes energy density?",
                "How can internal resistance be minimized?",
                "What electrolyte formulation optimizes safety and performance?",
                "How does electrode microstructure affect cycle life?",
                "What charging protocol minimizes degradation?",
            ],
            suggested_methodologies=[
                "Constant current discharge testing for capacity",
                "Electrochemical impedance spectroscopy (EIS)",
                "Accelerated cycle life testing",
                "Calorimetry for thermal analysis",
                "SEM/TEM for electrode microstructure characterization",
            ],
            interdisciplinary_links={
                "Materials Science": "Electrode material properties and degradation",
                "Thermodynamics": "Heat generation and thermal runaway",
                "Chemical Engineering": "Cell manufacturing and scale-up",
            },
        ),
    }
    
    if domain in analyses:
        return analyses[domain]
    
    # Generic fallback
    return DomainAnalysis(
        main_domain=f"Research and Development for: {problem_statement}",
        sub_domains=[
            "System Design and Optimization",
            "Performance Analysis",
            "Material Selection",
            "Experimental Validation",
        ],
        problem_breakdown=[
            "Performance optimization under constraints",
            "Material and component selection",
            "System integration and testing",
            "Reliability and durability assessment",
        ],
        governing_laws_and_equations=[
            "Fundamental physical and chemical principles",
            "Conservation laws (energy, mass, momentum)",
            "Thermodynamic equilibrium principles",
        ],
        key_research_questions=[
            "What are the optimal design parameters?",
            "How can performance be maximized within constraints?",
            "What are the key trade-offs in the design space?",
        ],
        suggested_methodologies=[
            "Computational simulation and modeling",
            "Prototype construction and testing",
            "Performance measurement under controlled conditions",
            "Statistical analysis of experimental data",
        ],
        interdisciplinary_links={
            "Physics": "Fundamental physical principles and constraints",
            "Materials Science": "Material properties and selection criteria",
            "Engineering": "Design optimization and manufacturing",
        },
    )


def analyze_research_problem(problem_statement: str) -> DomainAnalysis:
    system_prompt = (
        "You are an Elite Research Strategy AI. Your job is to perform an exhaustive, multi-dimensional "
        "structural decomposition of a problem statement. Extract deep scientific properties and governing rules.\n\n"
        "You MUST respond purely with a valid JSON object matching EXACTLY these keys:\n"
        "{\n"
        '  "main_domain": "string",\n'
        '  "sub_domains": ["string"],\n'
        '  "problem_breakdown": ["string"],\n'
        '  "governing_laws_and_equations": ["string"],\n'
        '  "key_research_questions": ["string"],\n'
        '  "suggested_methodologies": ["string"],\n'
        '  "interdisciplinary_links": {"Field Name": "Reason for link / Contribution"}\n'
        "}\n"
        "Do not insert outer wrapper keys, do not wrap in backticks, and omit any conversational filler."
    )

    try:
        result = llm_complete_json(
            system_prompt=system_prompt,
            user_prompt=f"Deconstruct this problem statement for an advanced scientific research lab. Output ONLY in the strict JSON template format:\n\n{problem_statement}",
            temperature=0.15
        )
        if result:
            return DomainAnalysis.model_validate_json(json.dumps(result))
        print("   LLM unavailable. Using deterministic fallback analysis...")
        return _generate_fallback_analysis(problem_statement)
    except ValidationError as ve:
        print(f"\n   Analyser validation error. Using fallback: {ve}")
        return _generate_fallback_analysis(problem_statement)
    except Exception as e:
        print(f"\n   Exception in Problem_analyser.py: {e}. Using fallback...")
        return _generate_fallback_analysis(problem_statement)

def print_analysis_report(analysis: DomainAnalysis):
    if not analysis: return
    print("\n" + "="*80)
    print("  PHASE 1: EXPANDED PROBLEM MATRIX DECONSTRUCTION")
    print("="*80)
    print(f"\n  MAIN DOMAIN: {analysis.main_domain}")
    print("\n  TARGET SUB-DOMAINS:")
    for sub in analysis.sub_domains: print(f"   - {sub}")
    print("\n  GRANULAR COMPONENT BREAKDOWN:")
    for item in analysis.problem_breakdown: print(f"   - {item}")
    print("\n  GOVERNING LAWS & THEORETICAL FORMULAS:")
    for law in analysis.governing_laws_and_equations: print(f"   - {law}")
    print("\n  PIVOTAL RESEARCH QUESTIONS:")
    for q in analysis.key_research_questions: print(f"   - {q}")
    print("\n  ADVANCED EXPERIMENTAL METHODOLOGIES:")
    for method in analysis.suggested_methodologies: print(f"   - {method}")
    print("\n  INTERDISCIPLINARY COUPLINGS:")
    for field, reason in analysis.interdisciplinary_links.items(): print(f"   - [{field}]: {reason}")
    print("\n" + "="*80)