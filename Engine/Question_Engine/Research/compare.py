"""
RESEARCH COMPARISON ENGINE (Groq-powered)
Takes AARL's final hypothesis and:
1. Imports REAL Python libraries for the domain (PyBaMM, Cantera, PySpice, etc.)
2. Builds a STANDARD/COMMON implementation (existing battery)
3. Builds the AARL-SUGGESTED implementation
4. Compares both with numerical metrics

This is the ONLY place Groq is used for domain-specific code generation.
Ollama handles everything else (DOSCAN, hypothesis generation, etc.)
"""

import os
import json
import sys
import subprocess
import importlib
from typing import Dict, Any, Optional, List, Tuple
from groq import Groq
from pydantic import BaseModel, Field
from datetime import datetime

# =========================================================
# DOMAIN → LIBRARY & BUILD INSTRUCTIONS
# =========================================================
DOMAIN_REGISTRY = {
    "battery": {
        "libraries": ["pybamm", "numpy", "scipy", "matplotlib"],
        "pip_packages": ["pybamm", "numpy", "scipy", "matplotlib"],
        "build_script": "build_battery.py",
        "description": "Electrochemical battery model (PyBaMM)"
    },
    "circuit": {
        "libraries": ["PySpice", "numpy", "scipy"],
        "pip_packages": ["PySpice", "numpy", "scipy"],
        "build_script": "build_circuit.py",
        "description": "Electronic circuit simulation (PySpice)"
    },
    "chemistry": {
        "libraries": ["cantera", "numpy", "scipy", "matplotlib"],
        "pip_packages": ["cantera", "numpy", "scipy", "matplotlib"],
        "build_script": "build_reaction.py",
        "description": "Chemical kinetics (Cantera)"
    },
    "materials": {
        "libraries": ["numpy", "scipy", "matplotlib"],
        "pip_packages": ["numpy", "scipy", "matplotlib"],
        "build_script": "build_material.py",
        "description": "Material property simulation"
    },
    "machine_learning": {
        "libraries": ["torch", "numpy", "sklearn"],
        "pip_packages": ["torch", "numpy", "scikit-learn"],
        "build_script": "build_model.py",
        "description": "Neural network training (PyTorch)"
    },
    "mathematics": {
        "libraries": ["numpy", "scipy", "matplotlib"],
        "pip_packages": ["numpy", "scipy", "matplotlib"],
        "build_script": "build_math.py",
        "description": "Numerical optimization (SciPy)"
    },
    "general": {
        "libraries": ["numpy", "scipy"],
        "pip_packages": ["numpy", "scipy"],
        "build_script": "build_general.py",
        "description": "General numerical simulation"
    }
}

# =========================================================
# PYDANTIC SCHEMA
# =========================================================
class BuildPlan(BaseModel):
    domain: str = Field(description="Detected research domain")
    libraries_needed: List[str] = Field(description="Python libraries to import")
    standard_approach: str = Field(description="Python code to build the standard/common implementation")
    aarl_approach: str = Field(description="Python code to build the AARL-suggested implementation")
    comparison_metrics: List[str] = Field(description="Metrics to compare (e.g., 'energy_density', 'cost')")
    expected_improvement: str = Field(description="What AARL expects to improve")

# =========================================================
# DOMAIN DETECTION
# =========================================================
DOMAIN_KEYWORDS = {
    "battery": ["battery", "electrode", "electrolyte", "lithium", "anode", "cathode", "cell", "voltage", "capacity", "energy density", "charge", "discharge"],
    "circuit": ["circuit", "amplifier", "transistor", "resistor", "capacitor", "oscillator", "filter", "op-amp", "impedance", "bandwidth"],
    "chemistry": ["catalyst", "reaction", "chemical", "kinetics", "thermodynamics", "combustion", "oxidation", "reduction", "molecule", "hydrogen"],
    "materials": ["material", "crystal", "polymer", "composite", "alloy", "graphene", "nanotube", "stress", "strain", "thermal conductivity"],
    "machine_learning": ["neural", "deep learning", "transformer", "classification", "regression", "training", "gradient", "optimizer", "loss function"],
    "mathematics": ["optimization", "differential equation", "linear algebra", "eigenvalue", "numerical", "convergence", "simulation"]
}

def detect_domain(problem: str) -> str:
    """Detect the research domain from the problem statement."""
    problem_lower = problem.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in problem_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)

def ensure_libraries(libraries: List[str], pip_names: List[str]) -> List[str]:
    """Check which libraries are installed, try to install missing ones."""
    missing = []
    for i, lib in enumerate(libraries):
        try:
            importlib.import_module(lib)
        except ImportError:
            missing.append(pip_names[i] if i < len(pip_names) else lib)
    
    if missing:
        print(f"   [Research] Installing missing libraries: {', '.join(missing)}")
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
                print(f"      ✅ Installed {pkg}")
            except Exception as e:
                print(f"      ⚠️  Failed to install {pkg}: {e}")
    
    return missing

# =========================================================
# BUILD PLAN GENERATION (Groq)
# =========================================================
def generate_build_plan(client: Groq, problem: str, hypothesis: str, domain: str) -> Optional[BuildPlan]:
    """Use Groq to generate Python code for both standard and AARL implementations."""
    domain_config = DOMAIN_REGISTRY.get(domain, DOMAIN_REGISTRY["general"])
    
    system_prompt = (
        f"You are a Research Engineer building two implementations for comparison.\n"
        f"Domain: {domain} ({domain_config['description']})\n"
        f"Available libraries: {', '.join(domain_config['libraries'])}\n\n"
        "Generate a build plan with REAL Python code for BOTH implementations.\n"
        "The code must be syntactically valid and use the imported libraries correctly.\n\n"
        "Respond with JSON:\n"
        "{\n"
        '  "domain": "battery",\n'
        '  "libraries_needed": ["pybamm", "numpy"],\n'
        '  "standard_approach": "import pybamm\\nmodel = pybamm.lithium_ion.SPM()\\n...",\n'
        '  "aarl_approach": "import pybamm\\n# AARL modified parameters\\nmodel = pybamm.lithium_ion.SPM()\\n...",\n'
        '  "comparison_metrics": ["energy_density", "power_density"],\n'
        '  "expected_improvement": "AARL expects 15% higher energy density"\n'
        "}\n"
        "IMPORTANT: The code must be complete and runnable."
    )
    
    user_prompt = (
        f"Research Problem: {problem}\n"
        f"AARL Hypothesis: {hypothesis}\n\n"
        f"Build two Python implementations:\n"
        f"1. STANDARD: The current common approach in {domain} research\n"
        f"2. AARL: The improved version suggested by the hypothesis\n\n"
        f"Use libraries: {', '.join(domain_config['libraries'])}\n"
        f"Generate code that actually runs."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return BuildPlan.model_validate_json(completion.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ Build plan generation failed: {e}")
        return None

# =========================================================
# EXECUTION & COMPARISON
# =========================================================
def execute_comparison(plan: BuildPlan) -> Dict[str, Any]:
    """
    Execute both implementations and compare results.
    Writes generated code to files, runs them, captures output.
    """
    results_dir = os.path.join("Research", "runs")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write standard implementation
    std_path = os.path.join(results_dir, f"standard_{timestamp}.py")
    with open(std_path, "w", encoding="utf-8") as f:
        # Add imports first
        for lib in plan.libraries_needed:
            f.write(f"import {lib}\n")
        f.write("\n")
        f.write(plan.standard_approach)
    
    # Write AARL implementation
    aarl_path = os.path.join(results_dir, f"aarl_{timestamp}.py")
    with open(aarl_path, "w", encoding="utf-8") as f:
        for lib in plan.libraries_needed:
            f.write(f"import {lib}\n")
        f.write("\n")
        f.write(plan.aarl_approach)
    
    # Run both and capture results
    results = {
        "standard": {"success": False, "output": "", "error": ""},
        "aarl": {"success": False, "output": "", "error": ""},
        "paths": {"standard": std_path, "aarl": aarl_path},
        "metrics": plan.comparison_metrics,
        "expected_improvement": plan.expected_improvement
    }
    
    for key, path in [("standard", std_path), ("aarl", aarl_path)]:
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=60
            )
            results[key]["success"] = result.returncode == 0
            results[key]["output"] = result.stdout[:2000]
            results[key]["error"] = result.stderr[:500]
        except subprocess.TimeoutExpired:
            results[key]["error"] = "Execution timed out (60s)"
        except Exception as e:
            results[key]["error"] = str(e)
    
    return results


# =========================================================
# MAIN ENTRY POINT
# =========================================================
def run_research_comparison(problem: str, hypothesis: str) -> Dict[str, Any]:
    """
    Main entry point for Research Comparison.
    1. Detect domain
    2. Ensure libraries are installed
    3. Generate build plan via Groq
    4. Execute both implementations
    5. Compare and report
    """
    print("\n" + "="*80)
    print("🔬 RESEARCH COMPARISON — Standard vs AARL-Suggested Implementation")
    print("="*80)
    
    # Step 1: Detect domain
    domain = detect_domain(problem)
    print(f"\n📋 Detected Domain: {domain}")
    print(f"   Registry: {DOMAIN_REGISTRY.get(domain, DOMAIN_REGISTRY['general'])['description']}")
    
    # Step 2: Ensure libraries
    domain_config = DOMAIN_REGISTRY.get(domain, DOMAIN_REGISTRY["general"])
    ensure_libraries(domain_config["libraries"], domain_config["pip_packages"])
    
    # Step 3: Generate build plan via Groq
    print(f"\n🧠 Generating build plan via Groq (llama-3.3-70b)...")
    client = Groq()
    plan = generate_build_plan(client, problem, hypothesis, domain)
    if not plan:
        return {"error": "Failed to generate build plan", "domain": domain}
    
    print(f"   Libraries: {', '.join(plan.libraries_needed)}")
    print(f"   Metrics: {', '.join(plan.comparison_metrics)}")
    print(f"   Expected: {plan.expected_improvement}")
    
    # Step 4: Execute both
    print(f"\n⚡ Executing comparison...")
    results = execute_comparison(plan)
    
    # Step 5: Report
    print(f"\n📊 Comparison Results:")
    print(f"   Standard: {'✅ PASS' if results['standard']['success'] else '❌ FAIL'}")
    if results['standard']['output']:
        print(f"      Output: {results['standard']['output'][:200]}")
    if results['standard']['error']:
        print(f"      Error: {results['standard']['error'][:200]}")
    
    print(f"   AARL:     {'✅ PASS' if results['aarl']['success'] else '❌ FAIL'}")
    if results['aarl']['output']:
        print(f"      Output: {results['aarl']['output'][:200]}")
    if results['aarl']['error']:
        print(f"      Error: {results['aarl']['error'][:200]}")
    
    # Save full report
    report = {
        "domain": domain,
        "problem": problem,
        "hypothesis": hypothesis,
        "plan": plan.model_dump() if plan else {},
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    report_path = os.path.join("Research", "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\n📄 Full report saved to '{report_path}'")
    print("="*80)
    
    return report


if __name__ == "__main__":
    test_problem = "Design a better battery with higher energy density"
    test_hypothesis = "Using graphene-enhanced electrodes increases battery capacity by 20%"
    run_research_comparison(test_problem, test_hypothesis)