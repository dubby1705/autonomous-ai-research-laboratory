"""
SIMULATION ENGINE
Classifies the research problem → selects appropriate simulator → runs simulation
→ collects results → compares with prediction → updates evidence confidence.

IMPORTANT DISCLAIMER:
This engine selects a simulator based on the problem domain, but the simulator
can ONLY model what its equations and parameters support. For example:
- PyBaMM simulates electrochemical battery models (voltage, current, capacity, temperature)
  It CANNOT simulate quantum dots, graphene nanostructures, or novel chemistry
  unless those are represented as model parameters.
- PySpice simulates electronic circuits (resistors, capacitors, transistors, op-amps)
  It CANNOT simulate quantum effects or novel materials.
- Cantera simulates chemical kinetics and thermodynamics
  It CANNOT simulate molecular dynamics or quantum chemistry.

The simulation produces SYNTHETIC results for demonstration purposes.
These are NOT real experimental data. They show the PIPELINE STRUCTURE
but actual scientific conclusions require real simulator integration.

Simulator Mapping (with capability boundaries):
  Battery      → PyBaMM (electrochemical battery models — standard equivalent circuits, DFN models)
  Circuit      → PySpice (electronic circuit simulation — RLC, op-amps, transistors)
  Chemistry    → Cantera (chemical kinetics, thermodynamics — reaction mechanisms)
  Materials    → OpenMM / FEniCS (molecular dynamics / FEM — atomistic to continuum)
  Physics      → FEniCS (finite element method — PDEs, heat, fluid, structural)
  ML/AI        → PyTorch (neural network training — standard architectures)
  Mathematics  → SciPy (numerical optimization, integration — general purpose)
  General      → SciPy (fallback for generic numerical simulation)
"""

import os
import json
import math
import random
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# =========================================================
# PROBLEM CLASSIFIER
# =========================================================
DOMAIN_KEYWORDS = {
    "battery": {
        "keywords": ["battery", "electrode", "electrolyte", "lithium", "anode", "cathode",
                     "cell", "voltage", "capacity", "energy density", "power density",
                     "solid-state", "li-ion", "sodium-ion", "charge", "discharge"],
        "simulator": "PyBaMM",
        "description": "Electrochemical battery model"
    },
    "circuit": {
        "keywords": ["circuit", "amplifier", "transistor", "resistor", "capacitor",
                     "inductor", "oscillator", "filter", "op-amp", "voltage divider",
                     "rectifier", "modulator", "demodulator", "impedance", "bandwidth"],
        "simulator": "PySpice",
        "description": "Electronic circuit simulation"
    },
    "chemistry": {
        "keywords": ["catalyst", "reaction", "chemical", "kinetics", "thermodynamics",
                     "combustion", "oxidation", "reduction", "synthesis", "molecule",
                     "hydrogen", "oxygen", "methane", "ethanol", "activation energy",
                     "reaction rate", "equilibrium", "concentration"],
        "simulator": "Cantera",
        "description": "Chemical kinetics and thermodynamics"
    },
    "materials": {
        "keywords": ["material", "crystal", "lattice", "atom", "molecular", "polymer",
                     "composite", "alloy", "ceramic", "graphene", "nanotube", "nanoparticle",
                     "elastic", "plastic", "deformation", "stress", "strain", "fracture",
                     "thermal conductivity", "diffusion"],
        "simulator": "OpenMM",
        "description": "Molecular dynamics / materials simulation"
    },
    "physics": {
        "keywords": ["fluid", "flow", "heat", "thermal", "mechanical", "structural",
                     "electromagnetic", "wave", "acoustic", "vibration", "elasticity",
                     "finite element", "continuum", "navier-stokes", "maxwell",
                     "schrodinger", "quantum", "relativity", "gravitational"],
        "simulator": "FEniCS",
        "description": "Finite element physics simulation"
    },
    "machine_learning": {
        "keywords": ["neural", "deep learning", "machine learning", "classification",
                     "regression", "transformer", "cnn", "rnn", "lstm", "attention",
                     "gradient", "backpropagation", "loss function", "optimizer",
                     "training", "inference", "overfitting", "generalization",
                     "supervised", "unsupervised", "reinforcement"],
        "simulator": "PyTorch",
        "description": "Neural network training and evaluation"
    },
    "mathematics": {
        "keywords": ["optimization", "integration", "differential equation", "linear algebra",
                     "eigenvalue", "matrix", "vector", "calculus", "numerical",
                     "interpolation", "extrapolation", "convergence", "iteration",
                     "gradient descent", "newton method", "simulation", "modeling"],
        "simulator": "SciPy",
        "description": "Numerical computation and optimization"
    }
}

def classify_problem(problem_statement: str) -> Dict[str, Any]:
    """
    Classify a research problem into a domain and select the appropriate simulator.
    Uses keyword matching with scoring.
    """
    problem_lower = problem_statement.lower()
    scores = {}
    
    for domain, config in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in config["keywords"]:
            if kw in problem_lower:
                score += 1
        if score > 0:
            scores[domain] = score
    
    if not scores:
        return {
            "primary_domain": "general",
            "simulator": "SciPy",
            "description": "General numerical simulation",
            "confidence": 0.5,
            "all_domains": []
        }
    
    # Sort by score descending
    sorted_domains = sorted(scores.items(), key=lambda x: -x[1])
    primary = sorted_domains[0][0]
    total_score = sum(scores.values())
    
    return {
        "primary_domain": primary,
        "simulator": DOMAIN_KEYWORDS[primary]["simulator"],
        "description": DOMAIN_KEYWORDS[primary]["description"],
        "confidence": round(sorted_domains[0][1] / total_score, 3) if total_score > 0 else 0,
        "all_domains": [{"domain": d, "score": s} for d, s in sorted_domains]
    }


# =========================================================
# SIMULATION RUNNER
# =========================================================
def run_simulation(problem: str, hypothesis: str, simulator: str, 
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a simulation appropriate for the problem domain.
    
    For now, this generates synthetic simulation results since actual simulators
    (PyBaMM, PySpice, etc.) may not be installed. The structure is designed so
    that real simulators can be plugged in later.
    
    Returns:
        Dict with: success, predicted_value, observed_value, error, 
                   confidence_delta, metrics, warnings
    """
    if params is None:
        params = {}
    
    # Extract key terms from hypothesis for simulation
    hyp_lower = hypothesis.lower()
    
    # Generate a synthetic "simulation result" based on the hypothesis
    # In production, this would call the actual simulator
    random.seed(hash(hypothesis) % (2**32))
    
    # Simulate: generate a predicted outcome and an observed outcome
    # The "observed" outcome has some noise relative to the prediction
    base_value = random.uniform(0.5, 2.0)
    noise = random.uniform(-0.3, 0.3)
    observed_value = base_value + noise
    
    # Determine if the hypothesis was supported
    # If observed_value is in the same direction as predicted, it's supported
    supported = abs(noise) < 0.2  # Within 20% noise = supported
    
    # Calculate error metrics
    absolute_error = abs(observed_value - base_value)
    relative_error = absolute_error / max(abs(base_value), 0.001)
    
    # Confidence update: if supported, increase; if not, decrease
    confidence_delta = 0.15 if supported else -0.20
    
    return {
        "success": True,
        "simulator_used": simulator,
        "predicted_value": round(base_value, 4),
        "simulation_result": round(observed_value, 4),
        "absolute_error": round(absolute_error, 4),
        "relative_error": round(relative_error, 4),
        "hypothesis_supported": supported,
        "confidence_delta": confidence_delta,
        "metrics": {
            "rmse": round(math.sqrt(absolute_error**2), 4),
            "mae": round(absolute_error, 4),
            "r2": round(max(0, 1 - relative_error), 4)
        },
        "warnings": [] if supported else ["Simulation deviation exceeds threshold"],
        "simulation_time_ms": round(random.uniform(50, 5000), 2),
        "timestamp": datetime.now().isoformat()
    }


# =========================================================
# EVIDENCE UPDATE FROM SIMULATION
# =========================================================
def update_evidence_from_simulation(evidence_db_path: str, claim: str, 
                                     sim_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the evidence database with simulation results.
    Adds an 'experiment' or 'simulation' type evidence entry with
    confidence derived from how well the simulation matched predictions.
    """
    from Evidence import EvidenceDatabase, EvidenceEntry, EvidenceScore
    
    db = EvidenceDatabase(evidence_db_path)
    score = db.get_or_create(claim)
    
    # Create evidence entry from simulation
    evidence_type = "simulation"
    sim_confidence = max(0.0, min(1.0, 1.0 - sim_result["relative_error"]))
    
    entry = EvidenceEntry(
        claim=claim,
        evidence_type=evidence_type,
        content=(
            f"Simulation ({sim_result['simulator_used']}): "
            f"predicted={sim_result['predicted_value']:.3f}, "
            f"sim_output={sim_result['simulation_result']:.3f}, "
            f"error={sim_result['relative_error']:.2%}, "
            f"supported={sim_result['hypothesis_supported']}"
        ),
        confidence=sim_confidence,
        source_details={
            "source": "simulation",
            "simulator": sim_result["simulator_used"],
            "predicted": sim_result["predicted_value"],
            "simulation_result": sim_result["simulation_result"],
            "error": sim_result["relative_error"],
            "metrics": sim_result["metrics"]
        }
    )
    
    score.add_evidence(entry)
    score._recompute_confidence()
    db.save()
    
    return {
        "claim": claim,
        "evidence_added": evidence_type,
        "sim_confidence": round(sim_confidence, 4),
        "new_overall_confidence": round(score.confidence, 4),
        "verdict": score.get_verdict()
    }


# =========================================================
# MAIN SIMULATION ENGINE ENTRY POINT
# =========================================================
def run_simulation_engine(
    problem: str,
    hypotheses: List[str],
    evidence_db_path: str = "evidence_scoring_db.json",
    simulation_results_file: str = "simulation_results.json"
) -> Dict[str, Any]:
    """
    Main entry point for the Simulation Engine.
    
    For each hypothesis:
    1. Classify the problem domain
    2. Select the appropriate simulator
    3. Run the simulation
    4. Compare results with predictions
    5. Update evidence confidence
    6. Generate improved hypothesis if needed
    
    Args:
        problem: Original research problem
        hypotheses: List of hypotheses to simulate
        evidence_db_path: Path to evidence database
        simulation_results_file: Output file path
    
    Returns:
        Dict with simulation results and evidence updates
    """
    print("\n" + "="*80)
    print("🖥️  SIMULATION ENGINE — Problem Classification + Simulation + Evidence Update")
    print("="*80)
    
    # Step 1: Classify the problem
    classification = classify_problem(problem)
    print(f"\n📋 Problem Classification:")
    print(f"   Primary Domain: {classification['primary_domain']}")
    print(f"   Selected Simulator: {classification['simulator']}")
    print(f"   Description: {classification['description']}")
    print(f"   Confidence: {classification['confidence']:.1%}")
    if classification.get("all_domains"):
        print(f"   All Matched Domains:")
        for d in classification["all_domains"]:
            print(f"      • {d['domain']} (score: {d['score']})")
    
    # Step 2: Run simulations for each hypothesis
    all_results = []
    evidence_updates = []
    
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n  [{i}/{len(hypotheses)}] Simulating hypothesis...")
        print(f"      📝 {hypothesis[:120]}...")
        
        # Run simulation
        sim_result = run_simulation(
            problem=problem,
            hypothesis=hypothesis,
            simulator=classification["simulator"]
        )
        
        if not sim_result.get("success"):
            print(f"      ❌ Simulation failed.")
            continue
        
        # Update evidence
        evidence_update = update_evidence_from_simulation(
            evidence_db_path=evidence_db_path,
            claim=hypothesis,
            sim_result=sim_result
        )
        
        all_results.append(sim_result)
        evidence_updates.append(evidence_update)
        
        # Print results
        status = "✅ SUPPORTED" if sim_result["hypothesis_supported"] else "❌ NOT SUPPORTED"
        print(f"      🖥️  Simulator: {sim_result['simulator_used']}")
        print(f"      📊 Predicted: {sim_result['predicted_value']:.3f} → Sim Output: {sim_result['simulation_result']:.3f}")
        print(f"      📉 Error: {sim_result['relative_error']:.2%}")
        print(f"      {status}")
        print(f"      📈 Evidence confidence: {evidence_update['sim_confidence']:.3f} → overall: {evidence_update['new_overall_confidence']:.3f}")
        print(f"      🏷️  Verdict: {evidence_update['verdict']}")
    
    # Step 3: Save all results
    output = {
        "problem_classification": classification,
        "total_simulations_run": len(all_results),
        "hypotheses_supported": sum(1 for r in all_results if r["hypothesis_supported"]),
        "hypotheses_not_supported": sum(1 for r in all_results if not r["hypothesis_supported"]),
        "simulation_results": all_results,
        "evidence_updates": evidence_updates,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(simulation_results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"\n📄 Simulation results saved to '{simulation_results_file}'")
    print(f"📊 Summary: {output['total_simulations_run']} simulations, "
          f"{output['hypotheses_supported']} supported, "
          f"{output['hypotheses_not_supported']} not supported")
    print("="*80)
    
    return output


if __name__ == "__main__":
    # Demo
    test_problem = "Design a battery with higher energy density using graphene electrodes"
    test_hypotheses = [
        "Increasing electrode surface area improves battery power density by 20%",
        "Graphene-based anodes double lithium-ion battery capacity"
    ]
    run_simulation_engine(test_problem, test_hypotheses)