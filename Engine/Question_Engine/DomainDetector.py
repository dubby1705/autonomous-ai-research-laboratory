#!/usr/bin/env python3
"""
DomainDetector.py — Scientific Domain Detection for AARL
==========================================================
Analyzes the research problem, knowledge graph, and literature review
to detect which scientific domains are involved.

A problem may activate multiple domains simultaneously.
Example:
  Battery           -> Physics + Chemistry
  Carbon Capture    -> Chemistry
  Solar Cell        -> Physics + Chemistry
  Semiconductor     -> Quantum + Physics
  Superconductor    -> Quantum + Physics
  Aircraft Wing     -> Physics
  Catalyst Design   -> Chemistry
"""

import os
import json
import re
from typing import Dict, List, Set, Tuple, Any

# =========================================================
# DOMAIN DEFINITIONS
# =========================================================
DOMAIN_REGISTRY = {
    "physics": {
        "keywords": {
            "mechanics", "force", "motion", "velocity", "acceleration", "mass",
            "energy", "work", "power", "momentum", "newton",
            "thermodynamics", "heat", "temperature", "entropy", "enthalpy",
            "heat transfer", "conduction", "convection", "radiation", "fourier",
            "fluid dynamics", "bernoulli", "navier-stokes", "flow", "pressure",
            "electromagnetism", "maxwell", "electric field", "magnetic field",
            "circuit", "voltage", "current", "resistance", "capacitance",
            "material mechanics", "stress", "strain", "elasticity", "modulus",
            "aerodynamics", "lift", "drag", "airfoil", "wind tunnel",
            "structural", "beam", "load", "deflection", "buckling",
            "acoustics", "wave", "frequency", "amplitude", "resonance",
            "optics", "lens", "refraction", "diffraction", "interference",
        },
        "topics": [
            "Classical Mechanics", "Thermodynamics", "Heat Transfer",
            "Fluid Dynamics", "Electromagnetism", "Material Mechanics",
            "Acoustics", "Optics", "Aerodynamics",
        ],
        "math_engine": "physics",
        "simulator": "physics",
    },
    "chemistry": {
        "keywords": {
            "chemistry", "chemical", "reaction", "kinetics", "catalyst",
            "thermodynamics", "gibbs free energy", "enthalpy", "entropy",
            "electrochemistry", "nernst", "butler-volmer", "electrode",
            "battery", "lithium", "anode", "cathode", "electrolyte",
            "adsorption", "langmuir", "isotherm", "fick", "diffusion",
            "molecular", "molecule", "bond", "orbital", "electron",
            "ph", "acid", "base", "buffer", "titration",
            "polymer", "monomer", "crosslink", "polymerization",
            "carbon capture", "co2", "sorbent", "mof", "zeolite",
            "combustion", "oxidation", "reduction", "redox",
            "crystallization", "precipitation", "solubility",
        },
        "topics": [
            "Chemical Kinetics", "Thermodynamics", "Electrochemistry",
            "Adsorption & Diffusion", "Catalysis", "Battery Chemistry",
            "Polymer Chemistry", "Analytical Chemistry",
        ],
        "math_engine": "chemistry",
        "simulator": "chemistry",
    },
    "quantum": {
        "keywords": {
            "quantum", "schrodinger", "wave function", "hamiltonian",
            "electron", "spin", "orbital", "band structure", "band gap",
            "semiconductor", "diode", "transistor", "doping",
            "superconductor", "critical temperature", "meissner",
            "tunneling", "quantum state", "quantum computing", "qubit",
            "photoelectric", "photon", "planck", "quantum mechanics",
            "density functional", "dft", "ab initio", "hartree-fock",
            "condensed matter", "crystal structure", "lattice",
            "graphene", "nanotube", "nanostructure", "quantum dot",
        },
        "topics": [
            "Quantum Mechanics", "Band Structure Theory",
            "Semiconductor Physics", "Superconductivity",
            "Quantum Chemistry", "Condensed Matter Physics",
        ],
        "math_engine": "quantum",
        "simulator": "physics",
    },
    "biology": {
        "keywords": {
            "biology", "dna", "rna", "protein", "enzyme",
            "gene", "genome", "mutation", "evolution",
            "microbiology", "bacteria", "virus", "microorganism",
            "biochemistry", "metabolism", "pathway",
            "neuroscience", "neuron", "synapse", "brain",
            "bioinformatics", "genomics", "proteomics",
            "biomedical", "tissue", "organ", "physiology",
        },
        "topics": [
            "Molecular Biology", "Biochemistry", "Genetics",
            "Microbiology", "Ecology", "Neuroscience",
        ],
        "math_engine": "chemistry",
        "simulator": "chemistry",
    },
    "materials_science": {
        "keywords": {
            "material", "composite", "alloy", "ceramic", "polymer",
            "nanomaterial", "nanoparticle", "nanostructure",
            "crystal", "crystalline", "amorphous", "grain boundary",
            "mechanical property", "tensile", "hardness", "fracture",
            "thermal conductivity", "electrical conductivity",
            "corrosion", "oxidation", "degradation",
            "coating", "thin film", "surface", "interface",
            "metallurgy", "phase diagram", "diffusion",
        },
        "topics": [
            "Nanomaterials", "Composites", "Metallurgy",
            "Surface Science", "Mechanical Properties",
        ],
        "math_engine": "physics",
        "simulator": "physics",
    },
    "computer_architecture": {
        "keywords": {
            "cpu", "processor", "architecture", "instruction set", "pipeline",
            "cache", "branch prediction", "superscalar", "out-of-order",
            "execution", "core", "thread", "clock", "frequency", "ipc",
            "instructions per cycle", "microarchitecture", "register", "alu",
            "fpga", "asic", "soc", "chip", "semiconductor", "transistor",
            "power consumption", "area usage", "memory hierarchy", "tlb",
            "prefetch", "speculative", "reorder buffer", "issue width",
            "decode", "fetch", "hardware", "computer", "computing",
            "processor design", "cpu design", "power efficiency",
            "thermal design", "tdp", "moore", "amdahl", "bottleneck",
        },
        "topics": [
            "Instruction Set Architecture", "Microarchitecture",
            "Pipeline Design", "Cache Hierarchy", "Branch Prediction",
            "Power-Efficient Computing", "Performance Analysis",
        ],
        "math_engine": "physics",
        "simulator": "general",
    },
}

# =========================================================
# DOMAIN KEYWORD -> SIMULATOR MAPPING
# =========================================================
DOMAIN_SIMULATOR_MAP = {
    # Battery / Energy Storage
    "battery": "battery_simulator.py",
    "lithium": "battery_simulator.py",
    "electrode": "battery_simulator.py",
    "electrolyte": "battery_simulator.py",
    "energy density": "battery_simulator.py",
    "supercapacitor": "battery_simulator.py",
    "fuel cell": "battery_simulator.py",

    # Carbon Capture
    "carbon capture": "carbon_capture_simulator.py",
    "co2": "carbon_capture_simulator.py",
    "sorbent": "carbon_capture_simulator.py",
    "carbon dioxide": "carbon_capture_simulator.py",
    "flue gas": "carbon_capture_simulator.py",

    # Solar / Photovoltaics
    "solar": "solar_simulator.py",
    "photovoltaic": "solar_simulator.py",
    "solar cell": "solar_simulator.py",
    "perovskite": "solar_simulator.py",

    # Thermal / Heat Transfer
    "heat transfer": "thermal_simulator.py",
    "thermal": "thermal_simulator.py",
    "heat exchanger": "thermal_simulator.py",
    "cooling": "thermal_simulator.py",
    "thermoelectric": "thermal_simulator.py",

    # Structural / Civil
    "bridge": "structural_simulator.py",
    "building": "structural_simulator.py",
    "structural": "structural_simulator.py",
    "beam": "structural_simulator.py",
    "column": "structural_simulator.py",
    "seismic": "structural_simulator.py",

    # Aerodynamics
    "wing": "aerodynamics_simulator.py",
    "airfoil": "aerodynamics_simulator.py",
    "aerodynamics": "aerodynamics_simulator.py",
    "wind turbine": "aerodynamics_simulator.py",
    "propeller": "aerodynamics_simulator.py",
    "turbine": "aerodynamics_simulator.py",
    "lift": "aerodynamics_simulator.py",
    "drag": "aerodynamics_simulator.py",
    "flight": "aerodynamics_simulator.py",
    "aircraft": "aerodynamics_simulator.py",
    "rocket": "aerodynamics_simulator.py",
    "nozzle": "aerodynamics_simulator.py",

    # Computer Architecture / CPU Design
    "cpu": "general_simulator.py",
    "processor": "general_simulator.py",
    "architecture": "general_simulator.py",
    "instruction set": "general_simulator.py",
    "pipeline": "general_simulator.py",
    "cache": "general_simulator.py",
    "branch prediction": "general_simulator.py",
    "superscalar": "general_simulator.py",
    "out-of-order": "general_simulator.py",
    "microarchitecture": "general_simulator.py",
    "ipc": "general_simulator.py",
    "instructions per cycle": "general_simulator.py",
    "fpga": "general_simulator.py",
    "asic": "general_simulator.py",
    "soc": "general_simulator.py",
    "chip": "general_simulator.py",
    "semiconductor": "general_simulator.py",
    "transistor": "general_simulator.py",
    "clock": "general_simulator.py",
    "tdp": "general_simulator.py",
}


def detect_domains(problem: str, knowledge_data: dict = None) -> List[dict]:
    """
    Detect which scientific domains are involved in the research problem.
    Uses BOTH the research problem text AND the knowledge graph data.
    """
    problem_lower = problem.lower()
    corpus = problem_lower
    
    if knowledge_data:
        for key, value in knowledge_data.items():
            if isinstance(value, str):
                corpus += " " + value.lower()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        corpus += " " + item.lower()
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str):
                                corpus += " " + v.lower()
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        corpus += " " + v.lower()
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                corpus += " " + item.lower()

    scores = []
    for domain_name, domain_info in DOMAIN_REGISTRY.items():
        score = 0
        matched_keywords = set()
        for keyword in domain_info["keywords"]:
            if keyword in corpus:
                score += 1
                matched_keywords.add(keyword)
        if score > 0:
            scores.append({
                "domain": domain_name,
                "score": score,
                "matched_keywords": sorted(matched_keywords)[:10],
                "topics": domain_info["topics"],
                "math_engine": domain_info["math_engine"],
                "simulator": domain_info["simulator"],
            })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def detect_simulator(problem: str, knowledge_data: dict = None) -> str:
    """
    Detect which domain-specific simulator to use for Phase 9.
    """
    corpus = problem.lower()
    if knowledge_data:
        for key, value in knowledge_data.items():
            if isinstance(value, str):
                corpus += " " + value.lower()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        corpus += " " + item.lower()

    sim_scores = {}
    for keyword, simulator in DOMAIN_SIMULATOR_MAP.items():
        if keyword in corpus:
            sim_scores[simulator] = sim_scores.get(simulator, 0) + 1

    if sim_scores:
        return max(sim_scores, key=sim_scores.get)
    return "general_simulator.py"


def get_math_engines(domains: List[dict]) -> List[str]:
    """Get unique math engines needed based on detected domains."""
    engines = set()
    for domain in domains:
        engines.add(domain["math_engine"])
    return list(engines)


def format_domain_report(domains: List[dict]) -> str:
    """Format domain detection results for display."""
    lines = []
    lines.append("  Domain Detection Results:")
    if not domains:
        lines.append("    (No specific domains detected)")
        return "\n".join(lines)
        
    for d in domains:
        kw = ", ".join(d["matched_keywords"][:5])
        lines.append(f"    [{d['domain'].upper():20s}] score={d['score']}, engine={d['math_engine']}")
        lines.append(f"     topics: {', '.join(d['topics'][:3])}")
        lines.append(f"     keywords: {kw}")
    return "\n".join(lines)


if __name__ == "__main__":
    test_cases = [
        "Designing a novel carbon capture material with reduced energy consumption",
        "High-energy-density lithium-ion battery with graphene-enhanced electrodes",
        "Perovskite solar cell with improved efficiency and stability",
        "Quantum computing with superconducting qubits",
        "Aircraft wing design with reduced drag and improved lift",
        "Novel catalyst for hydrogen production through water splitting",
    ]
    
    for problem in test_cases:
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}")
        detected_domains = detect_domains(problem)
        print(format_domain_report(detected_domains))
        sim = detect_simulator(problem)
        print(f"  -> Simulator: {sim}")
        print(f"  -> Math Engines: {get_math_engines(detected_domains)}")