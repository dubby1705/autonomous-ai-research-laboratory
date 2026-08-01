#!/usr/bin/env python3
"""
domain_metrics.py — Domain-Specific Simulation Metrics Registry
=================================================================
Defines scientifically meaningful metrics, baseline values, units,
and hypothesis-to-parameter mappings for each research domain.

Each domain has:
  - Detection keywords (for classifying the research problem)
  - Metrics: {metric_name: {unit, baseline, direction, description}}
  - Parameter map: hypothesis keywords → domain-specific parameters
  - Simulation function: computes baseline and hypothesis-driven results
"""

import math
import re
from typing import Dict, Any, List, Tuple, Optional


# =========================================================
# DOMAIN DETECTION KEYWORDS
# =========================================================
DOMAIN_KEYWORDS = {
    "chemistry": [
        "acid", "base", "ph", "h2so4", "sulfuric", "reaction", "catalyst",
        "chemical", "kinetics", "thermodynamics", "combustion", "oxidation",
        "reduction", "molecule", "hydrogen", "synthesis", "yield", "selectivity",
        "molar", "concentration", "titration", "buffer", "corrosion",
        "electrolysis", "polymerization", "solubility", "precipitate",
        "distillation", "organic", "inorganic", "compound", "substance",
        "acidic", "alkaline", "neutralization", "enthalpy", "entropy",
        "gibbs", "activation energy", "rate constant", "equilibrium",
    ],
    "battery": [
        "battery", "electrode", "electrolyte", "lithium", "anode", "cathode",
        "cell", "voltage", "capacity", "energy density", "charge", "discharge",
        "solid-state", "supercapacitor", "ion", "graphene", "power density",
        "cycle life", "charging", "storage", "sodium-ion", "fuel cell",
    ],
    "aerospace": [
        "wing", "airfoil", "aerodynamic", "lift", "drag", "thrust", "flight",
        "aircraft", "rocket", "nozzle", "propeller", "turbine", "velocity",
        "mach", "altitude", "payload", "propulsion", "aviation", "drone",
        "glider", "supersonic", "hypersonic", "stall", "maneuver",
    ],
    "robotics": [
        "robot", "actuator", "manipulator", "gripper", "autonomous", "sensor",
        "control", "navigation", "kinematics", "dynamics", "end-effector",
        "servo", "motor", "encoder", "feedback", "trajectory", "obstacle",
        "motion planning", "teleoperation", "swarm", "arm", "precision",
        "accuracy", "repeatability", "payload", "end effector", "mechanical",
        "automation", "machine", "device", "mechanism",
    ],
    "civil_engineering": [
        "bridge", "building", "structural", "beam", "column", "seismic",
        "concrete", "steel", "foundation", "load", "deflection", "buckling",
        "earthquake", "wind load", "infrastructure", "highway", "tunnel",
        "dam", "retaining wall", "truss", "reinforced",
    ],
    "medicine": [
        "drug", "pharmaceutical", "therapeutic", "bioavailability", "toxicity",
        "clinical", "patient", "disease", "treatment", "dosage", "pharmacokinetic",
        "pharmacodynamic", "antibiotic", "vaccine", "cancer", "tumor",
        "enzyme inhibitor", "receptor", "metabolism", "efficacy", "side effect",
    ],
    "materials": [
        "material", "crystal", "polymer", "composite", "alloy", "graphene",
        "nanotube", "stress", "strain", "thermal conductivity", "tensile",
        "durability", "strength", "hardness", "fracture", "fatigue",
        "corrosion resistance", "elastic modulus", "yield strength",
        "microstructure", "coating", "thin film",
    ],
    "machine_learning": [
        "neural", "deep learning", "transformer", "classification", "regression",
        "training", "gradient", "optimizer", "loss function", "accuracy",
        "inference", "model", "dataset", "overfitting", "generalization",
        "convolutional", "attention", "embedding", "fine-tuning", "pretrained",
        "machine learning", "ml", "ai", "artificial intelligence", "algorithm",
        "neural network", "deep neural", "latency", "throughput", "distillation",
        "quantization", "f1", "precision recall", "supervised", "unsupervised",
        "reinforcement", "computer vision", "nlp", "language model",
    ],
    "computer_architecture": [
        "cpu", "processor", "architecture", "instruction set", "pipeline",
        "cache", "branch prediction", "superscalar", "out-of-order", "execution",
        "core", "thread", "clock", "frequency", "ipc", "instructions per cycle",
        "microarchitecture", "register", "alu", "fpga", "asic", "soc",
        "chip", "semiconductor", "transistor", "power consumption", "area usage",
        "throughput", "latency", "memory hierarchy", "tlb", "prefetch",
        "speculative", "reorder buffer", "issue width", "decode", "fetch",
        "hardware", "computer", "computing", "processor design", "cpu design",
        "performance", "power efficiency", "thermal design", "tdp",
    ],
}


# =========================================================
# DOMAIN-SPECIFIC METRIC DEFINITIONS
# =========================================================
# Each metric: {
#   "unit": str,
#   "baseline": float,
#   "direction": "higher_better" | "lower_better",
#   "description": str,
# }
DOMAIN_METRICS = {
    "chemistry": {
        "ph_value": {
            "unit": "pH",
            "baseline": 1.0,
            "direction": "lower_better",
            "description": "Acidity level (lower = more acidic)",
        },
        "acid_strength_ka": {
            "unit": "Ka",
            "baseline": 1.0e-2,
            "direction": "higher_better",
            "description": "Acid dissociation constant (higher = stronger acid)",
        },
        "reaction_rate_constant": {
            "unit": "s⁻¹",
            "baseline": 0.05,
            "direction": "higher_better",
            "description": "Rate constant of the chemical reaction",
        },
        "yield_percent": {
            "unit": "%",
            "baseline": 78.5,
            "direction": "higher_better",
            "description": "Percentage yield of the target product",
        },
        "selectivity_percent": {
            "unit": "%",
            "baseline": 80.0,
            "direction": "higher_better",
            "description": "Selectivity toward the desired product",
        },
        "activation_energy_kj_mol": {
            "unit": "kJ/mol",
            "baseline": 85.0,
            "direction": "lower_better",
            "description": "Activation energy barrier for the reaction",
        },
        "turnover_frequency": {
            "unit": "s⁻¹",
            "baseline": 0.8,
            "direction": "higher_better",
            "description": "Catalytic turnover frequency (molecules per active site per second)",
        },
        "thermal_stability_c": {
            "unit": "°C",
            "baseline": 250.0,
            "direction": "higher_better",
            "description": "Decomposition temperature of the substance",
        },
        "solubility_g_l": {
            "unit": "g/L",
            "baseline": 45.0,
            "direction": "higher_better",
            "description": "Solubility in water at standard conditions",
        },
        "purity_percent": {
            "unit": "%",
            "baseline": 95.0,
            "direction": "higher_better",
            "description": "Purity of the synthesized compound",
        },
    },
    "battery": {
        "energy_density_wh_kg": {
            "unit": "Wh/kg",
            "baseline": 240.0,
            "direction": "higher_better",
            "description": "Gravimetric energy density",
        },
        "cycle_life": {
            "unit": "cycles",
            "baseline": 800,
            "direction": "higher_better",
            "description": "Number of charge/discharge cycles before 80% capacity",
        },
        "charging_efficiency_pct": {
            "unit": "%",
            "baseline": 90.0,
            "direction": "higher_better",
            "description": "Round-trip charging efficiency",
        },
        "degradation_per_cycle_pct": {
            "unit": "%/cycle",
            "baseline": 0.025,
            "direction": "lower_better",
            "description": "Capacity degradation per cycle",
        },
        "temperature_rise_c": {
            "unit": "°C",
            "baseline": 8.0,
            "direction": "lower_better",
            "description": "Temperature rise during operation",
        },
        "internal_resistance_ohm": {
            "unit": "Ω",
            "baseline": 0.15,
            "direction": "lower_better",
            "description": "Internal resistance of the cell",
        },
        "power_density_w_kg": {
            "unit": "W/kg",
            "baseline": 350.0,
            "direction": "higher_better",
            "description": "Gravimetric power density",
        },
    },
    "aerospace": {
        "lift_coefficient": {
            "unit": "CL",
            "baseline": 1.0,
            "direction": "higher_better",
            "description": "Lift coefficient of the airfoil",
        },
        "drag_coefficient": {
            "unit": "CD",
            "baseline": 0.35,
            "direction": "lower_better",
            "description": "Drag coefficient of the airfoil",
        },
        "lift_to_drag_ratio": {
            "unit": "L/D",
            "baseline": 2.86,
            "direction": "higher_better",
            "description": "Lift-to-drag ratio (aerodynamic efficiency)",
        },
        "thrust_force_n": {
            "unit": "N",
            "baseline": 500.0,
            "direction": "higher_better",
            "description": "Thrust force produced by the propulsion system",
        },
        "max_speed_mps": {
            "unit": "m/s",
            "baseline": 22.0,
            "direction": "higher_better",
            "description": "Maximum achievable speed",
        },
        "fuel_efficiency_km_kg": {
            "unit": "km/kg",
            "baseline": 1.2,
            "direction": "higher_better",
            "description": "Distance per unit fuel mass",
        },
        "stall_speed_mps": {
            "unit": "m/s",
            "baseline": 15.0,
            "direction": "lower_better",
            "description": "Minimum speed to maintain lift",
        },
        "structural_margin": {
            "unit": "index",
            "baseline": 1.5,
            "direction": "higher_better",
            "description": "Structural safety margin",
        },
    },
    "robotics": {
        "position_accuracy_mm": {
            "unit": "mm",
            "baseline": 2.5,
            "direction": "lower_better",
            "description": "End-effector position accuracy",
        },
        "repeatability_mm": {
            "unit": "mm",
            "baseline": 1.2,
            "direction": "lower_better",
            "description": "End-effector repeatability",
        },
        "payload_capacity_kg": {
            "unit": "kg",
            "baseline": 5.0,
            "direction": "higher_better",
            "description": "Maximum payload capacity",
        },
        "response_time_ms": {
            "unit": "ms",
            "baseline": 120.0,
            "direction": "lower_better",
            "description": "Control loop response time",
        },
        "energy_efficiency_pct": {
            "unit": "%",
            "baseline": 72.0,
            "direction": "higher_better",
            "description": "Energy conversion efficiency",
        },
        "trajectory_tracking_error_mm": {
            "unit": "mm",
            "baseline": 3.0,
            "direction": "lower_better",
            "description": "Trajectory tracking error",
        },
        "autonomy_level": {
            "unit": "index",
            "baseline": 0.6,
            "direction": "higher_better",
            "description": "Level of autonomous operation (0-1)",
        },
    },
    "civil_engineering": {
        "load_capacity_kn": {
            "unit": "kN",
            "baseline": 250.0,
            "direction": "higher_better",
            "description": "Maximum load-bearing capacity",
        },
        "deflection_mm": {
            "unit": "mm",
            "baseline": 12.0,
            "direction": "lower_better",
            "description": "Maximum deflection under load",
        },
        "safety_factor": {
            "unit": "index",
            "baseline": 1.5,
            "direction": "higher_better",
            "description": "Structural safety factor",
        },
        "seismic_resistance": {
            "unit": "index",
            "baseline": 0.7,
            "direction": "higher_better",
            "description": "Seismic resistance rating (0-1)",
        },
        "material_strength_mpa": {
            "unit": "MPa",
            "baseline": 220.0,
            "direction": "higher_better",
            "description": "Material compressive/tensile strength",
        },
        "service_life_years": {
            "unit": "years",
            "baseline": 50,
            "direction": "higher_better",
            "description": "Expected service life",
        },
        "construction_cost_index": {
            "unit": "index",
            "baseline": 1.0,
            "direction": "lower_better",
            "description": "Relative construction cost",
        },
    },
    "medicine": {
        "drug_efficacy_pct": {
            "unit": "%",
            "baseline": 65.0,
            "direction": "higher_better",
            "description": "Therapeutic efficacy percentage",
        },
        "bioavailability_pct": {
            "unit": "%",
            "baseline": 45.0,
            "direction": "higher_better",
            "description": "Fraction of drug reaching systemic circulation",
        },
        "toxicity_index": {
            "unit": "index",
            "baseline": 0.4,
            "direction": "lower_better",
            "description": "Toxicity index (0-1, higher = more toxic)",
        },
        "half_life_hours": {
            "unit": "hours",
            "baseline": 6.0,
            "direction": "higher_better",
            "description": "Drug elimination half-life",
        },
        "ic50_nm": {
            "unit": "nM",
            "baseline": 250.0,
            "direction": "lower_better",
            "description": "Half-maximal inhibitory concentration",
        },
        "selectivity_index": {
            "unit": "index",
            "baseline": 2.0,
            "direction": "higher_better",
            "description": "Therapeutic selectivity index",
        },
        "side_effect_severity": {
            "unit": "index",
            "baseline": 0.5,
            "direction": "lower_better",
            "description": "Side effect severity (0-1)",
        },
    },
    "materials": {
        "tensile_strength_mpa": {
            "unit": "MPa",
            "baseline": 250.0,
            "direction": "higher_better",
            "description": "Ultimate tensile strength",
        },
        "elastic_modulus_gpa": {
            "unit": "GPa",
            "baseline": 70.0,
            "direction": "higher_better",
            "description": "Young's modulus",
        },
        "thermal_conductivity_w_mk": {
            "unit": "W/m·K",
            "baseline": 45.0,
            "direction": "higher_better",
            "description": "Thermal conductivity",
        },
        "hardness_hv": {
            "unit": "HV",
            "baseline": 120.0,
            "direction": "higher_better",
            "description": "Vickers hardness",
        },
        "fracture_toughness_mpa_m05": {
            "unit": "MPa·m^0.5",
            "baseline": 25.0,
            "direction": "higher_better",
            "description": "Fracture toughness",
        },
        "corrosion_rate_mm_yr": {
            "unit": "mm/yr",
            "baseline": 0.5,
            "direction": "lower_better",
            "description": "Corrosion rate",
        },
        "density_g_cm3": {
            "unit": "g/cm³",
            "baseline": 2.7,
            "direction": "lower_better",
            "description": "Material density",
        },
    },
    "machine_learning": {
        "accuracy_pct": {
            "unit": "%",
            "baseline": 84.5,
            "direction": "higher_better",
            "description": "Model accuracy",
        },
        "inference_latency_ms": {
            "unit": "ms",
            "baseline": 45.0,
            "direction": "lower_better",
            "description": "Inference latency",
        },
        "training_time_min": {
            "unit": "min",
            "baseline": 120.0,
            "direction": "lower_better",
            "description": "Training time",
        },
        "memory_mb": {
            "unit": "MB",
            "baseline": 512.0,
            "direction": "lower_better",
            "description": "Memory footprint",
        },
        "generalization_error": {
            "unit": "index",
            "baseline": 0.15,
            "direction": "lower_better",
            "description": "Generalization error (lower = better)",
        },
        "f1_score": {
            "unit": "index",
            "baseline": 0.82,
            "direction": "higher_better",
            "description": "F1 score",
        },
        "throughput_samples_s": {
            "unit": "samples/s",
            "baseline": 1000.0,
            "direction": "higher_better",
            "description": "Inference throughput",
        },
    },
    "computer_architecture": {
        "ipc": {
            "unit": "IPC",
            "baseline": 1.5,
            "direction": "higher_better",
            "description": "Instructions per cycle (performance)",
        },
        "clock_frequency_ghz": {
            "unit": "GHz",
            "baseline": 3.5,
            "direction": "higher_better",
            "description": "Clock frequency",
        },
        "power_consumption_w": {
            "unit": "W",
            "baseline": 65.0,
            "direction": "lower_better",
            "description": "Power consumption (TDP)",
        },
        "performance_per_watt": {
            "unit": "GFLOPS/W",
            "baseline": 2.5,
            "direction": "higher_better",
            "description": "Performance per watt (efficiency)",
        },
        "area_mm2": {
            "unit": "mm²",
            "baseline": 150.0,
            "direction": "lower_better",
            "description": "Die area",
        },
        "cache_hit_rate_pct": {
            "unit": "%",
            "baseline": 92.0,
            "direction": "higher_better",
            "description": "Cache hit rate",
        },
        "branch_prediction_accuracy_pct": {
            "unit": "%",
            "baseline": 95.0,
            "direction": "higher_better",
            "description": "Branch prediction accuracy",
        },
        "pipeline_stalls_per_1000": {
            "unit": "stalls/1k",
            "baseline": 45.0,
            "direction": "lower_better",
            "description": "Pipeline stalls per 1000 instructions",
        },
        "memory_latency_ns": {
            "unit": "ns",
            "baseline": 80.0,
            "direction": "lower_better",
            "description": "Memory access latency",
        },
        "throughput_gops": {
            "unit": "GOPS",
            "baseline": 50.0,
            "direction": "higher_better",
            "description": "Throughput in giga-operations per second",
        },
    },
}


# =========================================================
# DOMAIN-SPECIFIC HYPOTHESIS → PARAMETER MAPPING
# =========================================================
# Each entry: (keyword_pattern, parameter_name, modification_type, strength)
# modification_type: "multiply" (compounds), "set" (overrides)
DOMAIN_PARAMETER_MAP = {
    "chemistry": [
        # --- Acidity / pH ---
        (r"acidic|more acidic|stronger acid|superacid", "acid_strength_ka", "multiply", 3.0),
        (r"ph\s*lower|lower\s*ph|decrease\s*ph", "ph_value", "multiply", 0.5),
        (r"proton|h\+|hydrogen ion", "acid_strength_ka", "multiply", 1.8),
        (r"lewis acid|electron acceptor", "acid_strength_ka", "multiply", 1.5),
        (r"superacid|magic acid|fluoroantimonic", "acid_strength_ka", "set", 10.0),
        (r"carborane|h(chb11cl11)", "acid_strength_ka", "set", 8.0),
        (r"triflic|trifluoromethanesulfonic", "acid_strength_ka", "set", 6.0),
        (r"fluorosulfuric|fluorosulfonic", "acid_strength_ka", "set", 5.0),
        (r"perchloric", "acid_strength_ka", "set", 4.0),
        (r"hydrochloric|hcl", "acid_strength_ka", "set", 3.0),
        (r"nitric|hno3", "acid_strength_ka", "set", 2.5),
        (r"sulfuric|h2so4", "acid_strength_ka", "set", 2.0),

        # --- Catalysis / Reaction ---
        (r"catalyst|catalytic", "turnover_frequency", "multiply", 3.0),
        (r"enzyme", "turnover_frequency", "multiply", 2.5),
        (r"nanoparticle|nano", "turnover_frequency", "multiply", 1.8),
        (r"surface area|porous", "turnover_frequency", "multiply", 1.5),
        (r"activation energy|lower.*barrier", "activation_energy_kj_mol", "multiply", 0.6),
        (r"rate|kinetics|faster", "reaction_rate_constant", "multiply", 2.0),
        (r"temperature|heat", "reaction_rate_constant", "multiply", 1.5),
        (r"pressure", "reaction_rate_constant", "multiply", 1.3),

        # --- Yield / Selectivity ---
        (r"yield|higher yield|improve yield", "yield_percent", "multiply", 1.2),
        (r"selectiv", "selectivity_percent", "multiply", 1.15),
        (r"purif|purity", "purity_percent", "multiply", 1.05),
        (r"solub|dissolve", "solubility_g_l", "multiply", 1.5),
        (r"stabil|thermal", "thermal_stability_c", "multiply", 1.3),
    ],
    "battery": [
        (r"graphene", "cathode_conductivity", "multiply", 3.0),
        (r"nanotube", "anode_conductivity", "multiply", 5.0),
        (r"solid.?state", "electrolyte_resistance", "multiply", 0.5),
        (r"polymer", "electrolyte_resistance", "multiply", 0.7),
        (r"conductivity", "ionic_conductivity", "multiply", 2.0),
        (r"internal.?resist", "internal_resistance", "multiply", 0.7),
        (r"interface", "charge_transfer_resistance", "multiply", 0.6),
        (r"coating", "electrode_protection", "multiply", 2.0),
        (r"doping", "carrier_density", "multiply", 1.8),
        (r"porous", "surface_area", "multiply", 2.5),
        (r"nano", "diffusion_coefficient", "multiply", 2.0),
        (r"silicon", "anode_capacity", "multiply", 3.0),
        (r"sulfur", "cathode_capacity", "multiply", 4.0),
        (r"lithium.?air", "energy_density_factor", "set", 3.0),
        (r"capacity", "capacity_factor", "multiply", 1.3),
        (r"voltage", "voltage_factor", "multiply", 1.15),
        (r"metal.?oxide", "cathode_stability", "multiply", 1.5),
        (r"aqueous", "electrolyte_safety", "multiply", 2.0),
        (r"composite", "mechanical_stability", "multiply", 1.5),
        (r"thermal.?conduct", "thermal_conductivity", "multiply", 1.5),
        (r"diffusion", "diffusion_coefficient", "multiply", 1.5),
    ],
    "aerospace": [
        (r"winglet|wing tip", "lift_coefficient", "multiply", 1.15),
        (r"airfoil|profile", "lift_coefficient", "multiply", 1.1),
        (r"lift", "lift_coefficient", "multiply", 1.2),
        (r"drag|streamline", "drag_coefficient", "multiply", 0.7),
        (r"aerodynamic|smooth", "drag_coefficient", "multiply", 0.8),
        (r"thrust|propulsion|engine", "thrust_force_n", "multiply", 1.3),
        (r"speed|velocity|faster", "max_speed_mps", "multiply", 1.2),
        (r"fuel|efficient", "fuel_efficiency_km_kg", "multiply", 1.3),
        (r"stall|low speed", "stall_speed_mps", "multiply", 0.8),
        (r"structural|reinforce", "structural_margin", "multiply", 1.2),
        (r"lightweight|weight", "structural_margin", "multiply", 1.1),
        (r"composite|material", "structural_margin", "multiply", 1.15),
    ],
    "robotics": [
        (r"precision|accurate", "position_accuracy_mm", "multiply", 0.6),
        (r"repeatab", "repeatability_mm", "multiply", 0.6),
        (r"payload|capacity|load", "payload_capacity_kg", "multiply", 1.3),
        (r"fast|speed|quick", "response_time_ms", "multiply", 0.6),
        (r"efficien", "energy_efficiency_pct", "multiply", 1.2),
        (r"tracking|trajectory", "trajectory_tracking_error_mm", "multiply", 0.6),
        (r"autonomous|self|adaptive", "autonomy_level", "multiply", 1.3),
        (r"control|feedback", "response_time_ms", "multiply", 0.7),
        (r"sensor|vision|perception", "position_accuracy_mm", "multiply", 0.7),
        (r"gripper|manipulat", "payload_capacity_kg", "multiply", 1.2),
    ],
    "civil_engineering": [
        (r"load|capacity|bearing", "load_capacity_kn", "multiply", 1.3),
        (r"deflection|bend|flex", "deflection_mm", "multiply", 0.6),
        (r"safe|factor", "safety_factor", "multiply", 1.2),
        (r"seismic|earthquake", "seismic_resistance", "multiply", 1.3),
        (r"strength|reinforce|steel", "material_strength_mpa", "multiply", 1.3),
        (r"durab|life|long", "service_life_years", "multiply", 1.2),
        (r"cost|econom", "construction_cost_index", "multiply", 0.8),
        (r"concrete|cement", "material_strength_mpa", "multiply", 1.2),
        (r"composite|fiber", "material_strength_mpa", "multiply", 1.4),
        (r"foundation|soil", "load_capacity_kn", "multiply", 1.2),
    ],
    "medicine": [
        (r"efficac|effective|potent", "drug_efficacy_pct", "multiply", 1.2),
        (r"bioavail|absorption|delivery", "bioavailability_pct", "multiply", 1.3),
        (r"toxic|safe", "toxicity_index", "multiply", 0.6),
        (r"half.life|duration|long.acting", "half_life_hours", "multiply", 1.4),
        (r"potency|ic50|inhibit", "ic50_nm", "multiply", 0.5),
        (r"selectiv|target", "selectivity_index", "multiply", 1.3),
        (r"side.effect|adverse", "side_effect_severity", "multiply", 0.6),
        (r"nanoparticle|nano|liposome", "bioavailability_pct", "multiply", 1.5),
        (r"targeted|specific", "selectivity_index", "multiply", 1.4),
        (r"controlled.release|sustained", "half_life_hours", "multiply", 1.5),
    ],
    "materials": [
        (r"tensile|strength|strong", "tensile_strength_mpa", "multiply", 1.3),
        (r"stiff|modulus|elastic", "elastic_modulus_gpa", "multiply", 1.2),
        (r"thermal|conduct|heat", "thermal_conductivity_w_mk", "multiply", 1.3),
        (r"hard|wear", "hardness_hv", "multiply", 1.3),
        (r"tough|fracture|impact", "fracture_toughness_mpa_m05", "multiply", 1.3),
        (r"corrosion|rust|oxidation", "corrosion_rate_mm_yr", "multiply", 0.5),
        (r"light|density|weight", "density_g_cm3", "multiply", 0.8),
        (r"composite|fiber|reinforce", "tensile_strength_mpa", "multiply", 1.5),
        (r"graphene|carbon|nanotube", "tensile_strength_mpa", "multiply", 1.6),
        (r"alloy|metal", "tensile_strength_mpa", "multiply", 1.2),
        (r"polymer|plastic", "tensile_strength_mpa", "multiply", 1.1),
        (r"ceramic", "hardness_hv", "multiply", 1.4),
    ],
    "machine_learning": [
        (r"accurate|accuracy|correct", "accuracy_pct", "multiply", 1.05),
        (r"fast|latency|speed|quick", "inference_latency_ms", "multiply", 0.5),
        (r"train|efficient", "training_time_min", "multiply", 0.6),
        (r"memory|lightweight|small", "memory_mb", "multiply", 0.6),
        (r"generaliz|overfit|robust", "generalization_error", "multiply", 0.6),
        (r"f1|balanced|precision.recall", "f1_score", "multiply", 1.08),
        (r"throughput|scale|parallel", "throughput_samples_s", "multiply", 1.5),
        (r"distill|prun|compress", "memory_mb", "multiply", 0.5),
        (r"attention|transformer", "accuracy_pct", "multiply", 1.08),
        (r"quantiz|int8|fp16", "memory_mb", "multiply", 0.4),
    ],
    "computer_architecture": [
        # --- Performance ---
        (r"performance|faster|speed|throughput|ipc", "ipc", "multiply", 1.2),
        (r"clock|frequency|ghz|hz", "clock_frequency_ghz", "multiply", 1.15),
        (r"pipeline|stall|hazard", "pipeline_stalls_per_1000", "multiply", 0.6),
        (r"branch|prediction", "branch_prediction_accuracy_pct", "multiply", 1.05),
        (r"cache|memory|hierarchy", "cache_hit_rate_pct", "multiply", 1.08),
        (r"latency|memory.*access", "memory_latency_ns", "multiply", 0.7),
        (r"out.of.order|superscalar|speculative", "ipc", "multiply", 1.3),
        (r"instruction.*set|isa|risc|cisc", "ipc", "multiply", 1.15),
        (r"parallel|multi.core|many.core|thread", "throughput_gops", "multiply", 1.5),
        (r"vector|simd|neon|avx", "throughput_gops", "multiply", 1.3),

        # --- Power / Efficiency ---
        (r"power|energy|efficien|tdp|watt", "power_consumption_w", "multiply", 0.7),
        (r"power.*efficien|performance.*watt|efficien", "performance_per_watt", "multiply", 1.3),
        (r"low.power|energy.efficient|green", "power_consumption_w", "multiply", 0.5),
        (r"thermal|heat|cool", "power_consumption_w", "multiply", 0.8),

        # --- Area / Size ---
        (r"area|size|die|footprint|small", "area_mm2", "multiply", 0.7),
        (r"compact|miniatur|dense", "area_mm2", "multiply", 0.6),
        (r"transistor|process|node|nm", "area_mm2", "multiply", 0.8),
    ],
}


# =========================================================
# DOMAIN-SPECIFIC SIMULATION FUNCTIONS
# =========================================================
def _simulate_chemistry(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate chemistry metrics from hypothesis-derived parameters."""
    p = params or {}

    # pH: lower is more acidic. Start from H2SO4 baseline (pH ~1.0)
    # Acid strength Ka: higher = stronger acid
    ka_base = 1.0e-2  # H2SO4 first dissociation
    ka = ka_base * p.get("acid_strength_ka", 1.0)
    # pH = -log10(sqrt(Ka * C)) for weak acid approximation
    # For strong acid: pH = -log10(C)
    conc = 1.0  # 1 M concentration
    if ka > 1.0:
        # Strong acid: pH = -log10(C)
        ph = -math.log10(conc)
    else:
        # Weak acid: pH = 0.5 * (pKa - log10(C))
        pka = -math.log10(ka)
        ph = 0.5 * (pka - math.log10(conc))
    ph = max(0.0, min(14.0, ph))

    # Reaction rate: Arrhenius-like
    ea_base = 85.0  # kJ/mol
    ea = ea_base * p.get("activation_energy_kj_mol", 1.0)
    rate_base = 0.05
    rate = rate_base * p.get("reaction_rate_constant", 1.0)
    # Lower activation energy → higher rate
    if p.get("activation_energy_kj_mol", 1.0) < 1.0:
        rate *= (ea_base / max(ea, 1.0)) ** 2

    # Yield
    yield_pct = min(99.0, 78.5 * p.get("yield_percent", 1.0))
    selectivity = min(99.0, 80.0 * p.get("selectivity_percent", 1.0))

    # Catalysis
    tof = 0.8 * p.get("turnover_frequency", 1.0)

    # Thermal stability
    thermal = 250.0 * p.get("thermal_stability_c", 1.0)

    # Solubility
    solubility = 45.0 * p.get("solubility_g_l", 1.0)

    # Purity
    purity = min(99.9, 95.0 * p.get("purity_percent", 1.0))

    results = {
        "ph_value": round(ph, 2),
        "acid_strength_ka": round(ka, 6),
        "reaction_rate_constant": round(rate, 4),
        "yield_percent": round(yield_pct, 2),
        "selectivity_percent": round(selectivity, 2),
        "activation_energy_kj_mol": round(ea, 2),
        "turnover_frequency": round(tof, 3),
        "thermal_stability_c": round(thermal, 1),
        "solubility_g_l": round(solubility, 2),
        "purity_percent": round(purity, 2),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_battery(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate battery metrics from hypothesis-derived parameters."""
    p = params or {}

    # Energy density
    E_base = 240.0
    surface = p.get("surface_area", 1.0)
    diffusion = p.get("diffusion_coefficient", 1.0)
    ionic = p.get("ionic_conductivity", 1.0)
    cap_factor = p.get("capacity_factor", 1.0)
    ed_factor = p.get("energy_density_factor", 1.0)
    anode_cap = p.get("anode_capacity", 1.0)
    cathode_cap = p.get("cathode_capacity", 1.0)

    transport_factor = (math.sqrt(surface) + math.sqrt(diffusion) + math.sqrt(ionic)) / 3.0
    capacity_mult = cap_factor * anode_cap * cathode_cap
    energy_density = E_base * transport_factor * capacity_mult * ed_factor

    # Internal resistance
    R_base = 0.15
    R_mod = p.get("internal_resistance", 1.0)
    cathode_cond = p.get("cathode_conductivity", 1.0)
    anode_cond = p.get("anode_conductivity", 1.0)
    electrolyte_R = p.get("electrolyte_resistance", 1.0)
    charge_transfer = p.get("charge_transfer_resistance", 1.0)
    carrier = p.get("carrier_density", 1.0)

    cond_factor = 1.0 / max(0.1, cathode_cond * anode_cond)
    R_int = R_base * cond_factor * electrolyte_R * charge_transfer * R_mod / max(0.1, carrier)

    # Power density
    P_base = 350.0
    power_mult = (math.sqrt(cathode_cond) + math.sqrt(anode_cond) + math.sqrt(surface)) / 3.0
    power_density = P_base * power_mult / max(0.5, R_int / max(R_base, 0.001))

    # Cycle life
    d_base = 0.025
    protection = p.get("electrode_protection", 1.0)
    mechanical = p.get("mechanical_stability", 1.0)
    cathode_stab = p.get("cathode_stability", 1.0)
    safety = p.get("electrolyte_safety", 1.0)
    d = d_base / (protection * mechanical * cathode_stab * safety)
    cycle_life = int(20.0 / max(d, 0.0001))

    # Temperature rise
    temp_rise = 8.0 * (R_int / R_base) * (1.0 / max(0.5, p.get("thermal_conductivity", 1.0)))

    # Charging efficiency
    charging_efficiency = max(0, (1 - (R_int / max(R_base, 0.001)) * 0.1)) * 100

    results = {
        "energy_density_wh_kg": round(energy_density, 2),
        "cycle_life": cycle_life,
        "charging_efficiency_pct": round(charging_efficiency, 2),
        "degradation_per_cycle_pct": round(d * 100, 4),
        "temperature_rise_c": round(temp_rise, 2),
        "internal_resistance_ohm": round(R_int, 4),
        "power_density_w_kg": round(power_density, 2),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_aerospace(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate aerospace metrics from hypothesis-derived parameters."""
    p = params or {}

    lift = 1.0 * p.get("lift_coefficient", 1.0)
    drag = 0.35 * p.get("drag_coefficient", 1.0)
    ld_ratio = lift / max(drag, 0.01)
    thrust = 500.0 * p.get("thrust_force_n", 1.0)
    speed = 22.0 * p.get("max_speed_mps", 1.0)
    fuel_eff = 1.2 * p.get("fuel_efficiency_km_kg", 1.0)
    stall = 15.0 * p.get("stall_speed_mps", 1.0)
    structural = 1.5 * p.get("structural_margin", 1.0)

    results = {
        "lift_coefficient": round(lift, 4),
        "drag_coefficient": round(drag, 4),
        "lift_to_drag_ratio": round(ld_ratio, 3),
        "thrust_force_n": round(thrust, 1),
        "max_speed_mps": round(speed, 2),
        "fuel_efficiency_km_kg": round(fuel_eff, 3),
        "stall_speed_mps": round(stall, 2),
        "structural_margin": round(structural, 3),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_robotics(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate robotics metrics from hypothesis-derived parameters."""
    p = params or {}

    accuracy = 2.5 * p.get("position_accuracy_mm", 1.0)
    repeatability = 1.2 * p.get("repeatability_mm", 1.0)
    payload = 5.0 * p.get("payload_capacity_kg", 1.0)
    response = 120.0 * p.get("response_time_ms", 1.0)
    efficiency = min(99.0, 72.0 * p.get("energy_efficiency_pct", 1.0))
    tracking = 3.0 * p.get("trajectory_tracking_error_mm", 1.0)
    autonomy = min(1.0, 0.6 * p.get("autonomy_level", 1.0))

    results = {
        "position_accuracy_mm": round(accuracy, 3),
        "repeatability_mm": round(repeatability, 3),
        "payload_capacity_kg": round(payload, 2),
        "response_time_ms": round(response, 1),
        "energy_efficiency_pct": round(efficiency, 2),
        "trajectory_tracking_error_mm": round(tracking, 3),
        "autonomy_level": round(autonomy, 3),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_civil_engineering(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate civil engineering metrics from hypothesis-derived parameters."""
    p = params or {}

    load = 250.0 * p.get("load_capacity_kn", 1.0)
    deflection = 12.0 * p.get("deflection_mm", 1.0)
    safety = 1.5 * p.get("safety_factor", 1.0)
    seismic = min(1.0, 0.7 * p.get("seismic_resistance", 1.0))
    strength = 220.0 * p.get("material_strength_mpa", 1.0)
    life = 50 * p.get("service_life_years", 1.0)
    cost = 1.0 * p.get("construction_cost_index", 1.0)

    results = {
        "load_capacity_kn": round(load, 1),
        "deflection_mm": round(deflection, 2),
        "safety_factor": round(safety, 3),
        "seismic_resistance": round(seismic, 3),
        "material_strength_mpa": round(strength, 1),
        "service_life_years": int(life),
        "construction_cost_index": round(cost, 3),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_medicine(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate medicine metrics from hypothesis-derived parameters."""
    p = params or {}

    efficacy = min(99.0, 65.0 * p.get("drug_efficacy_pct", 1.0))
    bioavailability = min(99.0, 45.0 * p.get("bioavailability_pct", 1.0))
    toxicity = min(1.0, 0.4 * p.get("toxicity_index", 1.0))
    half_life = 6.0 * p.get("half_life_hours", 1.0)
    ic50 = 250.0 * p.get("ic50_nm", 1.0)
    selectivity = 2.0 * p.get("selectivity_index", 1.0)
    side_effects = min(1.0, 0.5 * p.get("side_effect_severity", 1.0))

    results = {
        "drug_efficacy_pct": round(efficacy, 2),
        "bioavailability_pct": round(bioavailability, 2),
        "toxicity_index": round(toxicity, 3),
        "half_life_hours": round(half_life, 2),
        "ic50_nm": round(ic50, 2),
        "selectivity_index": round(selectivity, 3),
        "side_effect_severity": round(side_effects, 3),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_materials(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate materials science metrics from hypothesis-derived parameters."""
    p = params or {}

    tensile = 250.0 * p.get("tensile_strength_mpa", 1.0)
    modulus = 70.0 * p.get("elastic_modulus_gpa", 1.0)
    thermal = 45.0 * p.get("thermal_conductivity_w_mk", 1.0)
    hardness = 120.0 * p.get("hardness_hv", 1.0)
    toughness = 25.0 * p.get("fracture_toughness_mpa_m05", 1.0)
    corrosion = 0.5 * p.get("corrosion_rate_mm_yr", 1.0)
    density = 2.7 * p.get("density_g_cm3", 1.0)

    results = {
        "tensile_strength_mpa": round(tensile, 1),
        "elastic_modulus_gpa": round(modulus, 1),
        "thermal_conductivity_w_mk": round(thermal, 2),
        "hardness_hv": round(hardness, 1),
        "fracture_toughness_mpa_m05": round(toughness, 2),
        "corrosion_rate_mm_yr": round(corrosion, 4),
        "density_g_cm3": round(density, 3),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_machine_learning(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate machine learning metrics from hypothesis-derived parameters."""
    p = params or {}

    accuracy = min(99.0, 84.5 * p.get("accuracy_pct", 1.0))
    latency = 45.0 * p.get("inference_latency_ms", 1.0)
    training = 120.0 * p.get("training_time_min", 1.0)
    memory = 512.0 * p.get("memory_mb", 1.0)
    gen_error = 0.15 * p.get("generalization_error", 1.0)
    f1 = min(1.0, 0.82 * p.get("f1_score", 1.0))
    throughput = 1000.0 * p.get("throughput_samples_s", 1.0)

    results = {
        "accuracy_pct": round(accuracy, 2),
        "inference_latency_ms": round(latency, 2),
        "training_time_min": round(training, 2),
        "memory_mb": round(memory, 1),
        "generalization_error": round(gen_error, 4),
        "f1_score": round(f1, 4),
        "throughput_samples_s": round(throughput, 1),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


def _simulate_computer_architecture(params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Simulate computer architecture metrics from hypothesis-derived parameters."""
    p = params or {}

    # Performance
    ipc = 1.5 * p.get("ipc", 1.0)
    clock = 3.5 * p.get("clock_frequency_ghz", 1.0)
    stalls = 45.0 * p.get("pipeline_stalls_per_1000", 1.0)
    branch_acc = min(99.0, 95.0 * p.get("branch_prediction_accuracy_pct", 1.0))
    cache_hit = min(99.0, 92.0 * p.get("cache_hit_rate_pct", 1.0))
    mem_latency = 80.0 * p.get("memory_latency_ns", 1.0)
    throughput = 50.0 * p.get("throughput_gops", 1.0)

    # Power / Efficiency
    power = 65.0 * p.get("power_consumption_w", 1.0)
    perf_per_watt = 2.5 * p.get("performance_per_watt", 1.0)

    # Area
    area = 150.0 * p.get("area_mm2", 1.0)

    results = {
        "ipc": round(ipc, 3),
        "clock_frequency_ghz": round(clock, 2),
        "power_consumption_w": round(power, 1),
        "performance_per_watt": round(perf_per_watt, 3),
        "area_mm2": round(area, 1),
        "cache_hit_rate_pct": round(cache_hit, 2),
        "branch_prediction_accuracy_pct": round(branch_acc, 2),
        "pipeline_stalls_per_1000": round(stalls, 1),
        "memory_latency_ns": round(mem_latency, 1),
        "throughput_gops": round(throughput, 1),
    }
    if is_baseline:
        results["applied_modifications"] = {}
        results["simulation_type"] = "baseline"
    else:
        results["applied_modifications"] = {k: round(v, 3) for k, v in p.items()}
        results["simulation_type"] = "hypothesis_driven"
    return results


# =========================================================
# DOMAIN SIMULATION DISPATCH
# =========================================================
DOMAIN_SIMULATORS = {
    "chemistry": _simulate_chemistry,
    "battery": _simulate_battery,
    "aerospace": _simulate_aerospace,
    "robotics": _simulate_robotics,
    "civil_engineering": _simulate_civil_engineering,
    "medicine": _simulate_medicine,
    "materials": _simulate_materials,
    "machine_learning": _simulate_machine_learning,
    "computer_architecture": _simulate_computer_architecture,
}


# =========================================================
# PUBLIC API
# =========================================================
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


def get_domain_metrics(domain: str) -> Dict[str, Dict[str, Any]]:
    """Get the metric definitions for a domain."""
    return DOMAIN_METRICS.get(domain, DOMAIN_METRICS.get("general", {}))


def extract_domain_params(domain: str, hypothesis: str) -> Dict[str, float]:
    """Extract domain-specific parameter modifications from a hypothesis."""
    hyp_lower = hypothesis.lower()
    params = {}
    param_map = DOMAIN_PARAMETER_MAP.get(domain, [])

    for keyword, param, mod_type, value in param_map:
        if re.search(keyword, hyp_lower):
            if mod_type == "multiply":
                if param in params:
                    params[param] *= value
                else:
                    params[param] = value
            elif mod_type == "set":
                params[param] = value

    return params


def simulate_domain(domain: str, params: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
    """Run the domain-specific simulation."""
    simulator = DOMAIN_SIMULATORS.get(domain)
    if simulator is None:
        # Fallback: use chemistry simulator for unknown domains
        simulator = _simulate_chemistry
    return simulator(params, is_baseline)


def get_domain_metric_names(domain: str) -> List[str]:
    """Get the list of metric names for a domain."""
    return list(DOMAIN_METRICS.get(domain, {}).keys())


def get_domain_description(domain: str) -> str:
    """Get a human-readable description of the domain."""
    descriptions = {
        "chemistry": "Chemistry — reactions, acidity, catalysis, synthesis",
        "battery": "Battery & Energy Storage — electrochemistry, capacity, cycle life",
        "aerospace": "Aerospace — aerodynamics, propulsion, flight dynamics",
        "robotics": "Robotics — control, precision, autonomy, manipulation",
        "civil_engineering": "Civil Engineering — structures, materials, load-bearing",
        "medicine": "Medicine & Pharmacology — drug efficacy, bioavailability, toxicity",
        "materials": "Materials Science — mechanical, thermal, structural properties",
        "machine_learning": "Machine Learning — model performance, efficiency, accuracy",
        "computer_architecture": "Computer Architecture — CPU design, performance, power efficiency, area",
        "general": "General — domain-agnostic engineering analysis",
    }
    return descriptions.get(domain, descriptions["general"])
