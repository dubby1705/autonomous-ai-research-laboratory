"""
CHEMISTRY MODULE — Pure Python Chemical Knowledge
No API keys. No LLM. Pure numpy/scipy computation.

Concepts: reaction kinetics, thermodynamics, electrochemistry, acid-base, etc.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class ChemistryConcept:
    def __init__(self, name: str, symbol: str, unit: str, description: str,
                 formulas: Dict[str, Dict[str, Any]], relationships: List[str]):
        self.name = name
        self.symbol = symbol
        self.unit = unit
        self.description = description
        self.formulas = formulas
        self.relationships = relationships
    
    def get_all_formulas_text(self) -> str:
        lines = [f"=== {self.name} ({self.symbol}) ==="]
        lines.append(f"Unit: {self.unit}")
        lines.append(f"Description: {self.description}")
        lines.append("Formulas:")
        for fname, fdata in self.formulas.items():
            lines.append(f"  • {fname}: {fdata.get('text', '')}")
        lines.append(f"Relationships: {', '.join(self.relationships)}")
        return "\n".join(lines)
    
    def compute(self, formula_name: str, **kwargs) -> Optional[float]:
        if formula_name not in self.formulas:
            return None
        try:
            return self.formulas[formula_name]["func"](**kwargs)
        except Exception:
            return None
    
    def random_valid_params(self, formula_name: str) -> Dict[str, float]:
        if formula_name not in self.formulas:
            return {}
        return self.formulas[formula_name].get("random_params", lambda: {})()


def _build_chemistry_db() -> Dict[str, ChemistryConcept]:
    db = {}
    
    # --- REACTION RATES ---
    db["reaction_kinetics"] = ChemistryConcept(
        name="Reaction Kinetics", symbol="k, r", unit="mol/(L·s)",
        description="Study of reaction rates and the factors affecting them.",
        formulas={
            "rate_law": {
                "text": "r = k * [A]^m * [B]^n",
                "func": lambda k=1, A=1, B=1, m=1, n=1: k * (A**m) * (B**n),
                "random_params": lambda: {"k": np.random.uniform(0.001, 10), "A": np.random.uniform(0.1, 2),
                                           "B": np.random.uniform(0.1, 2), "m": np.random.choice([0,1,2]),
                                           "n": np.random.choice([0,1,2])},
                "vars": {"k": "rate constant", "[A],[B]": "concentrations (M)", "m,n": "reaction orders"}
            },
            "arrhenius": {
                "text": "k = A * exp(-Ea / (R * T))",
                "func": lambda A=1, Ea=50000, R=8.314, T=298: A * math.exp(-Ea / (R * T)),
                "random_params": lambda: {"A": np.random.uniform(1e8, 1e12), "Ea": np.random.uniform(20000, 100000),
                                           "T": np.random.uniform(250, 500)},
                "vars": {"A": "frequency factor", "Ea": "activation energy (J/mol)", "R": "gas constant", "T": "temperature (K)"}
            },
            "half_life_first": {
                "text": "t_½ = ln(2) / k",
                "func": lambda k=1: math.log(2) / k,
                "random_params": lambda: {"k": np.random.uniform(0.001, 10)},
                "vars": {"t_½": "half-life (s)", "k": "rate constant (1/s)"}
            }
        },
        relationships=["thermodynamics", "equilibrium", "catalysis"]
    )
    
    # --- EQUILIBRIUM ---
    db["equilibrium"] = ChemistryConcept(
        name="Chemical Equilibrium", symbol="K_eq", unit="dimensionless",
        description="State where forward and reverse reaction rates are equal.",
        formulas={
            "equilibrium_constant": {
                "text": "K_eq = [C]^c * [D]^d / ([A]^a * [B]^b)",
                "func": lambda C=1, D=1, A=1, B=1, c=1, d=1, a=1, b=1: (C**c * D**d) / (A**a * B**b),
                "random_params": lambda: {"C": np.random.uniform(0.1, 2), "D": np.random.uniform(0.1, 2),
                                           "A": np.random.uniform(0.1, 2), "B": np.random.uniform(0.1, 2)},
                "vars": {"[C],[D]": "product concentrations", "[A],[B]": "reactant concentrations"}
            },
            "gibbs_equilibrium": {
                "text": "ΔG° = -R * T * ln(K_eq)",
                "func": lambda R=8.314, T=298, K=1: -R * T * math.log(K),
                "random_params": lambda: {"T": np.random.uniform(250, 500), "K": np.random.uniform(0.001, 1000)},
                "vars": {"ΔG°": "standard Gibbs free energy (J/mol)", "K": "equilibrium constant"}
            },
            "le_chatelier": {
                "text": "System shifts to counteract applied stress (qualitative)",
                "func": lambda: 0,
                "random_params": lambda: {},
                "vars": {}
            }
        },
        relationships=["reaction_kinetics", "thermodynamics", "acid_base"]
    )
    
    # --- THERMODYNAMICS ---
    db["thermodynamics"] = ChemistryConcept(
        name="Chemical Thermodynamics", symbol="ΔG, ΔH, ΔS", unit="J/mol",
        description="Energy changes in chemical reactions.",
        formulas={
            "gibbs_free": {
                "text": "ΔG = ΔH - T * ΔS",
                "func": lambda dH=1, T=298, dS=1: dH - T * dS,
                "random_params": lambda: {"dH": np.random.uniform(-200000, 200000), "T": np.random.uniform(200, 500),
                                           "dS": np.random.uniform(-500, 500)},
                "vars": {"ΔG": "Gibbs free energy (J/mol)", "ΔH": "enthalpy (J/mol)", "T": "temperature (K)", "ΔS": "entropy (J/(mol·K))"}
            },
            "enthalpy": {
                "text": "ΔH = Σ(n*ΔH_f_products) - Σ(n*ΔH_f_reactants)",
                "func": lambda prod=1, react=1: prod - react,
                "random_params": lambda: {"prod": np.random.uniform(-500, 0), "react": np.random.uniform(-500, 0)},
                "vars": {"ΔH": "reaction enthalpy"}
            },
            "entropy_change": {
                "text": "ΔS = Σ(n*S_products) - Σ(n*S_reactants)",
                "func": lambda prod=1, react=1: prod - react,
                "random_params": lambda: {"prod": np.random.uniform(100, 500), "react": np.random.uniform(100, 500)},
                "vars": {"ΔS": "entropy change"}
            }
        },
        relationships=["equilibrium", "reaction_kinetics", "electrochemistry"]
    )
    
    # --- ELECTROCHEMISTRY ---
    db["electrochemistry"] = ChemistryConcept(
        name="Electrochemistry", symbol="E, V", unit="V",
        description="Chemical reactions involving electron transfer.",
        formulas={
            "nernst": {
                "text": "E = E° - (R*T)/(n*F) * ln(Q)",
                "func": lambda E0=1, R=8.314, T=298, n=1, F=96485, Q=1: E0 - (R*T)/(n*F) * math.log(Q),
                "random_params": lambda: {"E0": np.random.uniform(-2, 2), "T": np.random.uniform(250, 350),
                                           "n": np.random.choice([1,2,3]), "Q": np.random.uniform(0.001, 100)},
                "vars": {"E": "cell potential (V)", "E°": "standard potential (V)", "n": "electrons transferred", "Q": "reaction quotient"}
            },
            "faraday": {
                "text": "m = (Q * M) / (n * F)",
                "func": lambda Q=1, M=1, n=1, F=96485: Q * M / (n * F),
                "random_params": lambda: {"Q": np.random.uniform(100, 10000), "M": np.random.uniform(10, 200),
                                           "n": np.random.choice([1,2,3])},
                "vars": {"m": "mass deposited (g)", "Q": "charge (C)", "M": "molar mass (g/mol)", "n": "electrons"}
            },
            "cell_potential": {
                "text": "E°_cell = E°_cathode - E°_anode",
                "func": lambda Ec=1, Ea=1: Ec - Ea,
                "random_params": lambda: {"Ec": np.random.uniform(0, 3), "Ea": np.random.uniform(-3, 0)},
                "vars": {"E°_cell": "standard cell potential (V)"}
            }
        },
        relationships=["thermodynamics", "equilibrium", "battery"]
    )
    
    # --- ACID-BASE ---
    db["acid_base"] = ChemistryConcept(
        name="Acid-Base Chemistry", symbol="pH, pKa", unit="dimensionless",
        description="Proton transfer reactions and solution acidity.",
        formulas={
            "ph": {
                "text": "pH = -log10([H+])",
                "func": lambda H=1e-7: -math.log10(H),
                "random_params": lambda: {"H": 10**np.random.uniform(-14, 0)},
                "vars": {"pH": "acidity", "[H+]": "hydrogen ion concentration (M)"}
            },
            "henderson": {
                "text": "pH = pKa + log10([A-]/[HA])",
                "func": lambda pKa=4.76, A=1, HA=1: pKa + math.log10(A/HA),
                "random_params": lambda: {"pKa": np.random.uniform(0, 14), "A": np.random.uniform(0.01, 1),
                                           "HA": np.random.uniform(0.01, 1)},
                "vars": {"pKa": "acid dissociation constant", "[A-]": "conjugate base (M)", "[HA]": "acid (M)"}
            },
            "kw": {
                "text": "Kw = [H+][OH-] = 1e-14 at 25°C",
                "func": lambda H=1e-7: 1e-14 / H,
                "random_params": lambda: {"H": 10**np.random.uniform(-14, 0)},
                "vars": {"Kw": "water dissociation constant", "[OH-]": "hydroxide concentration (M)"}
            }
        },
        relationships=["equilibrium", "buffer", "titration"]
    )
    
    # --- GAS LAWS ---
    db["gas_laws"] = ChemistryConcept(
        name="Gas Laws", symbol="P, V, T", unit="atm, L, K",
        description="Behavior of gases under various conditions.",
        formulas={
            "ideal_gas": {
                "text": "P * V = n * R * T",
                "func": lambda n=1, R=0.08206, T=273, V=1: n * R * T / V,
                "random_params": lambda: {"n": np.random.uniform(0.1, 5), "T": np.random.uniform(200, 500),
                                           "V": np.random.uniform(0.1, 50)},
                "vars": {"P": "pressure (atm)", "V": "volume (L)", "n": "moles", "T": "temperature (K)"}
            },
            "boyle": {
                "text": "P1 * V1 = P2 * V2",
                "func": lambda P1=1, V1=1, V2=1: P1 * V1 / V2,
                "random_params": lambda: {"P1": np.random.uniform(0.5, 5), "V1": np.random.uniform(1, 20),
                                           "V2": np.random.uniform(1, 20)},
                "vars": {"P1,V1": "initial pressure/volume", "P2,V2": "final pressure/volume"}
            },
            "charles": {
                "text": "V1/T1 = V2/T2",
                "func": lambda V1=1, T1=273, T2=1: V1 * T2 / T1,
                "random_params": lambda: {"V1": np.random.uniform(1, 20), "T1": np.random.uniform(200, 400),
                                           "T2": np.random.uniform(200, 400)},
                "vars": {"V1,T1": "initial volume/temperature", "V2,T2": "final volume/temperature"}
            }
        },
        relationships=["thermodynamics", "kinetic_theory", "density"]
    )
    
    return db

CHEMISTRY_DB = _build_chemistry_db()

def get_concept(name: str) -> Optional[ChemistryConcept]:
    return CHEMISTRY_DB.get(name.lower())

def get_all_concept_names() -> List[str]:
    return list(CHEMISTRY_DB.keys())

def generate_concept_data_points(concept_names: List[str]) -> List[str]:
    points = []
    for name in concept_names:
        concept = get_concept(name)
        if concept:
            points.append(concept.get_all_formulas_text())
    return points

def random_prediction(concept_name: str, formula_name: str) -> Dict[str, Any]:
    concept = get_concept(concept_name)
    if not concept or formula_name not in concept.formulas:
        return {"error": f"No such formula: {concept_name}.{formula_name}"}
    params = concept.random_valid_params(formula_name)
    value = concept.compute(formula_name, **params)
    return {
        "concept": concept_name, "formula": formula_name,
        "formula_text": concept.formulas[formula_name].get("text", ""),
        "parameters": params, "result": value,
        "unit": concept.unit, "type": "chemistry_prediction"
    }

if __name__ == "__main__":
    print("Chemistry Concepts:", len(CHEMISTRY_DB))
    for name in CHEMISTRY_DB:
        print(f"  • {name}")
    print("\n--- Nernst Equation ---")
    print(CHEMISTRY_DB["electrochemistry"].get_all_formulas_text())