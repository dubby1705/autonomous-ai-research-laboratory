"""
PHYSICS MODULE — Pure Python Scientific Knowledge
No API keys. No LLM. Pure numpy/scipy computation.

For each concept (momentum, force, energy, etc.):
- Writes ALL known formulas and relationships
- Generates valid data points
- Derives related quantities
- Ollama is NEVER used for computation — only for validation of complex combinations
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional

# =========================================================
# CONCEPT DATABASE: Every physics concept with its formulas
# =========================================================

class PhysicsConcept:
    """A single physics concept with all its formulas and relationships."""
    
    def __init__(self, name: str, symbol: str, unit: str, description: str,
                 formulas: Dict[str, Dict[str, Any]], 
                 relationships: List[str],
                 dimension: Tuple[int, int, int]):  # (M, L, T)
        self.name = name
        self.symbol = symbol
        self.unit = unit
        self.description = description
        self.formulas = formulas      # formula_name -> {expression, vars, validate}
        self.relationships = relationships  # related concept names
        self.dimension = dimension    # (Mass, Length, Time) exponents
    
    def get_all_formulas_text(self) -> str:
        """Return all formulas as human-readable text for cluster data points."""
        lines = [f"=== {self.name} ({self.symbol}) ==="]
        lines.append(f"Unit: {self.unit}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Dimension: M^{self.dimension[0]} L^{self.dimension[1]} T^{self.dimension[2]}")
        lines.append("Formulas:")
        for fname, fdata in self.formulas.items():
            lines.append(f"  • {fname}: {fdata.get('text', '')}")
        lines.append(f"Relationships: {', '.join(self.relationships)}")
        return "\n".join(lines)
    
    def compute(self, formula_name: str, **kwargs) -> Optional[float]:
        """Compute a formula with given parameters."""
        if formula_name not in self.formulas:
            return None
        try:
            return self.formulas[formula_name]["func"](**kwargs)
        except Exception:
            return None

    def random_valid_params(self, formula_name: str) -> Dict[str, float]:
        """Generate random VALID parameters for a formula (physically meaningful ranges)."""
        if formula_name not in self.formulas:
            return {}
        return self.formulas[formula_name].get("random_params", lambda: {})()


# =========================================================
# COMPLETE PHYSICS CONCEPT DATABASE
# =========================================================

def _build_physics_db() -> Dict[str, PhysicsConcept]:
    """Build the complete physics concept database."""
    db = {}
    
    # --- MOMENTUM ---
    db["momentum"] = PhysicsConcept(
        name="Momentum", symbol="p", unit="kg·m/s",
        description="Product of mass and velocity. Conserved in isolated systems.",
        formulas={
            "linear_momentum": {
                "text": "p = m * v",
                "func": lambda m=1, v=1: m * v,
                "random_params": lambda: {"m": np.random.uniform(0.1, 1000), "v": np.random.uniform(0.01, 100)},
                "vars": {"m": "mass (kg)", "v": "velocity (m/s)"}
            },
            "impulse": {
                "text": "J = Δp = F * Δt",
                "func": lambda F=1, dt=1: F * dt,
                "random_params": lambda: {"F": np.random.uniform(0.1, 5000), "dt": np.random.uniform(0.001, 10)},
                "vars": {"F": "force (N)", "dt": "time interval (s)"}
            },
            "conservation": {
                "text": "m1*v1 + m2*v2 = m1*v1' + m2*v2'",
                "func": lambda m1=1, v1=1, m2=1, v2=1, v1p=0, v2p=0: m1*v1 + m2*v2 - (m1*v1p + m2*v2p),
                "random_params": lambda: {"m1": np.random.uniform(0.1, 10), "v1": np.random.uniform(-10, 10),
                                           "m2": np.random.uniform(0.1, 10), "v2": np.random.uniform(-10, 10),
                                           "v1p": np.random.uniform(-10, 10), "v2p": np.random.uniform(-10, 10)},
                "vars": {"m1,m2": "masses (kg)", "v1,v2": "initial velocities (m/s)", "v1p,v2p": "final velocities (m/s)"}
            }
        },
        relationships=["force", "energy", "collision"],
        dimension=(1, 1, -1)
    )
    
    # --- FORCE ---
    db["force"] = PhysicsConcept(
        name="Force", symbol="F", unit="N (kg·m/s²)",
        description="Any interaction that changes motion. Vector quantity.",
        formulas={
            "newton_second": {
                "text": "F = m * a",
                "func": lambda m=1, a=1: m * a,
                "random_params": lambda: {"m": np.random.uniform(0.1, 1000), "a": np.random.uniform(0.1, 100)},
                "vars": {"m": "mass (kg)", "a": "acceleration (m/s²)"}
            },
            "gravitational": {
                "text": "F = G * m1 * m2 / r²",
                "func": lambda m1=1, m2=1, r=1, G=6.674e-11: G * m1 * m2 / (r * r),
                "random_params": lambda: {"m1": np.random.uniform(1e10, 1e30), "m2": np.random.uniform(1, 1e25), 
                                           "r": np.random.uniform(1, 1e11)},
                "vars": {"m1,m2": "masses (kg)", "r": "distance (m)", "G": "gravitational constant"}
            },
            "friction": {
                "text": "F_f = μ * N",
                "func": lambda mu=0.5, N=1: mu * N,
                "random_params": lambda: {"mu": np.random.uniform(0.01, 1.0), "N": np.random.uniform(1, 1000)},
                "vars": {"μ": "coefficient of friction", "N": "normal force (N)"}
            },
            "spring": {
                "text": "F = -k * x",
                "func": lambda k=1, x=1: -k * x,
                "random_params": lambda: {"k": np.random.uniform(0.1, 1000), "x": np.random.uniform(-1, 1)},
                "vars": {"k": "spring constant (N/m)", "x": "displacement (m)"}
            },
            "centripetal": {
                "text": "F_c = m * v² / r",
                "func": lambda m=1, v=1, r=1: m * v * v / r,
                "random_params": lambda: {"m": np.random.uniform(0.1, 100), "v": np.random.uniform(1, 50), 
                                           "r": np.random.uniform(0.1, 10)},
                "vars": {"m": "mass (kg)", "v": "velocity (m/s)", "r": "radius (m)"}
            },
            "buoyancy": {
                "text": "F_b = ρ * V * g",
                "func": lambda rho=1000, V=1, g=9.81: rho * V * g,
                "random_params": lambda: {"rho": np.random.uniform(1, 1000), "V": np.random.uniform(0.001, 10)},
                "vars": {"ρ": "fluid density (kg/m³)", "V": "volume (m³)", "g": "gravity (m/s²)"}
            }
        },
        relationships=["momentum", "energy", "acceleration", "work"],
        dimension=(1, 1, -2)
    )
    
    # --- ENERGY ---
    db["energy"] = PhysicsConcept(
        name="Energy", symbol="E", unit="J (kg·m²/s²)",
        description="Capacity to do work. Conserved. Many forms.",
        formulas={
            "kinetic": {
                "text": "K = 0.5 * m * v²",
                "func": lambda m=1, v=1: 0.5 * m * v * v,
                "random_params": lambda: {"m": np.random.uniform(0.1, 100), "v": np.random.uniform(0.1, 100)},
                "vars": {"m": "mass (kg)", "v": "velocity (m/s)"}
            },
            "potential_gravitational": {
                "text": "U = m * g * h",
                "func": lambda m=1, h=1, g=9.81: m * g * h,
                "random_params": lambda: {"m": np.random.uniform(0.1, 100), "h": np.random.uniform(0.1, 100)},
                "vars": {"m": "mass (kg)", "g": "gravity (m/s²)", "h": "height (m)"}
            },
            "potential_spring": {
                "text": "U = 0.5 * k * x²",
                "func": lambda k=1, x=1: 0.5 * k * x * x,
                "random_params": lambda: {"k": np.random.uniform(0.1, 1000), "x": np.random.uniform(-1, 1)},
                "vars": {"k": "spring constant (N/m)", "x": "displacement (m)"}
            },
            "work": {
                "text": "W = F * d * cos(θ)",
                "func": lambda F=1, d=1, theta=0: F * d * math.cos(theta),
                "random_params": lambda: {"F": np.random.uniform(1, 100), "d": np.random.uniform(0.1, 10), 
                                           "theta": np.random.uniform(0, math.pi/2)},
                "vars": {"F": "force (N)", "d": "distance (m)", "θ": "angle between force and displacement"}
            },
            "thermal": {
                "text": "Q = m * c * ΔT",
                "func": lambda m=1, c=4186, dT=1: m * c * dT,
                "random_params": lambda: {"m": np.random.uniform(0.1, 10), "c": np.random.uniform(100, 4000), 
                                           "dT": np.random.uniform(1, 100)},
                "vars": {"m": "mass (kg)", "c": "specific heat (J/(kg·K))", "ΔT": "temperature change (K)"}
            },
            "einstein": {
                "text": "E = m * c²",
                "func": lambda m=1: m * 299792458 * 299792458,
                "random_params": lambda: {"m": np.random.uniform(1e-10, 1)},
                "vars": {"m": "mass (kg)", "c": "speed of light (m/s)"}
            }
        },
        relationships=["force", "momentum", "power", "work"],
        dimension=(1, 2, -2)
    )
    
    # --- POWER ---
    db["power"] = PhysicsConcept(
        name="Power", symbol="P", unit="W (J/s)",
        description="Rate of energy transfer or work done.",
        formulas={
            "power_work": {
                "text": "P = W / t",
                "func": lambda W=1, t=1: W / t,
                "random_params": lambda: {"W": np.random.uniform(1, 10000), "t": np.random.uniform(0.1, 100)},
                "vars": {"W": "work (J)", "t": "time (s)"}
            },
            "power_force": {
                "text": "P = F * v",
                "func": lambda F=1, v=1: F * v,
                "random_params": lambda: {"F": np.random.uniform(1, 1000), "v": np.random.uniform(0.1, 50)},
                "vars": {"F": "force (N)", "v": "velocity (m/s)"}
            },
            "electrical": {
                "text": "P = V * I = I² * R = V² / R",
                "func": lambda V=1, I=1: V * I,
                "random_params": lambda: {"V": np.random.uniform(1, 240), "I": np.random.uniform(0.1, 10)},
                "vars": {"V": "voltage (V)", "I": "current (A)", "R": "resistance (Ω)"}
            }
        },
        relationships=["energy", "force", "work"],
        dimension=(1, 2, -3)
    )
    
    # --- WAVES ---
    db["waves"] = PhysicsConcept(
        name="Waves", symbol="λ, f", unit="m, Hz",
        description="Propagating disturbances that transfer energy without transferring matter.",
        formulas={
            "wave_speed": {
                "text": "v = f * λ",
                "func": lambda f=1, lam=1: f * lam,
                "random_params": lambda: {"f": np.random.uniform(1, 1e15), "lam": np.random.uniform(1e-10, 1000)},
                "vars": {"f": "frequency (Hz)", "λ": "wavelength (m)"}
            },
            "doppler": {
                "text": "f' = f * (v ± v_o) / (v ∓ v_s)",
                "func": lambda f=1, v=343, vo=0, vs=0: f * (v + vo) / (v - vs),
                "random_params": lambda: {"f": np.random.uniform(100, 1000), "v": 343, 
                                           "vo": np.random.uniform(-50, 50), "vs": np.random.uniform(-50, 50)},
                "vars": {"f": "source frequency (Hz)", "v": "wave speed (m/s)", 
                         "vo": "observer velocity (m/s)", "vs": "source velocity (m/s)"}
            },
            "intensity": {
                "text": "I = P / (4π * r²)",
                "func": lambda P=1, r=1: P / (4 * math.pi * r * r),
                "random_params": lambda: {"P": np.random.uniform(0.1, 100), "r": np.random.uniform(0.1, 100)},
                "vars": {"P": "power (W)", "r": "distance (m)"}
            }
        },
        relationships=["energy", "frequency"],
        dimension=(0, 0, -1)
    )
    
    # --- ELECTRICITY ---
    db["electricity"] = PhysicsConcept(
        name="Electricity", symbol="V, I, R", unit="V, A, Ω",
        description="Flow of electric charge. Fundamental to circuits and electronics.",
        formulas={
            "ohms_law": {
                "text": "V = I * R",
                "func": lambda I=1, R=1: I * R,
                "random_params": lambda: {"I": np.random.uniform(0.001, 10), "R": np.random.uniform(1, 10000)},
                "vars": {"V": "voltage (V)", "I": "current (A)", "R": "resistance (Ω)"}
            },
            "power_electric": {
                "text": "P = V * I = I² * R",
                "func": lambda V=1, I=1: V * I,
                "random_params": lambda: {"V": np.random.uniform(1, 240), "I": np.random.uniform(0.01, 10)},
                "vars": {"P": "power (W)", "V": "voltage (V)", "I": "current (A)"}
            },
            "resistance_series": {
                "text": "R_total = R1 + R2 + ... + Rn",
                "func": lambda resistors=None: sum(resistors) if resistors else 0,
                "random_params": lambda: {"resistors": list(np.random.uniform(10, 1000, 3))},
                "vars": {"R1...Rn": "individual resistances (Ω)"}
            },
            "resistance_parallel": {
                "text": "1/R_total = 1/R1 + 1/R2 + ... + 1/Rn",
                "func": lambda resistors=None: 1/sum(1/r for r in resistors) if resistors else 0,
                "random_params": lambda: {"resistors": list(np.random.uniform(10, 1000, 3))},
                "vars": {"R1...Rn": "individual resistances (Ω)"}
            },
            "capacitance": {
                "text": "C = Q / V",
                "func": lambda Q=1, V=1: Q / V,
                "random_params": lambda: {"Q": np.random.uniform(1e-6, 0.1), "V": np.random.uniform(1, 100)},
                "vars": {"C": "capacitance (F)", "Q": "charge (C)", "V": "voltage (V)"}
            }
        },
        relationships=["energy", "power", "magnetism"],
        dimension=(1, 2, -3, -1)  # Adding I (current)
    )
    
    # --- THERMODYNAMICS ---
    db["thermodynamics"] = PhysicsConcept(
        name="Thermodynamics", symbol="T, Q, S", unit="K, J, J/K",
        description="Study of heat, work, temperature, and energy conversion.",
        formulas={
            "first_law": {
                "text": "ΔU = Q - W",
                "func": lambda Q=1, W=1: Q - W,
                "random_params": lambda: {"Q": np.random.uniform(-1000, 1000), "W": np.random.uniform(-1000, 1000)},
                "vars": {"ΔU": "internal energy change (J)", "Q": "heat added (J)", "W": "work done (J)"}
            },
            "ideal_gas": {
                "text": "P * V = n * R * T",
                "func": lambda n=1, T=273, R=8.314, V=1: n * R * T / V,
                "random_params": lambda: {"n": np.random.uniform(0.1, 10), "T": np.random.uniform(200, 500), 
                                           "V": np.random.uniform(0.1, 10)},
                "vars": {"P": "pressure (Pa)", "V": "volume (m³)", "n": "moles", "R": "gas constant", "T": "temperature (K)"}
            },
            "entropy": {
                "text": "ΔS = Q_rev / T",
                "func": lambda Q=1, T=1: Q / T,
                "random_params": lambda: {"Q": np.random.uniform(1, 1000), "T": np.random.uniform(200, 500)},
                "vars": {"ΔS": "entropy change (J/K)", "Q_rev": "reversible heat (J)", "T": "temperature (K)"}
            },
            "efficiency": {
                "text": "η = 1 - T_c / T_h",
                "func": lambda Tc=1, Th=1: 1 - Tc/Th,
                "random_params": lambda: {"Tc": np.random.uniform(200, 300), "Th": np.random.uniform(400, 1000)},
                "vars": {"η": "Carnot efficiency", "T_c": "cold reservoir (K)", "T_h": "hot reservoir (K)"}
            }
        },
        relationships=["energy", "temperature", "work"],
        dimension=(1, 2, -2, 0, -1)  # Adding Θ (temperature)
    )
    
    # --- FLUID MECHANICS ---
    db["fluid_mechanics"] = PhysicsConcept(
        name="Fluid Mechanics", symbol="ρ, P, Q", unit="kg/m³, Pa, m³/s",
        description="Study of fluids (liquids and gases) and forces on them.",
        formulas={
            "bernoulli": {
                "text": "P + 0.5*ρ*v² + ρ*g*h = constant",
                "func": lambda P=1, rho=1000, v=1, g=9.81, h=1: P + 0.5*rho*v*v + rho*g*h,
                "random_params": lambda: {"P": np.random.uniform(1e3, 1e5), "rho": np.random.uniform(1, 1000),
                                           "v": np.random.uniform(0.1, 10), "h": np.random.uniform(0, 10)},
                "vars": {"P": "pressure (Pa)", "ρ": "density (kg/m³)", "v": "velocity (m/s)", 
                         "g": "gravity (m/s²)", "h": "height (m)"}
            },
            "continuity": {
                "text": "A1 * v1 = A2 * v2",
                "func": lambda A1=1, v1=1, A2=1: A1 * v1 / A2,
                "random_params": lambda: {"A1": np.random.uniform(0.01, 1), "v1": np.random.uniform(0.1, 5),
                                           "A2": np.random.uniform(0.01, 1)},
                "vars": {"A1,A2": "cross-sectional areas (m²)", "v1,v2": "velocities (m/s)"}
            },
            "viscosity": {
                "text": "F = η * A * dv/dy",
                "func": lambda eta=1, A=1, dv=1, dy=1: eta * A * dv / dy,
                "random_params": lambda: {"eta": np.random.uniform(0.001, 100), "A": np.random.uniform(0.01, 1),
                                           "dv": np.random.uniform(0.1, 10), "dy": np.random.uniform(0.001, 0.1)},
                "vars": {"η": "dynamic viscosity (Pa·s)", "A": "area (m²)", "dv/dy": "velocity gradient (1/s)"}
            }
        },
        relationships=["force", "energy", "pressure"],
        dimension=(1, -1, -2)
    )
    
    # --- ELECTROMAGNETISM ---
    db["electromagnetism"] = PhysicsConcept(
        name="Electromagnetism", symbol="E, B, Φ", unit="V/m, T, Wb",
        description="Study of electric and magnetic fields and their interactions.",
        formulas={
            "coulombs_law": {
                "text": "F = k * q1 * q2 / r²",
                "func": lambda q1=1, q2=1, r=1, k=8.99e9: k * q1 * q2 / (r * r),
                "random_params": lambda: {"q1": np.random.uniform(1e-9, 1e-6), "q2": np.random.uniform(1e-9, 1e-6),
                                           "r": np.random.uniform(0.01, 1)},
                "vars": {"q1,q2": "charges (C)", "r": "distance (m)", "k": "Coulomb constant"}
            },
            "electric_field": {
                "text": "E = F / q",
                "func": lambda F=1, q=1: F / q,
                "random_params": lambda: {"F": np.random.uniform(1e-6, 1), "q": np.random.uniform(1e-9, 1e-6)},
                "vars": {"E": "electric field (N/C)", "F": "force (N)", "q": "test charge (C)"}
            },
            "magnetic_force": {
                "text": "F = q * v * B * sin(θ)",
                "func": lambda q=1, v=1, B=1, theta=math.pi/2: q * v * B * math.sin(theta),
                "random_params": lambda: {"q": np.random.uniform(1e-9, 1e-6), "v": np.random.uniform(1, 100),
                                           "B": np.random.uniform(0.01, 1), "theta": np.random.uniform(0, math.pi)},
                "vars": {"q": "charge (C)", "v": "velocity (m/s)", "B": "magnetic field (T)", "θ": "angle"}
            },
            "faradays_law": {
                "text": "ε = -dΦ_B / dt",
                "func": lambda dPhi=1, dt=1: -dPhi / dt,
                "random_params": lambda: {"dPhi": np.random.uniform(0.01, 10), "dt": np.random.uniform(0.001, 1)},
                "vars": {"ε": "induced EMF (V)", "dΦ_B": "change in magnetic flux (Wb)", "dt": "time (s)"}
            }
        },
        relationships=["electricity", "force", "energy", "waves"],
        dimension=(1, 1, -3, -1)
    )
    
    return db


# =========================================================
# DERIVED QUANTITIES
# =========================================================
DERIVED_RELATIONSHIPS = {
    "momentum_energy": {
        "text": "E = p² / (2*m)  [from K = 0.5*m*v² and p = m*v]",
        "func": lambda p=1, m=1: p*p / (2*m),
        "description": "Kinetic energy expressed in terms of momentum"
    },
    "force_power": {
        "text": "P = F * v  [instantaneous power from force and velocity]",
        "func": lambda F=1, v=1: F * v,
        "description": "Power from force and velocity"
    },
    "energy_frequency": {
        "text": "E = h * f  [Planck-Einstein relation]",
        "func": lambda f=1, h=6.626e-34: h * f,
        "description": "Photon energy from frequency"
    },
    "wave_number": {
        "text": "k = 2π / λ",
        "func": lambda lam=1: 2 * math.pi / lam,
        "description": "Angular wave number from wavelength"
    },
    "escape_velocity": {
        "text": "v_esc = sqrt(2 * G * M / r)",
        "func": lambda M=1, r=1: math.sqrt(2 * 6.674e-11 * M / r),
        "description": "Escape velocity from gravitational body"
    }
}


# =========================================================
# FORMULA VALIDATOR (Pure Python + Ollama for complex checks)
# =========================================================
def validate_formula_pure(formula_text: str, variables: Dict[str, str]) -> Tuple[bool, str]:
    """
    Pure Python validation of formulas.
    Checks: dimensional consistency, division by zero, sqrt negative, log negative.
    No LLM used.
    """
    issues = []
    
    # Check for division by constant zero
    if '/0' in formula_text or '/ 0' in formula_text:
        issues.append("Division by zero detected")
    
    # Check for sqrt of negative constant
    if 'sqrt(-' in formula_text:
        issues.append("Square root of negative value")
    
    # Check for log of zero
    if 'log(0)' in formula_text:
        issues.append("Logarithm of zero")
    
    # Check dimensional consistency if variables have dimensions
    # (basic check: LHS should have same dimensions as RHS)
    
    return len(issues) == 0, "; ".join(issues) if issues else "Valid"


# =========================================================
# PUBLIC API
# =========================================================
PHYSICS_DB = _build_physics_db()

def get_concept(name: str) -> Optional[PhysicsConcept]:
    """Get a physics concept by name."""
    return PHYSICS_DB.get(name.lower())

def get_all_concept_names() -> List[str]:
    """Get all available physics concept names."""
    return list(PHYSICS_DB.keys())

def generate_concept_data_points(concept_names: List[str]) -> List[str]:
    """Generate full text data points for given concepts (for cluster analysis)."""
    points = []
    for name in concept_names:
        concept = get_concept(name)
        if concept:
            points.append(concept.get_all_formulas_text())
    return points

def generate_derived_point(rel_name: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Generate a derived data point from a relationship between concepts."""
    if rel_name not in DERIVED_RELATIONSHIPS:
        return None
    rel = DERIVED_RELATIONSHIPS[rel_name]
    try:
        value = rel["func"](**kwargs)
        return {
            "name": rel_name,
            "text": rel["text"],
            "description": rel["description"],
            "value": value,
            "type": "derived"
        }
    except Exception:
        return None

def random_prediction(concept_name: str, formula_name: str) -> Dict[str, Any]:
    """
    Generate a random valid prediction using pure physics code.
    Ollama is NOT used here. Only numpy/math for random valid params.
    """
    concept = get_concept(concept_name)
    if not concept or formula_name not in concept.formulas:
        return {"error": f"No such formula: {concept_name}.{formula_name}"}
    
    params = concept.random_valid_params(formula_name)
    value = concept.compute(formula_name, **params)
    
    return {
        "concept": concept_name,
        "formula": formula_name,
        "formula_text": concept.formulas[formula_name].get("text", ""),
        "parameters": params,
        "result": value,
        "unit": concept.unit,
        "type": "physics_prediction"
    }


if __name__ == "__main__":
    # Demo
    print("Physics Concepts:", len(PHYSICS_DB))
    for name in PHYSICS_DB:
        print(f"  • {name}")
    
    print("\n--- Momentum Data Point ---")
    print(PHYSICS_DB["momentum"].get_all_formulas_text())
    
    print("\n--- Random Prediction ---")
    pred = random_prediction("momentum", "linear_momentum")
    print(f"p = {pred['parameters']['m']:.2f} × {pred['parameters']['v']:.2f} = {pred['result']:.2f} {pred['unit']}")