"""
RESEARCH MATHEMATICS ENGINE
Derives equations, checks dimensional consistency, verifies mathematical constraints,
estimates complexity, and rejects impossible formulations.

Core capabilities:
- Equation derivation from natural language hypotheses
- Dimensional analysis (MLT: Mass, Length, Time)
- Mathematical constraint verification
- Complexity estimation (Big-O)
- Physical plausibility checking
- Unit consistency validation
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class DerivedEquation(BaseModel):
    equation_id: str = Field(description="Unique identifier for this equation.")
    name: str = Field(description="Short descriptive name for the equation.")
    equation_latex: str = Field(description="The equation in LaTeX notation.")
    equation_text: str = Field(description="The equation in plain text (e.g., 'F = m * a').")
    
    # Variables
    variables: Dict[str, str] = Field(description="Map of variable name -> description/units.")
    
    # Dimensional Analysis
    left_hand_units: str = Field(description="Dimensional units of the left-hand side (MLT format).")
    right_hand_units: str = Field(description="Dimensional units of the right-hand side (MLT format).")
    dimensionally_consistent: bool = Field(description="Whether LHS and RHS units match.")
    
    # Derivation
    derived_from_hypothesis: str = Field(description="The hypothesis this equation was derived from.")
    derivation_logic: str = Field(description="Step-by-step reasoning for how this equation was derived.")
    
    # Constraints
    domain_constraints: List[str] = Field(description="Constraints on when this equation holds (e.g., 'x > 0', 'Re < 2300').")
    boundary_conditions: List[str] = Field(description="Boundary conditions where the equation is valid.")
    
    # Complexity
    computational_complexity: str = Field(description="Big-O complexity to evaluate this equation.")
    num_operations: int = Field(description="Estimated number of floating-point operations per evaluation.")

class EquationValidation(BaseModel):
    equation_id: str = Field(description="Matching the derived equation.")
    is_physically_plausible: bool = Field(description="Whether the equation describes a physically possible relationship.")
    is_mathematically_consistent: bool = Field(description="Whether the math is internally consistent (no division by zero, etc.).")
    potential_issues: List[str] = Field(description="Mathematical issues found.")
    suggested_fixes: List[str] = Field(description="Suggested corrections for any issues.")
    plausibility_score: float = Field(description="Overall score 0.0-1.0 for how plausible this equation is.")

# =========================================================
# DIMENSIONAL ANALYSIS (MLT System)
# =========================================================

# Base dimensions: M (Mass), L (Length), T (Time), I (Current), Θ (Temperature), N (Amount), J (Luminosity)
# We use MLT as the primary set, extended with I and Θ where needed.

DIMENSION_SYMBOLS = {
    "M": "Mass",
    "L": "Length", 
    "T": "Time",
    "I": "Electric Current",
    "TH": "Temperature"
}

# Common physical quantities and their MLT dimensions
# Format: "quantity_name": (M, L, T, I, TH)
PHYSICAL_DIMENSIONS: Dict[str, Tuple[int, int, int, int, int]] = {
    "mass": (1, 0, 0, 0, 0),
    "length": (0, 1, 0, 0, 0),
    "distance": (0, 1, 0, 0, 0),
    "position": (0, 1, 0, 0, 0),
    "radius": (0, 1, 0, 0, 0),
    "area": (0, 2, 0, 0, 0),
    "volume": (0, 3, 0, 0, 0),
    "time": (0, 0, 1, 0, 0),
    "velocity": (0, 1, -1, 0, 0),
    "speed": (0, 1, -1, 0, 0),
    "acceleration": (0, 1, -2, 0, 0),
    "force": (1, 1, -2, 0, 0),
    "energy": (1, 2, -2, 0, 0),
    "work": (1, 2, -2, 0, 0),
    "power": (1, 2, -3, 0, 0),
    "pressure": (1, -1, -2, 0, 0),
    "density": (1, -3, 0, 0, 0),
    "frequency": (0, 0, -1, 0, 0),
    "wavelength": (0, 1, 0, 0, 0),
    "voltage": (1, 2, -3, -1, 0),
    "current": (0, 0, -1, 1, 0),
    "resistance": (1, 2, -3, -2, 0),
    "capacitance": (-1, -2, 4, 2, 0),
    "inductance": (1, 2, -2, -2, 0),
    "temperature": (0, 0, 0, 0, 1),
    "entropy": (1, 2, -2, 0, -1),
    "specific_heat": (0, 2, -2, 0, -1),
    "viscosity": (1, -1, -1, 0, 0),
    "diffusion_coefficient": (0, 2, -1, 0, 0),
    "thermal_conductivity": (1, 1, -3, 0, -1),
    "charge": (0, 0, 1, 1, 0),
    "magnetic_field": (1, 0, -2, -1, 0),
    "dimensionless": (0, 0, 0, 0, 0),
    "ratio": (0, 0, 0, 0, 0),
    "efficiency": (0, 0, 0, 0, 0),
    "probability": (0, 0, 0, 0, 0),
    "count": (0, 0, 0, 0, 0),
    "number": (0, 0, 0, 0, 0),
    "strain": (0, 0, 0, 0, 0),
    "refractive_index": (0, 0, 0, 0, 0),
    "poisson_ratio": (0, 0, 0, 0, 0),
    "coefficient": (0, 0, 0, 0, 0),
    "constant": (0, 0, 0, 0, 0),
    "factor": (0, 0, 0, 0, 0),
    "index": (0, 0, 0, 0, 0),
    # Learning rate specific
    "learning_rate": (0, 0, -1, 0, 0),  # per time step
    "accuracy": (0, 0, 0, 0, 0),
    "loss": (0, 0, 0, 0, 0),
    "gradient": (0, 0, 0, 0, 0),
    "parameter": (0, 0, 0, 0, 0),
    "weight": (0, 0, 0, 0, 0),
    "bias": (0, 0, 0, 0, 0),
    "epoch": (0, 0, 1, 0, 0),
    "batch_size": (0, 0, 0, 0, 0),
    "dataset_size": (0, 0, 0, 0, 0),
}

def parse_dimensional_string(dim_str: str) -> Tuple[int, int, int, int, int]:
    """
    Parses an MLT dimensional string like "M^1 L^2 T^-2" into a tuple (M, L, T, I, TH).
    """
    dims = [0, 0, 0, 0, 0]
    labels = ["M", "L", "T", "I", "TH"]
    
    if not dim_str or dim_str == "dimensionless":
        return tuple(dims)
    
    # Match patterns like "M^1", "L^-2", "T^3", or just "M", "L", etc.
    pattern = r'([A-Za-z]+)\^?(-?\d+)?'
    for match in re.finditer(pattern, dim_str):
        symbol = match.group(1)
        exponent = int(match.group(2)) if match.group(2) else 1
        if symbol in labels:
            idx = labels.index(symbol)
            dims[idx] = exponent
    
    return tuple(dims)


def dimensions_match(dims1: Tuple[int, int, int, int, int], 
                     dims2: Tuple[int, int, int, int, int]) -> bool:
    """Check if two MLT dimension tuples are identical."""
    return dims1 == dims2


def format_dimensions(dims: Tuple[int, int, int, int, int]) -> str:
    """Format dimension tuple as human-readable string."""
    labels = ["M", "L", "T", "I", "TH"]
    parts = []
    for i, d in enumerate(dims):
        if d != 0:
            if d == 1:
                parts.append(labels[i])
            else:
                parts.append(f"{labels[i]}^{d}")
    return " · ".join(parts) if parts else "dimensionless"


def get_dimensions_for_variable(var_name: str) -> Tuple[int, int, int, int, int]:
    """
    Look up or infer the dimensions of a variable based on its name.
    Falls back to dimensionless if unknown.
    """
    var_lower = var_name.lower().strip()
    
    # Direct lookup
    if var_lower in PHYSICAL_DIMENSIONS:
        return PHYSICAL_DIMENSIONS[var_lower]
    
    # Check common prefixes/suffixes
    for key, dims in PHYSICAL_DIMENSIONS.items():
        if key in var_lower or var_lower in key:
            return dims
    
    # Check by common variable names
    var_symbols = {
        'm': (1, 0, 0, 0, 0),    # mass
        't': (0, 0, 1, 0, 0),    # time
        'x': (0, 1, 0, 0, 0),    # position
        'y': (0, 1, 0, 0, 0),    # position
        'z': (0, 1, 0, 0, 0),    # position
        'r': (0, 1, 0, 0, 0),    # radius
        'v': (0, 1, -1, 0, 0),   # velocity
        'u': (0, 1, -1, 0, 0),   # velocity
        'a': (0, 1, -2, 0, 0),   # acceleration
        'f': (1, 1, -2, 0, 0),   # force
        'e': (1, 2, -2, 0, 0),   # energy
        'p': (1, -1, -2, 0, 0),  # pressure (or power)
        'w': (1, 2, -2, 0, 0),   # work
        'i': (0, 0, -1, 1, 0),   # current
        'q': (0, 0, 1, 1, 0),    # charge
        'c': (0, 0, 0, 0, 0),    # constant
        'k': (0, 0, 0, 0, 0),    # constant
        'g': (0, 1, -2, 0, 0),   # acceleration due to gravity
        'h': (1, 2, -1, 0, 0),   # Planck constant / height
        'd': (0, 1, 0, 0, 0),    # distance
        's': (0, 0, 0, 0, 0),    # usually dimensionless or entropy
    }
    
    if len(var_lower) == 1 and var_lower in var_symbols:
        return var_symbols[var_lower]
    
    return (0, 0, 0, 0, 0)  # default dimensionless


# =========================================================
# COMPLEXITY ANALYSIS
# =========================================================

def estimate_complexity(equation_text: str, num_vars: int) -> Dict[str, Any]:
    """
    Estimates computational complexity of evaluating an equation.
    Uses heuristic: sum over operations in the expression.
    """
    # Count operations
    num_adds = equation_text.count('+') + equation_text.count('-')
    num_muls = equation_text.count('*') + equation_text.count('·')
    num_divs = equation_text.count('/')
    num_pows = equation_text.count('^') + equation_text.count('**')
    num_funcs = sum(equation_text.count(func) for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs'])
    
    total_ops = num_adds + num_muls + num_divs + num_pows + num_funcs
    
    # Determine Big-O
    if num_vars <= 1:
        complexity = "O(1)" if total_ops <= 10 else "O(1) with large constant"
    elif num_vars <= 3:
        complexity = "O(n)" if total_ops <= 20 else "O(n)"
    else:
        complexity = "O(n²)" if num_pows > 2 else f"O(n^{min(num_vars, 3)})"
    
    return {
        "big_o": complexity,
        "total_operations": total_ops,
        "num_additions": num_adds,
        "num_multiplications": num_muls,
        "num_divisions": num_divs,
        "num_powers": num_pows,
        "num_function_calls": num_funcs,
        "num_variables": num_vars
    }


def check_numerical_stability(equation_text: str) -> List[str]:
    """
    Checks for potential numerical stability issues in an equation.
    """
    issues = []
    
    # Check for division by small numbers
    if '/' in equation_text:
        issues.append("Contains division — check for denominator approaching zero.")
    
    # Check for exponentials that could overflow
    if 'exp' in equation_text or '^' in equation_text:
        issues.append("Contains exponentiation — check for overflow with large exponents.")
    
    # Check for subtraction that could cause catastrophic cancellation
    # Simple heuristic: look for " - " pattern
    if ' - ' in equation_text or ')-' in equation_text:
        issues.append("Contains subtraction — potential catastrophic cancellation.")
    
    # Check for log of negative
    if 'log' in equation_text or 'ln' in equation_text:
        issues.append("Contains logarithm — ensure argument is strictly positive.")
    
    # Check for sqrt of negative
    if 'sqrt' in equation_text:
        issues.append("Contains square root — ensure argument is non-negative.")
    
    return issues


# =========================================================
# EQUATION DERIVATION (LLM-based)
# =========================================================

def derive_equation(client: Groq, hypothesis: str, 
                    known_variables: Optional[List[str]] = None) -> Optional[DerivedEquation]:
    """
    Uses the LLM to derive a mathematical equation from a hypothesis.
    Also validates dimensional consistency.
    """
    system_prompt = (
        "You are a Mathematical Physicist. Your job is to derive rigorous mathematical equations "
        "from hypothesis statements. You MUST follow these rules:\n"
        "1. Derive at least one equation that captures the core relationship\n"
        "2. Use proper LaTeX notation\n"
        "3. Identify every variable and its physical dimensions in MLT format (M=Mass, L=Length, T=Time)\n"
        "4. Check dimensional consistency: LHS units must equal RHS units\n"
        "5. List all domain constraints and boundary conditions\n"
        "6. Estimate computational complexity\n\n"
        "You MUST respond with a valid JSON object matching this schema:\n"
        "{\n"
        '  "equation_id": "EQ-001",\n'
        '  "name": "Descriptive name",\n'
        '  "equation_latex": "F = m \\\\cdot a",\n'
        '  "equation_text": "F = m * a",\n'
        '  "variables": {"F": "Force (M^1 L^1 T^-2)", "m": "Mass (M^1)", "a": "Acceleration (L^1 T^-2)"},\n'
        '  "left_hand_units": "M^1 L^1 T^-2",\n'
        '  "right_hand_units": "M^1 L^1 T^-2",\n'
        '  "dimensionally_consistent": true,\n'
        '  "derived_from_hypothesis": "string",\n'
        '  "derivation_logic": "By Newton\'s second law...",\n'
        '  "domain_constraints": ["x > 0"],\n'
        '  "boundary_conditions": ["as t -> 0"],\n'
        '  "computational_complexity": "O(1)",\n'
        '  "num_operations": 10\n'
        "}\n"
        "Be specific and mathematically precise."
    )
    
    vars_context = ""
    if known_variables:
        vars_context = f"Known relevant variables: {', '.join(known_variables)}\n"
    
    prompt = (
        f"Hypothesis to derive equation from: {hypothesis}\n\n"
        f"{vars_context}"
        "Derive the governing equation(s). Include ALL variables with their dimensions. "
        "Verify dimensional consistency."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,  # Low temp for mathematical precision
        )
        raw_json = completion.choices[0].message.content
        return DerivedEquation.model_validate_json(raw_json)
    except ValidationError as ve:
        print(f"   ❌ Equation validation error: {ve}")
        return None
    except Exception as e:
        print(f"   ❌ Error deriving equation: {e}")
        return None


def validate_equation(equation: DerivedEquation) -> EquationValidation:
    """
    Validates a derived equation for physical and mathematical correctness.
    """
    issues = []
    fixes = []
    
    # 1. Check dimensional consistency from the LLM's own claim
    if not equation.dimensionally_consistent:
        issues.append(f"Dimensional inconsistency: LHS={equation.left_hand_units}, RHS={equation.right_hand_units}")
        fixes.append("Add a dimensional constant to balance units.")
    
    # 2. Independent dimensional check
    lhs_dims = parse_dimensional_string(equation.left_hand_units)
    rhs_dims = parse_dimensional_string(equation.right_hand_units)
    our_consistency_check = dimensions_match(lhs_dims, rhs_dims)
    
    if our_consistency_check != equation.dimensionally_consistent:
        issues.append(f"Dimensional analysis mismatch: LLM says {equation.dimensionally_consistent}, "
                      f"our check says {our_consistency_check} (LHS={format_dimensions(lhs_dims)}, "
                      f"RHS={format_dimensions(rhs_dims)})")
        fixes.append("Reconcile the dimensional analysis.")
    
    # 3. Check for division by zero in constraints
    for constraint in equation.domain_constraints:
        if '= 0' in constraint or '== 0' in constraint:
            # Check if denominator of any fraction could be zero
            if '/' in equation.equation_text:
                issues.append(f"Constraint {constraint} could cause division by zero.")
                fixes.append(f"Add condition that denominator ≠ 0 when {constraint}.")
    
    # 4. Check variable count vs complexity
    num_vars = len(equation.variables)
    complexity_info = estimate_complexity(equation.equation_text, num_vars)
    
    # 5. Numerical stability check
    stability_issues = check_numerical_stability(equation.equation_text)
    issues.extend(stability_issues)
    for si in stability_issues:
        fixes.append(f"Add guard clauses for: {si}")
    
    # 6. Plausibility score
    score = 1.0
    if not equation.dimensionally_consistent:
        score -= 0.5
    if not our_consistency_check:
        score -= 0.2
    if issues:
        score -= min(0.3, len(issues) * 0.1)
    if not equation.domain_constraints:
        score -= 0.1
    if not equation.boundary_conditions:
        score -= 0.1
    score = max(0.0, score)
    
    is_plausible = score >= 0.5
    is_consistent = len([i for i in issues if 'dimension' in i.lower()]) == 0
    
    return EquationValidation(
        equation_id=equation.equation_id,
        is_physically_plausible=is_plausible,
        is_mathematically_consistent=is_consistent,
        potential_issues=issues,
        suggested_fixes=fixes,
        plausibility_score=round(score, 3)
    )


# =========================================================
# MAIN MATHEMATICS ENGINE ENTRY POINT
# =========================================================

def run_mathematics_engine(
    hypotheses: List[str],
    equations_file: str = "derived_equations.json"
) -> Dict[str, Any]:
    """
    Main entry point for the Mathematics Engine.
    
    For each hypothesis:
    1. Derives one or more equations via LLM
    2. Validates dimensional consistency (double-checked)
    3. Checks mathematical constraints
    4. Estimates computational complexity
    5. Rejects impossible formulations
    6. Saves all derived equations
    
    Args:
        hypotheses: List of hypotheses to derive equations for
        equations_file: Output file path
    
    Returns:
        Dict with derived equations and validation results
    """
    print("\n" + "="*80)
    print("📐 MATHEMATICS ENGINE — Deriving Equations, Validating Dimensions, Checking Constraints")
    print("="*80)
    
    client = Groq()
    
    equations = []
    rejected = []
    
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n  [{i}/{len(hypotheses)}] Deriving equation for hypothesis...")
        print(f"      📝 {hypothesis[:120]}...")
        
        # Step 1: Derive equation
        equation = derive_equation(client, hypothesis)
        if not equation:
            print(f"      ❌ Failed to derive equation.")
            rejected.append({
                "hypothesis": hypothesis,
                "reason": "LLM derivation failed"
            })
            continue
        
        # Step 2: Validate
        validation = validate_equation(equation)
        
        # Step 3: Complexity analysis
        complexity = estimate_complexity(equation.equation_text, len(equation.variables))
        
        # Step 4: Stability check
        stability = check_numerical_stability(equation.equation_text)
        
        entry = {
            "equation": equation.model_dump(),
            "validation": validation.model_dump(),
            "complexity": complexity,
            "numerical_stability_issues": stability,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 5: Reject if impossible
        is_impossible = (
            not validation.is_physically_plausible and 
            not validation.is_mathematically_consistent
        )
        
        if is_impossible:
            print(f"      ❌ REJECTED — Impossible formulation")
            rejected.append({
                "hypothesis": hypothesis,
                "equation_id": equation.equation_id,
                "reason": f"Not physically plausible ({validation.plausibility_score}) and mathematically inconsistent"
            })
        
        equations.append(entry)
        
        print(f"      📐 Equation: {equation.name}")
        print(f"      🧮 LaTeX: {equation.equation_latex}")
        print(f"      📏 Dimensional consistency: {'✅' if equation.dimensionally_consistent else '❌'}")
        print(f"         LHS: {equation.left_hand_units} | RHS: {equation.right_hand_units}")
        print(f"      🔬 Physically plausible: {'✅' if validation.is_physically_plausible else '❌'} "
              f"(score: {validation.plausibility_score:.2f})")
        print(f"      🎯 Complexity: {complexity['big_o']} ({complexity['total_operations']} ops)")
        print(f"      ⚠️  Issues: {len(validation.potential_issues)}")
        if validation.potential_issues:
            for issue in validation.potential_issues[:2]:
                print(f"          • {issue}")
    
    # Save to disk
    output = {
        "total_derived": len(equations),
        "total_rejected": len(rejected),
        "equations": equations,
        "rejected": rejected,
        "summary": {
            "dimensionally_consistent": sum(
                1 for e in equations if e["equation"]["dimensionally_consistent"]
            ),
            "physically_plausible": sum(
                1 for e in equations if e["validation"]["is_physically_plausible"]
            ),
            "avg_plausibility": round(
                sum(e["validation"]["plausibility_score"] for e in equations) / len(equations), 3
            ) if equations else 0.0,
            "total_operations_avg": round(
                sum(e["complexity"]["total_operations"] for e in equations) / len(equations), 1
            ) if equations else 0.0
        }
    }
    
    with open(equations_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"\n📄 All equations saved to '{equations_file}'")
    print(f"📊 Summary: {output['summary']['dimensionally_consistent']} dimensionally consistent, "
          f"{output['summary']['physically_plausible']} physically plausible, "
          f"{output['summary']['avg_plausibility']:.2f} avg plausibility")
    print("="*80)
    
    return output


if __name__ == "__main__":
    test_hypotheses = [
        "Increasing electrode surface area increases battery capacity linearly with area",
        "Battery energy density is inversely proportional to internal resistance squared",
        "The gradient of the loss function with respect to weights decreases exponentially with depth"
    ]
    run_mathematics_engine(test_hypotheses)