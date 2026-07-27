"""
MATHS MODULE — Pure Python Mathematical Knowledge
No API keys. No LLM. Pure numpy/scipy computation.

Concepts: calculus, linear algebra, optimization, statistics, differential equations.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Any

class MathsConcept:
    def __init__(self, name: str, symbol: str, description: str,
                 formulas: Dict[str, Dict[str, Any]], relationships: List[str]):
        self.name = name
        self.symbol = symbol
        self.description = description
        self.formulas = formulas
        self.relationships = relationships
    
    def get_all_formulas_text(self) -> str:
        lines = [f"=== {self.name} ({self.symbol}) ==="]
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


def _build_maths_db() -> Dict[str, MathsConcept]:
    db = {}
    
    # --- DERIVATIVES ---
    db["derivatives"] = MathsConcept(
        name="Derivatives", symbol="dy/dx, f'(x)", 
        description="Rate of change of a function with respect to a variable.",
        formulas={
            "power_rule": {
                "text": "d/dx(x^n) = n * x^(n-1)",
                "func": lambda x=1, n=2: n * (x ** (n-1)),
                "random_params": lambda: {"x": np.random.uniform(0.1, 10), "n": np.random.uniform(-5, 5)},
                "vars": {"x": "variable", "n": "exponent"}
            },
            "exponential": {
                "text": "d/dx(e^x) = e^x",
                "func": lambda x=1: math.exp(x),
                "random_params": lambda: {"x": np.random.uniform(-5, 5)},
                "vars": {"x": "variable"}
            },
            "product_rule": {
                "text": "d/dx(u*v) = u*dv/dx + v*du/dx",
                "func": lambda u=1, v=1, du=1, dv=1: u*dv + v*du,
                "random_params": lambda: {"u": np.random.uniform(1, 10), "v": np.random.uniform(1, 10),
                                           "du": np.random.uniform(-5, 5), "dv": np.random.uniform(-5, 5)},
                "vars": {"u,v": "functions", "du,dv": "derivatives"}
            },
            "chain_rule": {
                "text": "d/dx(f(g(x))) = f'(g(x)) * g'(x)",
                "func": lambda fg=1, gx=1: fg * gx,
                "random_params": lambda: {"fg": np.random.uniform(-5, 5), "gx": np.random.uniform(-5, 5)},
                "vars": {"f'(g)": "outer derivative", "g'(x)": "inner derivative"}
            }
        },
        relationships=["integrals", "differential_equations", "optimization", "series"]
    )
    
    # --- INTEGRALS ---
    db["integrals"] = MathsConcept(
        name="Integrals", symbol="∫ f(x) dx", 
        description="Area under a curve, accumulation of quantities.",
        formulas={
            "power_integral": {
                "text": "∫ x^n dx = x^(n+1)/(n+1) + C, for n ≠ -1",
                "func": lambda x=1, n=2: (x ** (n+1)) / (n+1),
                "random_params": lambda: {"x": np.random.uniform(0.1, 10), "n": np.random.uniform(0, 5)},
                "vars": {"x": "variable", "n": "exponent", "C": "integration constant"}
            },
            "definite": {
                "text": "∫_a^b f(x) dx = F(b) - F(a) [Fundamental Theorem]",
                "func": lambda Fb=1, Fa=0: Fb - Fa,
                "random_params": lambda: {"Fb": np.random.uniform(0, 100), "Fa": np.random.uniform(0, 100)},
                "vars": {"F(b)": "antiderivative at b", "F(a)": "antiderivative at a"}
            },
            "integration_by_parts": {
                "text": "∫ u dv = u*v - ∫ v du",
                "func": lambda uv=1, vdu=0: uv - vdu,
                "random_params": lambda: {"uv": np.random.uniform(1, 50), "vdu": np.random.uniform(0, 50)},
                "vars": {"u,v": "functions"}
            }
        },
        relationships=["derivatives", "differential_equations", "probability"]
    )
    
    # --- LINEAR ALGEBRA ---
    db["linear_algebra"] = MathsConcept(
        name="Linear Algebra", symbol="A·x = b", 
        description="Study of vectors, matrices, and linear transformations.",
        formulas={
            "dot_product": {
                "text": "a · b = Σ(ai * bi) for i=1 to n",
                "func": lambda a1=1, a2=1, a3=1, b1=2, b2=2, b3=2: a1*b1 + a2*b2 + a3*b3,
                "random_params": lambda: {"a1": np.random.uniform(-10, 10), "a2": np.random.uniform(-10, 10),
                                           "a3": np.random.uniform(-10, 10), "b1": np.random.uniform(-10, 10),
                                           "b2": np.random.uniform(-10, 10), "b3": np.random.uniform(-10, 10)},
                "vars": {"ai,bi": "vector components"}
            },
            "matrix_multiply": {
                "text": "(A·B){ij} = Σ(k=1 to n) A_{ik} * B_{kj}",
                "func": lambda a11=1, a12=1, a21=1, a22=1, b11=1, b12=1, b21=1, b22=1: {
                    "c11": a11*b11 + a12*b21, "c12": a11*b12 + a12*b22,
                    "c21": a21*b11 + a22*b21, "c22": a21*b12 + a22*b22
                },
                "random_params": lambda: {f"a{i}{j}": np.random.uniform(-5, 5) for i in [1,2] for j in [1,2]} | {f"b{i}{j}": np.random.uniform(-5, 5) for i in [1,2] for j in [1,2]},
                "vars": {"A,B": "matrices", "i,j,k": "indices"}
            },
            "determinant_2x2": {
                "text": "det(A) = a11*a22 - a12*a21",
                "func": lambda a11=1, a12=1, a21=1, a22=1: a11*a22 - a12*a21,
                "random_params": lambda: {"a11": np.random.uniform(-10, 10), "a12": np.random.uniform(-10, 10),
                                           "a21": np.random.uniform(-10, 10), "a22": np.random.uniform(-10, 10)},
                "vars": {"a11,a22": "diagonal elements", "a12,a21": "off-diagonal"}
            },
            "eigenvalue": {
                "text": "A·v = λ·v",
                "func": lambda: 0,
                "random_params": lambda: {},
                "vars": {"A": "matrix", "v": "eigenvector", "λ": "eigenvalue"}
            }
        },
        relationships=["derivatives", "optimization", "statistics", "numerical_methods"]
    )
    
    # --- OPTIMIZATION ---
    db["optimization"] = MathsConcept(
        name="Optimization", symbol="min f(x)", 
        description="Finding the best solution from all feasible solutions.",
        formulas={
            "gradient_descent": {
                "text": "x_{n+1} = x_n - α * ∇f(x_n)",
                "func": lambda x=1, alpha=0.1, grad=1: x - alpha * grad,
                "random_params": lambda: {"x": np.random.uniform(-10, 10), "alpha": np.random.uniform(0.001, 0.5),
                                           "grad": np.random.uniform(-5, 5)},
                "vars": {"x_n": "current point", "α": "learning rate", "∇f": "gradient"}
            },
            "newton_method": {
                "text": "x_{n+1} = x_n - f(x_n)/f'(x_n)",
                "func": lambda x=1, fx=1, fpx=1: x - fx/fpx,
                "random_params": lambda: {"x": np.random.uniform(-10, 10), "fx": np.random.uniform(-5, 5),
                                           "fpx": np.random.uniform(0.1, 10)},
                "vars": {"f(x)": "function value", "f'(x)": "derivative"}
            },
            "lagrange": {
                "text": "L(x, λ) = f(x) - λ*g(x) [constrained optimization]",
                "func": lambda fx=1, lam=1, gx=1: fx - lam*gx,
                "random_params": lambda: {"fx": np.random.uniform(-10, 10), "lam": np.random.uniform(-5, 5),
                                           "gx": np.random.uniform(-10, 10)},
                "vars": {"f(x)": "objective", "λ": "Lagrange multiplier", "g(x)": "constraint"}
            }
        },
        relationships=["derivatives", "linear_algebra", "numerical_methods"]
    )
    
    # --- STATISTICS ---
    db["statistics"] = MathsConcept(
        name="Statistics", symbol="μ, σ, r", 
        description="Collection, analysis, interpretation of data.",
        formulas={
            "mean": {
                "text": "μ = (1/n) * Σ(x_i) for i=1 to n",
                "func": lambda *args: sum(args) / len(args) if args else 0,
                "random_params": lambda: {"args": tuple(np.random.uniform(0, 100, 10))},
                "vars": {"μ": "mean", "x_i": "data points", "n": "sample size"}
            },
            "variance": {
                "text": "σ² = (1/n) * Σ(x_i - μ)²",
                "func": lambda *args: sum((x - sum(args)/len(args))**2 for x in args) / len(args) if args else 0,
                "random_params": lambda: {"args": tuple(np.random.uniform(0, 100, 10))},
                "vars": {"σ²": "variance", "μ": "mean"}
            },
            "normal_dist": {
                "text": "f(x) = 1/(σ*√(2π)) * exp(-(x-μ)²/(2σ²))",
                "func": lambda x=0, mu=0, sigma=1: (1/(sigma*math.sqrt(2*math.pi))) * math.exp(-((x-mu)**2)/(2*sigma**2)),
                "random_params": lambda: {"x": np.random.uniform(-5, 5), "mu": 0, "sigma": np.random.uniform(0.5, 3)},
                "vars": {"μ": "mean", "σ": "standard deviation", "π": "pi"}
            },
            "bayes": {
                "text": "P(A|B) = P(B|A) * P(A) / P(B)",
                "func": lambda pBA=0.5, pA=0.5, pB=0.5: pBA * pA / pB,
                "random_params": lambda: {"pBA": np.random.uniform(0.1, 0.9), "pA": np.random.uniform(0.1, 0.9),
                                           "pB": np.random.uniform(0.1, 0.9)},
                "vars": {"P(A|B)": "posterior", "P(B|A)": "likelihood", "P(A)": "prior", "P(B)": "evidence"}
            },
            "linear_regression": {
                "text": "y = m*x + b, where m = cov(x,y)/var(x)",
                "func": lambda cov=1, varx=1: cov/varx,
                "random_params": lambda: {"cov": np.random.uniform(-10, 10), "varx": np.random.uniform(0.1, 10)},
                "vars": {"m": "slope", "b": "intercept", "cov": "covariance", "var": "variance"}
            }
        },
        relationships=["probability", "linear_algebra", "optimization"]
    )
    
    # --- DIFFERENTIAL EQUATIONS ---
    db["differential_equations"] = MathsConcept(
        name="Differential Equations", symbol="dy/dx = f(x,y)", 
        description="Equations involving derivatives of unknown functions.",
        formulas={
            "first_order_separable": {
                "text": "dy/dx = g(x)*h(y) → ∫ dy/h(y) = ∫ g(x) dx",
                "func": lambda: 0,
                "random_params": lambda: {},
                "vars": {"g(x)": "function of x only", "h(y)": "function of y only"}
            },
            "euler_method": {
                "text": "y_{n+1} = y_n + h * f(x_n, y_n)",
                "func": lambda yn=0, h=0.1, f=1: yn + h * f,
                "random_params": lambda: {"yn": np.random.uniform(-5, 5), "h": np.random.uniform(0.01, 0.5),
                                           "f": np.random.uniform(-5, 5)},
                "vars": {"y_n": "current value", "h": "step size", "f": "slope"}
            },
            "runge_kutta_4": {
                "text": "y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)",
                "func": lambda yn=0, h=0.1, k1=1, k2=1, k3=1, k4=1: yn + (h/6)*(k1 + 2*k2 + 2*k3 + k4),
                "random_params": lambda: {"yn": np.random.uniform(-5, 5), "h": 0.1,
                                           "k1": np.random.uniform(-3, 3), "k2": np.random.uniform(-3, 3),
                                           "k3": np.random.uniform(-3, 3), "k4": np.random.uniform(-3, 3)},
                "vars": {"k1...k4": "RK4 coefficients", "h": "step size"}
            }
        },
        relationships=["derivatives", "integrals", "physics", "numerical_methods"]
    )
    
    # --- SERIES & SEQUENCES ---
    db["series"] = MathsConcept(
        name="Series & Sequences", symbol="Σ an, lim", 
        description="Sum of terms in a sequence.",
        formulas={
            "taylor_series": {
                "text": "f(x) = Σ f^(n)(a)*(x-a)^n / n! for n=0 to ∞",
                "func": lambda: 0,
                "random_params": lambda: {},
                "vars": {"f^(n)(a)": "nth derivative at a", "n!": "n factorial"}
            },
            "geometric_series": {
                "text": "Σ r^n = 1/(1-r) for |r| < 1, n=0 to ∞",
                "func": lambda r=0.5: 1/(1-r),
                "random_params": lambda: {"r": np.random.uniform(-0.9, 0.9)},
                "vars": {"r": "common ratio"}
            },
            "arithmetic_series": {
                "text": "S_n = n*(a1 + an)/2 = n*(2a1 + (n-1)*d)/2",
                "func": lambda n=10, a1=1, d=1: n * (2*a1 + (n-1)*d) / 2,
                "random_params": lambda: {"n": np.random.randint(2, 100), "a1": np.random.uniform(-10, 10),
                                           "d": np.random.uniform(-5, 5)},
                "vars": {"S_n": "sum of n terms", "a1": "first term", "d": "common difference"}
            }
        },
        relationships=["derivatives", "integrals", "numerical_methods"]
    )
    
    # --- FOURIER ANALYSIS ---
    db["fourier"] = MathsConcept(
        name="Fourier Analysis", symbol="F(ω), a_n, b_n", 
        description="Decomposition of functions into sinusoidal components.",
        formulas={
            "fourier_series": {
                "text": "f(x) = a0/2 + Σ(an*cos(nx) + bn*sin(nx)) for n=1 to ∞",
                "func": lambda a0=1: a0/2,
                "random_params": lambda: {"a0": np.random.uniform(-5, 5)},
                "vars": {"an,bn": "Fourier coefficients"}
            },
            "fourier_transform": {
                "text": "F(ω) = ∫ f(t)*e^(-iωt) dt",
                "func": lambda: 0,
                "random_params": lambda: {},
                "vars": {"F(ω)": "frequency domain", "f(t)": "time domain"}
            },
            "convolution": {
                "text": "(f*g)(t) = ∫ f(τ)*g(t-τ) dτ",
                "func": lambda: 0,
                "random_params": lambda: {},
                "vars": {"f*g": "convolution"}
            }
        },
        relationships=["series", "differential_equations", "signal_processing"]
    )
    
    return db

MATHS_DB = _build_maths_db()

def get_concept(name: str) -> Optional[MathsConcept]:
    return MATHS_DB.get(name.lower())

def get_all_concept_names() -> List[str]:
    return list(MATHS_DB.keys())

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
        "type": "maths_prediction"
    }

if __name__ == "__main__":
    print("Maths Concepts:", len(MATHS_DB))
    for name in MATHS_DB:
        print(f"  • {name}")