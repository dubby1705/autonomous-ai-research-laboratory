"""
AARL Showcase — Gradient Descent Optimisation Suite
====================================================
A SAFE, predefined set of gradient-descent optimisers plus a controlled,
reproducible synthetic problem.

The LLM does NOT generate code here: it only selects an optimiser from this
registry and tunes a few bounded, validated parameters. All numbers reported by
the pipeline come from actual execution of these routines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optimiser parameter registry (safe, bounded).
# ---------------------------------------------------------------------------
_OPT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "sgd": {"lr": 0.05},
    "momentum": {"lr": 0.03, "momentum": 0.9},
    "nesterov": {"lr": 0.03, "momentum": 0.9},
    "adagrad": {"lr": 0.1, "eps": 1e-8},
    "rmsprop": {"lr": 0.01, "rho": 0.9, "eps": 1e-8},
    "adam": {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "amsgrad": {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "adam_cosine": {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "sgd_cosine": {"lr": 0.1, "lr_min": 0.001},
    "sgd_warm_restart": {"lr": 0.1, "lr_min": 0.001, "period": 500},
}

# Relative bounds used to clamp values (fractional of default is overkill; use
# per-key numeric bounds instead).
_OPT_BOUNDS: Dict[str, tuple] = {
    "lr": (1e-6, 1.0),
    "lr_min": (1e-8, 1.0),
    "momentum": (0.0, 0.999),
    "rho": (0.1, 0.999),
    "beta1": (0.1, 0.999),
    "beta2": (0.9, 0.999),
    "eps": (1e-12, 1e-2),
    "period": (2, 1000),
}


def sanitise_params(optimizer: str, params: Dict[str, float]) -> Dict[str, float]:
    """Merge user params over safe defaults and clamp to bounds."""
    if optimizer not in _OPT_DEFAULTS:
        raise ValueError(f"Unknown optimiser '{optimizer}'.")
    merged = dict(_OPT_DEFAULTS[optimizer])
    for k, v in (params or {}).items():
        if k not in merged:
            continue  # ignore unknown / malicious keys
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = _OPT_BOUNDS.get(k, (1e-6, 1.0))
        merged[k] = float(min(max(v, lo), hi))
    return merged


def known_optimizers() -> List[str]:
    return sorted(_OPT_DEFAULTS.keys())
# ---------------------------------------------------------------------------
# Optimiser state
# ---------------------------------------------------------------------------
@dataclass
class _State:
    method: str = "sgd"
    params: Dict[str, float] = field(default_factory=dict)
    t: float = 0.0
    m: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    vhat: Optional[np.ndarray] = None
    mhat: Optional[np.ndarray] = None
    g2: Optional[np.ndarray] = None
    iteration: int = 0

    def reset(self, size: int):
        z = np.zeros(size)
        self.m, self.v, self.vhat, self.mhat, self.g2 = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()


def _update(state: _State, grad: np.ndarray, lr_now: float) -> np.ndarray:
    """Yield a single first-order parameter update."""
    s, m, p = state, state.method, state.params

    if m in ("sgd", "sgd_cosine", "sgd_warm_restart"):
        return -lr_now * grad
    if m == "momentum":
        s.m = p["momentum"] * s.m - lr_now * grad
        return s.m.copy()
    if m == "nesterov":
        s.v = p["momentum"] * s.v - lr_now * grad
        return p["momentum"] * s.v - lr_now * grad
    if m == "adagrad":
        s.g2 += grad * grad
        return -(lr_now * grad) / (np.sqrt(s.g2) + p["eps"])
    if m == "rmsprop":
        s.g2 = p["rho"] * s.g2 + (1 - p["rho"]) * (grad * grad)
        return -(lr_now * grad) / (np.sqrt(s.g2) + p["eps"])
    if m in ("adam", "adam_cosine", "amsgrad"):
        b1, b2 = p["beta1"], p["beta2"]
        s.m = b1 * s.m + (1 - b1) * grad
        s.v = b2 * s.v + (1 - b2) * (grad * grad)
        s.t += 1
        mhat = s.m / (1 - b1 ** s.t)
        vhat = s.v / (1 - b2 ** s.t)
        if m == "amsgrad":
            s.vhat = np.maximum(s.vhat, vhat)
            vhat = s.vhat
        return -(lr_now * mhat) / (np.sqrt(vhat) + p["eps"])
    raise ValueError(f"Unsupported method '{m}'")


def _lr_schedule(state: _State, max_iterations: int) -> float:
    """Current learning-rate for schedule-based methods."""
    m, p, t = state.method, state.params, state.iteration
    if m in ("sgd_cosine", "adam_cosine"):
        frac = min(1.0, t / max(1, max_iterations))
        return p["lr_min"] + 0.5 * (p["lr"] - p["lr_min"]) * (1 + math.cos(math.pi * frac))
    if m == "sgd_warm_restart":
        period = max(1, int(p.get("period", 500)))
        pos = t % period
        return p["lr_min"] + 0.5 * (p["lr"] - p["lr_min"]) * (1 + math.cos(math.pi * pos / period))
    return p.get("lr", 0.01)


# ---------------------------------------------------------------------------
# Objective function helpers
# ---------------------------------------------------------------------------
def _loss(x: np.ndarray, curv: np.ndarray, x0: np.ndarray, extra: Dict[str, float]) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        base = 0.5 * float(np.sum(curv * ((x - x0) ** 2)))
        if extra:
            base = base + float(extra.get("a", 0.0)) * float(np.sum(np.sin(extra.get("b", 1.0) * x)))
    return base


def _grad(x: np.ndarray, curv: np.ndarray, x0: np.ndarray, extra: Dict[str, float]) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        g = curv * (x - x0)
        if extra:
            g = g + float(extra.get("a", 0.0)) * float(extra.get("b", 1.0)) * np.cos(extra.get("b", 1.0) * x)
    return g


def make_quadratic_problem(dim: int, seed: int) -> Dict[str, object]:
    """Controlled ill-conditioned quadratic with a well-offset optimum.

    Curvature condition ~ [0.05, 1.0] (condition number ~20): ill-conditioned
    enough to reward adaptive/momentum methods, mild enough that plain SGD
    converges reliably within a small iteration budget.
    """
    rng = np.random.default_rng(seed)
    curvature = np.exp(rng.uniform(math.log(0.05), math.log(1.0), dim))
    curvature = rng.permutation(curvature)
    x0 = rng.normal(0.0, 0.5, dim)
    return {"curvatures": curvature, "x0": x0, "dim": dim}


def make_nonconvex_problem(dim: int, seed: int) -> Dict[str, object]:
    """Mildly nonconvex (sin-perturbed) function with a single clean basin."""
    rng = np.random.default_rng(seed + 1)
    curvature = np.linspace(0.2, 8.0, dim)
    x0 = rng.normal(0.0, 0.5, dim)
    return {"curvatures": curvature, "x0": x0, "dim": dim, "a": 0.05, "b": 2.0}


PROBLEMS: Dict[str, Dict[str, object]] = {
    "quadratic": {"name": "Ill-conditioned quadratic", "make": make_quadratic_problem},
    "nonconvex_smooth": {"name": "Smooth nonconvex (sin-perturbed)", "make": make_nonconvex_problem},
}
# ---------------------------------------------------------------------------
# The executable optimiser run
# ---------------------------------------------------------------------------
@dataclass
class OptimiseResult:
    method: str
    converged: bool
    iterations: int
    final_loss: float
    loss_history: List[float]


def run_optimiser(
    problem: Dict[str, object],
    method: str,
    params: Dict[str, float],
    max_iterations: int,
    tol: float,
    start: np.ndarray,
) -> OptimiseResult:
    """Run one full optimisation trajectory (deterministic input -> output).

    Returns a real, measured result. Never a fabricated number.
    """
    method = method.lower().strip()
    if method not in _OPT_DEFAULTS:
        raise ValueError(f"Unknown optimiser '{method}'")

    params = sanitise_params(method, params)
    state = _State(method=method, params=params)
    state.reset(int(problem["dim"]))

    curv = problem["curvatures"]
    x0 = problem["x0"]
    extra = {k: v for k, v in problem.items() if k in ("a", "b")}
    dim = int(problem["dim"])

    x = np.array(start, dtype=float).reshape(-1)[:dim].astype(float)
    if x.size < dim:
        x = np.concatenate([x, np.zeros(dim - x.size)])
    loss_history: List[float] = []

    initial_loss = _loss(x, curv, x0, extra)
    loss_history.append(float(initial_loss))
    if initial_loss <= tol:
        return OptimiseResult(method, True, 0, float(initial_loss), loss_history)

    for it in range(1, max_iterations + 1):
        state.iteration = it
        lr = _lr_schedule(state, max_iterations)
        grad = _grad(x, curv, x0, extra)
        update = _update(state, grad, lr)
        if not np.all(np.isfinite(update)):
            return OptimiseResult(method, False, it, float(_loss(x, curv, x0, extra)), loss_history)
        x = x + update
        loss = _loss(x, curv, x0, extra)
        if it % 5 == 0:
            loss_history.append(float(loss))
        if loss <= tol:
            return OptimiseResult(method, True, it, float(loss), loss_history)

    final = float(_loss(x, curv, x0, extra))
    loss_history.append(final)
    return OptimiseResult(method, False, max_iterations, final, loss_history)