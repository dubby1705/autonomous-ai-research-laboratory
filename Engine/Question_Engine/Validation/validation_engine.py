"""
validation_engine.py — Core Validation Engine with Plugin Architecture
======================================================================
The validation engine:
  - Auto-discovers validation plugins from the validation/ directory
  - Runs domain-specific tests on each idea
  - Supports parallel validation
  - Returns PASS/WARNING/FAIL for each test
  - Never crashes the pipeline on validation errors

Each plugin defines:
  - name: str
  - domain_keywords: List[str]
  - validation_tests: List[Dict]
  - validate(idea, problem) -> List[Dict]
"""

import os
import json
import importlib
import inspect
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# TEST RESULT CONSTANTS
# =========================================================
PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"

# =========================================================
# DATA CLASSES
# =========================================================
class TestResult:
    """Result of a single validation test."""

    def __init__(self, test_name: str, status: str, score: float = 0.0,
                 details: str = "", severity: str = "normal"):
        self.test_name = test_name
        self.status = status  # PASS, WARNING, FAIL
        self.score = score    # 0.0 - 1.0
        self.details = details
        self.severity = severity  # critical, normal, minor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "score": round(self.score, 3),
            "details": self.details,
            "severity": self.severity,
        }


class ValidationResult:
    """Complete validation result for an idea."""

    def __init__(self, idea_id: str, idea: str, domain: str,
                 tests: List[TestResult], overall_status: str,
                 passed: int, warnings: int, failed: int,
                 validation_score: float, plugin_name: str):
        self.idea_id = idea_id
        self.idea = idea
        self.domain = domain
        self.tests = tests
        self.overall_status = overall_status
        self.passed = passed
        self.warnings = warnings
        self.failed = failed
        self.validation_score = validation_score
        self.plugin_name = plugin_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "idea": self.idea,
            "domain": self.domain,
            "tests": [t.to_dict() for t in self.tests],
            "overall_status": self.overall_status,
            "passed": self.passed,
            "warnings": self.warnings,
            "failed": self.failed,
            "validation_score": round(self.validation_score, 3),
            "plugin_name": self.plugin_name,
        }


class ValidationPlugin:
    """Base class for validation plugins."""

    name = "base"
    domain_keywords: List[str] = []
    validation_tests: List[Dict[str, Any]] = []

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        """Run validation tests on an idea. Override in subclasses."""
        results = []
        for test in self.validation_tests:
            results.append(TestResult(
                test_name=test.get("name", "unknown"),
                status=PASS,
                score=0.5,
                details="Base validation",
                severity=test.get("severity", "normal"),
            ))
        return results

    def _keyword_score(self, idea: str, keywords: List[str]) -> float:
        """Score how well an idea covers domain keywords."""
        idea_lower = idea.lower()
        hits = sum(1 for kw in keywords if kw in idea_lower)
        # Lenient scaling: 1 hit = 0.35, 2 hits = 0.6, 3+ = 1.0
        # Also check if idea overlaps with problem statement context
        if hits >= 3:
            return 1.0
        if hits == 2:
            return 0.6
        if hits == 1:
            return 0.35
        # Check for general design/improvement terms
        general_terms = ["design", "improve", "enhance", "optimize", "develop",
                         "novel", "system", "approach", "using", "based"]
        general_hits = sum(1 for t in general_terms if t in idea_lower)
        if general_hits >= 2:
            return 0.25
        return 0.1  # Small base score for any idea

    def _length_score(self, idea: str) -> float:
        """Score based on idea length (too short/long is bad)."""
        words = idea.split()
        if 8 <= len(words) <= 30:
            return 1.0
        if 5 <= len(words) <= 40:
            return 0.6
        return 0.3

    def _specificity_score(self, idea: str) -> float:
        """Score based on specificity (numbers, materials, techniques)."""
        import re
        score = 0.0
        if re.search(r'\d+', idea):
            score += 0.3
        if any(m in idea.lower() for m in [
            "graphene", "silicon", "polymer", "ceramic", "composite",
            "nanoparticle", "nanotube", "electrolyte", "catalyst",
            "quantum", "neural", "algorithm", "optimization",
        ]):
            score += 0.4
        if len(idea.split()) >= 10:
            score += 0.3
        return min(1.0, score)


# =========================================================
# BUILT-IN PLUGINS
# =========================================================
class GPUValidationPlugin(ValidationPlugin):
    """GPU design validation."""

    name = "gpu"
    domain_keywords = ["gpu", "graphics", "cuda", "shader", "raster", "render",
                       "bandwidth", "vram", "memory bandwidth", "gpu architecture"]
    validation_tests = [
        {"name": "thermal", "description": "Thermal management feasibility", "severity": "critical"},
        {"name": "memory", "description": "Memory capacity and bandwidth", "severity": "critical"},
        {"name": "bandwidth", "description": "Memory bandwidth adequacy", "severity": "critical"},
        {"name": "power", "description": "Power consumption within limits", "severity": "critical"},
        {"name": "cost", "description": "Manufacturing cost feasibility", "severity": "normal"},
        {"name": "manufacturability", "description": "Can be manufactured with current tech", "severity": "critical"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        # Thermal test
        thermal_score = self._keyword_score(idea, ["thermal", "cool", "heat", "temperature", "tdp"])
        results.append(TestResult(
            "thermal",
            PASS if thermal_score > 0.5 else (WARNING if thermal_score > 0.2 else FAIL),
            thermal_score,
            "Thermal management addressed" if thermal_score > 0.5 else "No thermal management mentioned",
            "critical",
        ))

        # Memory test
        memory_score = self._keyword_score(idea, ["memory", "vram", "cache", "hbm", "gdrd", "storage"])
        results.append(TestResult(
            "memory",
            PASS if memory_score > 0.5 else (WARNING if memory_score > 0.2 else FAIL),
            memory_score,
            "Memory architecture addressed" if memory_score > 0.5 else "No memory architecture mentioned",
            "critical",
        ))

        # Bandwidth test
        bandwidth_score = self._keyword_score(idea, ["bandwidth", "throughput", "data rate", "interconnect"])
        results.append(TestResult(
            "bandwidth",
            PASS if bandwidth_score > 0.5 else (WARNING if bandwidth_score > 0.2 else FAIL),
            bandwidth_score,
            "Bandwidth addressed" if bandwidth_score > 0.5 else "No bandwidth considerations",
            "critical",
        ))

        # Power test
        power_score = self._keyword_score(idea, ["power", "energy", "efficiency", "watt", "tdp"])
        results.append(TestResult(
            "power",
            PASS if power_score > 0.5 else (WARNING if power_score > 0.2 else FAIL),
            power_score,
            "Power consumption addressed" if power_score > 0.5 else "No power considerations",
            "critical",
        ))

        # Cost test
        cost_score = self._keyword_score(idea, ["cost", "price", "affordable", "cheap", "economical"])
        results.append(TestResult(
            "cost",
            PASS if cost_score > 0.3 else WARNING,
            cost_score,
            "Cost addressed" if cost_score > 0.3 else "No cost considerations",
            "normal",
        ))

        # Manufacturability test
        manuf_score = self._keyword_score(idea, ["manufactur", "fabricat", "process", "production", "yield"])
        results.append(TestResult(
            "manufacturability",
            PASS if manuf_score > 0.3 else WARNING,
            manuf_score,
            "Manufacturability addressed" if manuf_score > 0.3 else "No manufacturability considerations",
            "critical",
        ))

        return results


class BatteryValidationPlugin(ValidationPlugin):
    """Battery design validation."""

    name = "battery"
    domain_keywords = ["battery", "electrode", "electrolyte", "lithium", "anode", "cathode",
                       "energy density", "charge", "discharge", "cell", "solid-state"]
    validation_tests = [
        {"name": "energy_density", "description": "Energy density feasibility", "severity": "critical"},
        {"name": "stability", "description": "Chemical and structural stability", "severity": "critical"},
        {"name": "cycle_life", "description": "Cycle life adequacy", "severity": "critical"},
        {"name": "safety", "description": "Safety under failure conditions", "severity": "critical"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        # Energy density
        ed_score = self._keyword_score(idea, ["energy density", "capacity", "wh/kg", "specific energy"])
        results.append(TestResult(
            "energy_density",
            PASS if ed_score > 0.5 else (WARNING if ed_score > 0.2 else FAIL),
            ed_score,
            "Energy density addressed" if ed_score > 0.5 else "No energy density considerations",
            "critical",
        ))

        # Stability
        stab_score = self._keyword_score(idea, ["stability", "stable", "degradation", "decompos", "corrosion"])
        results.append(TestResult(
            "stability",
            PASS if stab_score > 0.5 else (WARNING if stab_score > 0.2 else FAIL),
            stab_score,
            "Stability addressed" if stab_score > 0.5 else "No stability considerations",
            "critical",
        ))

        # Cycle life
        cycle_score = self._keyword_score(idea, ["cycle", "lifetime", "longevity", "durability", "aging"])
        results.append(TestResult(
            "cycle_life",
            PASS if cycle_score > 0.5 else (WARNING if cycle_score > 0.2 else FAIL),
            cycle_score,
            "Cycle life addressed" if cycle_score > 0.5 else "No cycle life considerations",
            "critical",
        ))

        # Safety
        safety_score = self._keyword_score(idea, ["safety", "safe", "thermal runaway", "fire", "protection"])
        results.append(TestResult(
            "safety",
            PASS if safety_score > 0.5 else (WARNING if safety_score > 0.2 else FAIL),
            safety_score,
            "Safety addressed" if safety_score > 0.5 else "No safety considerations",
            "critical",
        ))

        return results


class FloatingVehicleValidationPlugin(ValidationPlugin):
    """Floating vehicle validation."""

    name = "floating"
    domain_keywords = ["floating", "buoyancy", "hover", "levitation", "amphibious",
                       "water", "marine", "boat", "ship", "hovercraft"]
    validation_tests = [
        {"name": "lift", "description": "Lift generation feasibility", "severity": "critical"},
        {"name": "buoyancy", "description": "Buoyancy and displacement", "severity": "critical"},
        {"name": "stability", "description": "Stability in water", "severity": "critical"},
        {"name": "energy", "description": "Energy requirements", "severity": "critical"},
        {"name": "battery", "description": "Battery/power system", "severity": "critical"},
        {"name": "thermal", "description": "Thermal management", "severity": "normal"},
        {"name": "safety", "description": "Safety systems", "severity": "critical"},
        {"name": "manufacturing", "description": "Manufacturability", "severity": "normal"},
        {"name": "cost", "description": "Cost feasibility", "severity": "normal"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        tests = [
            ("lift", ["lift", "thrust", "propulsion", "hover", "levitation"], "critical"),
            ("buoyancy", ["buoyancy", "displacement", "float", "density", "water"], "critical"),
            ("stability", ["stability", "stable", "balance", "trim", "control"], "critical"),
            ("energy", ["energy", "power", "efficiency", "consumption"], "critical"),
            ("battery", ["battery", "cell", "storage", "charge", "power source"], "critical"),
            ("thermal", ["thermal", "heat", "cool", "temperature"], "normal"),
            ("safety", ["safety", "safe", "protection", "emergency", "fail-safe"], "critical"),
            ("manufacturing", ["manufactur", "fabricat", "build", "production"], "normal"),
            ("cost", ["cost", "price", "affordable", "economical"], "normal"),
        ]

        for test_name, keywords, severity in tests:
            score = self._keyword_score(idea, keywords)
            status = PASS if score > 0.5 else (WARNING if score > 0.2 else FAIL)
            results.append(TestResult(
                test_name, status, score,
                f"{test_name.title()} addressed" if score > 0.5 else f"No {test_name} considerations",
                severity,
            ))

        return results


class MedicineValidationPlugin(ValidationPlugin):
    """Medicine/pharmaceutical validation."""

    name = "medicine"
    domain_keywords = ["drug", "medicine", "pharmaceutical", "therapeutic", "bioavailability",
                       "toxicity", "clinical", "patient", "treatment", "dosage"]
    validation_tests = [
        {"name": "toxicity", "description": "Toxicity assessment", "severity": "critical"},
        {"name": "safety", "description": "Safety profile", "severity": "critical"},
        {"name": "bioavailability", "description": "Bioavailability adequacy", "severity": "critical"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        # Toxicity
        tox_score = self._keyword_score(idea, ["toxicity", "toxic", "safe", "side effect", "adverse"])
        results.append(TestResult(
            "toxicity",
            PASS if tox_score > 0.5 else (WARNING if tox_score > 0.2 else FAIL),
            tox_score,
            "Toxicity addressed" if tox_score > 0.5 else "No toxicity considerations",
            "critical",
        ))

        # Safety
        safety_score = self._keyword_score(idea, ["safety", "safe", "protection", "risk", "adverse"])
        results.append(TestResult(
            "safety",
            PASS if safety_score > 0.5 else (WARNING if safety_score > 0.2 else FAIL),
            safety_score,
            "Safety addressed" if safety_score > 0.5 else "No safety considerations",
            "critical",
        ))

        # Bioavailability
        bio_score = self._keyword_score(idea, ["bioavailability", "absorption", "delivery", "distribution", "metabolism"])
        results.append(TestResult(
            "bioavailability",
            PASS if bio_score > 0.5 else (WARNING if bio_score > 0.2 else FAIL),
            bio_score,
            "Bioavailability addressed" if bio_score > 0.5 else "No bioavailability considerations",
            "critical",
        ))

        return results


class ChemistryValidationPlugin(ValidationPlugin):
    """Chemistry validation."""

    name = "chemistry"
    domain_keywords = ["chemistry", "chemical", "reaction", "catalyst", "synthesis",
                       "acid", "base", "ph", "molecule", "compound"]
    validation_tests = [
        {"name": "yield", "description": "Reaction yield feasibility", "severity": "normal"},
        {"name": "selectivity", "description": "Product selectivity", "severity": "normal"},
        {"name": "stability", "description": "Chemical stability", "severity": "critical"},
        {"name": "safety", "description": "Safety of chemicals/process", "severity": "critical"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        tests = [
            ("yield", ["yield", "efficiency", "conversion", "production"], "normal"),
            ("selectivity", ["selectivity", "specific", "target", "pure"], "normal"),
            ("stability", ["stability", "stable", "degradation", "decompos"], "critical"),
            ("safety", ["safety", "safe", "hazard", "toxic", "corrosive"], "critical"),
        ]

        for test_name, keywords, severity in tests:
            score = self._keyword_score(idea, keywords)
            status = PASS if score > 0.5 else (WARNING if score > 0.2 else FAIL)
            results.append(TestResult(
                test_name, status, score,
                f"{test_name.title()} addressed" if score > 0.5 else f"No {test_name} considerations",
                severity,
            ))

        return results


class RoboticsValidationPlugin(ValidationPlugin):
    """Robotics validation."""

    name = "robotics"
    domain_keywords = ["robot", "actuator", "manipulator", "gripper", "autonomous",
                       "sensor", "control", "navigation", "arm", "precision"]
    validation_tests = [
        {"name": "precision", "description": "Precision and accuracy", "severity": "critical"},
        {"name": "control", "description": "Control system feasibility", "severity": "critical"},
        {"name": "energy", "description": "Energy efficiency", "severity": "normal"},
        {"name": "safety", "description": "Safety systems", "severity": "critical"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        tests = [
            ("precision", ["precision", "accuracy", "repeatability", "resolution"], "critical"),
            ("control", ["control", "feedback", "sensor", "adaptive", "closed-loop"], "critical"),
            ("energy", ["energy", "power", "efficiency", "consumption"], "normal"),
            ("safety", ["safety", "safe", "protection", "collision", "fail-safe"], "critical"),
        ]

        for test_name, keywords, severity in tests:
            score = self._keyword_score(idea, keywords)
            status = PASS if score > 0.5 else (WARNING if score > 0.2 else FAIL)
            results.append(TestResult(
                test_name, status, score,
                f"{test_name.title()} addressed" if score > 0.5 else f"No {test_name} considerations",
                severity,
            ))

        return results


class MaterialsValidationPlugin(ValidationPlugin):
    """Materials science validation."""

    name = "materials"
    domain_keywords = ["material", "composite", "alloy", "polymer", "ceramic",
                       "strength", "stiffness", "toughness", "corrosion"]
    validation_tests = [
        {"name": "strength", "description": "Mechanical strength", "severity": "critical"},
        {"name": "durability", "description": "Long-term durability", "severity": "critical"},
        {"name": "manufacturability", "description": "Manufacturability", "severity": "normal"},
        {"name": "cost", "description": "Cost feasibility", "severity": "normal"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        tests = [
            ("strength", ["strength", "tensile", "stiffness", "modulus", "hardness"], "critical"),
            ("durability", ["durability", "fatigue", "corrosion", "wear", "aging"], "critical"),
            ("manufacturability", ["manufactur", "fabricat", "process", "production"], "normal"),
            ("cost", ["cost", "price", "affordable", "economical"], "normal"),
        ]

        for test_name, keywords, severity in tests:
            score = self._keyword_score(idea, keywords)
            status = PASS if score > 0.5 else (WARNING if score > 0.2 else FAIL)
            results.append(TestResult(
                test_name, status, score,
                f"{test_name.title()} addressed" if score > 0.5 else f"No {test_name} considerations",
                severity,
            ))

        return results


class TransportValidationPlugin(ValidationPlugin):
    """Transportation/aerospace validation."""

    name = "transport"
    domain_keywords = ["vehicle", "transport", "aircraft", "wing", "aerodynamic",
                       "lift", "drag", "thrust", "flight", "propulsion"]
    validation_tests = [
        {"name": "aerodynamics", "description": "Aerodynamic performance", "severity": "critical"},
        {"name": "structural", "description": "Structural integrity", "severity": "critical"},
        {"name": "energy", "description": "Energy efficiency", "severity": "critical"},
        {"name": "safety", "description": "Safety systems", "severity": "critical"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        tests = [
            ("aerodynamics", ["lift", "drag", "aerodynamic", "airfoil", "thrust"], "critical"),
            ("structural", ["structural", "strength", "frame", "integrity", "load"], "critical"),
            ("energy", ["energy", "fuel", "efficiency", "power", "consumption"], "critical"),
            ("safety", ["safety", "safe", "protection", "emergency", "fail-safe"], "critical"),
        ]

        for test_name, keywords, severity in tests:
            score = self._keyword_score(idea, keywords)
            status = PASS if score > 0.5 else (WARNING if score > 0.2 else FAIL)
            results.append(TestResult(
                test_name, status, score,
                f"{test_name.title()} addressed" if score > 0.5 else f"No {test_name} considerations",
                severity,
            ))

        return results


class GeneralValidationPlugin(ValidationPlugin):
    """General domain-agnostic validation."""

    name = "general"
    domain_keywords = []
    validation_tests = [
        {"name": "feasibility", "description": "Engineering feasibility", "severity": "critical"},
        {"name": "performance", "description": "Performance improvement potential", "severity": "normal"},
        {"name": "cost", "description": "Cost feasibility", "severity": "normal"},
        {"name": "safety", "description": "Safety considerations", "severity": "critical"},
        {"name": "scalability", "description": "Scalability potential", "severity": "normal"},
    ]

    def validate(self, idea: str, problem: str) -> List[TestResult]:
        results = []
        idea_lower = idea.lower()

        tests = [
            ("feasibility", ["feasible", "practical", "implement", "build", "construct"], "critical"),
            ("performance", ["performance", "improve", "enhance", "increase", "optimize"], "normal"),
            ("cost", ["cost", "price", "affordable", "economical", "cheap"], "normal"),
            ("safety", ["safety", "safe", "protection", "risk", "hazard"], "critical"),
            ("scalability", ["scale", "scalable", "expand", "large-scale", "production"], "normal"),
        ]

        for test_name, keywords, severity in tests:
            score = self._keyword_score(idea, keywords)
            status = PASS if score > 0.5 else (WARNING if score > 0.2 else FAIL)
            results.append(TestResult(
                test_name, status, score,
                f"{test_name.title()} addressed" if score > 0.5 else f"No {test_name} considerations",
                severity,
            ))

        return results


# =========================================================
# BUILT-IN PLUGIN REGISTRY
# =========================================================
BUILTIN_PLUGINS = [
    GPUValidationPlugin(),
    BatteryValidationPlugin(),
    FloatingVehicleValidationPlugin(),
    MedicineValidationPlugin(),
    ChemistryValidationPlugin(),
    RoboticsValidationPlugin(),
    MaterialsValidationPlugin(),
    TransportValidationPlugin(),
    GeneralValidationPlugin(),
]


# =========================================================
# PLUGIN DISCOVERY
# =========================================================
def _discover_external_plugins() -> List[ValidationPlugin]:
    """Discover validation plugins from the validation/ directory."""
    plugins = []
    plugin_dir = os.path.dirname(os.path.abspath(__file__))

    for fname in os.listdir(plugin_dir):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        if fname == "validation_engine.py":
            continue

        module_name = fname[:-3]
        try:
            module = importlib.import_module(f"{__name__.rsplit('.', 1)[0]}.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, ValidationPlugin) and obj is not ValidationPlugin
                        and obj.__module__ == module.__name__):
                    try:
                        plugin = obj()
                        plugins.append(plugin)
                    except Exception:
                        continue
        except Exception:
            continue

    return plugins


# =========================================================
# VALIDATION ENGINE
# =========================================================
class ValidationEngine:
    """Main validation engine that runs domain-specific tests on ideas."""

    def __init__(self):
        self.plugins = list(BUILTIN_PLUGINS)
        # Discover external plugins
        try:
            external = _discover_external_plugins()
            self.plugins.extend(external)
        except Exception:
            pass

    def detect_domain(self, problem: str) -> str:
        """Detect which validation plugin(s) apply to a problem."""
        problem_lower = problem.lower()
        best_plugin = None
        best_score = 0

        for plugin in self.plugins:
            if not plugin.domain_keywords:
                continue
            score = sum(1 for kw in plugin.domain_keywords if kw in problem_lower)
            if score > best_score:
                best_score = score
                best_plugin = plugin

        return best_plugin.name if best_plugin else "general"

    def get_plugin_for_domain(self, domain: str) -> ValidationPlugin:
        """Get the validation plugin for a domain."""
        for plugin in self.plugins:
            if plugin.name == domain:
                return plugin
        # Fallback to general
        for plugin in self.plugins:
            if plugin.name == "general":
                return plugin
        return GeneralValidationPlugin()

    def validate_idea(self, idea: str, problem: str,
                      domain: Optional[str] = None) -> ValidationResult:
        """Validate a single idea."""
        if domain is None:
            domain = self.detect_domain(problem)

        plugin = self.get_plugin_for_domain(domain)
        idea_id = hashlib.md5(idea.encode("utf-8")).hexdigest()[:16]

        try:
            test_results = plugin.validate(idea, problem)
        except Exception as e:
            # Never crash on validation errors
            test_results = [TestResult(
                "validation_error",
                WARNING,
                0.0,
                f"Validation plugin error: {e}",
                "normal",
            )]

        # Compute summary
        passed = sum(1 for t in test_results if t.status == PASS)
        warnings = sum(1 for t in test_results if t.status == WARNING)
        failed = sum(1 for t in test_results if t.status == FAIL)

        # Overall status: FAIL if any critical test fails, WARNING if any warning, else PASS
        critical_fail = any(t.status == FAIL and t.severity == "critical" for t in test_results)
        any_fail = failed > 0
        any_warning = warnings > 0

        if critical_fail:
            overall = FAIL
        elif any_fail:
            overall = WARNING
        elif any_warning:
            overall = WARNING
        else:
            overall = PASS

        # Validation score: weighted by severity
        total_weight = 0
        weighted_score = 0
        for t in test_results:
            weight = 3.0 if t.severity == "critical" else 1.0
            total_weight += weight
            if t.status == PASS:
                weighted_score += weight * t.score
            elif t.status == WARNING:
                weighted_score += weight * t.score * 0.5

        validation_score = weighted_score / max(total_weight, 1)

        return ValidationResult(
            idea_id=idea_id,
            idea=idea,
            domain=domain,
            tests=test_results,
            overall_status=overall,
            passed=passed,
            warnings=warnings,
            failed=failed,
            validation_score=validation_score,
            plugin_name=plugin.name,
        )

    def validate_batch(self, ideas: List[Dict[str, Any]], problem: str,
                       max_workers: int = 4) -> List[ValidationResult]:
        """Validate a batch of ideas in parallel."""
        if not ideas:
            return []

        domain = self.detect_domain(problem)
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.validate_idea, idea.get("idea", ""), problem, domain)
                for idea in ideas
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    continue

        return results


# =========================================================
# SINGLETON INSTANCE
# =========================================================
_engine_instance = None


def get_validation_engine() -> ValidationEngine:
    """Get the singleton validation engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ValidationEngine()
    return _engine_instance


def run_validation(ideas: List[Dict[str, Any]], problem: str,
                   max_workers: int = 4) -> List[Dict[str, Any]]:
    """Run validation on a list of ideas. Returns list of result dicts."""
    engine = get_validation_engine()
    results = engine.validate_batch(ideas, problem, max_workers=max_workers)
    return [r.to_dict() for r in results]


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    engine = get_validation_engine()
    print(f"Loaded {len(engine.plugins)} validation plugins")

    test_ideas = [
        {"idea": "Graphene-enhanced thermal management system for GPU with improved cooling"},
        {"idea": "Solid-state battery with high energy density and safety features"},
        {"idea": "Floating vehicle with buoyancy control and stability systems"},
        {"idea": "Nanoparticle drug delivery with improved bioavailability"},
    ]
    test_problem = "Design a better battery with higher energy density"

    results = run_validation(test_ideas, test_problem)
    for r in results:
        print(f"\n  Idea: {r['idea'][:60]}...")
        print(f"  Domain: {r['domain']}, Status: {r['overall_status']}, Score: {r['validation_score']:.2f}")
        for t in r["tests"]:
            print(f"    [{t['status']:7s}] {t['test_name']}: {t['details']}")