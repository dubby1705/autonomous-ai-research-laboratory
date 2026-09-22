"""
Validation Plugin Architecture for AARL
========================================
Each scientific field has its own validation engine.
New domains can be added without modifying the core system.

To add a new domain:
  1. Create a new file in this directory (e.g., quantum.py)
  2. Define a ValidationPlugin class with:
     - name: str
     - domain_keywords: List[str]
     - validation_tests: List[Dict]  # Each test has name, description, severity
     - validate(idea, problem) -> List[Dict]  # Returns test results
  3. The plugin is auto-discovered by the ValidationEngine
"""

from .validation_engine import (
    ValidationEngine,
    ValidationPlugin,
    ValidationResult,
    TestResult,
    PASS,
    WARNING,
    FAIL,
    get_validation_engine,
    run_validation,
)

__all__ = [
    "ValidationEngine",
    "ValidationPlugin",
    "ValidationResult",
    "TestResult",
    "PASS",
    "WARNING",
    "FAIL",
    "get_validation_engine",
    "run_validation",
]