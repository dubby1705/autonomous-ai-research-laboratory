"""
AARL Lab — the research core of the Autonomous AI Research Laboratory.

The package is deliberately import-light: importing ``lab`` pulls in nothing but
this docstring, so entry points can decide which sub-package to load. Every
sub-module documents the honesty rules it enforces; the central ones are:

* simulations come from executed engines, never from prose;
* every knowledge item carries provenance and a ``claim_type``;
* failed ideas are kept, never deleted;
* AARL never converts semantic similarity into scientific evidence.
"""

from __future__ import annotations

__all__ = [
    "config",
    "memory",
    "knowledge",
    "hypotheses",
    "experiments",
    "failures",
    "papers",
    "realworld",
    "reporting",
    "discovery",
    "web",
]
