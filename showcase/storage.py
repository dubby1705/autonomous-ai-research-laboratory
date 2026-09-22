"""
AARL Showcase — Run Storage
============================
Persists the complete research history of each run into a unique directory:

    runs/<YYYY-MM-DD_<seq>>/
        research.json      <- full structured state (all hypotheses/records)
        hypotheses.json    <- hypotheses + evaluations
        experiments.json   <- experiment specs + measured results
        results.json       <- aggregated results summary
        report.md          <- human-readable markdown report
        plots/             <- visualisations (if matplotlib available)

Previous runs are never overwritten.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List

from .records import ResearchRun, to_dict

log = logging.getLogger("aarl.storage")


class RunDirectory:
    """Represents a single unique research run directory."""

    def __init__(self, root: str = "runs"):
        self.root = os.path.abspath(root)
        self.run_id = ""
        self.path = ""
        self.plots_dir = ""

    def create(self) -> str:
        os.makedirs(self.root, exist_ok=True)
        seq = self._next_sequence()
        self.run_id = f"{datetime.date.today().isoformat()}_{seq:03d}"
        self.path = os.path.join(self.root, self.run_id)
        self.plots_dir = os.path.join(self.path, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        return self.path

    def _next_sequence(self) -> int:
        today = datetime.date.today().isoformat()
        prefix = today + "_"
        highest = 0
        try:
            for name in os.listdir(self.root):
                if name.startswith(prefix):
                    m = re.search(r"_(\d+)$", name)
                    if m:
                        highest = max(highest, int(m.group(1)))
        except FileNotFoundError:
            pass
        return highest + 1

    # ---- persistence ------------------------------------------------------
    def _write(self, filename: str, data: Any) -> str:
        path = os.path.join(self.path, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        return path

    def save_run(self, run: ResearchRun, experiments: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, str]:
        """Persist the complete research state. Returns written file paths."""
        run.run_id = self.run_id
        full = run.to_dict()

        # research.json — the single source of truth
        research_path = self._write("research.json", full)

        # hypotheses.json — all hypotheses incl. rejected, with evaluations
        hyp_path = self._write(
            "hypotheses.json",
            {
                "run_id": run.run_id,
                "research_question": run.research_question,
                "hypotheses": full["hypotheses"],
            },
        )

        # experiments.json — specs + measured results
        exp_path = self._write("experiments.json", experiments)

        # results.json — aggregated findings
        results_path = self._write("results.json", results)

        return {
            "research": research_path,
            "hypotheses": hyp_path,
            "experiments": exp_path,
            "results": results_path,
        }

    def save_text(self, filename: str, content: str) -> str:
        path = os.path.join(self.path, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path


def build_experiments_payload(run: ResearchRun) -> Dict[str, Any]:
    """Serialise experiment specs + results for experiments.json."""
    entries: List[Dict[str, Any]] = []
    for h in run.hypotheses:
        entries.append(
            {
                "hypothesis_id": h.hypothesis_id,
                "experiment_spec": to_dict(h.experiment_spec),
                "experiment_result": to_dict(h.experiment),
            }
        )
    return {
        "run_id": run.run_id,
        "problem": run.config.get("problem_name", "quadratic"),
        "experiments": entries,
    }


def build_results_payload(run: ResearchRun) -> Dict[str, Any]:
    """Serialise aggregated results + findings for results.json."""
    exp_rows: List[Dict[str, Any]] = []
    for h in run.hypotheses:
        if not h.passed:
            continue
        exp_rows.append(
            {
                "hypothesis_id": h.hypothesis_id,
                "title": h.title,
                "proposed_method": h.experiment_spec.proposed_method,
                "status": h.experiment.status,
                "baseline": h.experiment.baseline,
                "proposed": h.experiment.proposed,
                "improvement_pct": h.experiment.improvement_pct,
                "error": h.experiment.error,
            }
        )
    return {
        "run_id": run.run_id,
        "research_question": run.research_question,
        "created_at": run.created_at,
        "experiments": exp_rows,
        "findings": run.findings,
    }