"""
AARL Lab — Final Reporting Layer (10 reports + final result)
============================================================
ALL reports are generated from the MASTER RESEARCH STATE
(``research_memory/master_state.json``) — the single source of truth. No report
invents information; every number traces back to a Simulation JSON, a knowledge
item, or a human-provided record.

Files produced (researcher-facing layer)::

    00_FINAL_RESEARCH_RESULT.txt      top-level synthesis
    01_Research_Summary.txt
    02_Final_Design_Report.txt
    03_Design_Architecture.txt
    04_Engineering_Calculations.txt
    05_Materials_and_Technologies.txt
    06_Simulation_Report.txt
    07_Evidence_and_Confidence.txt
    08_Hypotheses_Analysis.txt
    09_Knowledge_Graph_Summary.txt
    10_Limitations_and_Next_Steps.txt

Claim-type discipline (labels used throughout)::

    [FACT]               sourced knowledge (papers / researcher data)
    [INFERENCE]          derived from multiple evidence pieces
    [HYPOTHESIS]         AI- or researcher-generated proposal
    [SIMULATION RESULT]  model outputs — never real-world evidence
    [REAL-WORLD RESULT]  human-provided observations
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_RULE = "=" * 78
_SUB = "-" * 78

_SUCCESS = {"success", "promising", "completed"}


class FinalReportGenerator:
    def __init__(self, state: Dict[str, Any], output_dir: str):
        self.state = state
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def generate_all(self, memory_root: Optional[str] = None) -> Dict[str, str]:
        builders = {
            "01_Research_Summary.txt": self.report_research_summary,
            "02_Final_Design_Report.txt": self.report_final_design,
            "03_Design_Architecture.txt": self.report_design_architecture,
            "04_Engineering_Calculations.txt": self.report_engineering_calculations,
            "05_Materials_and_Technologies.txt": self.report_materials_technologies,
            "06_Simulation_Report.txt": self.report_simulation,
            "07_Evidence_and_Confidence.txt": self.report_evidence_confidence,
            "08_Hypotheses_Analysis.txt": self.report_hypotheses_analysis,
            "09_Knowledge_Graph_Summary.txt": self.report_knowledge_graph,
            "10_Limitations_and_Next_Steps.txt": self.report_limitations_next_steps,
        }
        for filename, builder in builders.items():
            self._write(filename, builder())
        # The final synthesis is written last, from the same state.
        self._write("00_FINAL_RESEARCH_RESULT.txt", self.report_final_result())
        # Machine-readable graph export (same edges that report 09 prints).
        self.export_graph(self._graph_target(memory_root))
        return dict(self.paths)

    def _graph_target(self, memory_root: Optional[str]) -> str:
        """Where graph.json belongs: the research memory root, when it exists."""
        candidate = memory_root or self.state.get("memory_root") or "research_memory"
        try:
            if memory_root or os.path.isdir(candidate):
                os.makedirs(candidate, exist_ok=True)
                return os.path.join(candidate, "graph.json")
        except OSError:
            pass
        return os.path.join(self.output_dir, "graph.json")

    def export_graph(self, path: str) -> str:
        """Write ``graph.json`` — nodes + edges, derived only from the state."""
        nodes: List[Dict[str, Any]] = []
        for p in self._papers():
            nodes.append({"id": p.get("paper_id"), "type": "paper",
                          "label": str(p.get("title"))[:160],
                          "provided_by": p.get("provided_by", "unknown")})
        for i in self.knowledge_items():
            nodes.append({"id": i.get("item_id"), "type": "knowledge",
                          "claim_type": i.get("claim_type"),
                          "section": i.get("section"),
                          "source_paper": i.get("source_paper"),
                          "statement": str(i.get("statement"))[:200]})
        for h in self._hypotheses():
            nodes.append({"id": h.get("hypothesis_id"), "type": "hypothesis",
                          "domain": h.get("domain"),
                          "claim_label": "[HYPOTHESIS]",
                          "status": h.get("status"),
                          "assumptions": h.get("assumptions") or []})
        for sim in self._simulations():
            nodes.append({"id": sim.get("experiment_id"), "type": "simulation",
                          "label": "[SIMULATION RESULT] not real-world evidence",
                          "status": sim.get("status")})
        for f in self._failures():
            nodes.append({"id": f.get("failure_id"), "type": "failure",
                          "failure_type": f.get("failure_type")})
        for ls in self._lessons():
            nodes.append({"id": ls.get("lesson_id"), "type": "lesson"})
        for r in self.state.get("real_world", {}).get("proposals", []):
            nodes.append({"id": r.get("proposal_id"), "type": "real_world_proposal",
                          "status": r.get("status")})
        for r in self._rwr():
            nodes.append({"id": r.get("result_id"), "type": "real_world_result",
                          "label": "[REAL-WORLD RESULT] human-provided",
                          "researcher": r.get("researcher", "unknown")})

        payload = {
            "generated_at": self.state.get("generated_at", ""),
            "research_question": self._q(),
            "primary_domain": self.state.get("primary_domain", ""),
            "node_count": len(nodes),
            "edge_count": len(self._graph_edges()),
            "claim_discipline": {
                "FACT": "sourced knowledge (papers / researcher data)",
                "INFERENCE": "derived from multiple evidence pieces",
                "HYPOTHESIS": "AI- or researcher-generated proposal",
                "SIMULATION RESULT": "model output — never real-world evidence",
                "REAL-WORLD RESULT": "human-provided observation",
            },
            "nodes": nodes,
            "edges": [e.strip() for e in self._graph_edges()],
            "note": ("Counts and edges are descriptive only; they are never used as "
                     "evidence strength or hypothesis confidence."),
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        self.paths["graph.json"] = path
        return path

    def _write(self, filename: str, text: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        self.paths[filename] = path
        return path

    # ------------------------------------------------------------- helpers
    def _q(self) -> str:
        return str(self.state.get("research_question", "") or "")

    def _header(self, title: str) -> List[str]:
        return [_RULE, f"AARL — {title}", _RULE, "", f"Research question: {self._q()}",
                f"Generated: {self.state.get('generated_at', '')}",
                "Source: MASTER RESEARCH STATE (research_memory/master_state.json)", ""]

    def _domain_names(self) -> List[str]:
        return [
            str(d.get("domain", d) if isinstance(d, dict) else d)
            for d in self.state.get("relevant_domains", [])
        ]

    def _papers(self) -> List[Dict[str, Any]]:
        return list(self.state.get("papers") or [])

    def _hypotheses(self) -> List[Dict[str, Any]]:
        return list(self.state.get("hypotheses") or [])

    def _simulations(self) -> List[Dict[str, Any]]:
        return list(self.state.get("simulations") or [])

    def _failures(self) -> List[Dict[str, Any]]:
        return list(self.state.get("failures") or [])

    def _lessons(self) -> List[Dict[str, Any]]:
        return list(self.state.get("lessons") or [])

    def _best(self) -> Dict[str, Any]:
        return dict(self.state.get("best_candidate") or {})

    def _rwr(self) -> List[Dict[str, Any]]:
        return list((self.state.get("real_world") or {}).get("results") or [])

    # ============================================================ report 01
    def report_research_summary(self) -> str:
        L = self._header("01 — RESEARCH SUMMARY")
        L += [
            "1. RESEARCH QUESTION", _SUB, self._q(), "",
            "2. RESEARCH OBJECTIVE", _SUB,
            "Determine, through literature-grounded hypothesis generation and "
            "simulation, the most promising candidate approach for the research "
            "question, and identify what must be validated in the real world.", "",
            "3. RELEVANT RESEARCH DOMAINS", _SUB,
        ]
        for d in self._domain_names():
            L.append(f"  - {d}")
        L += ["", "4. PAPERS ANALYSED", _SUB]
        papers = self._papers()
        if not papers:
            L.append("  (no papers ingested in this cycle)")
        for p in papers:
            authors = ", ".join(p.get("authors", [])[:4])
            L.append(
                f"  - {p.get('paper_id')}: {p.get('title')}"
                + (f" — {authors}" if authors else "")
                + f" [provided by {p.get('provided_by', 'unknown')}]"
            )
        L += ["", "  Researcher-provided papers:", _SUB]
        researcher_papers = [p for p in papers if p.get("provided_by") == "researcher"]
        if not researcher_papers:
            L.append("  (none)")
        for p in researcher_papers:
            L.append(f"  - {p.get('paper_id')}: {p.get('title')}")

        L += ["", "5. MAJOR FINDINGS", _SUB]
        findings = self.knowledge_items("findings")
        if not findings:
            L.append("  (none recorded)")
        for i in findings[:12]:
            L.append(f"  [FACT] {i['item_id']}: {i['statement'][:220]}")

        L += ["", "6. IMPORTANT EVIDENCE", _SUB]
        for i in self.knowledge_items("evidence")[:10]:
            L.append(f"  [FACT] {i['item_id']}: {i['statement'][:220]}")

        L += ["", "7. KEY RESEARCH GAPS", _SUB]
        for i in self.knowledge_items("limitations")[:10]:
            L.append(f"  - {i['statement'][:220]}")

        L += ["", "8. OVERALL RESEARCH PROGRESSION", _SUB]
        for it in self.state.get("iterations", []):
            for h in it.get("hypotheses", []):
                L.append(
                    f"  Iteration {it.get('iteration')}: {h.get('hypothesis_id')} -> "
                    f"{h.get('experiment_id')} [{h.get('status')}]"
                    + (f" -> {h.get('failure_id')} / {h.get('lesson_id')}" if h.get("failure_id") else "")
                )
        L += ["", _RULE, ""]
        return "\n".join(L)

    def knowledge_items(self, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Flatten the knowledge items embedded in the master state."""
        items = [
            i for i in (self.state.get("knowledge_items") or []) if isinstance(i, dict)
        ]
        if section is None:
            return items
        return [i for i in items if i.get("section") == section]

    def knowledge_get(self, item_id: str) -> Optional[Dict[str, Any]]:
        for i in self.knowledge_items():
            if i.get("item_id") == item_id:
                return i
        return None

    def memory_get(self, collection: str, record_id: str) -> Optional[Dict[str, Any]]:
        for r in self.state.get(collection, []) if isinstance(self.state.get(collection), list) else []:
            if r.get("record_id") == record_id or r.get("hypothesis_id") == record_id \
                    or r.get("experiment_id") == record_id:
                return r
        return None

    # ============================================================ report 02
    def report_final_design(self) -> str:
        L = self._header("02 — FINAL DESIGN REPORT")
        best = self._best()
        rwr = self._rwr()
        if not best:
            L += [
                "NO FINAL DESIGN CAN BE REPORTED YET.",
                "",
                "No simulation produced a measurable improvement over the baseline,",
                "so no candidate is selected. This report will not fabricate a design.",
                "Failed attempts remain recorded in 06 and 08.",
                "",
                _RULE, "",
            ]
            return "\n".join(L)

        hyp = self.memory_get("hypotheses", best.get("hypothesis_id", ""))
        L += [
            "1. FINAL HYPOTHESIS / DESIGN", _SUB,
            f"  [HYPOTHESIS] {best.get('hypothesis_id')}: {best.get('hypothesis')}", "",
            "2. WHY IT WAS SELECTED", _SUB,
            f"  - Highest measured improvement among all simulated candidates: "
            f"{best.get('best_improvement_pct')}% (best metric).",
            f"  - Selection basis: [SIMULATION RESULT] {best.get('status')} "
            f"({best.get('experiment_id')}).", "",
            "3. SUPPORTING RESEARCH", _SUB,
        ]
        supporting = (hyp or {}).get("supporting_evidence", [])
        if not supporting:
            L.append("  (no sourced knowledge items linked)")
        for item_id in supporting:
            item = self.knowledge_get(item_id)
            if item:
                L.append(f"  [FACT] {item_id}: {item['statement'][:220]}")

        L += ["", "4. SIMULATION EVIDENCE", _SUB,
              "  [SIMULATION RESULT] metrics from the stored Simulation JSON:"]
        shown = False
        for name, m in (best.get("metrics") or {}).items():
            if isinstance(m, dict) and "improvement_pct" in m:
                shown = True
                L.append(f"  - {name}: observed {m.get('observed_value')} vs baseline "
                         f"{m.get('baseline_value')} ({m.get('improvement_pct')}%)")
        if not shown:
            L.append("  (see the full Simulation JSON in 06_Simulation_Report)")

        L += ["", "5. PREVIOUS FAILED APPROACHES", _SUB]
        failures = self._failures()
        if not failures:
            L.append("  (none)")
        for f in failures:
            L.append(f"  - {f.get('failure_id')} ({f.get('failure_type')}): {str(f.get('cause'))[:180]}")

        L += ["", "6. IMPROVEMENTS MADE OVER ITERATIONS", _SUB]
        improvements = [
            (it, h) for it in self.state.get("iterations", [])
            for h in it.get("hypotheses", []) if h.get("lesson_id")
        ]
        if not improvements:
            L.append("  (first iteration — no lesson-driven improvements yet)")
        for it, h in improvements:
            L.append(
                f"  Iteration {it.get('iteration')}: {h.get('hypothesis_id')} applied "
                f"{h.get('lesson_id')} after {h.get('failure_id')}"
            )

        L += ["", "7. IMPORTANT ASSUMPTIONS", _SUB]
        assumptions = (hyp or {}).get("assumptions", [])
        if not assumptions:
            L.append("  (none declared in the stored hypothesis record)")
        for a in assumptions:
            L.append(f"  - {a}")

        L += ["", "8. REMAINING UNCERTAINTY", _SUB]
        if rwr:
            for r in rwr:
                for c in r.get("simulation_vs_real", []):
                    L.append(
                        f"  [REAL-WORLD RESULT] {c.get('metric')}: predicted "
                        f"{c.get('prediction')}, observed {c.get('real_world_result')}, "
                        f"error {c.get('error')}"
                    )
        else:
            L.append("  - REAL-WORLD VALIDATION IS PENDING (see the RWE proposal).")
            L.append("  - The selection is a SIMULATION RESULT; it must not be treated "
                     "as real-world evidence.")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # ============================================================ report 03
    def report_design_architecture(self) -> str:
        L = self._header("03 — DESIGN ARCHITECTURE")
        best = self._best()
        if not best:
            L += [
                "No architecture can be described: no candidate was selected by "
                "simulation. See 02_Final_Design_Report.", "", _RULE, "",
            ]
            return "\n".join(L)
        hyp = self.memory_get("hypotheses", best.get("hypothesis_id", ""))
        sim = self.memory_get("simulations", best.get("experiment_id", ""))
        params = (hyp or {}).get("parameters") or (sim or {}).get("parameters") or {}

        L += [
            "1. SYSTEM STRUCTURE", _SUB,
            "  Derived strictly from the selected candidate's parameters and the",
            "  simulation record — no invented components.", "",
            "2. COMPONENTS AND PARAMETERS", _SUB,
        ]
        if params:
            for key, value in params.items():
                L.append(f"  - {key}: {value}")
        else:
            L.append("  (the hypothesis declared no explicit parameters; see its record)")

        L += ["", "3. CONNECTIONS / DATA FLOW", _SUB]
        for step in [
            "Baseline configuration -> measured reference metrics",
            "Hypothesis parameters -> modified configuration -> measured metrics",
            "Comparison (baseline vs hypothesis) -> improvement per metric",
        ]:
            L.append(f"  - {step}")
        obs = (sim or {}).get("observed_results") or {}
        if isinstance(obs, dict) and obs.get("applied_modifications"):
            L += ["", "4. APPLIED MODIFICATIONS (from the Simulation JSON)", _SUB]
            for key, value in dict(obs["applied_modifications"]).items():
                L.append(f"  - {key}: {value}")

        L += ["", "5. DESIGN DECISIONS", _SUB]
        decisions = []
        for f in self._failures():
            for ch in f.get("recommended_changes", [])[:2]:
                decisions.append(f"{ch} (from {f.get('failure_id')})")
        if not decisions:
            L.append("  (no failure-driven decisions recorded)")
        for d in decisions[:8]:
            L.append(f"  - {d}")

        L += ["", "6. CHANGES FROM PREVIOUS ITERATIONS", _SUB]
        parent_id = (hyp or {}).get("parent_hypothesis", "")
        parent = self.memory_get("hypotheses", parent_id) if parent_id else None
        if parent:
            before, after = parent.get("parameters") or {}, (hyp or {}).get("parameters") or {}
            keys = set(before) | set(after)
            changed = [k for k in sorted(keys) if before.get(k) != after.get(k)]
            L.append(f"  - Parent: {parent_id}")
            for key in changed:
                L.append(f"    - '{key}': {before.get(key)} -> {after.get(key)}")
            if not changed:
                L.append("    (parameter values unchanged)")
        else:
            L.append("  (first candidate in this cycle)")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # ============================================================ report 04
    def report_engineering_calculations(self) -> str:
        L = self._header("04 — ENGINEERING CALCULATIONS")
        L += [
            "Every value below is copied from a stored record (equations from",
            "sourced knowledge; calculated values from Simulation JSONs).",
            "No numerical result is invented here.", "",
            "1. EQUATIONS (sourced)", _SUB,
        ]
        equations = self.knowledge_items("equations")
        if not equations:
            L.append("  (no equations extracted from papers yet)")
        for i in equations[:12]:
            prov = i.get("provenance") or {}
            L.append(f"  [FACT] {i['item_id']}: {i['statement'][:200]}")
            L.append(f"         source: {prov.get('citation', i.get('source_paper', ''))}")

        L += ["", "2. VARIABLES AND INPUT VALUES (declared hypothesis parameters)", _SUB]
        best = self._best()
        if best:
            hyp = self.memory_get("hypotheses", best.get("hypothesis_id", ""))
            params = (hyp or {}).get("parameters") or {}
            if params:
                for key, value in params.items():
                    L.append(f"  - {key} = {value}")
            else:
                L.append("  (the selected hypothesis declared no explicit parameters)")
        else:
            L.append("  (no selected candidate)")

        L += ["", "3. CALCULATED VALUES (simulation outputs) [SIMULATION RESULT]", _SUB]
        if not self._simulations():
            L.append("  (no simulations recorded)")
        for s in self._simulations():
            if str(s.get("status", "")).lower() == "simulation_unavailable":
                L.append(f"  - {s.get('experiment_id')}: simulation_unavailable "
                         "(no calculated values — none may be claimed)")
                continue
            for name, m in (s.get("metrics") or {}).items():
                if isinstance(m, dict) and "observed_value" in m:
                    L.append(
                        f"  - {s.get('experiment_id')} {name}: baseline "
                        f"{m.get('baseline_value')} -> observed {m.get('observed_value')} "
                        f"({m.get('improvement_pct')}%) unit={m.get('unit', '')}"
                    )

        L += ["", "4. ASSUMPTIONS USED IN CALCULATIONS", _SUB]
        assumptions = self.knowledge_items("assumptions")
        if not assumptions:
            L.append("  (none recorded)")
        for i in assumptions[:10]:
            L.append(f"  - {i['statement'][:200]}")

        L += ["", "5. EXPECTED RANGES AND VALIDATION STATUS", _SUB]
        rwr = self._rwr()
        if rwr:
            for r in rwr:
                for c in r.get("simulation_vs_real", []):
                    L.append(
                        f"  - {c.get('metric')}: predicted {c.get('prediction')} vs "
                        f"real {c.get('real_world_result')} -> error {c.get('error')} "
                        "[REAL-WORLD RESULT]"
                    )
        else:
            L.append("  - NOT YET VALIDATED IN THE REAL WORLD (pending RWE execution).")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # ============================================================ report 05
    def report_materials_technologies(self) -> str:
        L = self._header("05 — MATERIALS AND TECHNOLOGIES")
        domain = str(self.state.get("primary_domain", "") or "")
        methods = self.knowledge_items("methods")
        physical = domain in ("chemistry", "materials", "battery", "aerospace",
                              "robotics", "civil_engineering", "medicine")
        if not physical:
            L += [
                f"Primary domain: {domain or 'unspecified'}.",
                "",
                "This research does not primarily involve physical materials, so this",
                "report lists the METHODS AND TECHNOLOGIES the research relies on",
                "instead of manufacturing details (no irrelevant material lists are",
                "invented).", "",
            ]
        else:
            L += [
                f"Primary domain: {domain}. Materials/technologies below are taken",
                "from sourced knowledge items only.", "",
            ]
        L += ["1. METHODS / TECHNOLOGIES (sourced from papers)", _SUB]
        if not methods:
            L.append("  (no methods extracted from papers yet)")
        for i in methods[:12]:
            prov = i.get("provenance") or {}
            L.append(f"  [FACT] {i['item_id']}: {i['statement'][:200]}")
            if prov.get("citation"):
                L.append(f"         source: {prov['citation']}")

        L += ["", "2. SELECTED APPROACH AND ITS EVIDENCE", _SUB]
        best = self._best()
        if best:
            L.append(f"  - {best.get('hypothesis_id')}: {best.get('hypothesis')[:220]}")
            L.append(f"  - Simulation evidence: {best.get('experiment_id')} "
                     f"[SIMULATION RESULT, {best.get('best_improvement_pct')}% best metric]")
        else:
            L.append("  (no candidate selected)")

        L += ["", "3. ALTERNATIVES CONSIDERED", _SUB]
        hyps = self._hypotheses()
        if len(hyps) <= 1:
            L.append("  (no alternatives generated)")
        for h in hyps:
            if best and h.get("hypothesis_id") == best.get("hypothesis_id"):
                continue
            L.append(f"  - {h.get('hypothesis_id')}: {str(h.get('hypothesis'))[:200]}")

        L += ["", "4. LIMITATIONS OF THE SELECTED APPROACH", _SUB]
        lims = self.knowledge_items("limitations")
        if not lims:
            L.append("  (none recorded)")
        for i in lims[:8]:
            L.append(f"  - {i['statement'][:200]}")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # =============================================== research-status discipline
    # Every important result carries exactly ONE status level.  Levels are never
    # promoted automatically: only a human-provided real-world result can move a
    # claim to EXPERIMENT / VALIDATED.
    #
    #   IDEA       AARL-generated research direction
    #   PREDICTION expected result derived from assumptions or a predictive model
    #   SIMULATION result produced by an EXECUTED computational simulation
    #   EXPERIMENT result measured through a controlled real experiment
    #   VALIDATED  result with sufficient experimental evidence to satisfy the
    #              predefined validation criteria
    STATUS_LEVELS = ("IDEA", "PREDICTION", "SIMULATION", "EXPERIMENT", "VALIDATED")

    _CLAIM_LABEL = {
        "sourced": "[FACT]",
        "inferred": "[INFERENCE]",
        "generated_hypothesis": "[HYPOTHESIS]",
        "simulation_observation": "[SIMULATION RESULT]",
        "real_world_observation": "[REAL-WORLD RESULT]",
    }

    @staticmethod
    def _adapter_label(adapter: Any) -> str:
        """Human-readable simulation-engine label (adapter is a dict)."""
        if isinstance(adapter, dict) and adapter:
            name = str(adapter.get("name", "") or "unnamed")
            version = str(adapter.get("version", "") or "")
            module = str(adapter.get("module", "") or "")
            label = f"{name}{(' v' + version) if version else ''}"
            return f"{label} [{module}]" if module else label
        return str(adapter or "n/a")

    def _label(self, claim_type: str) -> str:
        return self._CLAIM_LABEL.get(str(claim_type or ""), "[UNCLASSIFIED]")

    # ------------------------------------------- evidence -> hypothesis mapping
    def evidence_for_hypothesis(self, hyp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Only the knowledge items PROVENANCE-LINKED to this hypothesis.

        A knowledge item qualifies as evidence for a hypothesis only when the
        stored hypothesis record lists it in ``supporting_evidence`` or
        ``contradictory_evidence``.  Large knowledge bases, equations or graph
        node counts are NEVER converted into evidence strength.
        """
        linked_support = set(hyp.get("supporting_evidence") or [])
        linked_contra = set(hyp.get("contradictory_evidence") or [])
        linked = linked_support | linked_contra
        if not linked:
            return []
        out: List[Dict[str, Any]] = []
        for item in self.knowledge_items():
            item_id = item.get("item_id")
            if item_id not in linked:
                continue
            prov = item.get("provenance") or {}
            claim_type = str(item.get("claim_type") or "")
            out.append({
                "evidence_id": item_id,
                "evidence_type": claim_type,
                "source": (prov.get("citation") or item.get("source")
                           or item.get("source_paper") or prov.get("origin") or ""),
                "source_locator": prov.get("locator") or prov.get("source_locator") or "",
                "domain": prov.get("domain") or (item.get("metadata") or {}).get("domain", ""),
                "relevance_to_hypothesis": "DIRECT",
                "supports_or_contradicts": (
                    "contradicts" if item_id in linked_contra else "supports"
                ),
                "verification_status": (
                    "REAL_WORLD" if claim_type == "real_world_observation" else "UNVERIFIED"
                ),
                "statement": str(item.get("statement") or ""),
                "confidence": float(item.get("confidence") or 0.0),
                "claim_type": claim_type,
            })
        return out

    def _simulation_for_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """Best-status Simulation JSON recorded for a hypothesis (never merged)."""
        rank = {"success": 3, "promising": 3, "completed": 3,
                "partial": 2, "failure": 1, "simulation_unavailable": 0}
        best: Dict[str, Any] = {}
        for sim in self._simulations():
            if sim.get("hypothesis_id") != hypothesis_id:
                continue
            if not best or rank.get(str(sim.get("status", "")).lower(), -1) > \
                    rank.get(str(best.get("status", "")).lower(), -1):
                best = sim
        return best

    # ------------------------------------------------------------ confidence
    def _confidence_breakdown(self, hyp: Dict[str, Any],
                              evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic, evidence-gated confidence, guaranteed within [0, 10].

            final = novelty*0.15 + evidence*0.25 + feasibility*0.15
                  + consistency*0.20 + testability*0.15 + (10-risk)*0.10

        Every dimension is clamped to [1, 10] and the result to [0, 10], so an
        impossible score such as 30/10 cannot occur.  ``risk`` enters only as
        ``(10 - risk)`` and therefore can only REDUCE confidence.

        Missing evidence is not neutral: it is penalised and the total is capped
        while the evidence status is INSUFFICIENT_DIRECT_EVIDENCE.  Contradictory
        evidence lowers both the evidence and consistency dimensions.
        """
        supporting = [e for e in evidence if e["supports_or_contradicts"] == "supports"]
        contra = [e for e in evidence if e["supports_or_contradicts"] == "contradicts"]

        if not supporting:
            evidence_dim, evidence_status = 1, "INSUFFICIENT_DIRECT_EVIDENCE"
        else:
            evidence_dim = 1 + min(9, int(round(
                max(e["confidence"] for e in supporting) * 10)))
            evidence_status = "UNVERIFIED_DIRECT_EVIDENCE"
        if contra:
            evidence_dim = max(1, evidence_dim - 2 * len(contra))

        # consistency: contradictory evidence directly reduces agreement
        consistency = 10 if not contra else max(1, 10 - 3 * len(contra))

        # testability: only counts when the hypothesis names measurable metrics
        testability = 8 if (hyp.get("measurable_metrics") or []) else 3

        # novelty: derived from whether the hypothesis builds on earlier lessons
        novelty = 6 if (hyp.get("based_on_lessons") or hyp.get("parent_hypothesis")) else 4

        # risk: 3 base + 1.5 per declared risk (higher risk reduces confidence)
        risk = min(10, 3 + int(round(1.5 * len(hyp.get("risks") or []))))

        # feasibility: from the recorded simulation status (no simulation = low)
        sim = self._simulation_for_hypothesis(str(hyp.get("hypothesis_id", "")))
        sim_status = str(sim.get("status", "") or "").lower()
        feasibility = {"success": 9, "promising": 9, "completed": 9,
                       "partial": 6, "failure": 2,
                       "simulation_unavailable": 1}.get(sim_status, 1)

        dims = {"novelty": novelty, "evidence": evidence_dim,
                "feasibility": feasibility, "consistency": consistency,
                "testability": testability, "risk": risk}
        dims = {k: max(1, min(10, int(v))) for k, v in dims.items()}

        weighted = (dims["novelty"] * 0.15 + dims["evidence"] * 0.25
                    + dims["feasibility"] * 0.15 + dims["consistency"] * 0.20
                    + dims["testability"] * 0.15 + (10 - dims["risk"]) * 0.10)
        final = round(max(0.0, min(10.0, weighted)), 2)

        # Evidence gating: no directly-linked evidence caps the score.
        capped = False
        if evidence_status == "INSUFFICIENT_DIRECT_EVIDENCE" and final > 4.0:
            final, capped = 4.0, True
        # Contradiction gating: cannot be "strong" while contradictions remain.
        if contra and final > 6.0:
            final, capped = 6.0, True

        return {
            "dimensions": dims,
            "final_score": final,
            "capped": capped,
            "evidence_status": evidence_status,
            "n_supporting": len(supporting),
            "n_contradicting": len(contra),
            "simulation_status": sim_status or "none",
            "verdict": (
                "UNVERIFIED — requires experimental validation" if final < 3.5 else
                "SPECULATIVE" if final < 5.5 else
                "PRELIMINARY / PLAUSIBLE" if final < 7.5 else
                "PRELIMINARY — promising, still unvalidated"
            ),
        }

    # ------------------------------------------------------- primary weakness
    def primary_weakness(self, hyp: Dict[str, Any], evidence: List[Dict[str, Any]],
                         breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Weakness with an explicit basis.  Never asserted as established fact."""
        contra = [e for e in evidence if e["supports_or_contradicts"] == "contradicts"]
        sim = self._simulation_for_hypothesis(str(hyp.get("hypothesis_id", "")))
        if contra:
            return {
                "weakness": f"Directly contradictory evidence exists ({len(contra)} item(s)).",
                "basis": "Knowledge items linked as 'contradictory_evidence' to this hypothesis.",
                "evidence": [e["evidence_id"] for e in contra],
                "confidence": "HIGH (for the existence of the contradiction)",
                "status": "DEMONSTRATED (a stored contradiction record exists)",
                "verification_required": "Resolve the contradiction experimentally.",
            }
        if breakdown["evidence_status"] == "INSUFFICIENT_DIRECT_EVIDENCE":
            return {
                "weakness": "No directly linked evidence; the hypothesis rests on assumptions.",
                "basis": "The hypothesis record links zero supporting knowledge items.",
                "evidence": [],
                "confidence": "HIGH (for the absence of linked evidence)",
                "status": "DEMONSTRATED (absence of provenance links)",
                "verification_required": ("Source or produce direct evidence before "
                                          "relying on this idea."),
            }
        if str(sim.get("status", "")).lower() in ("failure", "simulation_unavailable", "partial"):
            return {
                "weakness": f"Simulation outcome was '{sim.get('status')}'.",
                "basis": (f"Simulation JSON {sim.get('experiment_id', '')} "
                          f"(constraints violated: "
                          f"{sim.get('constraints_violated') or 'none recorded'})."),
                "evidence": sim.get("errors") or [],
                "confidence": "MEDIUM (single simulation, one parameter set)",
                "status": "SIMULATION (not experimentally confirmed)",
                "verification_required": "Confirm under real experimental conditions.",
            }
        return {
            "weakness": "No demonstrated weakness identified from stored records.",
            "basis": "No contradictions, no failing simulation and linked evidence exists.",
            "evidence": [],
            "confidence": "LOW",
            "status": "INFERRED / UNVERIFIED",
            "verification_required": "Adversarial testing of the assumption set.",
        }

    # ------------------------------------------------------- consistency check
    def consistency_checks(self) -> List[str]:
        """Run BEFORE the final synthesis (safety rule 12): problems found.

        Each finding forces the affected claim to be downgraded rather than
        reported more strongly than the stored data supports.
        """
        problems: List[str] = []
        for h in self._hypotheses():
            breakdown = self._confidence_breakdown(h, self.evidence_for_hypothesis(h))
            score = breakdown["final_score"]
            if not (0.0 <= score <= 10.0):
                problems.append(
                    f"{h.get('hypothesis_id')}: confidence {score} outside [0,10]")
            if not breakdown["n_supporting"] and score > 4.0:
                problems.append(
                    f"{h.get('hypothesis_id')}: confidence {score} despite no direct evidence")
            if breakdown["n_contradicting"] and score > 6.0:
                problems.append(
                    f"{h.get('hypothesis_id')}: confidence {score} despite contradictions")
        # simulation statuses must stay within the declared vocabulary
        for sim in self._simulations():
            if str(sim.get("status", "")).lower() not in (
                    "success", "failure", "partial", "simulation_unavailable"):
                problems.append(
                    f"{sim.get('experiment_id')}: unknown simulation status "
                    f"'{sim.get('status')}'")
        # an unvalidated cycle must not claim real-world validation
        if not (self.state.get("real_world") or {}).get("validated"):
            for r in self.state.get("iterations", []):
                for h in r.get("hypotheses", []):
                    if str(h.get("status", "")).lower() in ("validated", "experiment"):
                        problems.append(
                            f"{h.get('hypothesis_id')}: labelled '{h.get('status')}' "
                            "without a real-world result")
        return problems

    # ============================================================ report 06
    @staticmethod
    def _kv_block(title: str, data: Any, indent: str = "      ") -> List[str]:
        """Render a stored dict/list without altering any stored value."""
        out = [f"  {title}:"]
        if isinstance(data, dict) and data:
            for k, v in data.items():
                out.append(f"{indent}{k}: {v}")
        elif isinstance(data, (list, tuple)) and data:
            for v in data:
                out.append(f"{indent}- {v}")
        else:
            out.append(f"{indent}(none recorded)")
        return out

    def report_simulation(self) -> str:
        """Generated DIRECTLY from Simulation JSON records (no invented results)."""
        L = self._header("06 — SIMULATION REPORT")
        sims = self._simulations()

        if not sims:
            L += [
                "1. SIMULATION AVAILABILITY", _SUB,
                "  simulation_unavailable",
                "  No Simulation JSON record exists for this research cycle. AARL does",
                "  NOT claim that any simulation was performed, and no simulated",
                "  result is reported below.", "",
                _RULE, "",
            ]
            return "\n".join(L)

        L += [
            "1. SIMULATION RECORDS (verbatim from Simulation JSON)", _SUB,
            "  Every value below is copied from research_memory/simulations/. Values",
            "  are NOT rounded, averaged or re-derived.", "",
        ]
        for sim in sims:
            status = str(sim.get("status", "")).lower()
            level = ("SIMULATION" if status in ("success", "failure", "partial")
                     else "PREDICTION" if status == "simulation_unavailable" else "IDEA")
            L += [
                _SUB,
                f"  Experiment ID:    {sim.get('experiment_id', '')}",
                f"  Hypothesis ID:    {sim.get('hypothesis_id', '')}",
                f"  Simulation type:  {sim.get('simulation_type', '')}",
                f"  Adapter:          {self._adapter_label(sim.get('adapter'))}",
                f"  Status:           {sim.get('status', '')}",
                f"  Research status:  {level}",
                f"  Iteration:        {sim.get('iteration', '')}",
                f"  Timestamp:        {sim.get('timestamp', '')}",
                "",
            ]
            L += self._kv_block("Parameters", sim.get("parameters"))
            L += self._kv_block("Expected results", sim.get("expected_results"))
            L += self._kv_block("Observed results", sim.get("observed_results"))
            L += ["  Metrics (per metric — never averaged):"]
            metrics = sim.get("metrics") or {}
            if isinstance(metrics, dict) and metrics:
                for name, m in metrics.items():
                    if isinstance(m, dict):
                        L.append(
                            f"      {name}: baseline={m.get('baseline_value')} "
                            f"observed={m.get('observed_value')} "
                            f"change={m.get('improvement_pct')}%"
                        )
                    else:
                        L.append(f"      {name}: {m}")
            else:
                L.append("      (none recorded)")
            L += self._kv_block("Errors", sim.get("errors"))
            L += self._kv_block("Constraints violated", sim.get("constraints_violated"))
            L += self._kv_block("Unexpected results", sim.get("unexpected_results"))
            repro = sim.get("reproducibility")
            if repro:
                L += self._kv_block("Reproducibility", repro)
            if sim.get("notes"):
                L += [f"  Notes: {sim.get('notes')}"]
            L.append("")

        L += ["2. ITERATION HISTORY (how the research improved over time)", _SUB]
        iterations = self.state.get("iterations", [])
        if not iterations:
            L.append("  (no iteration records)")
        for it in iterations:
            L.append(f"  Iteration {it.get('iteration')}:")
            for h in it.get("hypotheses", []):
                outcome = str(h.get("status", "") or "")
                L.append(f"    - {h.get('hypothesis_id')} -> {h.get('experiment_id')} "
                         f"[{outcome}]")
                if h.get("failure_id") or h.get("lesson_id"):
                    L.append(f"        failure={h.get('failure_id') or '-'} "
                             f"lesson={h.get('lesson_id') or '-'}")
        L += ["", "3. SIMULATION AVAILABILITY PER DOMAIN", _SUB]
        avail = (self.state.get("simulation_availability") or {})
        engines = avail.get("engines") or []
        L.append(f"  Registered engines: {', '.join(str(e) for e in engines) if engines else 'none'}")
        if avail.get("note"):
            L.append(f"  Note: {avail.get('note')}")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # ============================================================ report 07
    def evidence_eligibility(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Domain/relevance gate applied BEFORE evidence scoring.

            evidence -> domain recorded? -> matches primary domain or hypothesis?
                     -> HYPOTHESIS LINK present? -> ELIGIBLE / NOT ELIGIBLE

        An item that is provenance-linked but sits in an unrelated domain is
        labelled ``VALID_BUT_NOT_RELEVANT`` and EXCLUDED from confidence.  A
        mathematically valid equation from the wrong discipline is not evidence.
        """
        primary = str(self.state.get("primary_domain", "") or "").lower()
        best = self._best()
        hyp_text = " ".join(
            str(v).lower() for v in (
                best.get("hypothesis", ""),
                (self.memory_get("hypotheses", best.get("hypothesis_id", "")) or {})
                .get("hypothesis", ""),
            )
        )
        eligible, excluded = [], []
        for e in evidence:
            item_domain = str(e.get("domain") or "").strip().lower()
            if not item_domain:
                eligible.append(e)                      # domain unknown -> link is enough
                continue
            if item_domain == primary or item_domain in hyp_text or primary in item_domain:
                eligible.append(e)
                continue
            e = dict(e)
            e["relevance_to_hypothesis"] = "VALID_BUT_NOT_RELEVANT"
            e["exclusion_reason"] = (
                f"evidence domain '{item_domain}' does not match the primary "
                f"research domain '{primary or 'unspecified'}'"
            )
            excluded.append(e)
        return {"eligible": eligible, "excluded": excluded}

    def _section_literature(self) -> List[str]:
        """Section 1 of report 07 — literature evidence, claim-type labelled."""
        L: List[str] = [
            "1. LITERATURE EVIDENCE", _SUB,
            "  Claim types are labelled explicitly. 'sourced' items come from supplied",
            "  documents only; 'inferred' items are model/rule reasoning and are NOT",
            "  established facts.", "",
        ]
        papers = self._papers()
        if not papers:
            L.append("  (no papers ingested in this cycle)")
        for p in papers:
            L.append(f"  [FACT] {p.get('paper_id')}: {p.get('title')} "
                     f"({p.get('year', 'n/a')}) provided_by={p.get('provided_by', 'unknown')}")
        for section in ("findings", "methods", "evidence", "limitations", "contradictions"):
            items = self.knowledge_items(section)
            L.append(f"  {section.upper()} ({len(items)} item(s)):")
            if not items:
                L.append("      (none recorded)")
            for i in items[:6]:
                L.append(f"      {self._label(i.get('claim_type'))} {i.get('item_id')}: "
                         f"{str(i.get('statement'))[:160]}")
        L.append("")
        return L

    def focus_hypothesis(self) -> Dict[str, Any]:
        """The hypothesis the reports focus on (best candidate, else best score)."""
        best = self._best()
        focus_id = best.get("hypothesis_id") or ""
        focus = self.memory_get("hypotheses", focus_id) if focus_id else None
        if focus:
            return focus
        hypotheses = self._hypotheses()
        if not hypotheses:
            return {}
        return max(
            hypotheses,
            key=lambda h: self._confidence_breakdown(
                h, self.evidence_for_hypothesis(h))["final_score"],
        )

    def _section_evidence_mapping(self) -> List[str]:
        """Section 2 of report 07 — hypothesis <-> evidence mapping."""
        L: List[str] = ["2. HYPOTHESIS AND EVIDENCE MAPPING", _SUB]
        focus = self.focus_hypothesis()
        if not focus:
            L += ["  No hypotheses were generated in this cycle.", ""]
            return L

        linked = self.evidence_for_hypothesis(focus)
        gate = self.evidence_eligibility(linked)
        eligible = gate["eligible"]
        breakdown = self._confidence_breakdown(focus, eligible)

        L += [
            f"  Focus hypothesis: {focus.get('hypothesis_id')}",
            f"  Statement: {str(focus.get('hypothesis'))[:300]}",
            "",
            "  Evidence-eligibility gate (relevance is checked BEFORE scoring):",
            f"      provenance-linked items : {len(linked)}",
            f"      eligible (relevant)     : {len(eligible)}",
            f"      VALID_BUT_NOT_RELEVANT  : {len(gate['excluded'])}",
            "",
            "  Evidence items (only directly relevant, provenance-linked items count):",
        ]
        if not eligible:
            L.append("      (none) — Evidence Status: INSUFFICIENT_DIRECT_EVIDENCE")
        for e in eligible:
            L += [
                f"      evidence_id            : {e['evidence_id']}",
                f"      evidence_type          : {e['evidence_type']}",
                f"      source                 : {e['source'] or '(unrecorded)'}",
                f"      source_locator         : {e['source_locator'] or '(unrecorded)'}",
                f"      domain                 : {e['domain'] or '(unrecorded)'}",
                f"      relevance_to_hypothesis: {e['relevance_to_hypothesis']}",
                f"      supports_or_contradicts: {e['supports_or_contradicts']}",
                f"      verification_status    : {e['verification_status']}",
                "",
            ]
        if gate["excluded"]:
            L.append("  Excluded from confidence (VALID_BUT_NOT_RELEVANT):")
            for e in gate["excluded"]:
                L.append(f"      {e['evidence_id']} [{e['evidence_type']}] "
                         f"domain={e['domain']} — {e['exclusion_reason']}")
            L.append("")

        L += [
            "  Evidence score derivation (document/equation counts are NEVER used):",
            f"      Evidence Score  : {breakdown['dimensions']['evidence']}/10",
            f"      Evidence Status : {breakdown['evidence_status']}",
            f"      Supporting      : {breakdown['n_supporting']} item(s)",
            f"      Contradicting   : {breakdown['n_contradicting']} item(s)",
        ]
        if breakdown["evidence_status"] == "INSUFFICIENT_DIRECT_EVIDENCE":
            L += [
                "      NOTE: no directly relevant evidence exists, so the Evidence",
                "      dimension is 1/10 and the overall confidence score is capped.",
            ]
        L.append("")
        return L

    def _section_math_validation(self) -> List[str]:
        """Section 3 of report 07 — mathematical validity vs research relevance."""
        L: List[str] = [
            "3. MATHEMATICAL VALIDATION", _SUB,
            "  Mathematical validity is reported SEPARATELY from research relevance.",
            "  A valid equation unrelated to the hypothesis is NOT evidence for that",
            "  hypothesis, and never contributes to its confidence.", "",
        ]
        focus = self.focus_hypothesis()
        equations = self.knowledge_items("equations")
        linked_ids = set((focus.get("supporting_evidence") or [])
                         + (focus.get("contradictory_evidence") or []))
        linked_eq = [i for i in equations if i.get("item_id") in linked_ids]
        rejected = sum(
            1 for i in equations
            if str((i.get("metadata") or {}).get("validated", "")).lower()
            in ("false", "no", "rejected")
        )
        L += [
            "  Mathematical Validation:",
            f"      Equation items in knowledge base : {len(equations)}",
            f"      Rejected equations               : {rejected}",
            "      Mathematical validity status     : NOT COMPUTED IN THIS LAYER",
            "",
            "  Research Relevance:",
            f"      Relevant equations linked to the current hypothesis : {len(linked_eq)}",
            f"      Irrelevant / unlinked equations                     : "
            f"{max(0, len(equations) - len(linked_eq))}",
            "",
            "  No mathematical validation engine is attached to this layer, so no",
            "  equation is promoted to research evidence. Equation counts are shown",
            "  for traceability only and are EXPLICITLY EXCLUDED from confidence.",
            "",
        ]
        return L

    def _section_simulation(self) -> List[str]:
        """Section 4 of report 07 — executed simulation vs assumption projection."""
        L: List[str] = ["4. PRELIMINARY SIMULATION / PREDICTION", _SUB]
        focus = self.focus_hypothesis()
        sim = self._simulation_for_hypothesis(str(focus.get("hypothesis_id", "")))
        if not sim:
            L += [
                "  Simulation Type: SIMULATION_UNAVAILABLE",
                "  No executable simulation was run for this hypothesis. No projected",
                "  numbers are reported, because AARL does not fabricate results.",
                "  Experimental Status: NOT EXPERIMENTALLY VALIDATED", "",
            ]
            return L
        status = str(sim.get("status", "") or "").lower()
        sim_type = str(sim.get("simulation_type", "") or "")
        if status == "simulation_unavailable":
            L += [
                f"  Experiment ID: {sim.get('experiment_id', '')}",
                "  Simulation Type: SIMULATION_UNAVAILABLE",
                "  The router found no engine able to model this hypothesis. AARL",
                "  records this explicitly instead of pretending a simulation ran.",
                "  Experimental Status: NOT EXPERIMENTALLY VALIDATED", "",
            ]
            return L

        L += [
            "  Simulation Type: EXECUTED_SIMULATION",
            f"  Experiment ID:   {sim.get('experiment_id', '')}",
            f"  Adapter:         {self._adapter_label(sim.get('adapter'))}",
            f"  Model:           {sim_type or '(unnamed)'}",
            "  Research status: SIMULATION (a computational model was executed)",
            "  Experimental Status: NOT EXPERIMENTALLY VALIDATED",
            "",
        ]
        L += self._kv_block("Inputs (parameters)", sim.get("parameters"))
        L += self._kv_block("Expected results", sim.get("expected_results"))
        L += self._kv_block("Observed results", sim.get("observed_results"))
        L += self._kv_block("Runs / seed / reproducibility", sim.get("reproducibility"))
        L += ["  Metrics (each metric reported separately — never averaged):"]
        metrics = sim.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            for name, m in metrics.items():
                if isinstance(m, dict):
                    L.append(f"      {name}:")
                    L.append(f"        baseline:   {m.get('baseline_value')}")
                    L.append(f"        hypothesis: {m.get('observed_value')}")
                    L.append(f"        change:     {m.get('improvement_pct')}%")
                else:
                    L.append(f"      {name}: {m}")
            L += [
                "",
                "  NOTE: no single 'Average Improvement' is computed. Averaging",
                "  percentages from unrelated metrics is scientifically invalid; the",
                "  per-metric changes above must be read individually.",
            ]
        else:
            L.append("      (none recorded)")
        L += self._kv_block("Errors", sim.get("errors"))
        L += self._kv_block("Constraints violated", sim.get("constraints_violated"))
        L += self._kv_block("Unexpected results", sim.get("unexpected_results"))
        L.append("")
        return L

    def report_evidence_confidence(self) -> str:
        """07 — EVIDENCE AND CONFIDENCE (8-section, evidence-gated structure).

        Never presents assumptions, predictions, mathematical validity or
        preliminary simulation output as experimentally proven evidence.
        """
        L = self._header("07 — EVIDENCE AND CONFIDENCE")
        L += ["RESEARCH PROBLEM", _SUB, self._q() or "Not specified.", ""]

        focus = self.focus_hypothesis()
        linked = self.evidence_for_hypothesis(focus) if focus else []
        gate = self.evidence_eligibility(linked)
        eligible = gate["eligible"]
        breakdown = self._confidence_breakdown(focus, eligible) if focus else None

        L += self._section_literature()
        L += self._section_evidence_mapping()
        L += self._section_math_validation()
        L += self._section_simulation()

        # ---- 5. confidence assessment -------------------------------------
        L += ["5. CONFIDENCE ASSESSMENT", _SUB]
        if not focus or breakdown is None:
            L.append("  No hypotheses were generated, so no confidence is reported.")
        else:
            L += [
                f"  Focus hypothesis: {focus.get('hypothesis_id')}",
                f"  Final Confidence Score: {breakdown['final_score']:.2f}/10",
                f"  Confidence Verdict:     {breakdown['verdict']}",
                f"  Research status:        {self._research_status(focus, breakdown)}",
                "",
                "  Explicit, deterministic aggregation formula:",
                "      final_score = novelty*0.15 + evidence*0.25 + feasibility*0.15",
                "                  + consistency*0.20 + testability*0.15 + (10-risk)*0.10",
                "      every dimension clamped to [1,10]; result clamped to [0,10];",
                "      risk enters only as (10 - risk) and can ONLY reduce confidence.",
                "",
                "  Dimension breakdown:",
            ]
            for dim in ("novelty", "evidence", "feasibility",
                        "consistency", "testability", "risk"):
                L.append(f"      {dim.capitalize():15s}: {breakdown['dimensions'][dim]}/10")
            L += [
                f"      {'Final':15s}: {breakdown['final_score']:.2f}/10",
                "",
                "  Evidence gating:",
                f"      Evidence status : {breakdown['evidence_status']}",
                f"      score capped    : {'YES' if breakdown['capped'] else 'no'}",
            ]
            if breakdown["evidence_status"] == "INSUFFICIENT_DIRECT_EVIDENCE":
                L += [
                    "      No directly relevant evidence is linked to this hypothesis, so",
                    "      the Evidence dimension is 1/10 and the final score is capped at",
                    "      4.00/10.  The score is NOT a statement of scientific proof.",
                ]
            L.append("")

        # ---- 6. limitations ----------------------------------------------
        L += [
            "6. LIMITATIONS", _SUB,
            "  • Confidence is computed from stored records, not from replicated",
            "    experiments; it is an internal ranking aid only.",
            "  • Mathematical validity is reported separately from research relevance.",
            "  • Any simulation output is a model result, NOT real-world evidence.",
            "  • Cross-domain knowledge is filtered out when it is irrelevant.",
            "  • All evidence is UNVERIFIED unless a human provided real-world results.",
            "  • Absence of evidence is reported as INSUFFICIENT, never as support.",
            "",
        ]
        if focus and breakdown:
            pw = self.primary_weakness(focus, eligible, breakdown)
            L += [
                "  Primary Identified Weakness:",
                f"      Weakness    : {pw['weakness']}",
                f"      Basis       : {pw['basis']}",
                f"      Evidence    : {pw['evidence'] or '(none)'}",
                f"      Confidence  : {pw['confidence']}",
                f"      Status      : {pw['status']}",
                f"      Verification Required: {pw['verification_required']}",
                "",
            ]

        # ---- 7. what still requires human validation ----------------------
        L += ["7. WHAT STILL REQUIRES HUMAN RESEARCHER VALIDATION", _SUB]
        rwr = self._rwr()
        if rwr:
            for r in rwr:
                L.append(f"  [REAL-WORLD RESULT] {r.get('result_id')} by "
                         f"{r.get('researcher', 'unknown')}")
                for c in r.get("simulation_vs_real", []):
                    L += [
                        f"      metric {c.get('metric')}:",
                        f"        prediction       : {c.get('prediction')}",
                        f"        real_world_result: {c.get('real_world_result')}",
                        f"        error            : {c.get('error')}",
                    ]
        else:
            L += [
                "  Real-world validation: NOT YET AVAILABLE.",
                "  Status: REQUIRES_HUMAN_RESEARCHER_VALIDATION.",
                "  The following therefore remain UNVERIFIED:",
                "      • the candidate's predicted effect under real conditions;",
                "      • every assumption listed in the hypothesis record;",
                "      • the accuracy of the simulation model itself.",
            ]
        L.append("")

        # ---- 8. researcher handoff ----------------------------------------
        L += [
            "8. RESEARCHER HANDOFF", _SUB,
            "  AARL provides a candidate research direction and a preliminary",
            "  computational assessment. The result is NOT presented as a",
            "  scientifically validated discovery.", "",
            "  A researcher should independently review the evidence and the",
            "  confidence scoring, modify the proposed approach where appropriate,",
            "  and perform controlled experiments before drawing scientific",
            "  conclusions. AARL does not perform real-world experiments.", "",
        ]

        # ---- automatic consistency check before finalisation --------------
        problems = self.consistency_checks()
        if problems:
            L += ["CONSISTENCY CHECK (automatic downgrade applied)", _SUB,
                  "  The following issues were detected before finalisation:"]
            for p in problems:
                L.append(f"      - {p}")
            L += ["      Affected claims are reported as UNVERIFIED / PRELIMINARY.", ""]
        L += [_RULE, ""]
        return "\n".join(L)

    @staticmethod
    def _research_status(hyp: Dict[str, Any], breakdown: Dict[str, Any]) -> str:
        """IDEA / PREDICTION / SIMULATION / EXPERIMENT / VALIDATED (never promoted)."""
        evidence = hyp.get("supporting_evidence") or []
        if any(str(e).startswith("RWR") for e in evidence):
            return "EXPERIMENT"
        sim_status = breakdown.get("simulation_status", "none")
        if sim_status in ("success", "failure", "partial"):
            return "SIMULATION"
        if sim_status == "simulation_unavailable":
            return "PREDICTION"
        return "IDEA"

    # ============================================================ report 08
    def report_hypotheses_analysis(self) -> str:
        """08 — complete hypothesis evolution, iteration by iteration."""
        L = self._header("08 — HYPOTHESES ANALYSIS")
        hypotheses = self._hypotheses()
        if not hypotheses:
            L += ["No hypotheses were generated in this cycle.", "", _RULE, ""]
            return "\n".join(L)

        L += [
            "Every hypothesis is a [HYPOTHESIS] — an AARL-generated proposal, never",
            "an established fact. Its status is derived from the stored Simulation",
            "JSON, never from the wording of the hypothesis text.", "",
            "1. HYPOTHESIS EVOLUTION", _SUB,
        ]
        by_id = {h.get("hypothesis_id"): h for h in hypotheses}
        chain: List[str] = []
        for it in self.state.get("iterations", []):
            for entry in it.get("hypotheses", []):
                chain.append(entry.get("hypothesis_id", ""))
        if not chain:
            chain = [h.get("hypothesis_id", "") for h in hypotheses]
        for hid in chain:
            h = by_id.get(hid) or {}
            sim = self._simulation_for_hypothesis(str(hid))
            status = str(sim.get("status", "no simulation")).upper()
            L.append(f"  {hid}")
            L.append(f"    |-> Simulation ({sim.get('experiment_id') or 'n/a'})")
            L.append(f"    |-> {status}")
            for f in self._failures():
                if f.get("hypothesis_id") == hid:
                    L.append(f"    |-> Failure {f.get('failure_id')}: "
                             f"{str(f.get('failure_type'))}")
            for ls in self._lessons():
                if ls.get("hypothesis_id") == hid:
                    L.append(f"    |-> {ls.get('lesson_id')}: "
                             f"{str(ls.get('lesson'))[:140]}")
            L.append("")

        L += ["2. HYPOTHESIS DETAIL", _SUB]
        for h in hypotheses:
            L += self._hypothesis_detail_block(h, hypotheses)
        L += ["", "3. ITERATION SUMMARY", _SUB]
        for it in self.state.get("iterations", []):
            L.append(f"  Iteration {it.get('iteration')}: "
                     f"{len(it.get('hypotheses', []))} hypothesis(es), "
                     f"{len(it.get('experiments', []))} experiment(s)"
                     + (" [threshold met]" if it.get("threshold_met") else ""))
        L += ["", _RULE, ""]
        return "\n".join(L)

    def _hypothesis_detail_block(self, h: Dict[str, Any],
                                 hypotheses: List[Dict[str, Any]]) -> List[str]:
        """One hypothesis, in full, with its evidence, simulation and lesson chain."""
        hid = h.get("hypothesis_id")
        linked = self.evidence_for_hypothesis(h)
        gate = self.evidence_eligibility(linked)
        breakdown = self._confidence_breakdown(h, gate["eligible"])
        sim = self._simulation_for_hypothesis(str(hid))
        failures = [f for f in self._failures() if f.get("hypothesis_id") == hid]
        successors = [x for x in hypotheses if x.get("parent_hypothesis") == hid]

        L: List[str] = [
            _SUB,
            f"  Hypothesis ID:        {hid}",
            f"  Description:          {str(h.get('hypothesis'))[:300]}",
            f"  Iteration:            {h.get('iteration')}",
            f"  Domain:               {h.get('domain')}",
            f"  Research status:      {self._research_status(h, breakdown)}",
            f"  Confidence:           {breakdown['final_score']:.2f}/10 "
            f"({breakdown['verdict']})",
            "",
            f"  Supporting knowledge: {h.get('supporting_evidence') or '(none linked)'}",
            f"  Contradictory:        {h.get('contradictory_evidence') or '(none linked)'}",
            f"  Evidence status:      {breakdown['evidence_status']}",
            "",
            "  Assumptions:",
        ]
        for a in (h.get("assumptions") or ["(none declared)"]):
            L.append(f"      - {a}")
        L += ["", "  Expected result (declared BEFORE the simulation):"]
        for k, v in (h.get("expected_outcome") or {}).items():
            L.append(f"      - {k}: {v}")
        if not h.get("expected_outcome"):
            L.append("      (none recorded)")
        L += ["", "  Measurable metrics:"]
        for m in (h.get("measurable_metrics") or ["(none declared)"]):
            L.append(f"      - {m}")
        L += ["", "  Actual result (from the Simulation JSON):"]
        if sim:
            for k, v in (sim.get("observed_results") or {}).items():
                L.append(f"      - {k}: {v}")
            for name, m in (sim.get("metrics") or {}).items():
                if isinstance(m, dict):
                    L.append(f"      - {name}: {m.get('improvement_pct')}% "
                             f"(baseline {m.get('baseline_value')} -> "
                             f"{m.get('observed_value')})")
        else:
            L.append("      (no simulation record)")
        L += [
            "",
            f"  Simulation status:    {sim.get('status', 'simulation_unavailable')}",
            "  Failure reason:",
        ]
        for f in failures:
            L += [
                f"      {f.get('failure_id')} ({f.get('failure_type')}, "
                f"severity {f.get('severity')}): {str(f.get('cause'))[:200]}",
                f"        failed assumption: {f.get('failed_assumption')}",
            ]
        if not failures:
            L.append("      (no failure was recorded for this hypothesis)")
        L += ["", "  Lessons learned:"]
        lessons = [ls for ls in self._lessons() if ls.get("hypothesis_id") == hid]
        for ls in lessons:
            L.append(f"      {ls.get('lesson_id')}: {str(ls.get('lesson'))[:200]}")
            for ch in ls.get("recommended_changes", []):
                L.append(f"        recommended change: {ch}")
        if not lessons:
            L.append("      (none)")
        L += ["", "  Changes made in the next iteration:"]
        if successors:
            for nxt in successors:
                before, after = h.get("parameters") or {}, nxt.get("parameters") or {}
                changed = [k for k in sorted(set(before) | set(after))
                           if before.get(k) != after.get(k)]
                L.append(f"      {nxt.get('hypothesis_id')} (parent {hid})")
                L.append(f"        lessons applied: "
                         f"{nxt.get('based_on_lessons') or '(none)'}")
                for k in changed:
                    L.append(f"        '{k}': {before.get(k)} -> {after.get(k)}")
                if not changed:
                    L.append("        (no parameter change recorded)")
        else:
            L.append("      (no successor hypothesis — this is the latest iteration)")
        L.append("")
        return L

    # ============================================================ report 09
    def _graph_edges(self) -> List[str]:
        """PAPER -> FINDING -> HYP -> EXP -> FAIL -> LESSON -> HYP -> RWE/RWR edges."""
        edges: List[str] = []
        for p in self._papers():
            edges.append(f"  {p.get('paper_id')} --(provides knowledge)--> KNOWLEDGE")
        for i in self.knowledge_items():
            ct = str(i.get("claim_type") or "")
            if ct == "sourced" and i.get("source_paper"):
                edges.append(f"  {i.get('source_paper')} --(supports)--> {i.get('item_id')}")
            for rel in (i.get("relationships") or []):
                tgt = rel.get("target") or rel.get("item_id") or rel.get("to")
                if tgt:
                    edges.append(f"  {i.get('item_id')} --"
                                 f"({rel.get('type', i.get('relationship_type', 'related'))})"
                                 f"--> {tgt}")
        for h in self._hypotheses():
            hid = h.get("hypothesis_id")
            for item in (h.get("supporting_evidence") or []):
                edges.append(f"  {item} --(supports)--> {hid}")
            for item in (h.get("contradictory_evidence") or []):
                edges.append(f"  {item} --(contradicts)--> {hid}")
            for ls in (h.get("based_on_lessons") or []):
                edges.append(f"  {ls} --(influenced)--> {hid}")
            if h.get("parent_hypothesis"):
                edges.append(f"  {h.get('parent_hypothesis')} --(improved into)--> {hid}")
        for sim in self._simulations():
            edges.append(f"  {sim.get('hypothesis_id')} --(tested by)--> "
                         f"{sim.get('experiment_id')}")
        for f in self._failures():
            edges.append(f"  {f.get('experiment_id')} --(failed because)--> "
                         f"{f.get('failure_id')}")
        for ls in self._lessons():
            edges.append(f"  {ls.get('failure_id')} --(generated)--> "
                         f"{ls.get('lesson_id')}")
        for r in self.state.get("real_world", {}).get("proposals", []):
            edges.append(f"  {r.get('hypothesis_id')} --(proposed for real world)--> "
                         f"{r.get('proposal_id')}")
        for r in self._rwr():
            edges.append(f"  {r.get('proposal_id')} --(validated by)--> {r.get('result_id')}")
        return edges

    def report_knowledge_graph(self) -> str:
        """09 — knowledge-graph summary, derived from the master research state."""
        L = self._header("09 — KNOWLEDGE GRAPH SUMMARY")
        summary = self.state.get("knowledge_summary") or {}
        by_section = summary.get("by_section") or {}
        by_claim = summary.get("by_claim_type") or {}
        knowledge = self.knowledge_items()
        edges = self._graph_edges()

        node_kinds = {
            "papers": len(self._papers()),
            "knowledge_items": len(knowledge),
            "hypotheses": len(self._hypotheses()),
            "simulations": len(self._simulations()),
            "failures": len(self._failures()),
            "lessons": len(self._lessons()),
            "real_world_records": (len(self.state.get("real_world", {}).get("proposals", []))
                                   + len(self._rwr())),
        }
        total_nodes = sum(node_kinds.values())
        L += [
            "1. GRAPH SIZE", _SUB,
            f"  Number of nodes          : {total_nodes}",
            f"  Number of relationships  : {len(edges)}",
            "  Counts are descriptive only — they are NEVER used as evidence strength",
            "  or as hypothesis confidence.", "",
            "  Nodes by type:",
        ]
        for k, v in node_kinds.items():
            L.append(f"      {k:22s}: {v}")
        L += ["", "2. IMPORTANT ENTITIES", _SUB,
              "  Knowledge sections (from knowledge.json):"]
        for section, count in (by_section or {}).items():
            L.append(f"      {section:16s}: {count}")
        L += ["", "  Claim-type separation (critical distinction):"]
        for claim, count in (by_claim or {}).items():
            L.append(f"      {claim:24s}: {count}")
        L += [
            "      sourced = from documents | inferred = model reasoning |",
            "      generated_hypothesis = AARL proposal | simulation_observation = model",
            "      output | real_world_observation = human-provided.", "",
            "3. IMPORTANT CLUSTERS / DOMAINS", _SUB,
            f"  Primary domain: {self.state.get('primary_domain', '')}",
            "  Domains investigated:",
        ]
        for d in self._domain_names():
            L.append(f"      - {d}")
        L += [
            "",
            "4. SUPPORTING RELATIONSHIPS", _SUB,
            f"  supporting edges:  {sum(1 for e in edges if '--(supports)-->' in e)}",
            f"  improvement edges: {sum(1 for e in edges if '--(improved into)-->' in e)}",
            f"  lesson edges:      {sum(1 for e in edges if '--(influenced)-->' in e)}",
            "",
            "5. CONTRADICTORY RELATIONSHIPS", _SUB,
            f"  contradicting edges: {sum(1 for e in edges if '--(contradicts)-->' in e)}",
            f"  contradiction records preserved: "
            f"{summary.get('contradiction_records', 0)}",
            "  Contradictions are preserved as explicit relationships; no source is",
            "  silently deleted when it disagrees with another source.", "",
            "6. RESEARCHER-PROVIDED KNOWLEDGE", _SUB,
        ]
        rp = [p for p in self._papers() if p.get("provided_by") == "researcher"]
        L.append(f"  researcher-provided papers: {len(rp)}")
        for p in rp:
            L.append(f"      - {p.get('paper_id')}: {p.get('title')}")
        researcher_nodes = sum(
            1 for i in knowledge
            if "researcher" in str((i.get("provenance") or {}).get("origin", "")).lower()
        )
        L.append(f"  knowledge nodes with researcher provenance: {researcher_nodes}")
        L += ["", "7. EXPERIMENT HISTORY", _SUB]
        for it in self.state.get("iterations", []):
            L.append(f"  Iteration {it.get('iteration')}:")
            for h in it.get("hypotheses", []):
                L.append(f"      {h.get('hypothesis_id')} -> {h.get('experiment_id')} "
                         f"[{h.get('status')}]")
        L += ["", "8. FULL RELATIONSHIP LIST", _SUB]
        if not edges:
            L.append("  (no relationships recorded)")
        for e in edges[:80]:
            L.append(e)
        if len(edges) > 80:
            L.append(f"  ... {len(edges) - 80} more relationship(s) in "
                     f"research_memory/graph.json")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # ============================================================ report 10
    def report_limitations_next_steps(self) -> str:
        """10 — limitations and next steps. Failures are never hidden."""
        L = self._header("10 — LIMITATIONS AND NEXT STEPS")
        focus = self.focus_hypothesis()
        primary = str(self.state.get("primary_domain", "") or "")

        L += ["1. CURRENT LIMITATIONS", _SUB,
              "  • AARL generates candidate research directions and preliminary",
              "    computational assessments; it does not produce scientifically",
              "    validated discoveries.",
              "  • Confidence scores are internal ranking aids derived from stored",
              "    records, not from replicated experiments.",
              "  • All generated hypotheses are [HYPOTHESIS], never established fact.",
              ""]

        L += ["2. UNSUPPORTED ASSUMPTIONS", _SUB]
        if focus:
            assumptions = focus.get("assumptions") or []
            if not assumptions:
                L.append("  (the focus hypothesis declared no assumptions)")
            for a in assumptions:
                L.append(f"  - [UNVERIFIED] {a}")
            ev = self.evidence_for_hypothesis(focus)
            if not [e for e in ev if e["supports_or_contradicts"] == "supports"]:
                L += ["  - [UNVERIFIED] The hypothesis is not linked to any supporting",
                      "    evidence item, so every claim in it is unsupported."]
        else:
            L.append("  (no hypothesis available)")
        L.append("")

        L += ["3. SIMULATION LIMITATIONS", _SUB]
        sims = self._simulations()
        if not sims:
            L += ["  • No simulation was available for this cycle (simulation_unavailable).",
                  "    No simulated result is claimed anywhere in these reports."]
        else:
            for sim in sims:
                status = str(sim.get("status", "") or "")
                L.append(f"  • {sim.get('experiment_id')} "
                         f"({self._adapter_label(sim.get('adapter'))}): "
                         f"status={status}")
                if status == "simulation_unavailable":
                    L.append("      no engine could model this hypothesis — recorded honestly")
                if sim.get("constraints_violated"):
                    L.append(f"      constraints violated: {sim.get('constraints_violated')}")
                if sim.get("errors"):
                    L.append(f"      errors: {sim.get('errors')}")
            L += [
                "  • Simulation output depends on the model's assumptions; it predicts",
                "    behaviour under those assumptions only and is NOT real-world evidence.",
                "  • Reproducibility is limited by the recorded seed and parameters only.",
            ]
        L.append("")

        L += ["4. EVIDENCE GAPS", _SUB]
        gaps = self.knowledge_items("limitations")
        if not gaps:
            L.append("  (no limitation items extracted from papers)")
        for i in gaps[:10]:
            L.append(f"  - {self._label(i.get('claim_type'))} {i.get('item_id')}: "
                     f"{str(i.get('statement'))[:180]}")
        if focus and not self.evidence_for_hypothesis(focus):
            L.append("  - CRITICAL: the focus hypothesis has NO directly linked evidence.")
        L.append("")

        L += ["5. REAL-WORLD VALIDATION GAPS", _SUB]
        if self._rwr():
            for r in self._rwr():
                L.append(f"  - [REAL-WORLD RESULT] {r.get('result_id')} by "
                         f"{r.get('researcher', 'unknown')}")
                for c in r.get("simulation_vs_real", []):
                    L.append(f"      {c.get('metric')}: simulation predicted "
                             f"{c.get('prediction')}, real world "
                             f"{c.get('real_world_result')}, error {c.get('error')}")
                for cause in (r.get("possible_causes") or []):
                    L.append(f"      possible cause: {cause}")
        else:
            L += [
                "  - No real-world result has been provided. Everything reported here",
                "    remains UNVERIFIED outside simulation.",
                "  - The simulation model's own accuracy is UNKNOWN until a real-world",
                "    result is compared against its prediction.",
            ]
        L.append("")

        L += ["6. KNOWN FAILURES (never hidden)", _SUB]
        failures = self._failures()
        if not failures:
            L.append("  (no simulation failure was recorded as a Failure JSON)")
        for f in failures:
            L += [
                f"  - {f.get('failure_id')} (hypothesis {f.get('hypothesis_id')}, "
                f"experiment {f.get('experiment_id')})",
                f"      failure_type     : {f.get('failure_type')}",
                f"      cause            : {str(f.get('cause'))[:220]}",
                f"      failed_assumption: {f.get('failed_assumption')}",
                f"      severity         : {f.get('severity')}",
                f"      recommended      : {f.get('recommended_changes')}",
            ]
        L.append("")

        L += ["7. MANUFACTURING / PRACTICAL LIMITATIONS", _SUB]
        physical = primary.lower() in ("chemistry", "materials", "battery", "aerospace",
                                       "robotics", "civil_engineering", "medicine")
        if physical:
            L += [f"  Primary domain '{primary}' involves physical construction; AARL has",
                  "  NOT assessed manufacturability, supply chain, cost or tolerances."]
        else:
            L += [f"  Primary domain '{primary or 'unspecified'}' does not primarily",
                  "  involve physical materials, so manufacturing constraints are not",
                  "  applicable; no irrelevant material list has been generated."]
        L.append("")

        L += ["8. OPEN RESEARCH QUESTIONS", _SUB]
        for q in [
            "Do the declared assumptions hold under real conditions?",
            "Does the predicted effect survive without the simulation's simplifications?",
            "How sensitive is the result to the parameters not varied in simulation?",
            "Which contradictory evidence can be resolved experimentally?",
        ]:
            L.append(f"  - {q}")
        L.append("")

        L += ["9. RECOMMENDED FUTURE EXPERIMENTS", _SUB]
        proposals = self.state.get("real_world", {}).get("proposals", [])
        for p in proposals:
            L += [
                f"  - {p.get('proposal_id')} ({p.get('status')}): "
                f"{str(p.get('objective'))[:200]}",
                f"      required equipment: {p.get('required_equipment')}",
                f"      controls:           {p.get('controls')}",
                f"      safety:             {p.get('safety_considerations')}",
                f"      metrics:            {p.get('measurable_metrics')}",
                f"      limitations:        {p.get('known_limitations')}",
            ]
        if not proposals:
            L += ["  (no real-world proposal was generated — no candidate passed",
                  "   simulation with a measurable improvement)"]
        L.append("")

        L += ["10. NEXT RESEARCH DIRECTIONS", _SUB,
              "  - Extend the knowledge base with directly relevant sourced evidence."]
        for ls in self._lessons():
            L.append(f"  - Apply {ls.get('lesson_id')}: {str(ls.get('lesson'))[:170]}")
            for ch in ls.get("recommended_changes", []):
                L.append(f"      change: {ch}")
        if not self._lessons():
            L.append("  - No lesson was recorded yet (no failing simulation in this cycle).")
        L += ["", _RULE, ""]
        return "\n".join(L)

    # ================================================== final result (report 00)
    def report_final_result(self) -> str:
        """00 — the document a researcher reads first.

        Synthesised from the SAME master research state as reports 01-10; it is
        NOT a concatenation of them, but it links back to each one.
        """
        focus = self.focus_hypothesis()
        linked = self.evidence_for_hypothesis(focus) if focus else []
        eligible = self.evidence_eligibility(linked)["eligible"]
        breakdown = self._confidence_breakdown(focus, eligible) if focus else None
        best = self._best()
        rwr = self._rwr()
        problems = self.consistency_checks()

        L = [_RULE, "AARL FINAL RESEARCH RESULT", _RULE, "",
             f"Research question: {self._q()}",
             f"Generated: {self.state.get('generated_at', '')}",
             "Synthesised from research_memory/master_state.json (single source of truth)",
             "",
             "CLAIM LABELS USED IN THIS DOCUMENT:",
             "  [FACT] sourced knowledge | [INFERENCE] derived conclusion |",
             "  [HYPOTHESIS] AARL proposal | [SIMULATION RESULT] model output |",
             "  [REAL-WORLD RESULT] human-provided observation",
             _RULE, ""]

        # 1-3
        L += ["1. RESEARCH QUESTION", _SUB, self._q() or "Not specified.", "",
              "2. RESEARCH OBJECTIVE", _SUB,
              "Identify a candidate approach for the research question from literature-",
              "grounded hypothesis generation and simulation, and determine what must be",
              "validated experimentally. AARL does not prove discoveries autonomously.", "",
              "3. DOMAINS INVESTIGATED", _SUB,
              f"Primary domain: {self.state.get('primary_domain', '')}"]
        for d in self._domain_names():
            L.append(f"  - {d}")

        # 4
        L += ["", "4. RESEARCH BASE", _SUB]
        papers = self._papers()
        L.append(f"  Papers ingested: {len(papers)}")
        for p in papers:
            L.append(f"    - {p.get('paper_id')}: {p.get('title')} "
                     f"(provided_by={p.get('provided_by', 'unknown')})")
        rp = [p for p in papers if p.get("provided_by") == "researcher"]
        L.append(f"  Researcher-provided papers: "
                 f"{[p.get('paper_id') for p in rp] or '(none)'}")
        summary = self.state.get("knowledge_summary") or {}
        L.append(f"  Existing knowledge: {summary.get('total_items', 0)} item(s)")
        for claim, count in (summary.get("by_claim_type") or {}).items():
            L.append(f"    {claim:24s}: {count}")

        # 5
        L += ["", "5. KEY DISCOVERIES", _SUB]
        facts = self.knowledge_items("findings")
        if not facts:
            L.append("  (no sourced findings recorded)")
        for i in facts[:8]:
            L.append(f"  {self._label(i.get('claim_type'))} {i.get('item_id')}: "
                     f"{str(i.get('statement'))[:220]}")
        L.append("  NOTE: 'discoveries' here means findings extracted from the supplied")
        L.append("  literature, not results AARL established itself.")
        # 6
        L += ["", "6. HYPOTHESIS EVOLUTION", _SUB]
        hypotheses = self._hypotheses()
        if not hypotheses:
            L.append("  (no hypotheses generated)")
        by_id = {h.get("hypothesis_id"): h for h in hypotheses}
        for it in self.state.get("iterations", []):
            for entry in it.get("hypotheses", []):
                hid = entry.get("hypothesis_id", "")
                h = by_id.get(hid) or {}
                sim = self._simulation_for_hypothesis(str(hid))
                note = ""
                if h.get("based_on_lessons"):
                    note = f" [applied {', '.join(h['based_on_lessons'])}]"
                elif h.get("parent_hypothesis"):
                    note = f" [parent {h['parent_hypothesis']}]"
                L.append(f"  Iteration {it.get('iteration')}: {hid} -> "
                         f"{entry.get('experiment_id')} [{entry.get('status')}]{note}")
                if entry.get("failure_id"):
                    L.append(f"      failure {entry.get('failure_id')} -> "
                             f"lesson {entry.get('lesson_id')}")
                if not h.get("based_on_lessons") and entry.get("lesson_id") is None \
                        and not h.get("parent_hypothesis"):
                    L.append("      (first candidate — no prior lesson applied)")

        # 7
        L += ["", "7. SIMULATION RESULTS", _SUB]
        if not self._simulations():
            L += ["  simulation_unavailable — no engine could model these hypotheses.",
                  "  No simulated result is claimed."]
        for sim in self._simulations():
            L.append(f"  {sim.get('experiment_id')} "
                     f"(hypothesis {sim.get('hypothesis_id')}, "
                     f"{self._adapter_label(sim.get('adapter'))}): "
                     f"status={sim.get('status')}")
            for name, m in (sim.get("metrics") or {}).items():
                if isinstance(m, dict):
                    L.append(f"      {name}: baseline {m.get('baseline_value')} -> "
                             f"{m.get('observed_value')} "
                             f"({m.get('improvement_pct')}%)")
            if sim.get("constraints_violated"):
                L.append(f"      constraints violated: {sim.get('constraints_violated')}")
        L += ["  Metrics are listed per-metric: no 'average improvement' is computed,",
              "  because averaging unrelated metric percentages is invalid.",
              "  [SIMULATION RESULT] is NOT real-world evidence."]

        # 8-9
        L += ["", "8. FAILURES AND LESSONS", _SUB]
        if not self._failures():
            L.append("  (no simulation failure recorded in this cycle)")
        for f in self._failures():
            L.append(f"  {f.get('failure_id')} on {f.get('hypothesis_id')} "
                     f"({f.get('failure_type')}, severity {f.get('severity')})")
            L.append(f"      cause: {str(f.get('cause'))[:200]}")
            L.append(f"      failed assumption: {f.get('failed_assumption')}")
        for ls in self._lessons():
            L.append(f"  {ls.get('lesson_id')} (from {ls.get('failure_id')}): "
                     f"{str(ls.get('lesson'))[:200]}")
            for ch in ls.get("recommended_changes", []):
                L.append(f"      recommended change: {ch}")

        L += ["", "9. IMPROVEMENTS MADE", _SUB]
        improvements = [(it, h) for it in self.state.get("iterations", [])
                        for h in it.get("hypotheses", []) if h.get("lesson_id")]
        if not improvements:
            L += ["  (no lesson-driven improvement recorded — either the first iteration",
                  "   succeeded or no failure produced a lesson)"]
        for it, h in improvements:
            L.append(f"  Iteration {it.get('iteration')}: {h.get('hypothesis_id')} applied "
                     f"{h.get('lesson_id')} after {h.get('failure_id')}")
        L.append("  See 08_Hypotheses_Analysis.txt for the parameter-level diff.")

        # 10
        L += ["", "10. FINAL CANDIDATE / DESIGN", _SUB]
        if not best:
            L += ["  NO CANDIDATE WAS SELECTED.",
                  "  No hypothesis produced a successful simulation with a measurable",
                  "  improvement, so AARL does NOT present a final design. Fabricating",
                  "  one would misrepresent the research.", "",
                  "  See 02_Final_Design_Report.txt and 10_Limitations_and_Next_Steps.txt."]
        else:
            L += [f"  Candidate hypothesis: {best.get('hypothesis_id')}",
                  f"  Statement: {str(best.get('hypothesis'))[:300]}",
                  f"  Basis: simulation {best.get('experiment_id')} "
                  f"[{best.get('status')}], best metric change "
                  f"{best.get('best_improvement_pct')}%",
                  f"  Research status: {self._research_status(focus, breakdown) if breakdown else 'SIMULATION'}",
                  "  This is a SIMULATED candidate selected on simulated metrics; it has",
                  "  NOT been validated in the real world."]
        # 11
        L += ["", "11. ENGINEERING ANALYSIS", _SUB]
        equations = self.knowledge_items("equations")
        assumptions = self.knowledge_items("assumptions")
        L += [f"  Equations extracted from literature: {len(equations)}",
              "  Mathematical validity is NOT computed in this layer and equation",
              "  counts are NOT evidence. See 04_Engineering_Calculations.txt.", "",
              f"  Assumptions recorded: {len(assumptions)}"]
        for i in assumptions[:8]:
            L.append(f"      - {str(i.get('statement'))[:180]}")
        if focus:
            for a in (focus.get("assumptions") or []):
                L.append(f"      - [UNVERIFIED] {a}")
        if best:
            sim = self.memory_get("simulations", best.get("experiment_id", ""))
            for name, m in ((sim or {}).get("metrics") or {}).items():
                if isinstance(m, dict):
                    L.append(f"      [SIMULATION RESULT] {name}: "
                             f"{m.get('baseline_value')} -> {m.get('observed_value')}")

        # 12
        L += ["", "12. EVIDENCE AND CONFIDENCE", _SUB]
        if not focus or breakdown is None:
            L.append("  No hypothesis to assess.")
        else:
            L += [f"  Focus hypothesis: {focus.get('hypothesis_id')}",
                  f"  Final Confidence: {breakdown['final_score']:.2f}/10 "
                  f"({breakdown['verdict']})",
                  f"  Research status:  {self._research_status(focus, breakdown)}",
                  f"  Evidence status:  {breakdown['evidence_status']}"]
            for dim in ("novelty", "evidence", "feasibility",
                        "consistency", "testability", "risk"):
                L.append(f"      {dim.capitalize():15s}: "
                         f"{breakdown['dimensions'][dim]}/10")
            L += ["  Formula: novelty*0.15 + evidence*0.25 + feasibility*0.15",
                  "           + consistency*0.20 + testability*0.15 + (10-risk)*0.10",
                  "  All dimensions clamped to [1,10]; result clamped to [0,10];",
                  "  risk can only REDUCE confidence; missing evidence caps the score.",
                  "  See 07_Evidence_and_Confidence.txt for the full mapping."]
            if breakdown["capped"]:
                L.append("  NOTE: the confidence score was CAPPED by evidence gating.")

        # 13
        L += ["", "13. KNOWLEDGE GRAPH SUMMARY", _SUB]
        summary = self.state.get("knowledge_summary") or {}
        L += [f"  Knowledge nodes: {summary.get('total_items', 0)}",
              f"  Relationships:   {len(self._graph_edges())}",
              "  Contradiction records preserved: "
              f"{summary.get('contradiction_records', 0)}",
              "  See 09_Knowledge_Graph_Summary.txt and research_memory/graph.json."]
        # 14
        L += ["", "14. REAL-WORLD VALIDATION", _SUB]
        if rwr:
            for r in rwr:
                L += [f"  [REAL-WORLD RESULT] {r.get('result_id')} provided by "
                      f"{r.get('researcher', 'unknown')}"]
                for c in r.get("simulation_vs_real", []):
                    L.append(f"      {c.get('metric')}: prediction "
                             f"{c.get('prediction')} vs observed "
                             f"{c.get('real_world_result')} (error {c.get('error')})")
        else:
            L += ["  VALIDATION IS PENDING. No human researcher has supplied real-world",
                  "  results for this cycle, so no claim here is experimentally validated.",
                  "  AARL does not perform real-world experiments and does not fabricate",
                  "  validation."]

        # 15-17
        L += ["", "15. LIMITATIONS", _SUB,
              "  • Every result is a candidate direction, not a validated discovery.",
              "  • Simulation output is model-dependent, not real-world evidence.",
              "  • Confidence is an internal ranking aid built from stored records.",
              "  • Cross-domain knowledge is excluded when it is not relevant."]
        if breakdown and breakdown["evidence_status"] == "INSUFFICIENT_DIRECT_EVIDENCE":
            L.append("  • The focus hypothesis has NO directly linked evidence.")
        if self._failures():
            L.append(f"  • {len(self._failures())} simulation failure(s) are documented "
                     "and not hidden.")

        L += ["", "16. NEXT STEPS", _SUB]
        for ls in self._lessons():
            L.append(f"  - Apply {ls.get('lesson_id')}: {str(ls.get('lesson'))[:170]}")
        for p in self.state.get("real_world", {}).get("proposals", []):
            L.append(f"  - Human review of {p.get('proposal_id')} "
                     f"({p.get('status')}) before any experiment")
        L += ["  - Extend the knowledge base with directly relevant sourced evidence.",
              "  - Run the next research cycle; prior failures and lessons are retained.",
              "  See 10_Limitations_and_Next_Steps.txt."]

        L += ["", "17. FINAL CONCLUSION", _SUB]
        if problems:
            L += ["  The automatic consistency check reported issues, so the affected",
                  "  claims are downgraded to UNVERIFIED / PRELIMINARY:"]
            for p in problems:
                L.append(f"      - {p}")
            L.append("")
        if not best:
            L += ["  AARL did NOT identify a validated candidate for this research",
                  "  question. The recorded failures and lessons are the useful output",
                  "  of this cycle and should drive the next iteration.", "",
                  "  STATUS: IDEA / UNVERIFIED — no candidate was established."]
        else:
            L += [f"  AARL identified {best.get('hypothesis_id')} as the best SIMULATED",
                  f"  candidate (best metric change {best.get('best_improvement_pct')}%).", "",
                  f"  STATUS: SIMULATION — confidence "
                  f"{breakdown['final_score']:.2f}/10, evidence "
                  f"{breakdown['evidence_status']}.", "",
                  "  This is a candidate research direction and a preliminary computational",
                  "  assessment. It is NOT a scientifically validated discovery. A",
                  "  researcher should independently review the evidence, modify the",
                  "  approach where appropriate, and perform controlled experiments",
                  "  before drawing scientific conclusions.", "",
                  "  TRACEABILITY: 00 -> 01-10 -> master_state.json -> knowledge.json",
                  "                -> papers/evidence -> hypotheses -> simulations",
                  "                -> failures -> lessons -> improved hypotheses"]
        L += ["", _RULE, ""]
        return "\n".join(L)

    # __PART7__





