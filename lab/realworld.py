"""
AARL Lab — Real-World Experiment Layer
======================================
After simulation identifies a promising candidate, AARL produces a *proposal*
for a real-world experiment (RWE-###). AARL never performs physical experiments:
the proposal is a structured recommendation for a qualified human researcher,
including safety considerations and controls.

When the researcher returns real-world results, AARL records them (RWR-###),
compares **simulation prediction vs real-world observation** for every shared
metric, and integrates the observations into the knowledge store as
``real_world_observation`` claims — with the prediction error recorded so AARL
learns when its simulation model is inaccurate.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

CLAIM_REAL_WORLD = "real_world_observation"
CLAIM_SIMULATION = "simulation_observation"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class RealWorldManager:
    def __init__(self, memory, knowledge, config):
        self.memory = memory
        self.knowledge = knowledge
        self.config = config

    # ------------------------------------------------------------- proposal
    def propose(self, hypothesis: Dict[str, Any], simulation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        proposal_id = self.memory.next_id("real_world_experiments", "RWE")
        metrics = self._shared_metrics(hypothesis, simulation)
        proposal = {
            "proposal_id": proposal_id,
            "hypothesis_id": hypothesis.get("hypothesis_id", ""),
            "status": "awaiting_human_approval",
            "disclaimer": (
                "This is a PROPOSAL generated from simulation evidence. A qualified "
                "human researcher must review, approve, and conduct the experiment. "
                "AARL does not perform real-world experiments autonomously."
            ),
            "objective": f"Validate in the real world: {hypothesis.get('hypothesis', '')}",
            "required_equipment": self._equipment(hypothesis),
            "parameters": dict(hypothesis.get("parameters") or {}),
            "expected_results": dict(hypothesis.get("expected_outcome") or {}),
            "measurable_metrics": list(metrics),
            "controls": self._controls(hypothesis),
            "safety_considerations": self._safety(hypothesis, simulation),
            "procedure_outline": self._procedure(hypothesis, simulation),
            "simulation_evidence": self._sim_evidence(simulation),
            "known_limitations": self._limitations(hypothesis, simulation),
            "provenance": {
                "derived_from": "simulation_record",
                "hypothesis_id": hypothesis.get("hypothesis_id", ""),
                "experiment_id": (simulation or {}).get("experiment_id", ""),
                "generated_by": "aarl.lab.realworld.RealWorldManager",
            },
            "timestamp": _now(),
        }
        self.memory.append_record("real_world_experiments", proposal_id, proposal)
        self.memory.append_history({
            "event": "real_world_proposal_created",
            "proposal_id": proposal_id,
            "hypothesis_id": proposal["hypothesis_id"],
            "experiment_id": proposal["provenance"]["experiment_id"],
        })
        return proposal

    @staticmethod
    def _shared_metrics(hypothesis: Dict[str, Any], simulation: Optional[Dict[str, Any]]) -> List[str]:
        metrics = set()
        for name in ((simulation or {}).get("metrics") or {}):
            metrics.add(str(name))
        for name in (hypothesis.get("expected_outcome") or {}):
            metrics.add(str(name))
        return sorted(metrics)

    @staticmethod
    def _equipment(hypothesis: Dict[str, Any]) -> List[str]:
        """Domain-aware equipment list; generic where the domain is unknown."""
        domain = str(hypothesis.get("domain", "") or "")
        if domain == "machine_learning":
            return [
                "Compute node matching the simulation's declared hardware assumptions",
                "Versioned dataset identical to the simulated problem scale",
                "Seeded, logged experiment tracking",
            ]
        if domain in ("chemistry", "materials", "battery"):
            return [
                "Laboratory bench with safety enclosure / fume hood",
                "Calibrated instrumentation for the reported metrics",
                "Sample preparation and characterisation equipment",
                "PPE and waste-handling per institutional safety rules",
            ]
        return [
            "Instrumentation able to measure the declared metrics with calibrated accuracy",
            "Prototype/bench setup matching the simulation's declared conditions",
            "Data acquisition and logging",
        ]

    @staticmethod
    def _controls(hypothesis: Dict[str, Any]) -> List[str]:
        return [
            "Run a baseline configuration identical to the simulation's baseline",
            "Repeat the proposed configuration under the same conditions/seeds where applicable",
            "Keep all variables fixed except those the hypothesis changes: "
            + (", ".join(sorted((hypothesis.get("parameters") or {}).keys())) or "(declare explicitly)"),
        ]

    @staticmethod
    def _safety(hypothesis: Dict[str, Any], simulation: Optional[Dict[str, Any]]) -> List[str]:
        considerations = [
            "Review by a qualified human researcher is REQUIRED before conducting this experiment.",
            "Do not exceed any operating limits the simulation flagged as constraints.",
        ]
        for c in (simulation or {}).get("constraints_violated") or []:
            considerations.append(f"Simulation violated: {c} — treat as a hard safety limit.")
        return considerations

    @staticmethod
    def _procedure(hypothesis: Dict[str, Any], simulation: Optional[Dict[str, Any]]) -> List[str]:
        steps = [
            "1. Reproduce the simulation's baseline configuration and record reference metrics.",
            "2. Apply the hypothesis parameters exactly as declared ('parameters').",
            "3. Measure every metric listed under 'measurable_metrics' with calibrated instruments.",
            "4. Repeat sufficient times to estimate variance.",
        ]
        for i, note in enumerate(((simulation or {}).get("notes") or [])[:3], start=5):
            steps.append(f"{i}. Simulation note to respect: {note}")
        steps.append(
            f"{len(steps) + 1}. Return the observed metric values to AARL for "
            "simulation-vs-real comparison (RWR record)."
        )
        return steps

    @staticmethod
    def _sim_evidence(simulation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not simulation:
            return {"note": "No simulation evidence available."}
        return {
            "experiment_id": simulation.get("experiment_id", ""),
            "simulation_type": simulation.get("simulation_type", ""),
            "simulation_status": simulation.get("status", ""),
            "observed_results": simulation.get("observed_results"),
            "metrics": simulation.get("metrics"),
            "note": "SIMULATION EVIDENCE ONLY — model outputs, not real-world results.",
        }

    @staticmethod
    def _limitations(hypothesis: Dict[str, Any], simulation: Optional[Dict[str, Any]]) -> List[str]:
        limitations = [
            "The simulation model may not capture all real-world effects; differences "
            "are expected and will be recorded when results are returned.",
        ]
        for u in (simulation or {}).get("unexpected_results") or []:
            limitations.append(f"Unexpected simulated behavior to watch for: {u}")
        limitations.extend(str(r) for r in (hypothesis.get("risks") or [])[:3])
        return limitations

    # --------------------------------------------------------- result intake
    def record_result(
        self,
        proposal_id: str,
        results: Dict[str, Any],
        researcher: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record human-provided real-world results and compare them with the
        simulation prediction.

        ``results`` accepts {"<metric_name>": value, ...} (optionally nested
        under "metrics"). Only numbers AARL can compare are analysed; everything
        else is stored verbatim. No result is ever invented here.
        """
        proposal = self.memory.get("real_world_experiments", proposal_id)
        if not proposal:
            raise ValueError(f"Unknown proposal id: {proposal_id}")

        provided = results.get("metrics") if isinstance(results.get("metrics"), dict) else results
        provided = dict(provided or {})

        simulation = self.memory.get("simulations", proposal["provenance"].get("experiment_id", ""))
        comparisons = self._compare_with_simulation(simulation, provided)

        result_id = self.memory.next_id("real_world_experiments", "RWR")
        record = {
            "result_id": result_id,
            "proposal_id": proposal_id,
            "hypothesis_id": proposal.get("hypothesis_id", ""),
            "experiment_id": proposal["provenance"].get("experiment_id", ""),
            "researcher": researcher,
            "results_provided": provided,
            "extra_results": {
                k: v for k, v in results.items() if k not in ("metrics",)
            },
            "simulation_vs_real": comparisons,
            "notes": notes,
            "provenance": {
                "origin": "human_provided_real_world_result",
                "researcher": researcher,
                "proposal_id": proposal_id,
                "generated_by": "aarl.lab.realworld.RealWorldManager.record_result",
                "claim_type_note": "REAL-WORLD observation provided by a human researcher",
            },
            "timestamp": _now(),
        }
        self.memory.append_record("real_world_experiments", result_id, record)

        # Mark the proposal as having a returned result (revision, not overwrite).
        proposal["status"] = "result_received"
        proposal["result_id"] = result_id
        self.memory.append_record(
            "real_world_experiments", proposal_id, proposal,
            revision_of="real-world result received",
        )

        self._integrate_real_world(record, proposal)
        self.memory.append_history({
            "event": "real_world_result_recorded",
            "result_id": result_id,
            "proposal_id": proposal_id,
            "hypothesis_id": record["hypothesis_id"],
            "researcher": researcher,
        })
        return record

    # ------------------------------------------------------------- analysis
    @staticmethod
    def _compare_with_simulation(
        simulation: Optional[Dict[str, Any]], provided: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simulation prediction vs real-world observation, per shared metric."""
        comparisons: List[Dict[str, Any]] = []
        sim_metrics = ((simulation or {}).get("metrics") or {})
        for name, real_value in provided.items():
            entry: Dict[str, Any] = {"metric": str(name), "real_world_result": real_value}
            sim_entry = sim_metrics.get(str(name))
            prediction = None
            if isinstance(sim_entry, dict):
                for key in ("observed_value", "improvement_pct", "value"):
                    if isinstance(sim_entry.get(key), (int, float)):
                        prediction = sim_entry[key]
                        break
                entry["prediction"] = prediction
            elif isinstance(sim_entry, (int, float)):
                prediction = sim_entry
                entry["prediction"] = prediction
            else:
                entry["prediction"] = None

            if isinstance(prediction, (int, float)) and isinstance(real_value, (int, float)):
                error = real_value - prediction
                entry["error"] = round(error, 6)
                entry["error_pct"] = (
                    round(100.0 * error / abs(prediction), 4) if prediction else None
                )
                entry["possible_causes"] = [
                    "The simulation model may omit real-world effects (friction, "
                    "noise, tolerances, measurement error).",
                    "The real-world configuration may differ from the declared "
                    "simulation parameters.",
                    "Measurement uncertainty in the real experiment.",
                ]
            elif prediction is None:
                entry["note"] = "No simulation prediction available for this metric."
            comparisons.append(entry)
        return comparisons

    def _integrate_real_world(self, record: Dict[str, Any], proposal: Dict[str, Any]) -> None:
        """Knowledge update from real-world results + sim-vs-real lesson."""
        citation = f"RWR {record['result_id']} (researcher: {record.get('researcher') or 'unknown'})"
        for entry in record.get("simulation_vs_real", []):
            real_value = entry.get("real_world_result")
            if not isinstance(real_value, (int, float)):
                continue
            statement = (
                f"Real-world result for metric '{entry['metric']}': {real_value} "
                f"(simulation prediction: {entry.get('prediction')}, "
                f"error: {entry.get('error')})"
            )
            item_id = self.knowledge.add(
                section="experiments",
                statement=statement,
                claim_type=CLAIM_REAL_WORLD,
                provenance={
                    "result_id": record["result_id"],
                    "proposal_id": record["proposal_id"],
                    "real_world_experiment_id": record["proposal_id"],
                    "provided_by": record.get("researcher") or "researcher",
                    "researcher": record.get("researcher", ""),
                },
                source=f"rwr:{record['result_id']}",
                source_paper=citation,
                confidence=0.9,  # human-provided observation
                metadata={
                    "hypothesis_id": record.get("hypothesis_id", ""),
                    "metric": entry["metric"],
                    "prediction": entry.get("prediction"),
                    "error": entry.get("error"),
                },
            )
            record.setdefault("knowledge_item_ids", []).append(item_id)

        # Sim-vs-real accuracy lesson when error is large (>25%).
        for entry in record.get("simulation_vs_real", []):
            error_pct = entry.get("error_pct")
            if isinstance(error_pct, (int, float)) and abs(error_pct) > 25.0:
                statement = (
                    f"The simulation overestimated metric '{entry['metric']}' by "
                    f"{error_pct:.1f}% relative to the real-world result "
                    f"(predicted {entry.get('prediction')}, observed "
                    f"{entry.get('real_world_result')}). Treat the simulation "
                    "model's accuracy for this metric with caution."
                )
                self.knowledge.add(
                    section="lessons",
                    statement=statement,
                    claim_type=CLAIM_REAL_WORLD,
                    provenance={
                        "derived_from": "simulation_vs_real_comparison",
                        "result_id": record["result_id"],
                        "real_world_experiment_id": record["proposal_id"],
                        "provided_by": record.get("researcher") or "researcher",
                        "metric": entry["metric"],
                    },
                    source=f"rwr:{record['result_id']}",
                    source_paper=citation,
                    confidence=0.8,
                    metadata={"hypothesis_id": record.get("hypothesis_id", "")},
                )



