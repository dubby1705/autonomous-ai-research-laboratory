"""
AARL Discovery — Problem Facets (the second level of the possibility space)
===========================================================================
A *facet* is a structural dimension of the research problem — architecture,
scheduling, memory, energy, materials, and so on. Facets are what make the
possibility space problem-general instead of a hard-coded tree: the same facet
taxonomy applies to a GPU, a battery, a catalyst or a clinical workflow.

Honesty rules
-------------
* A facet's ``limitation`` text is an **assumed** limitation of the current
  approach. It is labelled ``claim_type = "inference"``, and AARL first tries to
  find a *sourced* limitation for the facet in the knowledge base. Only when a
  sourced limitation exists is the limitation presented as literature-backed.
* Facet scores are keyword-overlap counts between the problem statement and the
  facet vocabulary. They decide *ordering*, never evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Facet:
    facet_id: str
    label: str
    #: Noun used to build direction titles ("Decentralized GPU Scheduling").
    title_noun: str
    #: Short description of what the facet is about.
    framing: str
    #: Metric this facet is usually judged on (feeds the core question).
    metric: str
    #: Assumed limitation of the conventional approach (NOT a fact).
    limitation: str
    keywords: Tuple[str, ...]
    #: Domains (as returned by the engine detector) this facet is central to.
    domains: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "label": self.label,
            "title_noun": self.title_noun,
            "framing": self.framing,
            "metric": self.metric,
            "assumed_limitation": self.limitation,
            "keywords": list(self.keywords),
            "domains": list(self.domains),
        }


def F(facet_id, label, noun, framing, metric, limitation, keywords, domains=()):
    """Compact positional builder (keeps the taxonomy readable as a table)."""
    return Facet(facet_id, label, noun, framing, metric, limitation,
                 tuple(keywords), tuple(domains))


#: The facet taxonomy (order = deterministic tie-break order).
FACETS: Tuple[Facet, ...] = (
    F("scheduling", "Scheduling", "Scheduling",
      "how work is assigned to the available resources",
      "resource utilization",
      "A single centralized scheduler must gather global state before it can decide, "
      "so its decisions are always slightly stale under fast-changing loads.",
      ("schedul", "dispatch", "queue", "load balanc", "assign", "task", "workload",
       "utilization", "throughput", "parallel", "concurren"),
      ("computer_architecture", "machine_learning", "robotics")),
    F("architecture", "Architecture", "Architecture",
      "how the major components are structured and connected",
      "performance per unit area",
      "A fixed microarchitecture is chosen at design time, so it is optimal for the "
      "workload it was designed around and suboptimal for everything else.",
      ("architect", "microarchitect", "core", "block", "component", "structure",
       "pipeline", "chip", "processor", "modular"),
      ("computer_architecture", "materials_science")),
    F("computing_model", "Computing Model", "Compute Model",
      "what kind of computation the system performs and how",
      "achievable operations per second",
      "The dominant computing model is uniform and synchronous, so it spends energy "
      "on operations whose result the application does not actually need.",
      ("compute", "comput", "arithmetic", "matrix", "tensor", "parallelism",
       "simd", "accelerat", "precision", "algorithm", "inference",
       "training", "operation"),
      ("computer_architecture", "machine_learning", "quantum")),
    F("memory", "Memory Hierarchy", "Memory Hierarchy",
      "where state lives and how it moves between levels",
      "effective data-movement cost",
      "Capacity and bandwidth are both limited, so data movement rather than "
      "arithmetic usually dominates the energy and time budget.",
      ("memory", "cache", "storage", "bandwidth", "dram", "hbm", "sram",
       "data movement", "hierarchy", "buffer", "register"),
      ("computer_architecture", "machine_learning")),
    F("interconnect_communication", "Interconnect", "Interconnect",
      "how components exchange information",
      "communication overhead",
      "Every additional participant increases the number of links that must be kept "
      "coherent, so coordination cost grows faster than the work added.",
      ("interconnect", "network", "bandwidth", "communicat", "link", "fabric",
       "topolog", "noc", "coheren", "synchroni", "latency", "message"),
      ("computer_architecture", "physics")),
    F("energy_thermal", "Power and Thermal", "Power Management",
      "how energy is consumed, converted and removed",
      "energy per operation",
      "Power density is limited by heat removal, so performance is capped by "
      "thermals rather than by the logic itself.",
      ("power", "energy", "thermal", "heat", "watt", "cool", "temperature",
       "efficiency", "sustainab", "consumption", "battery"),
      ("computer_architecture", "chemistry", "materials_science", "biology")),
    F("materials", "Material Stack", "Material Stack",
      "which materials carry the function and why",
      "material performance limit",
      "Material choice is constrained by what can be manufactured reliably at "
      "scale, not by the best measured property in the laboratory.",
      ("material", "alloy", "polymer", "ceramic", "composite", "crystal",
       "catalyst", "electrode", "substrate", "film", "dopant", "conduct"),
      ("materials_science", "chemistry", "physics")),
    F("manufacturing", "Fabrication", "Fabrication Approach",
      "how the artefact is produced and how reproducible it is",
      "yield and reproducibility",
      "Process variation is unavoidable, so the design must tolerate the spread of "
      "real fabrication instead of the nominal specification.",
      ("manufactur", "fabricat", "produc", "yield", "assembly", "process",
       "scalab", "toleranc", "defect", "lithograph"),
      ("materials_science", "chemistry")),
    F("reliability", "Reliability", "Reliability Scheme",
      "how the system keeps working when parts of it do not",
      "failure rate under load",
      "Components degrade with use, so the design point must be chosen for the end "
      "of life rather than the beginning.",
      ("reliab", "fault", "error", "robust", "degrad", "wear",
       "durab", "toleran", "recover", "corrupt", "safety"),
      ("computer_architecture", "materials_science", "biology")),
    F("software_stack", "Programming Model", "Programming Model",
      "what the user is allowed to express and how it is compiled",
      "developer-visible efficiency",
      "The application must be rewritten to exploit the hardware, so measured gains "
      "only materialise for teams willing to make that investment.",
      ("software", "program", "compil", "language", "framework", "api",
       "kernel", "sdk", "abstract", "code", "explainab"),
      ("computer_architecture", "machine_learning")),
    F("data_representation", "Data Representation", "Data Representation",
      "how information is encoded, stored and transported",
      "information density per unit cost",
      "The representation is uniform across the whole tensor or molecule, so a few "
      "significant elements pay for many insignificant ones.",
      ("represent", "encod", "format", "precision", "quantiz", "compress",
       "spars", "datatype", "bit", "tensor", "structure", "embed"),
      ("computer_architecture", "machine_learning", "chemistry")),
    F("control_adaptation", "Runtime Adaptation", "Runtime Adaptation",
      "how the system changes its own behaviour while running",
      "stability of achieved performance",
      "Design-time parameters are tuned for an average case, so the system is rarely "
      "at its best for the case actually running.",
      ("adapt", "runtime", "dynamic", "feedback", "control", "autotun",
       "reconfig", "monitor", "phase", "drift", "online"),
      ("computer_architecture", "machine_learning", "robotics", "physics")),
    F("cost_scaling", "Scaling Strategy", "Scaling Strategy",
      "what happens to cost and benefit as the system grows",
      "marginal cost of additional scale",
      "Benefit saturates before cost does, so adding more of the same resource stops "
      "buying proportionate improvement.",
      ("scal", "grow", "cost", "econom", "size", "number of", "thousand",
       "million", "large", "saturat", "diminish", "limit"),
      ("computer_architecture", "physics", "chemistry")),
)


#: Editorial tie-break weight per facet. Facets with the same keyword/domain
#: score are ordered by this weight, so structurally interesting dimensions
#: (scheduling, interconnect) are not pushed out by an alphabetical accident.
#: This is an editorial choice about *exploration order only*: it never touches
#: evidence, confidence or any dimension of a direction.
FACET_PRIORITY: Dict[str, int] = {
    "scheduling": 3,
    "interconnect_communication": 3,
    "memory": 2,
    "computing_model": 2,
    "architecture": 2,
    "control_adaptation": 2,
    "energy_thermal": 2,
    "data_representation": 2,
    "materials": 2,
    "manufacturing": 1,
    "reliability": 1,
    "software_stack": 1,
    "cost_scaling": 1,
}


#: Used when neither the problem text nor the detected domain selects a facet.
GENERIC_FACETS: Tuple[str, ...] = (
    "architecture", "computing_model", "energy_thermal", "cost_scaling",
)


def facet_index() -> Dict[str, Facet]:
    """``facet_id -> facet`` lookup."""
    return {f.facet_id: f for f in FACETS}


def score_facets(
    problem: str, domains: List[str], limit: int = 0
) -> List[Dict[str, Any]]:
    """Rank facets for a problem by keyword hits and domain affinity.

    ``score`` is a *selection* score used only to order the possibility space.
    It is never presented as evidence and never converted into confidence.
    """
    text = str(problem or "").lower()
    detected = {str(d or "").strip().lower() for d in domains if str(d or "").strip()}
    scored: List[Dict[str, Any]] = []
    for facet in FACETS:
        matched = [kw for kw in facet.keywords if kw in text]
        score = float(len(matched))
        domain_hits = [d for d in facet.domains if d.lower() in detected]
        score += 1.0 * len(domain_hits)
        scored.append(
            {
                "facet": facet,
                "facet_id": facet.facet_id,
                "score": round(score, 3),
                "matched_keywords": matched,
                "matched_domains": domain_hits,
                "priority": FACET_PRIORITY.get(facet.facet_id, 0),
                "selection_basis": (
                    "keyword overlap with the problem statement"
                    + (" + detected-domain affinity" if domain_hits else "")
                    + " + editorial exploration priority (order only)"
                ),
                "selection_score_note": (
                    "Selection score for ordering the possibility space only — "
                    "not evidence and not confidence."
                ),
            }
        )
    scored.sort(
        key=lambda r: (
            -r["score"],
            -FACET_PRIORITY.get(r["facet_id"], 0),
            r["facet_id"],
        )
    )
    if limit > 0:
        return scored[:limit]
    return scored


def select_facets(
    problem: str, domains: List[str], limit: int = 4
) -> List[Dict[str, Any]]:
    """The facets the possibility space will actually explore.

    Selection order: keyword-scored facets with a positive score, then
    facets central to the detected domain, then :data:`GENERIC_FACETS`.
    """
    ranked = score_facets(problem, domains)
    chosen = [r for r in ranked if r["score"] > 0]
    if len(chosen) < limit:
        seen = {r["facet_id"] for r in chosen}
        for record in ranked:
            if record["facet_id"] in seen:
                continue
            if record["matched_domains"]:
                chosen.append(record)
                seen.add(record["facet_id"])
            if len(chosen) >= limit:
                break
    if not chosen:
        index = facet_index()
        chosen = [
            {
                "facet": index[fid],
                "facet_id": fid,
                "score": 0.0,
                "matched_keywords": [],
                "matched_domains": [],
                "priority": FACET_PRIORITY.get(fid, 0),
                "selection_basis": "generic fallback facet (no keyword or domain signal)",
                "selection_score_note": (
                    "Selection score for ordering the possibility space only — "
                    "not evidence and not confidence."
                ),
            }
            for fid in GENERIC_FACETS
            if fid in index
        ]
    return chosen[:limit]


def facets_for_domain(domain: str) -> List[Facet]:
    """Facets that are central to a detected domain (used for selection only)."""
    return [f for f in FACETS if domain and str(domain).lower() in
            {d.lower() for d in f.domains}]
