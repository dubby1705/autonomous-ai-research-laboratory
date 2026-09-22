"""
AARL Discovery — Cross-Domain Transferable Concept Library
==========================================================
A curated, inspectable library of *transferable mechanisms*: ideas that already
exist in one field and can be carried into another. This is what lets AARL
propose a direction such as "decentralized scheduling" by combining

    distributed systems + swarm intelligence  ->  scheduling of <subject>

Honesty rules
-------------
* Every concept is a **library entry**, not a discovered fact. Directions built
  from it carry construction provenance (``provenance.constructed_by``) so the
  reader can see exactly how the label was assembled.
* A concept entry carries **no evidence claim of its own**. Evidence is attached
  later, only from provenance-linked knowledge items.
* ``testability`` / ``novelty`` / ``risk`` are *library metadata* describing the
  mechanism in the abstract. They are never presented as a score of scientific
  truth, and a direction's own dimensions are computed separately from that
  direction's own records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TransferableConcept:
    """One mechanism that can be transferred between domains."""

    concept_id: str
    name: str
    #: Adjective used to build direction titles ("Decentralized GPU Scheduling").
    title_adjective: str
    #: Where the mechanism comes from (shown in the "Inspired by" block).
    source_domains: Tuple[str, ...]
    description: str
    #: Ordered mechanism stages. ``{subject}`` / ``{facet}`` are substituted.
    mechanism_steps: Tuple[str, ...]
    #: Question template using ``{metric}``, ``{subject}`` and ``{facet}``.
    core_question: str
    #: What is typically unknown about this mechanism class (mechanism-level).
    uncertainties: Tuple[str, ...]
    #: Library metadata — descriptive, NOT a claim of truth.
    testability: str          # HIGH / MEDIUM / LOW
    novelty: str              # HIGH / MEDIUM / LOW
    risk: str                 # HIGH / MEDIUM / LOW
    #: Facet ids this mechanism can plausibly be applied to.
    facets: Tuple[str, ...]
    #: Terms used to look the mechanism up in the local paper library.
    literature_terms: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "title_adjective": self.title_adjective,
            "source_domains": list(self.source_domains),
            "description": self.description,
            "mechanism_steps": list(self.mechanism_steps),
            "core_question": self.core_question,
            "uncertainties": list(self.uncertainties),
            "library_metadata": {
                "testability": self.testability,
                "novelty": self.novelty,
                "implementation_risk": self.risk,
                "note": (
                    "Library metadata describing the mechanism in the abstract. "
                    "It is NOT a score of scientific truth and is NOT evidence "
                    "for any direction."
                ),
            },
            "facet_affinity": list(self.facets),
            "literature_terms": list(self.literature_terms),
        }


def C(concept_id, name, adjective, domains, description, steps, question,
      uncertainties, testability, novelty, risk, facets, terms=()):
    """Compact positional builder (keeps the library readable as a table)."""
    return TransferableConcept(
        concept_id=concept_id, name=name, title_adjective=adjective,
        source_domains=tuple(domains), description=description,
        mechanism_steps=tuple(steps), core_question=question,
        uncertainties=tuple(uncertainties), testability=testability,
        novelty=novelty, risk=risk, facets=tuple(facets), literature_terms=tuple(terms),
    )

#: The library. Order matters only for deterministic tie-breaking.
CONCEPTS: Tuple[TransferableConcept, ...] = (
    C(
        "decentralized_coordination", "Decentralized coordination", "Decentralized",
        ("distributed systems", "swarm intelligence", "multi-agent systems"),
        "Replace a single global controller with many local decision makers that "
        "coordinate through cheap local signals.",
        (
            "Work arrives at {subject} as a stream of independent units",
            "Each unit is placed on a participant that sees only its own local state",
            "Participants exchange bounded local signals instead of global state",
            "Local decisions redistribute work as conditions change",
            "Participants process concurrently",
            "Partial results are aggregated back into a single answer",
        ),
        "Can decentralized coordination improve {metric} for {subject} in "
        "situations where a single centralized controller cannot observe the whole "
        "system quickly enough?",
        (
            "Communication and synchronization overhead between participants",
            "Convergence time of local decisions versus one global decision",
            "Behaviour when participants are heterogeneous or unreliable",
        ),
        "HIGH", "MEDIUM", "MEDIUM",
        ("scheduling", "interconnect_communication", "control_adaptation",
         "reliability", "software_stack", "cost_scaling"),
        ("decentralized scheduling", "distributed coordination", "swarm scheduling"),
    ),
    C(
        "hierarchical_decomposition", "Hierarchical decomposition", "Hierarchical",
        ("systems engineering", "organisational theory", "graph partitioning"),
        "Use different decision granularities at different levels so local freedom "
        "and global coherence can coexist.",
        (
            "The {facet} problem is split into nested decision levels",
            "Low levels decide locally and fast from local information only",
            "A lightweight upper level sets only the boundaries between regions",
            "Each level reports a compact summary upward",
            "Upper-level corrections arrive as constraints, not commands",
            "The system re-partitions when the summary shows imbalance",
        ),
        "Can a two-level hierarchy improve {metric} for {subject} compared with "
        "either a purely global or a purely local {facet} decision rule?",
        (
            "Where the optimal boundary between levels lies",
            "How often the upper level must intervene before it becomes the bottleneck",
            "Whether re-partitioning cost is repaid by better balance",
        ),
        "HIGH", "LOW", "LOW",
        ("scheduling", "memory", "interconnect_communication", "architecture",
         "control_adaptation", "cost_scaling"),
        ("hierarchical scheduling", "two-level scheduling", "hybrid coordination"),
    ),
    C(
        "substrate_substitution", "Substrate substitution", "Alternative-Substrate",
        ("photonics", "neuromorphic engineering", "analog computing",
         "condensed matter physics"),
        "Change the physical carrier that performs the operation instead of "
        "optimising the existing carrier further.",
        (
            "Identify the operation in {subject} that dominates cost or energy",
            "Select a physical carrier better matched to that operation",
            "Design the interface between the new carrier and the existing stack",
            "Move only the dominant operation onto the new carrier",
            "Measure the offsetting cost of the new interface",
            "Decide whether the net budget improves",
        ),
        "Would moving the cost-dominant operation of {subject} onto a different "
        "physical carrier improve {metric} once interface overhead is included?",
        (
            "Interface and conversion overhead between carriers",
            "Availability and maturity of the alternative carrier",
            "Whether measured gains survive at production scale",
        ),
        "MEDIUM", "HIGH", "HIGH",
        ("computing_model", "materials", "architecture", "energy_thermal",
         "manufacturing", "data_representation"),
        ("photonic computing", "neuromorphic hardware", "analog in-memory computing"),
    ),
    C(
        "locality_exploitation", "Locality exploitation", "Locality-Aware",
        ("cache theory", "graph partitioning", "computational neuroscience"),
        "Restructure work so whatever is needed next is already nearby and no "
        "movement is required.",
        (
            "Profile which data movements dominate {subject} cost",
            "Partition work to match the natural data neighbourhoods",
            "Place each partition next to the data it repeatedly reads",
            "Reorder execution to keep the working set resident",
            "Track whether reuse actually increased",
            "Re-partition when the working set outgrows the local capacity",
        ),
        "Can locality-aware partitioning improve {metric} for {subject} whose "
        "access patterns are systematic but not uniform?",
        (
            "How much reuse the real workload actually exhibits",
            "Whether partitioning cost is repaid at realistic sizes",
            "Sensitivity to workload phase changes",
        ),
        "HIGH", "LOW", "MEDIUM",
        ("memory", "interconnect_communication", "scheduling", "data_representation",
         "computing_model", "architecture"),
        ("data locality", "cache-aware scheduling", "memory hierarchy"),
    ),
    C(
        "approximate_computing", "Approximate computing", "Approximate",
        ("signal processing", "cognitive science", "numerical analysis"),
        "Spend precision only where the final answer is sensitive to it.",
        (
            "Determine which parts of {subject} output are precision-sensitive",
            "Assign a precision budget per part instead of one global precision",
            "Execute low-sensitivity parts with reduced precision or fewer steps",
            "Detect when a declared error bound would be breached",
            "Escalate precision only for those parts",
            "Verify end-to-end accuracy stays inside the declared bound",
        ),
        "Can per-part precision budgets improve {metric} for {subject} while "
        "keeping a declared, measurable error bound?",
        (
            "How error propagates from parts to the end-to-end answer",
            "Whether the sensitivity analysis costs more than it saves",
            "Acceptability of the resulting error in the target application",
        ),
        "HIGH", "MEDIUM", "MEDIUM",
        ("computing_model", "data_representation", "energy_thermal", "architecture",
         "materials", "software_stack"),
        ("approximate computing", "precision scaling", "mixed precision"),
    ),
    C(
        "heterogeneity_awareness", "Heterogeneity awareness", "Heterogeneity-Aware",
        ("ecology / niche partitioning", "statistics", "scheduling theory"),
        "Treat non-uniform participants as exploitable structure instead of "
        "averaging them away.",
        (
            "Characterise how the participants of {subject} actually differ",
            "Derive which work type each participant class does best",
            "Route each work type toward its best-matching class",
            "Track the per-class queue the routing creates",
            "Rebalance when one class becomes the bottleneck",
            "Re-measure the characterisation, which changes under load",
        ),
        "Can capability-matched routing improve {metric} for {subject} made of "
        "deliberately non-uniform participants?",
        (
            "Stability of the capability profiles under real load",
            "Whether matching overhead outweighs the placement gain",
            "Tail behaviour when one class is starved",
        ),
        "HIGH", "MEDIUM", "LOW",
        ("scheduling", "architecture", "cost_scaling", "interconnect_communication",
         "control_adaptation"),
        ("heterogeneous workload scheduling", "capability matching"),
    ),
    C(
        "closed_loop_feedback", "Closed-loop feedback", "Closed-Loop",
        ("control theory", "physiology", "operations research"),
        "Measure the outcome and continuously correct the {facet} decision instead "
        "of computing it once.",
        (
            "Define the measurable signal that indicates healthy operation of {subject}",
            "Observe that signal at a bounded sampling rate",
            "Compare it against a target band rather than a fixed setpoint",
            "Apply a small correction when the signal leaves the band",
            "Damp the correction to avoid oscillation",
            "Log every correction so the controller's behaviour can be audited",
        ),
        "Can closed-loop correction of {facet} improve {metric} for {subject} whose "
        "operating conditions drift during execution?",
        (
            "Control latency versus how fast conditions actually change",
            "Risk of oscillation from over-correction",
            "Cost of the observation itself",
        ),
        "HIGH", "LOW", "MEDIUM",
        ("control_adaptation", "scheduling", "energy_thermal", "reliability",
         "software_stack", "cost_scaling"),
        ("runtime feedback control", "adaptive scheduling", "closed-loop management"),
    ),
    C(
        "market_mechanism", "Market-based allocation", "Market-Based",
        ("economics", "auction theory", "grid computing"),
        "Let participants bid for the scarce {facet} resource so value, not arrival "
        "order, decides allocation.",
        (
            "Define the scarce resource and who may request it in {subject}",
            "Require each request to declare a value or deadline",
            "Run a bounded bid-and-clear round",
            "Allocate to the highest declared value per unit of scarcity",
            "Publish clearing prices so requesters can adapt",
            "Detect and correct for strategic bidding",
        ),
        "Can market-based clearing improve {metric} for {subject} when request value "
        "and resource scarcity are both known at request time?",
        (
            "Truthfulness of declared values when participants can strategise",
            "Latency added by the clearing round",
            "Whether pricing stays stable under bursty arrival",
        ),
        "MEDIUM", "MEDIUM", "MEDIUM",
        ("scheduling", "cost_scaling", "control_adaptation",
         "interconnect_communication"),
        ("market-based scheduling", "auction-based resource allocation"),
    ),
    C(
        "oscillator_synchronization", "Oscillator synchronization",
        "Oscillator-Synchronized",
        ("coupled oscillator physics", "chronobiology", "clock distribution"),
        "Replace an explicit global clock or barrier with emergent phase alignment "
        "between participants.",
        (
            "Give each participant of {subject} a local timing source",
            "Let neighbours nudge each other's phase slightly",
            "Observe whether global phase alignment emerges",
            "Use alignment as the coordination signal instead of a broadcast barrier",
            "Bound the phase drift the system tolerates",
            "Measure the energy spent maintaining alignment",
        ),
        "Can emergent phase alignment replace an explicit global barrier to improve "
        "{metric} for {subject}?",
        (
            "Lock-in time versus workload duration",
            "Energy cost of maintaining alignment",
            "Behaviour when coupling is interrupted",
        ),
        "MEDIUM", "HIGH", "HIGH",
        ("interconnect_communication", "computing_model", "architecture",
         "control_adaptation", "energy_thermal"),
        ("coupled oscillator computing", "clockless synchronization"),
    ),
    C(
        "redundancy_error_correction", "Structured redundancy",
        "Redundancy-Protected",
        ("coding theory", "DNA repair biology", "fault-tolerant computing"),
        "Accept that components will be imperfect and encode the information so "
        "imperfection stays recoverable.",
        (
            "Identify which failure modes of {subject} are likely but not fatal",
            "Encode the computation or data with structured redundancy",
            "Detect deviation at the point where it appears",
            "Correct locally instead of restarting globally",
            "Bound the failure rate the encoding can absorb",
            "Measure the resource cost of the redundancy",
        ),
        "Can structured redundancy relax the manufacturing or reliability tolerance "
        "of {subject} enough to improve {metric}?",
        (
            "Overhead of redundancy against the yield or reliability actually gained",
            "Correction latency versus how fast faults accumulate",
            "Whether failure modes are independent as assumed",
        ),
        "MEDIUM", "LOW", "MEDIUM",
        ("reliability", "manufacturing", "materials", "memory", "architecture",
         "cost_scaling"),
        ("fault-tolerant architecture", "error correction hardware", "redundancy yield"),
    ),
    C(
        "predictive_prefetching", "Predictive prefetching", "Predictive",
        ("machine learning", "cognitive science", "queueing theory"),
        "Use a cheap learned predictor to act before the need becomes observable.",
        (
            "Log the sequence of {facet} events for {subject}",
            "Train a small, cheap predictor on those sequences",
            "Act speculatively on the prediction",
            "Validate speculatively produced work when the need really appears",
            "Roll back or discard when the prediction was wrong",
            "Monitor prediction accuracy and disable speculation below a floor",
        ),
        "Can a cheap predictor acting before demand is observable improve {metric} "
        "for {subject} with predictable-but-delayed access sequences?",
        (
            "Predictor accuracy on the real, non-stationary workload",
            "Cost of speculative work that turns out to be wasted",
            "Whether prediction cost is itself non-trivial",
        ),
        "HIGH", "MEDIUM", "MEDIUM",
        ("memory", "scheduling", "interconnect_communication", "control_adaptation",
         "software_stack", "data_representation"),
        ("prefetching", "speculative scheduling", "learned prediction cache"),
    ),
    C(
        "sparsity_compression", "Sparsity exploitation", "Sparsity-Driven",
        ("information theory", "compressed sensing", "neuroscience"),
        "Exploit measured structure to stop storing and moving what is effectively "
        "empty.",
        (
            "Measure how much of {subject}'s state or traffic is redundant",
            "Choose a representation that exploits the measured structure",
            "Skip work on structurally empty parts instead of masking them",
            "Preserve the metadata needed to reconstruct the full result",
            "Measure metadata cost against the work removed",
            "Handle the worst case where the structure disappears",
        ),
        "Can structure-aware representation improve {metric} for {subject} whose "
        "state is measurably sparse?",
        (
            "Metadata and bookkeeping overhead",
            "Behaviour on dense or irregular inputs",
            "Whether the structure survives across workload phases",
        ),
        "HIGH", "MEDIUM", "LOW",
        ("data_representation", "memory", "computing_model", "energy_thermal",
         "software_stack", "interconnect_communication"),
        ("sparsity exploitation", "sparse computation", "compressed representation"),
    ),
    C(
        "adaptive_reconfiguration", "Adaptive reconfiguration", "Self-Reconfiguring",
        ("control theory", "biology / homeostasis", "reconfigurable hardware"),
        "Let the {facet} structure change shape while the workload runs.",
        (
            "Classify the workload phase {subject} is currently in",
            "Define a small set of structural configurations, each tied to a phase",
            "Switch configuration on a phase change",
            "Bound the cost of the switch itself",
            "Prevent thrashing between two near-equal phases",
            "Audit how often each configuration is selected",
        ),
        "Can phase-matched reconfiguration improve {metric} for {subject} whose "
        "workload changes character during execution?",
        (
            "Cost and frequency of reconfiguration events",
            "Accuracy of phase classification",
            "Risk of oscillation between configurations",
        ),
        "HIGH", "MEDIUM", "HIGH",
        ("architecture", "control_adaptation", "computing_model", "energy_thermal",
         "software_stack", "scheduling"),
        ("reconfigurable architecture", "dynamic reconfiguration", "adaptive hardware"),
    ),
    C(
        "domain_specific_language", "Domain-specific representation", "Domain-Specific",
        ("compiler theory", "operations research", "type theory"),
        "Change what the system is allowed to express so unwanted cases become "
        "unrepresentable.",
        (
            "List the {facet} configurations that {subject} must never enter",
            "Choose a representation where those configurations cannot be written",
            "Compile or interpret that representation with the guarantee preserved",
            "Check the guarantee against the escaped, lower-level path",
            "Measure the expressiveness that was given up",
            "Collect the requests the representation could not express",
        ),
        "Can a restricted, domain-specific representation improve {metric} for "
        "{subject} by making invalid {facet} states unrepresentable?",
        (
            "Whether the surrendered expressiveness blocks real use cases",
            "Correctness of the compilation or interpretation step",
            "Cost of the escape hatch for unmet needs",
        ),
        "HIGH", "MEDIUM", "MEDIUM",
        ("software_stack", "data_representation", "architecture",
         "control_adaptation", "reliability"),
        ("domain-specific language hardware", "compiler scheduling"),
    ),
    C(
        "phase_change_bistability", "Nonlinear bistability", "Bistable",
        ("phase-change physics", "nonlinear dynamics", "memristor devices"),
        "Use a physical state that is stable without power, turning the {facet} "
        "state itself into free memory.",
        (
            "Identify the {facet} state of {subject} that must persist while idle",
            "Map it onto a physical state that holds without power",
            "Define the write and read procedures and their energy cost",
            "Measure retention against the required idle duration",
            "Measure endurance against the expected write count",
            "Compare against the volatile alternative on the same metric",
        ),
        "Can a non-volatile physical state improve {metric} for {subject} whose "
        "{facet} state is currently refreshed continuously?",
        (
            "Endurance under the real write frequency",
            "Retention time at the required temperature",
            "Write cost versus the refresh energy removed",
        ),
        "MEDIUM", "HIGH", "HIGH",
        ("memory", "data_representation", "materials", "energy_thermal",
         "manufacturing", "computing_model"),
        ("non-volatile memory", "phase change memory", "memristor computing"),
    ),
    C(
        "topological_protection", "Topological protection", "Topologically-Protected",
        ("condensed matter physics", "topology", "quantum error correction"),
        "Encode information in a global property that local perturbations cannot "
        "destroy.",
        (
            "Identify the local perturbation modes that threaten {subject}",
            "Express the information in a globally redundant property",
            "Confirm that local perturbation cannot change that property",
            "Define the readout procedure for the global property",
            "Measure the readout overhead",
            "Bound the perturbation strength the protection survives",
        ),
        "Can encoding {facet} in a locally-robust global property improve {metric} "
        "for {subject} operating in a noisy environment?",
        (
            "Readout overhead of the global property",
            "Physical feasibility at the required operating conditions",
            "Whether protection survives thermal and manufacturing variation",
        ),
        "LOW", "HIGH", "HIGH",
        ("reliability", "computing_model", "data_representation", "materials",
         "manufacturing", "architecture"),
        ("topological protection", "topological quantum computing"),
    ),
    C(
        "self_assembly_emergence", "Self-assembly", "Self-Assembling",
        ("chemistry", "crystallography", "molecular biology"),
        "Let the desired structure form from local interaction rules instead of "
        "being placed step by step.",
        (
            "Specify the local interaction rules that must produce the target "
            "structure of {subject}",
            "Verify that the rules have a single stable outcome",
            "Let the structure form without a global placement plan",
            "Detect and correct rule violations",
            "Measure defect density against a placement-based route",
            "Confirm process-variation tolerance",
        ),
        "Can rule-based self-assembly reach an acceptable {metric} for {subject} "
        "without step-by-step global placement?",
        (
            "Defect density versus a deterministic placement route",
            "Yield sensitivity to process variation",
            "Existence of a single stable outcome for the chosen rules",
        ),
        "MEDIUM", "HIGH", "HIGH",
        ("materials", "manufacturing", "architecture", "reliability"),
        ("self-assembly", "directed self-assembly fabrication"),
    ),
    C(
        "stochastic_sampling", "Stochastic sampling", "Stochastic-Sampling",
        ("statistical mechanics", "Monte Carlo methods", "population genetics"),
        "Replace exhaustive evaluation with a bounded, statistically controlled "
        "sample of the space.",
        (
            "Define the {facet} space that would otherwise be exhausted",
            "Choose a sampling distribution with bounded error",
            "Sample and evaluate a small subset",
            "Estimate the confidence interval of the result",
            "Enlarge the sample only while the interval is too wide",
            "Record the sample size that was actually used",
        ),
        "Can bounded stochastic sampling replace exhaustive {facet} search for "
        "{subject} while keeping a declared confidence level?",
        (
            "Whether the estimator's variance is acceptable given the budget",
            "Behaviour on rare but important cases that sampling misses",
            "Statistical validity when the search space is not stationary",
        ),
        "HIGH", "LOW", "LOW",
        ("control_adaptation", "scheduling", "cost_scaling", "software_stack",
         "data_representation"),
        ("stochastic search", "sampling-based planning", "Monte Carlo scheduling"),
    ),
    C(
        "reversible_computing", "Thermodynamic reversibility", "Reversible",
        ("thermodynamics", "Landauer principle", "adiabatic circuits"),
        "Remove the intrinsic energy cost of discarding information by making the "
        "operation reversible.",
        (
            "Identify which operations of {subject} irreversibly discard information",
            "Reformulate them as reversible operations with kept history",
            "Account for the storage the reversibility requires",
            "Drive the reversible operation slowly enough to stay near-adiabatic",
            "Measure whether the stored history costs less than the energy saved",
            "Bound the operating range where the trade remains favourable",
        ),
        "Can making the information-destroying operations of {subject} reversible "
        "improve {metric} once the required history storage is counted?",
        (
            "Whether the history storage costs less than the energy saved",
            "Speed limits imposed by near-adiabatic operation",
            "Physical realisation of reversible primitives",
        ),
        "LOW", "HIGH", "HIGH",
        ("energy_thermal", "computing_model", "materials", "manufacturing",
         "architecture"),
        ("reversible computing", "adiabatic circuits", "Landauer limit computing"),
    ),
)


def concept_index() -> Dict[str, TransferableConcept]:
    """``concept_id -> concept`` lookup."""
    return {c.concept_id: c for c in CONCEPTS}


def concepts_for_facet(facet_id: str) -> List[TransferableConcept]:
    """Concepts whose affinity list includes ``facet_id`` (library order kept)."""
    return [c for c in CONCEPTS if facet_id in c.facets]


def clean_phrases() -> List[str]:
    """Every free-text phrase in the library, for the self-check test."""
    out: List[str] = []
    for concept in CONCEPTS:
        out.extend([concept.name, concept.description, concept.core_question])
        out.extend(concept.mechanism_steps)
        out.extend(concept.uncertainties)
    return out

