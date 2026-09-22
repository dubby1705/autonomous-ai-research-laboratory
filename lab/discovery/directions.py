"""
AARL Discovery - Research Direction Discovery + Possibility Space
=================================================================
Turns a research problem into a set of **research directions** and the tree that
shows where they came from.

Construction (fully inspectable):

    research problem
      -> facets          (structural dimensions of the problem; facets.py)
      -> concepts         (cross-domain transferable mechanisms; concepts.py)
      -> directions       (facet x concept, titled and given a core question)
      -> possibility space (root -> facet -> direction, plus reconsidered branches)

Honesty rules
-------------
* A direction is a *construction*, and its provenance records exactly which
  facet, which library concept and which knowledge items produced its wording.
* Prior failure memory is consulted first: a candidate that AARL already failed is
  marked ``already_failed`` with the FAIL id, and its refined variant is proposed
  instead of silently repeating the same idea.
* Titles use the problem's own object of study (heuristic, documented in text.py).
  Nothing here claims the direction is correct or novel in the literature.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from .concepts import TransferableConcept, concepts_for_facet
from .facets import select_facets
from .mechanism import build_mechanism, build_why_chain
from .refine import build_refinement, choose_operator
from .text import content_token_set, display_term, subject_of

log = logging.getLogger("aarl.lab.discovery.directions")

#: Library metadata -> rank value (used only for concept selection order).
_LEVEL_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _overlap(left: str, right: str) -> float:
    a, b = content_token_set(left), content_token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def rank_concepts(facet_id: str) -> List[TransferableConcept]:
    """Concept candidates for a facet, ordered by library metadata then order.

    Ordering only: novelty first, then lower realisation risk, then higher
    testability. Library metadata never enters a direction's dimensions without
    being labelled as library metadata.
    """
    candidates = list(concepts_for_facet(facet_id))
    return sorted(
        candidates,
        key=lambda c: (
            -_LEVEL_RANK.get(c.novelty, 2),
            _LEVEL_RANK.get(c.risk, 2),
            -_LEVEL_RANK.get(c.testability, 2),
        ),
    )

def _failure_index(memory) -> Dict[str, List[Dict[str, Any]]]:
    """Map ``mechanism key`` -> the FAIL records AARL already holds for it.

    The key is ``<facet_id>:<concept_id>``. It is written onto every hypothesis a
    direction generates, so a stored failure can be traced back to the exact
    mechanism that produced it - no fuzzy matching is required for that path.
    """
    hypotheses = {str(h.get("hypothesis_id", "")): h
                  for h in memory.load_collection("hypotheses")}
    key_by_experiment: Dict[str, str] = {}
    for sim in memory.load_collection("simulations"):
        hyp = hypotheses.get(str(sim.get("hypothesis_id", "")))
        key = str((hyp or {}).get("direction_mechanism_key", "") or "")
        if key:
            key_by_experiment[str(sim.get("experiment_id", ""))] = key
    index: Dict[str, List[Dict[str, Any]]] = {}
    for failure in memory.load_collection("failures"):
        key = str(failure.get("direction_mechanism_key", "") or "")
        if not key:
            key = key_by_experiment.get(str(failure.get("experiment_id", "")), "")
        if not key:
            continue
        index.setdefault(key, []).append(failure)
    return index


class DirectionDiscovery:
    """Builds research directions and the possibility-space tree for a problem."""

    def __init__(self, memory, knowledge, config=None, llm_service=None) -> None:
        self.memory = memory
        self.knowledge = knowledge
        self.config = config
        self.llm = llm_service

    # ------------------------------------------------------------- discovery
    def discover(
        self,
        problem: str,
        domains: List[Dict[str, Any]],
        primary_domain: str = "",
        run_id: str = "",
        per_facet: int = 2,
        max_directions: int = 8,
        facet_limit: int = 4,
    ) -> Dict[str, Any]:
        """Discover directions for a problem and persist them (append-only)."""
        domain_names = [
            str(d.get("domain", d) if isinstance(d, dict) else d) for d in (domains or [])
        ]
        subject = display_term(subject_of(problem)) or "the system"
        knowledge_items = self._knowledge_dicts()
        failure_index = _failure_index(self.memory)

        facet_records = select_facets(problem, domain_names, limit=facet_limit)
        directions: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        considered_concepts = 0
        try:
            start_index = int(self.memory.next_id("directions", "DIR").split("-")[-1])
        except Exception:  # noqa: BLE001 - fall back to a fresh numbering
            start_index = 1

        for facet_record in facet_records:
            facet = facet_record["facet"]
            ranked = rank_concepts(facet.facet_id)
            considered_concepts += len(ranked)
            taken = 0
            for position, concept in enumerate(ranked):
                key = "%s:%s" % (facet.facet_id, concept.concept_id)
                prior = list(failure_index.get(key) or [])
                if taken >= per_facet or len(directions) >= max_directions:
                    rejected.append(self._rejected(
                        facet, concept, key,
                        reason="candidate limit reached for this run",
                        matched_keywords=facet_record.get("matched_keywords") or [],
                    ))
                    continue
                taken += 1
                directions.append(self._build_direction(
                    problem=problem, subject=subject, facet_record=facet_record,
                    concept=concept, key=key, prior_failures=prior,
                    knowledge_items=knowledge_items, run_id=run_id,
                    order=len(directions) + 1,
                    direction_id="DIR-%03d" % (start_index + len(directions)),
                ))

        space = possibility_space(problem, subject, facet_records, directions, rejected,
                                  domain_names)
        stats = {
            "facets_explored": len(facet_records),
            "concepts_considered": considered_concepts,
            "directions": len(directions),
            "candidates_rejected": len(rejected),
            "already_failed_in_memory": sum(1 for d in directions if d["already_failed"]),
            "subject_label": subject,
            "domains_considered": domain_names,
            "how_directions_were_built": (
                "facet x cross-domain concept. Facets are selected by keyword/domain "
                "overlap with the problem; concepts come from the curated transferable "
                "mechanism library. The provenance block of each direction lists the "
                "exact inputs."
            ),
        }
        for direction in directions:
            self.memory.append_record("directions", direction["direction_id"], direction)
            self.memory.append_history({
                "event": "direction_discovered",
                "direction_id": direction["direction_id"],
                "title": direction["title"],
                "run_id": run_id,
                "already_failed": direction["already_failed"],
            })
        return {
            "directions": directions,
            "space": space,
            "stats": stats,
            "rejected": rejected,
            "subject": subject,
        }
    # ------------------------------------------------------------- internals
    def _knowledge_dicts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            for item in self.knowledge.items():
                out.append(item.to_dict() if hasattr(item, "to_dict") else dict(item))
        except Exception as exc:
            log.warning("knowledge items unavailable: %s", exc)
        return out

    @staticmethod
    def _rejected(facet, concept, key, reason: str, matched_keywords=None) -> Dict[str, Any]:
        return {
            "mechanism_key": key,
            "facet_id": facet.facet_id,
            "facet_label": facet.label,
            "concept_id": concept.concept_id,
            "concept_name": concept.name,
            "title": "%s %s" % (concept.title_adjective, facet.title_noun),
            "reason": reason,
            "matched_keywords": list(matched_keywords or []),
            "note": (
                "This branch was considered and not explored in this run. It is "
                "recorded so the possibility space shows what was left out."
            ),
        }

    def _build_direction(
        self,
        problem: str,
        subject: str,
        facet_record: Dict[str, Any],
        concept,
        key: str,
        prior_failures: List[Dict[str, Any]],
        knowledge_items: List[Dict[str, Any]],
        run_id: str,
        order: int,
        direction_id: str = "",
    ) -> Dict[str, Any]:
        facet = facet_record["facet"]
        direction_id = direction_id or self.memory.next_id("directions", "DIR")
        title = "%s %s %s" % (concept.title_adjective, subject, facet.title_noun)
        core_question = str(concept.core_question).format(
            metric=facet.metric, subject=subject, facet=facet.title_noun.lower()
        )
        mechanism = build_mechanism(concept, facet, subject, knowledge_items)
        why_chain = build_why_chain(problem, facet, concept, "", subject, knowledge_items)

        already_failed = bool(prior_failures)
        refinement = None
        for failure in prior_failures[:1]:
            refinement = build_refinement(
                {"direction_id": direction_id, "title": title},
                failure,
                lesson=None,
                operator=choose_operator(failure),
            )
        title = refinement["title"] if refinement else title

        related_papers = self._related_papers(concept)
        unknowns = list(concept.uncertainties)
        speculative = [
            "The mechanism is transferred from %s and has not been demonstrated for %s."
            % (", ".join(concept.source_domains), subject),
            "AARL has no implemented prototype of this mechanism for this problem.",
        ]
        if facet_record.get("matched_keywords"):
            speculative.append(
                "The facet (%s) was selected using keyword overlap on: %s."
                % (facet.label, ", ".join(facet_record["matched_keywords"]))
            )
        direction = {
            "direction_id": direction_id,
            "run_id": run_id,
            "order": order,
            "title": title,
            "base_title": "%s %s %s" % (concept.title_adjective, subject, facet.title_noun),
            "one_line": core_question,
            "core_question": core_question,
            "research_problem": problem,
            "subject": subject,
            "claim_class": "DIRECTION_PROPOSAL",
            "claim_class_label": "RESEARCH DIRECTION (AARL PROPOSAL)",
            "facet": facet.to_dict(),
            "concept": concept.to_dict(),
            "mechanism_key": key,
            "inspiration": {
                "domains": list(concept.source_domains),
                "concepts": [concept.name],
                "library_entry": "aarl.discovery.concepts:%s" % concept.concept_id,
                "library_metadata": concept.to_dict()["library_metadata"],
                "papers": related_papers,
                "display": " + ".join(list(concept.source_domains)[:3]),
            },
            "why_interesting": self._why_interesting(subject, facet, concept),
            "why_this_idea": why_chain,
            "mechanism": mechanism,
            "already_failed": already_failed,
            "prior_failures": [
                {
                    "failure_id": f.get("failure_id", ""),
                    "experiment_id": f.get("experiment_id", ""),
                    "failure_type": f.get("failure_type", ""),
                    "severity": f.get("severity", ""),
                    "observed_behavior": f.get("observed_behavior", ""),
                    "failed_assumption": f.get("failed_assumption", ""),
                    "recommended_changes": list(f.get("recommended_changes") or []),
                }
                for f in prior_failures[:3]
            ],
            "refinement": refinement,
            "hypothesis_ids": [],
            "experiment_ids": [],
            "failure_ids": [f.get("failure_id", "") for f in prior_failures],
            "lesson_ids": [],
            "status": None,
            "dimensions": {},
            "evidence": {},
            "gaps": [],
            "speculative_aspects": speculative,
            "unknowns": unknowns,
            "risks": [
                "%s (open question for this mechanism class)" % u for u in unknowns[:3]
            ],
            "failure_conditions": self._failure_conditions(subject, facet, concept),
            "first_experiment": self._first_experiment(subject, facet, concept),
            "alternatives": [],
            "possibility_space_path": [
                "Research problem", facet.label, "%s (%s)" % (concept.name, concept.concept_id),
                title,
            ],
            "provenance": {
                "constructed_by": "aarl.discovery.directions.DirectionDiscovery",
                "research_question": problem,
                "facet_selection": {
                    "facet_id": facet.facet_id,
                    "score": facet_record.get("score"),
                    "matched_keywords": list(facet_record.get("matched_keywords") or []),
                    "matched_domains": list(facet_record.get("matched_domains") or []),
                    "selection_basis": facet_record.get("selection_basis"),
                    "selection_score_note": facet_record.get("selection_score_note"),
                },
                "concept_library_entry": "aarl.discovery.concepts:%s" % concept.concept_id,
                "knowledge_items_consulted": len(knowledge_items),
                "prior_failure_memory_consulted": len(prior_failures),
                "subject_label_heuristic": (
                    "The object of study was labelled '%s' by a documented text "
                    "heuristic (lab/discovery/text.py:subject_of). It is a label only."
                    % subject
                ),
                "what_this_is": (
                    "A constructed research direction, not a result. Evidence for it is "
                    "attached later and only from provenance-linked records."
                ),
            },
            "timestamp": _now(),
        }
        return direction
    # ------------------------------------------------------ helper builders
    def _related_papers(self, concept) -> List[Dict[str, Any]]:
        """Papers already in AARL's library whose text matches this mechanism.

        Purely a *lead*: matching titles/abstracts are never treated as evidence.
        """
        out: List[Dict[str, Any]] = []
        for paper in self.memory.load_collection("papers"):
            text = " ".join([
                str(paper.get("title") or ""),
                str((paper.get("extraction") or {}).get("research_problem") or ""),
            ])
            best = max([_overlap(text, term) for term in concept.literature_terms] or [0.0])
            if best >= 0.25:
                out.append({
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "match": round(best, 3),
                    "note": "Lead only - not evidence for the direction.",
                })
        out.sort(key=lambda r: (-r["match"], str(r["paper_id"])))
        return out[:5]

    @staticmethod
    def _why_interesting(subject: str, facet, concept) -> str:
        return (
            "The problem's '{}' dimension ({}) is where the conventional approach carries "
            "a structural assumption: {} AARL took a mechanism that already works in {} "
            "- {} - and asked what happens if it is applied to {}. If the mechanism "
            "transfers, the gain would come from removing that assumption rather than "
            "from tuning the existing design. Whether it transfers is exactly what is "
            "not established here."
        ).format(
            facet.label,
            facet.framing,
            facet.limitation,
            ", ".join(concept.source_domains),
            concept.description,
            subject,
        )

    @staticmethod
    def _failure_conditions(subject: str, facet, concept) -> List[Dict[str, str]]:
        out = [{
            "condition": "%s" % u,
            "why_it_would_break_the_direction": (
                "If this quantity grows faster than the benefit, the mechanism loses "
                "its advantage even though it stays technically correct."
            ),
        } for u in concept.uncertainties]
        out.append({
            "condition": (
                "The assumed limitation of the %s facet does not actually hold for "
                "this system." % facet.label
            ),
            "why_it_would_break_the_direction": (
                "The direction exists to remove that limitation; if the limitation is "
                "not real, the direction has no target (this is the first assumption "
                "to check)."
            ),
        })
        return out

    @staticmethod
    def _first_experiment(subject: str, facet, concept) -> Dict[str, Any]:
        return {
            "objective": (
                "Establish whether the mechanism moves %s for %s at all, before "
                "tuning anything." % (facet.metric, subject)
            ),
            "design": (
                "Baseline A: the conventional %s approach. Variant B: the %s "
                "mechanism applied only to the %s dimension. Keep every other "
                "parameter identical and use the same seed/warm-up."
                % (facet.label.lower(), concept.name, facet.title_noun.lower())
            ),
            "metrics": [facet.metric, "coordination or overhead cost introduced"],
            "decision_rule": (
                "Only claim support if B moves %s outside the run-to-run spread of A; "
                "if the spread is larger than the effect, report INCONCLUSIVE rather "
                "than a direction." % facet.metric
            ),
            "what_would_make_it_stop": (
                "If A is already at the physical or structural limit of %s, the "
                "comparison cannot discriminate and the design must change."
                % facet.metric
            ),
            "claim_class": "EXPERIMENT_PROPOSAL",
        }
    # ------------------------------------------------------------ evaluation
    def evaluate(
        self,
        direction: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        lessons: Optional[List[Dict[str, Any]]] = None,
        all_directions: Optional[List[Dict[str, Any]]] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Attach status, dimensions, evidence, gaps and result provenance.

        Everything attached here is read from stored records. Nothing is attached
        from model prose.
        """
        from .dimensions import dimensions_for
        from .evidence import classify_simulation, evidence_panel
        from .gaps import research_gaps
        from .status import EvidenceCounters, classify

        lessons = list(lessons or [])
        all_directions = list(all_directions or [])
        knowledge_items = self._knowledge_dicts()
        items_by_id = {str(i.get("item_id", "")): i for i in knowledge_items}
        papers_by_id = {str(p.get("paper_id", "")): p
                        for p in self.memory.load_collection("papers")}
        concept = _ConceptView(direction.get("concept") or {})
        facet = _FacetView(direction.get("facet") or {})

        counters = EvidenceCounters()
        for hyp in hypotheses:
            for item_id in hyp.get("supporting_evidence") or []:
                item = items_by_id.get(str(item_id))
                if item is None:
                    continue
                if str(item.get("claim_type")) in ("sourced", "real_world_observation"):
                    counters.direct += 1
                else:
                    counters.indirect += 1
            for item_id in hyp.get("contradictory_evidence") or []:
                if str(item_id) in items_by_id:
                    counters.contradicting += 1

        for sim in experiments:
            status = str(sim.get("status", "")).lower()
            if status in ("success", "promising", "completed"):
                counters.simulations_supporting += 1
            elif status == "partial":
                counters.simulations_partial += 1
            elif status == "simulation_unavailable":
                counters.simulations_unavailable += 1
            else:
                counters.simulations_failed += 1
            if sim.get("constraints_violated"):
                counters.simulations_failed += 0  # already failure; recorded in the panel

        panel = evidence_panel(
            hypotheses=hypotheses,
            knowledge_items=knowledge_items,
            papers_by_id=papers_by_id,
            direction_text="%s %s %s" % (direction.get("title", ""),
                                         direction.get("core_question", ""),
                                         concept.description),
            mechanism_steps=[s.get("text", "") for s in
                             (direction.get("mechanism") or {}).get("steps", [])],
            uncertainties=list(concept.uncertainties),
        )
        counters.missing = len(panel.get("missing") or [])

        hypothesis_primary = hypotheses[0] if hypotheses else {}
        knowledge_overlap = 0
        for item in knowledge_items:
            if _overlap(str(direction.get("core_question", "")),
                        str(item.get("statement") or "")) >= 0.2:
                knowledge_overlap += 1

        status = classify(counters)
        dimensions = dimensions_for(
            concept, facet, counters, experiments,
            prior_failures=list(direction.get("prior_failures") or []),
            knowledge_overlap=knowledge_overlap,
        )
        gaps = research_gaps(
            facet, concept, str(direction.get("subject") or ""),
            hypothesis=hypothesis_primary, experiments=experiments,
            failures=failures, panel=panel, lessons=lessons,
        )
        result_provenance = [classify_simulation(sim) for sim in experiments]
        assumption_based = any(
            r["result_class"] == "ASSUMPTION_BASED_RESULT" for r in result_provenance
        )
        alternatives = [
            {
                "direction_id": other.get("direction_id", ""),
                "title": other.get("title", ""),
                "status": (other.get("status") or {}).get("status", ""),
                "shared_facet": other.get("facet", {}).get("facet_id", ""),
                "why_related": (
                    "Same facet ('%s'), different mechanism."
                    % other.get("facet", {}).get("label", "")
                ),
            }
            for other in all_directions
            if other.get("direction_id") != direction.get("direction_id")
            and (other.get("facet") or {}).get("facet_id")
            == (direction.get("facet") or {}).get("facet_id")
        ]

        updated = dict(direction)
        updated.update({
            "hypothesis_ids": [h.get("hypothesis_id", "") for h in hypotheses],
            "experiment_ids": [e.get("experiment_id", "") for e in experiments],
            "failure_ids": sorted(set(
                [f.get("failure_id", "") for f in failures]
                + [f.get("failure_id", "") for f in (direction.get("prior_failures") or [])]
            )),
            "lesson_ids": [l.get("lesson_id", "") for l in lessons],
            "status": status,
            "dimensions": dimensions,
            "evidence": panel,
            "gaps": gaps,
            "alternatives": alternatives,
            "result_provenance": result_provenance,
            "is_assumption_based": assumption_based,
            "evaluated_at": _now(),
        })
        if not updated.get("one_line"):
            updated["one_line"] = updated.get("core_question", "")
        if persist:
            self.memory.append_record(
                "directions", updated["direction_id"], updated,
                revision_of="direction evaluated against stored records",
            )
        return updated

class _ConceptView:
    """Attribute view over a persisted concept dict (library metadata included)."""

    def __init__(self, data: Dict[str, Any]) -> None:
        meta = data.get("library_metadata") or {}
        self.concept_id = str(data.get("concept_id") or "")
        self.name = str(data.get("name") or "")
        self.title_adjective = str(data.get("title_adjective") or "")
        self.source_domains = list(data.get("source_domains") or [])
        self.description = str(data.get("description") or "")
        self.mechanism_steps = list(data.get("mechanism_steps") or [])
        self.core_question = str(data.get("core_question") or "")
        self.uncertainties = list(data.get("uncertainties") or [])
        self.testability = str(meta.get("testability") or "MEDIUM")
        self.novelty = str(meta.get("novelty") or "MEDIUM")
        self.risk = str(meta.get("implementation_risk") or "MEDIUM")
        self.facets = tuple(data.get("facet_affinity") or [])
        self.literature_terms = tuple(data.get("literature_terms") or [])


class _FacetView:
    """Attribute view over a persisted facet dict."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.facet_id = str(data.get("facet_id") or "")
        self.label = str(data.get("label") or "")
        self.title_noun = str(data.get("title_noun") or self.label)
        self.framing = str(data.get("framing") or "")
        self.metric = str(data.get("metric") or "the target metric")
        self.limitation = str(data.get("assumed_limitation") or "")
        self.keywords = tuple(data.get("keywords") or [])
        self.domains = tuple(data.get("domains") or [])


def possibility_space(
    problem: str,
    subject: str,
    facet_records: List[Dict[str, Any]],
    directions: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    domain_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The explorable tree: problem -> facet -> direction (+ reconsidered branches)."""
    nodes: List[Dict[str, Any]] = [{
        "id": "root",
        "label": problem,
        "kind": "research_question",
        "parent": None,
        "detail": "The problem AARL was given.",
        "domains": list(domain_names or []),
    }]
    edges: List[Dict[str, Any]] = []
    for record in facet_records:
        facet = record["facet"]
        node_id = "facet:%s" % facet.facet_id
        nodes.append({
            "id": node_id,
            "label": facet.label,
            "kind": "facet",
            "parent": "root",
            "detail": "%s (judged on %s)" % (facet.framing, facet.metric),
            "selection_score": record.get("score"),
            "matched_keywords": record.get("matched_keywords"),
            "basis": record.get("selection_basis"),
            "assumed_limitation": facet.limitation,
        })
        edges.append({"source": "root", "target": node_id, "type": "has_dimension",
                      "note": "Facet selected for this problem."})
    for direction in directions:
        facet_id = (direction.get("facet") or {}).get("facet_id", "")
        node_id = "dir:%s" % direction["direction_id"]
        status = (direction.get("status") or {})
        nodes.append({
            "id": node_id,
            "label": direction["title"],
            "kind": "direction",
            "parent": "facet:%s" % facet_id,
            "direction_id": direction["direction_id"],
            "status": status.get("status", ""),
            "status_label": status.get("label", ""),
            "status_dot": status.get("dot", "#8b94a7"),
            "concept_id": (direction.get("concept") or {}).get("concept_id", ""),
            "already_failed": bool(direction.get("already_failed")),
            "detail": direction.get("core_question", ""),
        })
        edges.append({"source": "facet:%s" % facet_id, "target": node_id,
                      "type": "inspired_by",
                      "note": "Cross-domain mechanism applied to this facet."})
    for item in rejected:
        node_id = "rejected:%s" % item["mechanism_key"].replace(":", "_")
        nodes.append({
            "id": node_id,
            "label": item["title"],
            "kind": "branch_not_taken",
            "parent": "facet:%s" % item["facet_id"],
            "detail": item["reason"],
            "concept_id": item["concept_id"],
            "ranked_out": True,
        })
        edges.append({"source": "facet:%s" % item["facet_id"], "target": node_id,
                      "type": "considered", "note": item["reason"]})
    return {
        "root": problem,
        "subject": subject,
        "nodes": nodes,
        "edges": edges,
        "counts": {
            "facets": len(facet_records),
            "directions": len(directions),
            "branches_not_taken": len(rejected),
            "nodes": len(nodes),
            "edges": len(edges),
        },
        "legend": {
            "research_question": "The problem AARL was given",
            "facet": "A structural dimension of the problem",
            "direction": "A discovered research direction",
            "branch_not_taken": "Considered but not explored in this run",
        },
        "edge_types": ["has_dimension", "inspired_by", "considered"],
        "note": (
            "This tree shows how the directions were constructed. Node counts are "
            "descriptive only and are never used as evidence strength."
        ),
    }