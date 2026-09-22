"""
AARL Lab — Hypothesis Engine
============================
Generates hypotheses from the combined knowledge base, explicitly retrieving
previous experiments and lessons so that each new hypothesis is an *improvement*
over the last iteration (Experiment -> Failure -> Lesson -> Improved Hypothesis).

Every hypothesis carries full provenance: which knowledge items, papers and
lessons influenced it. When an LLM is available it is the reasoning layer; when
it is not, a deterministic knowledge-grounded fallback is used. Nothing is
invented beyond what the knowledge base supports — claims are always labelled
``generated_hypothesis`` (never fact).
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

CLAIM_HYPOTHESIS = "generated_hypothesis"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class HypothesisEngine:
    def __init__(self, memory, knowledge, config, llm_service=None):
        self.memory = memory
        self.knowledge = knowledge
        self.config = config
        self.llm = llm_service

    # ------------------------------------------------------------------
    def _context_for_generation(self, research_question: str) -> Dict[str, Any]:
        k = int(getattr(self.config, "lesson_retrieval_k", 5) or 5)
        lessons = self._recent_lessons(k)
        experiments = self.memory.load_collection("simulations")
        previous = self.memory.load_collection("hypotheses")
        supporting = self.knowledge.search(research_question, section="findings", limit=5)
        limitations = self.knowledge.search(research_question, section="limitations", limit=5)
        return {
            "lessons": lessons,
            "experiments": experiments,
            "previous_hypotheses": previous,
            "supporting": supporting,
            "limitations": limitations,
        }

    def _recent_lessons(self, k: int) -> List[Dict[str, Any]]:
        try:
            from lab.failures import LessonManager
            return LessonManager(self.memory).retrieve(k)
        except Exception:  # noqa: BLE001 - lessons are optional
            lessons = self.memory.load_collection("lessons")
            lessons.sort(key=lambda r: str(r.get("stored_at", "")), reverse=True)
            return lessons[:k]

    # ------------------------------------------------------------------
    def _hypothesis_of(self, hypothesis_id: str) -> Dict[str, Any]:
        """The stored hypothesis a lesson points back to ({} when unavailable).

        Used only for provenance: an improved hypothesis must show which
        previous statement and parameters it is changing.
        """
        if not hypothesis_id:
            return {}
        try:
            record = self.memory.get("hypotheses", str(hypothesis_id))
        except Exception:  # noqa: BLE001 - provenance is optional
            record = None
        if isinstance(record, dict) and record:
            return record
        for previous in self.memory.load_collection("hypotheses"):
            if str(previous.get("hypothesis_id", "")) == str(hypothesis_id):
                return previous
        return {}

    # ------------------------------------------------------------------
    def generate(
        self,
        research_question: str,
        iteration: int,
        domain: str = "",
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses for one iteration, grounded in combined knowledge."""
        ctx = self._context_for_generation(research_question)
        hyps: List[Dict[str, Any]] = []

        if self.llm is not None:
            try:
                hyps = self._generate_with_llm(research_question, iteration, domain, ctx)
            except Exception:  # noqa: BLE001 - LLM failure falls back
                hyps = []

        if not hyps:
            hyps = self._generate_deterministic(research_question, iteration, domain, ctx)

        count = int(getattr(self.config, "hypotheses_per_iteration", 3) or 3)
        out: List[Dict[str, Any]] = []
        for draft in hyps[:count]:
            record = self._persist(draft, research_question, iteration, domain, ctx)
            out.append(record)
        return out

    # ------------------------------------------------------------------
    def _generate_deterministic(
        self,
        research_question: str,
        iteration: int,
        domain: str,
        ctx: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Knowledge- and lesson-grounded fallback generator (no LLM).

        Priority 1: *improved* hypotheses that explicitly apply the lesson from
        the most recent failures. Priority 2: hypotheses derived from the
        strongest supporting knowledge items not yet tested.
        """
        drafts: List[Dict[str, Any]] = []
        used_lessons = set()

        # ---- Priority 1: lesson-driven improvements ----------------------
        for lesson in ctx["lessons"]:
            rec = lesson.get("recommended_changes") or []
            lesson_id = lesson.get("lesson_id", "")
            if lesson_id in used_lessons or not lesson_id:
                continue
            used_lessons.add(lesson_id)

            base_hyp = self._hypothesis_of(lesson.get("hypothesis_id", ""))
            base_text = base_hyp.get("hypothesis", "") if base_hyp else ""
            change_text = "; ".join(str(r) for r in rec[:3]) or lesson.get("failure_type", "")
            statement = (
                f"Improved variant of {lesson.get('hypothesis_id', 'previous hypothesis')} "
                f"addressing lesson {lesson_id}: {change_text}. "
                + (f"Base idea: {base_text}" if base_text else "")
            ).strip()

            draft = self._draft(statement, domain, research_question)
            draft["based_on_lessons"] = [lesson_id]
            draft["related_experiments"] = (
                [lesson.get("experiment_id", "")] if lesson.get("experiment_id") else []
            )
            draft["parent_hypothesis"] = lesson.get("hypothesis_id", "")
            draft["parameters"] = self._apply_recommendations(
                (base_hyp or {}).get("parameters") or {}, rec
            )
            draft["risks"] = [
                f"Lesson {lesson_id} derived from a {lesson.get('severity', 'medium')}-severity "
                f"{lesson.get('failure_type', 'failure')}; the correction may not fully remove the cause."
            ]
            drafts.append(draft)

        # ---- Priority 2: knowledge-grounded new hypotheses ---------------
        tested = {str(h.get("hypothesis", "")) for h in ctx["previous_hypotheses"]}
        for item in ctx["supporting"]:
            statement = str(item.statement or "")[:400]
            if not statement or statement in tested:
                continue
            draft = self._draft(
                f"Based on finding {item.item_id}: {statement}", domain, research_question
            )
            draft["related_papers"] = [item.source_paper] if item.source_paper else []
            draft["based_on_knowledge"] = [item.item_id]
            drafts.append(draft)

        # ---- Priority 3: variation on the research question --------------
        if not drafts:
            drafts.append(self._draft(
                f"Investigate the primary engineering trade-off of: {research_question} "
                "by adjusting the dominant design parameter and measuring the impact "
                "on the domain metrics.",
                domain,
                research_question,
            ))
        return drafts

    # ------------------------------------------------------------------
    def _draft(self, statement: str, domain: str, research_question: str) -> Dict[str, Any]:
        """Build an empty draft with simulator-ready parameters."""
        draft: Dict[str, Any] = {
            "hypothesis": statement,
            "domain": domain,
            "parameters": {},
            "expected_outcome": {},
            "constraints": [],
            "risks": [],
            "based_on_lessons": [],
            "based_on_knowledge": [],
            "related_papers": [],
            "related_experiments": [],
            "parent_hypothesis": "",
        }
        if domain == "machine_learning":
            text = statement.lower()
            optimizer = "momentum"
            for keyword, method in (
                ("nesterov", "nesterov"), ("adam", "adam"), ("rmsprop", "rmsprop"),
                ("adagrad", "adagrad"), ("amsgrad", "amsgrad"), ("cosine", "sgd_cosine"),
                ("restart", "sgd_warm_restart"),
            ):
                if keyword in text:
                    optimizer = method
                    break
            draft["parameters"] = {"optimizer": optimizer, "lr": 0.1}
            draft["expected_outcome"] = {"improvement_pct": 5.0}
            draft["constraints"] = [{
                "name": "improvement_pct", "op": ">=", "value": 0.0,
                "source": "generated hypothesis expectation",
            }]
        return draft

    @staticmethod
    def _apply_recommendations(base_params: Dict[str, Any], recommendations: List[str]) -> Dict[str, Any]:
        """Apply lesson recommendations to numeric parameters where derivable."""
        import re as _re
        params = dict(base_params)
        for rec in recommendations:
            text = str(rec)
            for key in list(params):
                value = params[key]
                if str(key) in text and isinstance(value, (int, float)) and not isinstance(value, bool):
                    if any(w in text for w in ("reduce", "lower", "relax", "decrease")):
                        params[key] = round(value / 2.0, 6)
            m = _re.search(r"parameter '([^']+)'", text)
            if m and m.group(1) not in params:
                params[m.group(1)] = None  # unknown; the engine will report it
        return params

    # ------------------------------------------------------------------
    def _persist(
        self,
        draft: Dict[str, Any],
        research_question: str,
        iteration: int,
        domain: str,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        hyp_id = self.memory.next_id("hypotheses", "HYP")
        statement = str(draft.get("hypothesis", "") or "")

        # Supporting / contradictory evidence straight from the knowledge store.
        supporting = self.knowledge.search(statement, section="findings", limit=4)
        contradictory = self.knowledge.search(statement, section="contradictions", limit=3)

        record = {
            "hypothesis_id": hyp_id,
            "hypothesis": statement,
            "research_question": research_question,
            "iteration": iteration,
            "domain": domain,
            "supporting_evidence": [i.item_id for i in supporting],
            "contradictory_evidence": [i.item_id for i in contradictory],
            "assumptions": list(draft.get("assumptions") or []),
            "expected_outcome": dict(draft.get("expected_outcome") or {}),
            "measurable_metrics": list((draft.get("expected_outcome") or {}).keys()),
            "related_papers": list(draft.get("related_papers") or []),
            "related_experiments": list(draft.get("related_experiments") or []),
            "based_on_lessons": list(draft.get("based_on_lessons") or []),
            "based_on_knowledge": list(draft.get("based_on_knowledge") or []),
            "parent_hypothesis": str(draft.get("parent_hypothesis", "") or ""),
            "risks": list(draft.get("risks") or []),
            "confidence": 0.4,  # generated hypotheses are never "certain"
            "parameters": dict(draft.get("parameters") or {}),
            "constraints": list(draft.get("constraints") or []),
            "claim_type": CLAIM_HYPOTHESIS,
            "provenance": {
                "generated_by": "llm" if self.llm is not None else "deterministic_fallback",
                "iteration": iteration,
                "knowledge_items": list(draft.get("based_on_knowledge") or [])
                + [i.item_id for i in supporting],
                "lessons_applied": list(draft.get("based_on_lessons") or []),
                "experiments_considered": [e.get("experiment_id", "") for e in ctx["experiments"][-5:]],
                "claim_type_note": "AI-generated proposal — NOT established fact",
            },
            "timestamp": _now(),
        }
        self.memory.append_record("hypotheses", hyp_id, record)
        self.memory.append_history({
            "event": "hypothesis_generated",
            "hypothesis_id": hyp_id,
            "iteration": iteration,
            "based_on_lessons": record["based_on_lessons"],
            "generated_by": record["provenance"]["generated_by"],
        })

        # Register in the living knowledge store as a *generated* claim.
        item_id = self.knowledge.add(
            section="hypotheses",
            statement=statement,
            claim_type=CLAIM_HYPOTHESIS,
            provenance={
                "generated_by": record["provenance"]["generated_by"],
                "hypothesis_id": hyp_id,
                "iteration": iteration,
                "lessons_applied": record["based_on_lessons"],
            },
            source="aarl.hypothesis_engine",
            confidence=record["confidence"],
            evidence=record["supporting_evidence"],
            derived_from=record["based_on_knowledge"],
            metadata={"hypothesis_id": hyp_id},
        )
        record["knowledge_item_id"] = item_id

        # Mark the lessons this hypothesis applies.
        for lesson_id in record["based_on_lessons"]:
            try:
                from lab.failures import LessonManager
                LessonManager(self.memory).mark_applied(lesson_id, hyp_id)
            except Exception:  # noqa: BLE001 - marking is non-critical
                pass
        return record

    # ------------------------------------------------------------------
    def _generate_with_llm(
        self,
        research_question: str,
        iteration: int,
        domain: str,
        ctx: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """LLM-backed generation. The LLM proposes; AARL validates and labels.

        Returns drafts in the same shape as the deterministic generator. An
        empty/invalid response raises so the deterministic fallback takes over.
        """
        lesson_lines = "\n".join(
            f"- {l.get('lesson_id')}: {l.get('lesson', '')}" for l in ctx["lessons"]
        ) or "(no previous lessons)"
        experiment_lines = "\n".join(
            f"- {e.get('experiment_id')} ({e.get('status')}): metrics={e.get('metrics')}"
            for e in ctx["experiments"][-5:]
        ) or "(no previous experiments)"
        finding_lines = "\n".join(
            f"- {i.item_id}: {i.statement[:200]}" for i in ctx["supporting"]
        ) or "(no knowledge items)"

        system_prompt = (
            "You are the Hypothesis Engine of an autonomous research laboratory. "
            "Generate 3 testable research hypotheses for the given research question. "
            "You MUST incorporate the lessons from previous failed experiments and "
            "explicitly improve upon previous hypotheses. Never claim certainty. "
            'Respond ONLY with JSON: {"hypotheses": [{"hypothesis": str, '
            '"assumptions": [str], "expected_outcome": {}, "parameters": {}, '
            '"risks": [str], "based_on_lessons": [str], "parent_hypothesis": str}]}'
        )
        user_prompt = (
            f"Research question: {research_question}\n"
            f"Domain: {domain or 'auto-detect'}\n"
            f"Iteration: {iteration}\n\n"
            f"Previous lessons:\n{lesson_lines}\n\n"
            f"Previous experiments:\n{experiment_lines}\n\n"
            f"Supporting knowledge:\n{finding_lines}\n"
        )
        data = self.llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(getattr(self.config, "temperature", 0.4) or 0.4),
        )
        drafts: List[Dict[str, Any]] = []
        for h in (data.get("hypotheses") or [])[:6]:
            if not isinstance(h, dict):
                continue
            statement = str(h.get("hypothesis", "") or "").strip()
            if len(statement) < 20:
                continue
            draft = self._draft(statement, domain, research_question)
            draft["assumptions"] = [str(a) for a in (h.get("assumptions") or [])]
            draft["expected_outcome"] = dict(h.get("expected_outcome") or {})
            draft["parameters"] = dict(h.get("parameters") or {})
            draft["risks"] = [str(r) for r in (h.get("risks") or [])]
            draft["based_on_lessons"] = [str(l) for l in (h.get("based_on_lessons") or [])]
            draft["parent_hypothesis"] = str(h.get("parent_hypothesis", "") or "")
            drafts.append(draft)
        if not drafts:
            raise ValueError("LLM returned no usable hypotheses")
        return drafts





