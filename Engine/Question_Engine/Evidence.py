"""
EVIDENCE SCORING ENGINE — Multi-Factor Weighted Confidence
Replaces binary Approved/Rejected/Deferred with rich, structured evidence tracking.

Confidence is computed from MULTIPLE weighted factors:
  1. Paper/Literature support (from KB mining)         — weight 0.20
  2. Knowledge graph support (from DOSCAN insights)     — weight 0.20
  3. Equation validation (from Phase 8)                 — weight 0.15
  4. Simulation success (from Phase 9)                  — weight 0.20
  5. Experimental agreement (from historical data)      — weight 0.10
  6. Novelty score (conflict-based)                     — weight 0.10
  7. Falsifiability score (Popperian)                   — weight 0.05

This replaces the old approach that relied mostly on LLM judgment and
TF-IDF cosine similarity alone (which produced ~8% confidence).
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# =========================================================
# EVIDENCE DATABASE
# =========================================================
EVIDENCE_DB_FILE = "evidence_scoring_db.json"

class EvidenceEntry:
    """Structured evidence for one scientific claim."""
    
    def __init__(self, 
                 claim: str,
                 source_hypothesis: str = "",
                 evidence_type: str = "literature",
                 content: str = "",
                 confidence: float = 0.0,
                 source_details: Optional[Dict[str, Any]] = None):
        self.claim = claim
        self.source_hypothesis = source_hypothesis
        self.evidence_type = evidence_type  # literature | experiment | simulation | logical_deduction
        self.content = content
        self.confidence = min(1.0, max(0.0, confidence))
        self.source_details = source_details or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "source_hypothesis": self.source_hypothesis,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "confidence": self.confidence,
            "source_details": self.source_details,
            "timestamp": self.timestamp
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvidenceEntry":
        e = EvidenceEntry(
            claim=d["claim"],
            source_hypothesis=d.get("source_hypothesis", ""),
            evidence_type=d.get("evidence_type", "literature"),
            content=d.get("content", ""),
            confidence=d.get("confidence", 0.0),
            source_details=d.get("source_details", {})
        )
        e.timestamp = d.get("timestamp", e.timestamp)
        return e


class EvidenceScore:
    """
    Rich evidence score with MULTI-FACTOR weighted confidence.
    
    Confidence is computed from 7 weighted factors:
      1. Paper/Literature support (from KB mining)         — weight 0.20
      2. Knowledge graph support (from DOSCAN insights)     — weight 0.20
      3. Equation validation (from Phase 8)                 — weight 0.15
      4. Simulation success (from Phase 9)                  — weight 0.20
      5. Experimental agreement (from historical data)      — weight 0.10
      6. Novelty score (conflict-based)                     — weight 0.10
      7. Falsifiability score (Popperian)                   — weight 0.05
    
    Attributes:
        claim: The scientific claim being scored
        confidence: Aggregated 0.0-1.0
        evidence_sources: Dict mapping source type -> list of EvidenceEntry
        missing_evidence: List of evidence types that are missing
        contradictions: List of contradictory evidence
        next_action: Recommended next step
        historical_accuracy: How often similar claims were correct
        kg_support_score: Knowledge graph support (0.0-1.0)
        equation_validation_score: Equation validation (0.0-1.0)
        simulation_success_score: Simulation success (0.0-1.0)
        factor_breakdown: Dict showing each factor's contribution
    """
    
    def __init__(self, claim: str):
        self.claim = claim
        self.confidence = 0.0
        self.evidence_sources: Dict[str, List[EvidenceEntry]] = {
            "literature": [],
            "experiment": [],
            "simulation": [],
            "logical_deduction": []
        }
        self.missing_evidence: List[str] = []
        self.contradictions: List[str] = []
        self.next_action: str = "none"
        self.historical_accuracy: float = 0.0
        self.novelty_score: float = 0.0
        self.falsifiability_score: float = 0.0
        # New multi-factor scores
        self.kg_support_score: float = 0.0
        self.equation_validation_score: float = 0.0
        self.simulation_success_score: float = 0.0
        self.factor_breakdown: Dict[str, float] = {}

    def add_evidence(self, entry: EvidenceEntry):
        etype = entry.evidence_type
        if etype in self.evidence_sources:
            self.evidence_sources[etype].append(entry)
        self._recompute_confidence()

    def _recompute_confidence(self):
        """
        MULTI-FACTOR weighted confidence aggregation.
        
        Factors and weights:
          1. Literature support (from KB mining)         — 0.20
          2. Knowledge graph support (from DOSCAN)        — 0.20
          3. Equation validation (from Phase 8)           — 0.15
          4. Simulation success (from Phase 9)            — 0.20
          5. Experimental agreement (historical)          — 0.10
          6. Novelty score                                — 0.10
          7. Falsifiability score                         — 0.05
        
        Each factor contributes 0.0-1.0, multiplied by its weight.
        Missing evidence types are tracked but don't zero out other factors.
        """
        # Track which evidence types are missing
        self.missing_evidence = []
        for etype in ["literature", "experiment", "simulation", "logical_deduction"]:
            if not self.evidence_sources[etype]:
                self.missing_evidence.append(etype)

        # --- Factor 1: Literature Support (0.20) ---
        lit_entries = self.evidence_sources["literature"]
        if lit_entries:
            literature_score = sum(e.confidence for e in lit_entries) / len(lit_entries)
        else:
            literature_score = 0.0

        # --- Factor 2: Knowledge Graph Support (0.20) ---
        kg_score = self.kg_support_score

        # --- Factor 3: Equation Validation (0.15) ---
        eq_score = self.equation_validation_score

        # --- Factor 4: Simulation Success (0.20) ---
        sim_entries = self.evidence_sources["simulation"]
        if sim_entries:
            sim_from_entries = sum(e.confidence for e in sim_entries) / len(sim_entries)
            simulation_score = max(sim_from_entries, self.simulation_success_score)
        else:
            simulation_score = self.simulation_success_score

        # --- Factor 5: Experimental Agreement (0.10) ---
        exp_entries = self.evidence_sources["experiment"]
        if exp_entries:
            experimental_score = sum(e.confidence for e in exp_entries) / len(exp_entries)
        else:
            experimental_score = self.historical_accuracy

        # --- Factor 6: Novelty (0.10) ---
        novelty = self.novelty_score if self.novelty_score > 0 else 0.3  # default moderate

        # --- Factor 7: Falsifiability (0.05) ---
        falsifiability = self.falsifiability_score if self.falsifiability_score > 0 else 0.3

        # --- Weighted Sum ---
        weights = {
            "literature": 0.20,
            "knowledge_graph": 0.20,
            "equation_validation": 0.15,
            "simulation": 0.20,
            "experimental": 0.10,
            "novelty": 0.10,
            "falsifiability": 0.05,
        }

        factor_values = {
            "literature": literature_score,
            "knowledge_graph": kg_score,
            "equation_validation": eq_score,
            "simulation": simulation_score,
            "experimental": experimental_score,
            "novelty": novelty,
            "falsifiability": falsifiability,
        }

        # Compute weighted sum
        weighted_sum = sum(factor_values[k] * weights[k] for k in weights)

        # Small penalty for missing evidence types (max 15% penalty, reduced from 25%)
        # This is gentler so that having literature + KG + equations still gives decent confidence
        # even without experiment/simulation
        missing_penalty = min(0.15, len(self.missing_evidence) * 0.04)

        self.confidence = weighted_sum * (1.0 - missing_penalty)
        self.confidence = min(1.0, max(0.0, self.confidence))

        # Store breakdown for transparency
        self.factor_breakdown = {
            k: round(v * weights[k], 4) for k, v in factor_values.items()
        }
        self.factor_breakdown["_total_weighted"] = round(weighted_sum, 4)
        self.factor_breakdown["_missing_penalty"] = round(missing_penalty, 4)
        self.factor_breakdown["_final_confidence"] = round(self.confidence, 4)

    def set_novelty(self, conflict_ratio: float):
        """
        Novelty based on conflict vs support ratio.
        Higher conflict ratio = higher novelty.
        """
        self.novelty_score = min(1.0, 0.3 + (conflict_ratio * 0.7))

    def set_falsifiability(self, has_testable_prediction: bool, num_conditions: int):
        """
        Falsifiability score based on Popperian criterion.
        A claim is more falsifiable if it has many testable conditions.
        """
        base = 0.3 if has_testable_prediction else 0.0
        self.falsifiability_score = min(1.0, base + (num_conditions * 0.15))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "confidence": round(self.confidence, 4),
            "evidence_sources": {
                etype: [e.to_dict() for e in entries]
                for etype, entries in self.evidence_sources.items()
            },
            "missing_evidence": self.missing_evidence,
            "contradictions": self.contradictions,
            "next_action": self.next_action,
            "historical_accuracy": round(self.historical_accuracy, 4),
            "novelty_score": round(self.novelty_score, 4),
            "falsifiability_score": round(self.falsifiability_score, 4),
            "kg_support_score": round(self.kg_support_score, 4),
            "equation_validation_score": round(self.equation_validation_score, 4),
            "simulation_success_score": round(self.simulation_success_score, 4),
            "factor_breakdown": self.factor_breakdown
        }

    def get_verdict(self) -> str:
        """Returns a rich verdict string."""
        if self.confidence >= 0.70 and len(self.missing_evidence) <= 1:
            return "STRONGLY_SUPPORTED"
        elif self.confidence >= 0.50:
            return "WEAKLY_SUPPORTED"
        elif self.confidence >= 0.30:
            return "INCONCLUSIVE"
        elif self.contradictions:
            return "CONTRADICTED"
        else:
            return "INSUFFICIENT_EVIDENCE"


# =========================================================
# EVIDENCE DATABASE (Persistence Layer)
# =========================================================
class EvidenceDatabase:
    """
    Persistent storage for all evidence scores.
    Tracks historical accuracy to calibrate future scores.
    """
    
    def __init__(self, db_path: str = EVIDENCE_DB_FILE):
        self.db_path = db_path
        self.scores: Dict[str, EvidenceScore] = {}
        self.accuracy_history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for claim, score_dict in data.get("scores", {}).items():
                        score = EvidenceScore(claim)
                        score.confidence = score_dict.get("confidence", 0.0)
                        score.missing_evidence = score_dict.get("missing_evidence", [])
                        score.contradictions = score_dict.get("contradictions", [])
                        score.next_action = score_dict.get("next_action", "none")
                        score.historical_accuracy = score_dict.get("historical_accuracy", 0.0)
                        score.novelty_score = score_dict.get("novelty_score", 0.0)
                        score.falsifiability_score = score_dict.get("falsifiability_score", 0.0)
                        score.kg_support_score = score_dict.get("kg_support_score", 0.0)
                        score.equation_validation_score = score_dict.get("equation_validation_score", 0.0)
                        score.simulation_success_score = score_dict.get("simulation_success_score", 0.0)
                        score.factor_breakdown = score_dict.get("factor_breakdown", {})
                        # Restore evidence entries
                        for etype, entries in score_dict.get("evidence_sources", {}).items():
                            for e in entries:
                                score.evidence_sources[etype].append(EvidenceEntry.from_dict(e))
                        self.scores[claim] = score
                    self.accuracy_history = data.get("accuracy_history", [])
            except Exception:
                self.scores = {}
                self.accuracy_history = []

    def save(self):
        data = {
            "scores": {claim: score.to_dict() for claim, score in self.scores.items()},
            "accuracy_history": self.accuracy_history,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def get_or_create(self, claim: str) -> EvidenceScore:
        if claim not in self.scores:
            self.scores[claim] = EvidenceScore(claim)
        return self.scores[claim]

    def record_outcome(self, claim: str, was_correct: bool):
        """
        After a claim is tested, record whether it was correct.
        This updates the historical accuracy for calibration.
        """
        self.accuracy_history.append({
            "claim": claim,
            "was_correct": was_correct,
            "predicted_confidence": self.scores.get(claim, EvidenceScore(claim)).confidence,
            "timestamp": datetime.now().isoformat()
        })
        # Recompute historical accuracy across all claims
        if self.accuracy_history:
            correct = sum(1 for h in self.accuracy_history if h["was_correct"])
            total = len(self.accuracy_history)
            avg_accuracy = correct / total if total > 0 else 0.0
            for score in self.scores.values():
                score.historical_accuracy = avg_accuracy
        self.save()

    def get_next_action(self, score: EvidenceScore) -> str:
        """
        Determines the most useful next action based on evidence gaps.
        """
        if "experiment" in score.missing_evidence:
            return "Run_Experiment"
        elif "simulation" in score.missing_evidence:
            return "Run_Simulation"
        elif "literature" in score.missing_evidence:
            return "Search_Literature"
        elif score.confidence < 0.50:
            return "Gather_More_Evidence"
        elif score.contradictions:
            return "Resolve_Contradiction"
        else:
            return "Ready_For_Use"

    def calibrate_confidence_threshold(self) -> float:
        """
        Learns the optimal confidence threshold from historical data.
        Returns the threshold that would have maximized accuracy.
        """
        if not self.accuracy_history or len(self.accuracy_history) < 5:
            return 0.50  # default
        
        # Test thresholds from 0.3 to 0.9
        best_threshold = 0.50
        best_accuracy = 0.0
        
        for threshold in [x * 0.05 for x in range(6, 19)]:  # 0.30 to 0.90
            correct = 0
            total = 0
            for h in self.accuracy_history:
                predicted_true = h["predicted_confidence"] >= threshold
                actual_true = h["was_correct"]
                if predicted_true == actual_true:
                    correct += 1
                total += 1
            acc = correct / total if total > 0 else 0.0
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold
        
        return best_threshold


# =========================================================
# INTEGRATION WITH OLD DOSCAN SCORING
# =========================================================
def upgrade_doscan_evidence(evidence_score: float, 
                            novelty: float, 
                            confidence: float,
                            has_conflict: bool,
                            has_support: bool) -> EvidenceScore:
    """
    Converts the old DOSCAN scoring into the rich EvidenceScore format.
    This bridges the old system with the new evidence engine.
    """
    claim = "migrated_doscan_claim"
    escore = EvidenceScore(claim)
    
    # Map old scores to evidence entries
    if has_support:
        escore.add_evidence(EvidenceEntry(
            claim=claim,
            evidence_type="literature",
            content="Support found in knowledge base co-occurrence",
            confidence=confidence * 0.8  # discount because it's co-occurrence, not causation
        ))
    
    if has_conflict:
        escore.add_evidence(EvidenceEntry(
            claim=claim,
            evidence_type="literature",
            content="Conflict found in knowledge base co-occurrence",
            confidence=confidence * 0.5
        ))
        escore.contradictions.append("Conflicting co-occurrence in literature")
    
    escore.confidence = evidence_score
    escore.next_action = escore.get_next_action(escore) if hasattr(escore, 'get_next_action') else "Review"
    
    return escore


# =========================================================
# MAIN EVIDENCE ENGINE ENTRY POINT
# =========================================================
# =========================================================
# TF-IDF SEMANTIC EVIDENCE MATCHING
# =========================================================
BOUNDARY_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "for", "of", "with", "by", "on", "in",
    "at", "from", "as", "this", "that", "it", "can", "could", "would", "should", "using", "such",
    "without", "will", "has", "have", "had", "not", "no", "only", "but", "however", "and", "or",
    "which", "where", "when", "why", "how", "we", "they", "their", "our", "show", "shows",
    "demonstrate", "compare", "search", "find", "found", "use", "used", "requires", "assumes",
    "must", "always", "during", "between", "through", "under", "over", "into"
}

def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower()) if w not in BOUNDARY_WORDS]

def _tfidf_vector(text: str, all_texts: List[str]) -> Dict[str, float]:
    """Compute TF-IDF vector for a single text against a corpus."""
    tokens = _tokenize(text)
    N = max(len(all_texts), 1)
    # Compute document frequency across all texts
    df = Counter()
    for t in all_texts:
        df.update(set(_tokenize(t)))
    # TF-IDF for this text
    vec = {}
    tf = Counter(tokens)
    total_terms = max(len(tokens), 1)
    for word, count in tf.items():
        vec[word] = (count / total_terms) * math.log(N / (1 + df[word]))
    return vec

def _cosine_sim(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in intersection)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return 0.0 if mag1 == 0 or mag2 == 0 else dot / (mag1 * mag2)

def _mine_knowledge_base_for_evidence(claim: str) -> List[EvidenceEntry]:
    """
    Mines the knowledge base for evidence using TF-IDF cosine similarity.
    Only returns entries with similarity > 0.15 (semantically relevant).
    Confidence = cosine_similarity * section_weight.
    """
    entries = []
    kb_path = "deep_research_knowledge_base.json"
    if not os.path.exists(kb_path):
        return entries
    
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
    except Exception:
        return entries
    
    # Collect all KB texts for IDF computation
    all_kb_texts = []
    sections = {
        "proven_facts": (kb_data.get("proven_facts", []), 0.85),
        "state_of_the_art_prior_art": (kb_data.get("state_of_the_art_prior_art", []), 0.70),
        "critical_unanswered_unknowns": (kb_data.get("critical_unanswered_unknowns", []), 0.50),
    }
    for section_name, (items, _) in sections.items():
        all_kb_texts.extend(items)
    # Add hypothesis texts
    for hyp, method in kb_data.get("proposed_testable_hypotheses", {}).items():
        all_kb_texts.append(f"{hyp} {method}")
    
    if not all_kb_texts:
        return entries
    
    # Compute claim vector once
    claim_vec = _tfidf_vector(claim, all_kb_texts + [claim])
    
    # Threshold: only accept matches above this similarity
    SIMILARITY_THRESHOLD = 0.15
    
    # Search proven_facts (highest weight)
    for fact in kb_data.get("proven_facts", []):
        fact_vec = _tfidf_vector(fact, all_kb_texts + [fact])
        sim = _cosine_sim(claim_vec, fact_vec)
        if sim >= SIMILARITY_THRESHOLD:
            entries.append(EvidenceEntry(
                claim=claim,
                evidence_type="literature",
                content=f"Proven fact: {fact}",
                confidence=round(sim * 0.85, 3),
                source_details={"source": "knowledge_base", "section": "proven_facts", "similarity": round(sim, 3)}
            ))
    
    # Search state_of_the_art_prior_art
    for art in kb_data.get("state_of_the_art_prior_art", []):
        art_vec = _tfidf_vector(art, all_kb_texts + [art])
        sim = _cosine_sim(claim_vec, art_vec)
        if sim >= SIMILARITY_THRESHOLD:
            entries.append(EvidenceEntry(
                claim=claim,
                evidence_type="literature",
                content=f"Prior art: {art}",
                confidence=round(sim * 0.70, 3),
                source_details={"source": "knowledge_base", "section": "state_of_the_art_prior_art", "similarity": round(sim, 3)}
            ))
    
    # Search proposed_testable_hypotheses
    for hyp, method in kb_data.get("proposed_testable_hypotheses", {}).items():
        hyp_text = f"{hyp} {method}"
        hyp_vec = _tfidf_vector(hyp_text, all_kb_texts + [hyp_text])
        sim = _cosine_sim(claim_vec, hyp_vec)
        if sim >= SIMILARITY_THRESHOLD:
            entries.append(EvidenceEntry(
                claim=claim,
                evidence_type="logical_deduction",
                content=f"Related hypothesis: {hyp} (test: {method})",
                confidence=round(sim * 0.60, 3),
                source_details={"source": "knowledge_base", "section": "proposed_testable_hypotheses", "similarity": round(sim, 3)}
            ))
    
    # Search critical_unanswered_unknowns
    for gap in kb_data.get("critical_unanswered_unknowns", []):
        gap_vec = _tfidf_vector(gap, all_kb_texts + [gap])
        sim = _cosine_sim(claim_vec, gap_vec)
        if sim >= SIMILARITY_THRESHOLD:
            entries.append(EvidenceEntry(
                claim=claim,
                evidence_type="literature",
                content=f"Known gap: {gap}",
                confidence=round(sim * 0.50, 3),
                source_details={"source": "knowledge_base", "section": "critical_unanswered_unknowns", "similarity": round(sim, 3)}
            ))
    
    # Sort by confidence descending, keep top 5
    entries.sort(key=lambda e: e.confidence, reverse=True)
    return entries[:5]


# =========================================================
# CROSS-PHASE DATA LOADING (DOSCAN, Equations, Simulation)
# =========================================================

def _load_doscan_kg_support(claim: str) -> float:
    """
    Load knowledge graph support score from DOSCAN breakthroughs.
    Checks if the claim's concepts appear in DOSCAN insights with high composite scores.
    Returns 0.0-1.0.
    """
    doscan_path = "doscan_breakthroughs.json"
    if not os.path.exists(doscan_path):
        return 0.0

    try:
        with open(doscan_path, "r", encoding="utf-8") as f:
            breakthroughs = json.load(f)
    except Exception:
        return 0.0

    if not isinstance(breakthroughs, list):
        return 0.0

    claim_words = set(_tokenize(claim))
    if not claim_words:
        return 0.0

    max_score = 0.0
    match_count = 0

    for entry in breakthroughs:
        insights = entry.get("scientific_research_insights", [])
        for insight in insights:
            insight_text = insight.get("insight_title", "") + " " + \
                           insight.get("research_note", {}).get("what_was_discovered", "")
            insight_words = set(_tokenize(insight_text))
            overlap = len(claim_words & insight_words)
            if overlap >= 2:
                composite = insight.get("composite_score", 0.0)
                # Normalize: composite scores are typically 0.4-0.9
                normalized = min(1.0, composite / 0.8)
                max_score = max(max_score, normalized)
                match_count += 1

    # If multiple insights match, boost the score
    if match_count >= 3:
        max_score = min(1.0, max_score + 0.1)
    elif match_count >= 1:
        max_score = min(1.0, max_score + 0.05)

    return max_score


def _load_equation_validation_score(claim: str) -> float:
    """
    Load equation validation score from derived_equations.json.
    Checks if validated equations are associated with this claim's hypothesis.
    Returns 0.0-1.0.
    """
    eq_path = "derived_equations.json"
    if not os.path.exists(eq_path):
        return 0.0

    try:
        with open(eq_path, "r", encoding="utf-8") as f:
            eq_data = json.load(f)
    except Exception:
        return 0.0

    validated = eq_data.get("validated_equations", [])
    rejected = eq_data.get("rejected_ideas", [])

    if not validated and not rejected:
        return 0.0

    claim_words = set(_tokenize(claim))
    if not claim_words:
        return 0.0

    # Check how many validated equations relate to this claim
    relevant_validated = 0
    for eq in validated:
        hyp = eq.get("hypothesis", "")
        hyp_words = set(_tokenize(hyp))
        overlap = len(claim_words & hyp_words)
        if overlap >= 2:
            relevant_validated += 1

    # Check rejected equations for this claim
    relevant_rejected = 0
    for eq in rejected:
        hyp = eq.get("hypothesis", "")
        hyp_words = set(_tokenize(hyp))
        overlap = len(claim_words & hyp_words)
        if overlap >= 2:
            relevant_rejected += 1

    total = relevant_validated + relevant_rejected
    if total == 0:
        return 0.0

    # Score = ratio of validated to total, with bonus for absolute count
    validation_ratio = relevant_validated / total
    count_bonus = min(0.2, relevant_validated * 0.05)

    return min(1.0, validation_ratio * 0.8 + count_bonus)


def _load_simulation_success_score(claim: str) -> float:
    """
    Load simulation success score from comparison_report.json.
    If the simulation showed AARL is better, the claim gets support.
    Returns 0.0-1.0.
    """
    # Try multiple possible paths
    for report_path in ["Research/comparison_report.json",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Research", "comparison_report.json")]:
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                continue

            success = report.get("success", False)
            overall_better = report.get("overall_better", False)
            metrics_improved = report.get("metrics_improved", 0)
            metrics_total = report.get("metrics_total", 0)
            avg_improvement = report.get("average_improvement_pct", 0)

            if not success:
                return 0.1  # Simulation ran but didn't succeed

            if overall_better and metrics_total > 0:
                # Score based on fraction of metrics improved
                fraction = metrics_improved / metrics_total
                # Bonus for large average improvement
                improvement_bonus = min(0.2, max(0, avg_improvement) / 100)
                return min(1.0, fraction * 0.8 + improvement_bonus)
            else:
                return 0.3  # Simulation ran but AARL wasn't clearly better

    return 0.0  # No simulation report found


def run_evidence_engine(claims: List[str], 
                        doscan_scores: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Main entry point for the Evidence Engine with MULTI-FACTOR confidence.
    
    For each claim:
    1. Mines the knowledge base for relevant literature evidence
    2. Loads DOSCAN knowledge graph support (Phase 3)
    3. Loads equation validation scores (Phase 8)
    4. Loads simulation success scores (Phase 9)
    5. Computes multi-factor weighted confidence
    6. Identifies missing evidence gaps
    7. Recommends next action
    
    Args:
        claims: List of scientific claims to score
        doscan_scores: Optional list of old-format scores for migration
    
    Returns:
        Dict with evidence results per claim
    """
    print("\n" + "="*80)
    print("📊 EVIDENCE SCORING ENGINE — Multi-Factor Weighted Confidence")
    print("="*80)
    
    db = EvidenceDatabase()
    results = {}
    
    # Pre-load cross-phase data once
    print("   [Cross-Phase] Loading DOSCAN, Equation, and Simulation data...")
    
    for i, claim in enumerate(claims):
        score = db.get_or_create(claim)
        
        # Step 1: Mine knowledge base for evidence (only if not already populated)
        if not score.evidence_sources["literature"] and not score.evidence_sources["logical_deduction"]:
            kb_entries = _mine_knowledge_base_for_evidence(claim)
            for entry in kb_entries:
                score.add_evidence(entry)
            if kb_entries:
                print(f"   [KB Mining] Found {len(kb_entries)} evidence entries for claim {i+1}")
        
        # Step 2: Load DOSCAN knowledge graph support
        kg_score = _load_doscan_kg_support(claim)
        if kg_score > score.kg_support_score:
            score.kg_support_score = kg_score
            if kg_score > 0:
                print(f"   [DOSCAN] KG support score: {kg_score:.3f} for claim {i+1}")
        
        # Step 3: Load equation validation score
        eq_score = _load_equation_validation_score(claim)
        if eq_score > score.equation_validation_score:
            score.equation_validation_score = eq_score
            if eq_score > 0:
                print(f"   [Equations] Validation score: {eq_score:.3f} for claim {i+1}")
        
        # Step 4: Load simulation success score
        sim_score = _load_simulation_success_score(claim)
        if sim_score > score.simulation_success_score:
            score.simulation_success_score = sim_score
            if sim_score > 0:
                print(f"   [Simulation] Success score: {sim_score:.3f} for claim {i+1}")
        
        # Step 5: Set novelty based on contradictions
        if score.contradictions:
            score.set_novelty(len(score.contradictions) / max(1, len(score.evidence_sources["literature"])))
        else:
            score.set_novelty(0.3)  # default moderate novelty
        
        # Step 6: Set falsifiability (claims with testable hypotheses are more falsifiable)
        has_testable = bool(score.evidence_sources["logical_deduction"])
        num_conditions = len(score.evidence_sources["literature"]) + len(score.evidence_sources["logical_deduction"])
        score.set_falsifiability(has_testable, num_conditions)
        
        # If old scores provided, migrate them
        if doscan_scores and i < len(doscan_scores):
            old = doscan_scores[i]
            if not score.evidence_sources["literature"]:  # only if not already populated
                old_score = upgrade_doscan_evidence(
                    old.get("evidence_score", 0.0),
                    old.get("novelty", 0.0),
                    old.get("confidence", 0.0),
                    old.get("has_conflict", False),
                    old.get("has_support", False)
                )
                score = old_score
                db.scores[claim] = score
        
        # Force recompute confidence and missing_evidence
        score._recompute_confidence()
        
        # Determine next action
        score.next_action = db.get_next_action(score)
        
        verdict = score.get_verdict()
        results[claim] = {
            "verdict": verdict,
            "confidence": round(score.confidence, 4),
            "evidence_types_present": [
                etype for etype, entries in score.evidence_sources.items() if entries
            ],
            "missing_evidence": score.missing_evidence,
            "next_action": score.next_action,
            "novelty": round(score.novelty_score, 4),
            "falsifiability": round(score.falsifiability_score, 4),
            "kg_support": round(score.kg_support_score, 4),
            "equation_validation": round(score.equation_validation_score, 4),
            "simulation_success": round(score.simulation_success_score, 4),
            "factor_breakdown": score.factor_breakdown
        }
        
        print(f"\n  [{i+1}] Claim: {claim[:100]}...")
        print(f"      Verdict: {verdict}")
        print(f"      Confidence: {score.confidence:.1%}")
        print(f"      Factors: Lit={score.factor_breakdown.get('literature',0):.3f} "
              f"KG={score.factor_breakdown.get('knowledge_graph',0):.3f} "
              f"Eq={score.factor_breakdown.get('equation_validation',0):.3f} "
              f"Sim={score.factor_breakdown.get('simulation',0):.3f}")
        print(f"      Evidence: {', '.join(results[claim]['evidence_types_present']) or 'NONE'}")
        print(f"      Missing: {', '.join(score.missing_evidence) or 'NONE'}")
        print(f"      Next: {score.next_action}")
    
    # Calibrate
    threshold = db.calibrate_confidence_threshold()
    print(f"\n📈 Calibrated confidence threshold: {threshold:.2f}")
    if db.accuracy_history:
        acc_val = db.accuracy_history[-1].get("predicted_confidence")
        if isinstance(acc_val, (int, float)):
            print(f"📊 Historical accuracy: {acc_val:.2f}")
        else:
            print("📊 Historical accuracy: N/A")
    else:
        print("📊 Historical accuracy: N/A")
    
    db.save()
    
    avg_conf = round(
        sum(r["confidence"] for r in results.values()) / len(results), 4
    ) if results else 0.0
    
    summary = {
        "total_claims": len(claims),
        "verdicts": {claim: r["verdict"] for claim, r in results.items()},
        "average_confidence": avg_conf,
        "calibrated_threshold": round(threshold, 4),
        "action_items": [r["next_action"] for r in results.values()],
        "total_missing_evidence": sum(len(r["missing_evidence"]) for r in results.values()),
        "factor_weights": {
            "literature": 0.20,
            "knowledge_graph": 0.20,
            "equation_validation": 0.15,
            "simulation": 0.20,
            "experimental": 0.10,
            "novelty": 0.10,
            "falsifiability": 0.05,
        }
    }
    
    print(f"\n📋 Summary: {avg_conf:.1%} avg confidence, {summary['total_missing_evidence']} gaps identified")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    # Demo
    test_claims = [
        "Increasing electrode surface area improves battery power density",
        "Solid electrolytes eliminate thermal runaway in lithium-ion batteries",
        "Graphene anodes double the energy density of current lithium-ion cells"
    ]
    run_evidence_engine(test_claims)