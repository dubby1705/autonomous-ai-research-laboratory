"""
EVIDENCE SCORING ENGINE
Replaces binary Approved/Rejected/Deferred with rich, structured evidence tracking.

For each hypothesis-concept pair, stores:
- Evidence sources (Literature, Experiment, Simulation, Logical Deduction)
- Missing evidence gaps
- Confidence score (0.0 - 1.0)
- Next recommended action
- Historical accuracy tracking for self-calibration
"""

import os
import json
import math
import re
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
from groq import Groq

# Performance optimization settings
MAX_PARALLEL_EVIDENCE = 3  # Parallel evidence mining

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
    Rich evidence score replacing the old composite_score.
    
    Attributes:
        claim: The scientific claim being scored
        confidence: Aggregated 0.0-1.0
        evidence_sources: Dict mapping source type -> list of EvidenceEntry
        missing_evidence: List of evidence types that are missing
        contradictions: List of contradictory evidence
        next_action: Recommended next step
        historical_accuracy: How often similar claims were correct
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

    def add_evidence(self, entry: EvidenceEntry):
        etype = entry.evidence_type
        if etype in self.evidence_sources:
            self.evidence_sources[etype].append(entry)
        self._recompute_confidence()

    def _recompute_confidence(self):
        """
        Weighted confidence aggregation:
        - literature: 0.25 weight
        - experiment: 0.40 weight (most trustworthy)
        - simulation: 0.20 weight
        - logical_deduction: 0.15 weight
        
        Each evidence type's contribution = average confidence of its entries * weight.
        Missing types contribute 0.
        """
        weights = {
            "literature": 0.25,
            "experiment": 0.40,
            "simulation": 0.20,
            "logical_deduction": 0.15
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Track which types have evidence and which are missing
        self.missing_evidence = []
        
        for etype, weight in weights.items():
            entries = self.evidence_sources[etype]
            if entries:
                avg_conf = sum(e.confidence for e in entries) / len(entries)
                weighted_sum += avg_conf * weight
                total_weight += weight
            else:
                self.missing_evidence.append(etype)
        
        # Penalize for missing evidence types (max 25% penalty)
        missing_penalty = len(self.missing_evidence) * 0.06
        missing_penalty = min(0.25, missing_penalty)
        
        self.confidence = weighted_sum * (1.0 - missing_penalty) if total_weight > 0 else 0.0
        self.confidence = min(1.0, max(0.0, self.confidence))

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
            "falsifiability_score": round(self.falsifiability_score, 4)
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


def run_evidence_engine(claims: List[str], 
                        doscan_scores: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Main entry point for the Evidence Engine.
    
    For each claim:
    1. Mines the knowledge base for relevant literature evidence
    2. Loads any existing evidence from the database
    3. Computes weighted confidence across 4 evidence types
    4. Identifies missing evidence gaps
    5. Recommends next action
    
    Args:
        claims: List of scientific claims to score
        doscan_scores: Optional list of old-format scores for migration
    
    Returns:
        Dict with evidence results per claim
    """
    print("\n" + "="*80)
    print("📊 EVIDENCE SCORING ENGINE — Mining Knowledge Base + Structured Tracking")
    print("="*80)
    
    db = EvidenceDatabase()
    results = {}
    
    def process_claim(args):
        """Process a single claim for evidence - runs in parallel"""
        i, claim, doscan_scores = args
        score = db.get_or_create(claim)
        
        # Mine KB for evidence
        if not score.evidence_sources["literature"] and not score.evidence_sources["logical_deduction"]:
            kb_entries = _mine_knowledge_base_for_evidence(claim)
            for entry in kb_entries:
                score.add_evidence(entry)
        
        # Migrate old scores if provided
        if doscan_scores and i < len(doscan_scores):
            old = doscan_scores[i]
            if not score.evidence_sources["literature"]:
                old_score = upgrade_doscan_evidence(
                    old.get("evidence_score", 0.0),
                    old.get("novelty", 0.0),
                    old.get("confidence", 0.0),
                    old.get("has_conflict", False),
                    old.get("has_support", False)
                )
                score = old_score
                db.scores[claim] = score
        
        score._recompute_confidence()
        score.next_action = db.get_next_action(score)
        verdict = score.get_verdict()
        
        return (i, claim, verdict, score)
    
    # Process claims in parallel
    print("\n  🔄 Mining evidence in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EVIDENCE) as executor:
        futures = {executor.submit(process_claim, (i, claim, doscan_scores)): (i, claim) 
                   for i, claim in enumerate(claims)}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                i, claim, verdict, score = future.result()
                results[claim] = {
                    "verdict": verdict,
                    "confidence": round(score.confidence, 4),
                    "evidence_types_present": [
                        etype for etype, entries in score.evidence_sources.items() if entries
                    ],
                    "missing_evidence": score.missing_evidence,
                    "next_action": score.next_action,
                    "novelty": round(score.novelty_score, 4),
                    "falsifiability": round(score.falsifiability_score, 4)
                }
                
                print(f"\n  [{i+1}/{len(claims)}] {verdict} | Conf: {score.confidence:.2f}")
            except Exception as e:
                print(f"\n  ⚠️ Error processing claim: {e}")
    
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
    
    summary = {
        "total_claims": len(claims),
        "verdicts": {claim: r["verdict"] for claim, r in results.items()},
        "average_confidence": round(
            sum(r["confidence"] for r in results.values()) / len(results), 4
        ) if results else 0.0,
        "calibrated_threshold": round(threshold, 4),
        "action_items": [r["next_action"] for r in results.values()],
        "total_missing_evidence": sum(len(r["missing_evidence"]) for r in results.values())
    }
    
    print(f"\n📋 Summary: {summary['average_confidence']:.2f} avg confidence, {summary['total_missing_evidence']} gaps identified")
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