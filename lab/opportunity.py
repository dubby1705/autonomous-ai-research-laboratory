"""AARL opportunity report (research opportunity, not a finished paper)."""
from __future__ import annotations
from typing import Any, Dict, List
def opportunity_report(direction: Dict[str, Any]) -> Dict[str, Any]:
    st = direction.get("status") or {}
    return {"direction_id": direction.get("direction_id", ""), "title": direction.get("title", ""),
        "sections": [
            {"n": 1, "h": "Research Problem", "b": str(direction.get("research_problem", ""))},
            {"n": 2, "h": "Discovered Idea", "b": "%s\n%s" % (direction.get("title", ""), direction.get("core_question", ""))},
            {"n": 3, "h": "Why It Is Interesting", "b": "\n".join(str(x.get("point", x)) if isinstance(x, dict) else str(x) for x in (direction.get("why_interesting") or []))},
            {"n": 4, "h": "Mechanism (proposed, conceptual)", "b": "\n -> ".join(str(s.get("text", "")) for s in ((direction.get("mechanism") or {}).get("steps") or []))},
            {"n": 5, "h": "Inspiration / Cross-domain Connection", "b": str((direction.get("inspiration") or {}).get("display", ""))},
            {"n": 6, "h": "Existing Evidence", "b": "direct=%d indirect=%d contradicting=%d" % (((direction.get("evidence") or {}).get("counts") or {}).get("direct", 0), ((direction.get("evidence") or {}).get("counts") or {}).get("indirect", 0), ((direction.get("evidence") or {}).get("counts") or {}).get("contradicting", 0))},
            {"n": 7, "h": "Status", "b": "%s (%s). %s" % (st.get("label", ""), st.get("rule_id", ""), st.get("meaning", ""))},
            {"n": 8, "h": "What Is Speculative", "b": "\n".join(str(x) for x in (direction.get("speculative_aspects") or []))},
            {"n": 9, "h": "What AARL Does NOT Know", "b": "\n".join(str(g.get("question", "")) for g in (direction.get("gaps") or []))},
            {"n": 10, "h": "Major Risks / Failure Conditions", "b": "\n".join(str(f.get("condition", "")) for f in (direction.get("failure_conditions") or []))},
            {"n": 11, "h": "Suggested First Experiment", "b": str((direction.get("first_experiment") or {}).get("design", ""))},
            {"n": 12, "h": "Failure Conditions", "b": "See section 10."},
            {"n": 13, "h": "Related Alternative Ideas", "b": "\n".join("%s (%s)" % (a.get("title", ""), a.get("direction_id", "")) for a in (direction.get("alternatives") or []))},
            {"n": 14, "h": "Dimensions", "b": "; ".join("%s=%s" % (k, (v or {}).get("level", "")) for k, v in (direction.get("dimensions") or {}).items())},
        ],
        "honesty": "A research opportunity, not a finished scientific conclusion. Statuses describe stored evidence only."}
