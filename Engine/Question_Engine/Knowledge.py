import os
import json
from typing import List, Dict
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

class ResearchKnowledge(BaseModel):
    core_research_thesis: str = Field(description="The central objective and primary scientific assumption being investigated.")
    state_of_the_art_prior_art: List[str] = Field(description="Current leading solutions, industry standards, and academic baselines already achieved.")
    proven_facts: List[str] = Field(description="Irrefutable axioms, chemical, physical, or logical parameters established by empirical consensus.")
    critical_unanswered_unknowns: List[str] = Field(description="The precise technical gaps, unmapped variables, or physical limitations preventing a complete solution.")
    proposed_testable_hypotheses: Dict[str, str] = Field(description="Proposed actionable claims paired with their corresponding verification metric/test method.")
    failure_modes_and_risks: List[str] = Field(description="Expected technical degradation, safety concerns, environmental risks, or operational failures.")
    required_empirical_data_inputs: List[str] = Field(description="Granular telemetry, data streams, materials characterization, or academic papers required next.")

def build_knowledge_base(problem_statement: str, analysis_summary: str) -> dict:
    print("\n[Knowledge Engine] Initializing deep academic extraction pipeline...")
    client = Groq()

    system_prompt = (
        "You are the Core Knowledge and Epistemology Engine of an Autonomous Research Lab. "
        "Your duty is to generate exhaustive, highly specific domain knowledge while aggressively identifying gaps "
        "and defining precise experimental validation hooks.\n\n"
        "You MUST respond purely with a valid JSON object matching EXACTLY this schema structure:\n"
        "{\n"
        '  "core_research_thesis": "string",\n'
        '  "state_of_the_art_prior_art": ["string"],\n'
        '  "proven_facts": ["string"],\n'
        '  "critical_unanswered_unknowns": ["string"],\n'
        '  "proposed_testable_hypotheses": {"Hypothesis Statement": "Verification Metric/Test Method"},\n'
        '  "failure_modes_and_risks": ["string"],\n'
        '  "required_empirical_data_inputs": ["string"]\n'
        "}\n"
        "Ensure no wrapper objects, no markdown blocks, and no extra text."
    )

    prompt = (
        f"Problem Scope: {problem_statement}\n\n"
        f"Domain Structural Context: {analysis_summary}\n\n"
        "Execute deep extraction of parameters, unknowns, hypotheses, and vulnerabilities."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
        )

        raw_json = completion.choices[0].message.content
        knowledge_data = ResearchKnowledge.model_validate_json(raw_json)
        
        knowledge_dict = knowledge_data.model_dump()
        output_filename = "deep_research_knowledge_base.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(knowledge_dict, f, indent=4)
            
        print(f"[Knowledge Engine] ✅ Master dataset successfully written to '{output_filename}'")
        return knowledge_dict

    except ValidationError as ve:
        print(f"\n❌ KNOWLEDGE VALIDATION ERROR:\n{ve}")
        return {}
    except Exception as e:
        print(f"\n❌ Exception in Knowledge.py: {e}")
        return {}