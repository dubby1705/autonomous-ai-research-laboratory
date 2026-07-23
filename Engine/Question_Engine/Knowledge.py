import os
import json
import re
from typing import List, Dict
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

def _repair_json(raw: str) -> str:
    """
    Attempt to repair common JSON generation issues from LLMs:
    - Trailing unescaped newlines inside strings
    - Broken last array element
    - Missing closing brackets
    """
    # Remove any text before the first '{'
    first_brace = raw.find('{')
    if first_brace >= 0:
        raw = raw[first_brace:]
    
    # Remove any text after the last '}'
    last_brace = raw.rfind('}')
    if last_brace >= 0:
        raw = raw[:last_brace + 1]
    
    # Replace literal newlines inside strings with escaped newlines
    # First, find all string values and escape internal newlines
    lines = raw.split('\n')
    repaired_lines = []
    in_string = False
    for line in lines:
        # Count unescaped quotes to track string boundaries
        quote_count = line.count('"') - line.count('\\"')
        if quote_count % 2 != 0:
            in_string = not in_string
        if in_string:
            # This line is inside a string value, escape it
            repaired_lines.append(line.rstrip('\n\r'))
        else:
            repaired_lines.append(line)
    
    repaired = '\n'.join(repaired_lines)
    
    # Remove trailing comma before closing brace/bracket (common JSON error)
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    return repaired

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
        "You MUST respond with a valid JSON object matching EXACTLY this schema. "
        "CRITICAL: Do NOT use colons inside dictionary keys (like 'Hypothesis 1: ...'). "
        "Use simple keys instead. CRITICAL: All string values must be on a single line — no line breaks inside strings.\n\n"
        "{\n"
        '  "core_research_thesis": "string",\n'
        '  "state_of_the_art_prior_art": ["string"],\n'
        '  "proven_facts": ["string"],\n'
        '  "critical_unanswered_unknowns": ["string"],\n'
        '  "proposed_testable_hypotheses": {"simple key": "verification method"},\n'
        '  "failure_modes_and_risks": ["string"],\n'
        '  "required_empirical_data_inputs": ["string"]\n'
        "}\n"
        "No wrapper objects, no markdown blocks, no extra text."
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
        # Attempt to repair common JSON formatting issues from LLM output
        repaired_json = _repair_json(raw_json)
        if repaired_json != raw_json:
            print("   [JSON Repair] Applied fixes to LLM output")
        knowledge_data = ResearchKnowledge.model_validate_json(repaired_json)
        
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