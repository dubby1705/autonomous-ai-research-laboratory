import os
from typing import List, Dict
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

class DomainAnalysis(BaseModel):
    main_domain: str = Field(description="Primary overarching academic or technical field.")
    sub_domains: List[str] = Field(description="Highly specific technical sub-fields requiring literature review.")
    problem_breakdown: List[str] = Field(description="Deconstruction of variables, technical requirements, and target metrics.")
    governing_laws_and_equations: List[str] = Field(description="Scientific laws, mathematical theorems, or foundational equations that rule this problem.")
    key_research_questions: List[str] = Field(description="Deep scientific questions that must be answered to innovate or solve the issue.")
    suggested_methodologies: List[str] = Field(description="Advanced experimental setups, computational frameworks, simulation models, or algorithms needed.")
    interdisciplinary_links: Dict[str, str] = Field(description="Mapping of secondary fields to what specific insight or cross-over data they provide.")

def analyze_research_problem(problem_statement: str) -> DomainAnalysis:
    client = Groq()
    
    system_prompt = (
        "You are an Elite Research Strategy AI. Your job is to perform an exhaustive, multi-dimensional "
        "structural decomposition of a problem statement. Extract deep scientific properties and governing rules.\n\n"
        "You MUST respond purely with a valid JSON object matching EXACTLY these keys:\n"
        "{\n"
        '  "main_domain": "string",\n'
        '  "sub_domains": ["string"],\n'
        '  "problem_breakdown": ["string"],\n'
        '  "governing_laws_and_equations": ["string"],\n'
        '  "key_research_questions": ["string"],\n'
        '  "suggested_methodologies": ["string"],\n'
        '  "interdisciplinary_links": {"Field Name": "Reason for link / Contribution"}\n'
        "}\n"
        "Do not insert outer wrapper keys, do not wrap in backticks, and omit any conversational filler."
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Deconstruct this problem statement for an advanced scientific research lab. Output ONLY in the strict JSON template format:\n\n{problem_statement}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
        )
        raw_json = completion.choices[0].message.content
        return DomainAnalysis.model_validate_json(raw_json)
    except ValidationError as ve:
        print(f"\n❌ ANALYSER VALIDATION ERROR:\n{ve}")
        return None
    except Exception as e:
        print(f"\n❌ Exception in Problem_analyser.py: {e}")
        return None

def print_analysis_report(analysis: DomainAnalysis):
    if not analysis: return
    print("\n" + "="*80)
    print("🔬 PHASE 1: EXPANDED PROBLEM MATRIX DECONSTRUCTION")
    print("="*80)
    print(f"\n🟩 MAIN DOMAIN: {analysis.main_domain}")
    print("\n🔷 TARGET SUB-DOMAINS:")
    for sub in analysis.sub_domains: print(f"   • {sub}")
    print("\n🔍 GRANULAR COMPONENT BREAKDOWN:")
    for item in analysis.problem_breakdown: print(f"   • {item}")
    print("\n📐 GOVERNING LAWS & THEORETICAL FORMULAS:")
    for law in analysis.governing_laws_and_equations: print(f"   • {law}")
    print("\n❓ PIVOTAL RESEARCH QUESTIONS:")
    for q in analysis.key_research_questions: print(f"   • {q}")
    print("\n🛠️ ADVANCED EXPERIMENTAL METHODOLOGIES:")
    for method in analysis.suggested_methodologies: print(f"   • {method}")
    print("\n🌐 INTERDISCIPLINARY COUPLINGS:")
    for field, reason in analysis.interdisciplinary_links.items(): print(f"   • [{field}]: {reason}")
    print("\n" + "="*80)