"""
EXPERIMENT ENGINE
Automatically designs testable experiments from hypotheses.

For each hypothesis, produces:
- Dataset / Test Environment
- Variables (Independent, Dependent, Controlled)
- Hyperparameters (learning rate, optimizer, epochs, etc.)
- Baseline method
- Evaluation Metrics
- Number of runs for statistical significance
- Success Criteria
- Failure Criteria
- Predicted Outcomes
- Estimated Computational Cost
"""

import os
import json
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class ExperimentDesign(BaseModel):
    hypothesis: str = Field(description="The hypothesis being tested.")
    experiment_name: str = Field(description="Short descriptive name for this experiment.")
    
    # Data
    dataset: str = Field(description="Specific dataset, simulator, or test environment to use.")
    dataset_source: str = Field(description="Where the dataset comes from (benchmark, synthetic, real-world).")
    dataset_size: int = Field(description="Expected number of samples / data points.")
    
    # Variables
    independent_variable: str = Field(description="The variable being manipulated.")
    dependent_variable: str = Field(description="The variable being measured.")
    controlled_variables: List[str] = Field(description="Variables held constant.")
    
    # Model / Setup
    architecture_or_method: str = Field(description="The model architecture, algorithm, or physical setup.")
    optimizer: str = Field(description="Optimizer used (SGD, Adam, AdamW, L-BFGS, etc.) or 'N/A' for non-ML.")
    learning_rate: float = Field(description="Learning rate or equivalent step size.")
    batch_size: int = Field(description="Batch size (or 1 if not applicable).")
    epochs_or_iterations: int = Field(description="Number of training epochs or experimental iterations.")
    
    # Baseline
    baseline_method: str = Field(description="The current best known method to compare against.")
    baseline_performance: str = Field(description="Known performance of the baseline (numeric if available).")
    
    # Metrics
    primary_metric: str = Field(description="The main evaluation metric (accuracy, F1, MSE, energy density, etc.).")
    secondary_metrics: List[str] = Field(description="Additional metrics to track.")
    
    # Statistical Rigor
    num_runs: int = Field(description="Number of independent runs for statistical significance (>=3).")
    confidence_interval: float = Field(description="Desired confidence interval (0.95 for 95%).")
    
    # Criteria
    success_criteria: str = Field(description="Quantifiable condition that confirms the hypothesis.")
    failure_criteria: str = Field(description="Quantifiable condition that refutes the hypothesis.")
    
    # Predictions
    predicted_outcome: str = Field(description="Expected result based on current knowledge.")
    predicted_effect_size: str = Field(description="Expected magnitude of improvement (e.g., '+15% accuracy').")
    predicted_compute_cost: str = Field(description="Estimated compute hours or FLOPs.")
    predicted_failure_modes: List[str] = Field(description="Anticipated ways the experiment could fail.")

class ExperimentResult(BaseModel):
    experiment_name: str = Field(description="Name matching the design.")
    hypothesis_supported: bool = Field(description="Whether the experiment supported the hypothesis.")
    observed_effect_size: str = Field(description="What was actually observed.")
    confidence_in_result: float = Field(description="Statistical confidence in the result (0.0-1.0).")
    unexpected_findings: List[str] = Field(description="Anything surprising that occurred.")
    did_fail_as_predicted: bool = Field(description="Whether failure modes materialized.")
    lessons_learned: str = Field(description="What this experiment teaches us.")


# =========================================================
# EXPERIMENT DESIGNER (LLM-based)
# =========================================================

def design_experiment(client: Groq, hypothesis: str, domain_context: str = "") -> Optional[ExperimentDesign]:
    """
    Uses the LLM to design a rigorous, testable experiment for a given hypothesis.
    """
    system_prompt = (
        "You are an Expert Experimental Scientist. Your job is to design a rigorous, "
        "reproducible experiment to test a given hypothesis. Be extremely specific.\n\n"
        "You MUST respond with a valid JSON object matching this schema exactly:\n"
        "{\n"
        '  "hypothesis": "string",\n'
        '  "experiment_name": "string",\n'
        '  "dataset": "string",\n'
        '  "dataset_source": "string",\n'
        '  "dataset_size": 1000,\n'
        '  "independent_variable": "string",\n'
        '  "dependent_variable": "string",\n'
        '  "controlled_variables": ["string", "string"],\n'
        '  "architecture_or_method": "string",\n'
        '  "optimizer": "string",\n'
        '  "learning_rate": 0.001,\n'
        '  "batch_size": 32,\n'
        '  "epochs_or_iterations": 100,\n'
        '  "baseline_method": "string",\n'
        '  "baseline_performance": "string",\n'
        '  "primary_metric": "string",\n'
        '  "secondary_metrics": ["string"],\n'
        '  "num_runs": 5,\n'
        '  "confidence_interval": 0.95,\n'
        '  "success_criteria": "string",\n'
        '  "failure_criteria": "string",\n'
        '  "predicted_outcome": "string",\n'
        '  "predicted_effect_size": "string",\n'
        '  "predicted_compute_cost": "string",\n'
        '  "predicted_failure_modes": ["string"]\n'
        "}\n"
        "Include specific numbers, dataset names, and concrete metrics."
    )
    
    prompt = (
        f"Hypothesis to test: {hypothesis}\n\n"
        f"Domain context: {domain_context if domain_context else 'General scientific domain.'}\n\n"
        "Design a rigorous experiment. Be specific about datasets, hyperparameters, and criteria."
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Low temp for precision
        )
        raw_json = completion.choices[0].message.content
        return ExperimentDesign.model_validate_json(raw_json)
    except ValidationError as ve:
        print(f"   ❌ Experiment validation error: {ve}")
        return None
    except Exception as e:
        print(f"   ❌ Error designing experiment: {e}")
        return None


def validate_experiment_design(design: ExperimentDesign) -> Tuple[bool, List[str]]:
    """
    Validates that an experiment design has internal consistency.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    
    # Check num_runs for statistical significance
    if design.num_runs < 3:
        issues.append(f"num_runs={design.num_runs} < 3. Need at least 3 runs for statistical significance.")
    
    # Check learning rate is reasonable
    if design.learning_rate <= 0 or design.learning_rate > 10:
        issues.append(f"Learning rate {design.learning_rate} is outside reasonable range (0, 10].")
    
    # Check batch size
    if design.batch_size < 1:
        issues.append(f"Batch size {design.batch_size} < 1.")
    
    # Check epochs
    if design.epochs_or_iterations < 1:
        issues.append(f"Epochs {design.epochs_or_iterations} < 1.")
    
    # Check confidence interval
    if not (0.5 <= design.confidence_interval <= 1.0):
        issues.append(f"Confidence interval {design.confidence_interval} outside [0.5, 1.0].")
    
    # Check dataset size
    if design.dataset_size < 10:
        issues.append(f"Dataset size {design.dataset_size} < 10. Too small for meaningful results.")
    
    # Check we have both success and failure criteria
    if not design.success_criteria:
        issues.append("No success criteria defined.")
    if not design.failure_criteria:
        issues.append("No failure criteria defined.")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def estimate_compute_cost(design: ExperimentDesign) -> Dict[str, Any]:
    """
    Estimates computational cost of the experiment.
    Uses formula: cost ≈ num_runs × epochs × batch_processing_time
    """
    # Rough FLOPs estimation
    # Assume ~1e12 FLOPs per epoch per batch for a typical deep learning model
    estimated_flops = design.num_runs * design.epochs_or_iterations * 1e12
    
    # Rough GPU hours (assuming one A100-equivalent at 312 TFLOPS)
    gpu_hours = estimated_flops / (312e12 * 3600) if estimated_flops > 0 else 0
    
    # Rough cost in USD (assuming $3/hr for A100)
    estimated_cost_usd = gpu_hours * 3.0
    
    return {
        "estimated_flops": f"{estimated_flops:.2e}",
        "estimated_gpu_hours": round(gpu_hours, 2),
        "estimated_cost_usd": round(estimated_cost_usd, 2),
        "total_runs": design.num_runs,
        "total_epochs_across_runs": design.num_runs * design.epochs_or_iterations,
        "feasibility": "high" if gpu_hours < 10 else ("medium" if gpu_hours < 100 else "low")
    }


# =========================================================
# MAIN EXPERIMENT ENGINE ENTRY POINT
# =========================================================

def run_experiment_engine(
    hypotheses: List[str], 
    domain_context: str = "",
    designs_file: str = "experiment_designs.json"
) -> Dict[str, Any]:
    """
    Main entry point for the Experiment Engine.
    
    For each hypothesis:
    1. Designs a rigorous experiment via LLM
    2. Validates the design for consistency
    3. Estimates compute cost
    4. Saves all designs to disk
    
    Args:
        hypotheses: List of hypotheses to design experiments for
        domain_context: Optional domain-specific context
    
    Returns:
        Dict with experiment designs and analysis
    """
    print("\n" + "="*80)
    print("🧪 EXPERIMENT ENGINE — Designing Rigorous, Testable Experiments")
    print("="*80)
    
    client = Groq()
    
    experiments = []
    validation_results = []
    
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n  [{i}/{len(hypotheses)}] Designing experiment for hypothesis...")
        print(f"      📝 {hypothesis[:120]}...")
        
        # Step 1: Design
        design = design_experiment(client, hypothesis, domain_context)
        if not design:
            print(f"      ❌ Failed to design experiment.")
            continue
        
        # Step 2: Validate
        is_valid, issues = validate_experiment_design(design)
        if not is_valid:
            print(f"      ⚠️  Design has issues:")
            for issue in issues:
                print(f"          • {issue}")
        
        # Step 3: Estimate cost
        cost = estimate_compute_cost(design)
        
        experiment_entry = {
            "design": design.model_dump(),
            "validation": {
                "is_valid": is_valid,
                "issues": issues
            },
            "compute_cost": cost,
            "timestamp": datetime.now().isoformat()
        }
        
        experiments.append(experiment_entry)
        validation_results.append({
            "hypothesis": hypothesis,
            "experiment_name": design.experiment_name,
            "valid": is_valid,
            "issues": issues,
            "cost_feasibility": cost["feasibility"]
        })
        
        print(f"      ✅ Experiment: {design.experiment_name}")
        print(f"      📊 Dataset: {design.dataset} ({design.dataset_source})")
        print(f"      🎯 Metric: {design.primary_metric}")
        print(f"      🔄 Runs: {design.num_runs} @ {design.confidence_interval*100:.0f}% CI")
        print(f"      💰 Est. Cost: {cost['estimated_cost_usd']} USD ({cost['feasibility']} feasibility)")
        print(f"      ✅ Valid: {'YES' if is_valid else 'NO - ' + '; '.join(issues[:2])}")
    
    # Save to disk
    output = {
        "total_experiments_designed": len(experiments),
        "experiments": experiments,
        "validation_summary": {
            "total_valid": sum(1 for v in validation_results if v["valid"]),
            "total_invalid": sum(1 for v in validation_results if not v["valid"]),
            "feasibility_counts": {
                "high": sum(1 for v in validation_results if v["cost_feasibility"] == "high"),
                "medium": sum(1 for v in validation_results if v["cost_feasibility"] == "medium"),
                "low": sum(1 for v in validation_results if v["cost_feasibility"] == "low")
            }
        },
        "recommendation": "Proceed with valid, high-feasibility experiments first."
    }
    
    with open(designs_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"\n📄 All experiment designs saved to '{designs_file}'")
    print(f"📊 Summary: {output['validation_summary']['total_valid']} valid, "
          f"{output['validation_summary']['total_invalid']} invalid, "
          f"{output['validation_summary']['feasibility_counts']['high']} high feasibility")
    print("="*80)
    
    return output


if __name__ == "__main__":
    test_hypotheses = [
        "Increasing electrode surface area improves battery power density by 20%",
        "AdamW optimizer generalizes better than Adam for transformer architectures",
        "Graphene-based anodes double lithium-ion battery capacity"
    ]
    run_experiment_engine(test_hypotheses, domain_context="battery technology and deep learning")