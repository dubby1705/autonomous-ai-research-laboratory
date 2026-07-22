"""
AARL DEMO — Quick Test Script
Run this to see a full AARL pipeline end-to-end with a sample problem.
No user input required.

Usage:
    python demo.py
"""

import sys
import os

# Make sure Engine modules are importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
engine_path = os.path.join(ROOT_DIR, "Engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)
qe_path = os.path.join(ROOT_DIR, "Engine", "Question_Engine")
if qe_path not in sys.path:
    sys.path.insert(0, qe_path)

SAMPLE_PROBLEMS = {
    "battery": "Design a battery with higher energy density and better safety using solid-state electrolytes and graphene electrodes",
    "circuit": "Design a more efficient audio amplifier with lower total harmonic distortion using negative feedback",
    "chemistry": "Find a better catalyst for hydrogen production through water splitting using transition metal oxides",
    "ml": "Improve the training efficiency of large language models by reducing memory requirements during backpropagation"
}

def main():
    print("=" * 80)
    print("  AARL DEMO — Autonomous AI Research Laboratory")
    print("  Select a sample problem to run through the pipeline")
    print("=" * 80)
    print()
    
    # List available problems
    options = list(SAMPLE_PROBLEMS.keys())
    for i, key in enumerate(options, 1):
        print(f"  [{i}] {key.upper():12s} → {SAMPLE_PROBLEMS[key][:60]}...")
    print(f"  [{len(options)+1}] Custom problem (enter manually)")
    print()
    
    # Get selection
    try:
        choice = input("  Select [1-5], default=1: ").strip()
        if not choice:
            choice = "1"
        idx = int(choice) - 1
        if idx < 0 or idx >= len(options):
            problem = input("  Enter your research problem: ")
        else:
            problem = SAMPLE_PROBLEMS[options[idx]]
    except (ValueError, IndexError):
        problem = SAMPLE_PROBLEMS["battery"]
    
    print(f"\n  Selected problem: {problem}")
    print()
    
    # Run the AARL pipeline
    from Engine.Question_Engine.AARL_run import main as aarl_main
    
    # Monkey-patch the input calls to use our problem
    import builtins
    original_input = builtins.input
    
    input_responses = iter([
        problem,        # First input: research problem
        "0"             # Second input: deepening cycles (0 = skip for demo speed)
    ])
    
    def mock_input(prompt=""):
        try:
            return next(input_responses)
        except StopIteration:
            return original_input(prompt)
    
    builtins.input = mock_input
    
    try:
        aarl_main()
    finally:
        builtins.input = original_input


if __name__ == "__main__":
    main()