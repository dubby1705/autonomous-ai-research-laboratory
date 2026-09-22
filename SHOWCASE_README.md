# AARL Showcase — Autonomous AI Research Laboratory

A reliable, research-focused showcase built **on top of** the existing AARL
project. It demonstrates a complete autonomous research cycle in one controlled
domain — **machine-learning optimisation** ("How can gradient descent
convergence be improved?").

The existing `Engine/` pipeline is untouched; this showcase lives in
`showcase/` with new top-level entry points.

## Quick start

```bash
# 1. configure your Groq key (never commit .env)
copy .env.example .env      # then edit .env: GROQ_API_KEY=YOUR_KEY

# 2. test the Groq connection
python test_groq.py

# 3. run a complete research showcase cycle
python main.py

# optional tuning
python main.py --hypotheses 8 --runs 3 --seed 11 --max-iterations 4000
python main.py --self-test          # run the bundled test suite
```

Without a Groq key the planning stages use a clearly-labelled deterministic
fallback so the whole cycle still runs; **experiments are always real**.

## Pipeline

```
RESEARCH PROBLEM
      ↓
PHASE 1  HYPOTHESIS GENERATION   (LLM plans; safe fallback otherwise)
      ↓
PHASE 2  CRITICAL EVALUATION     (novelty / feasibility / reasoning scores)
      ↓
PHASE 3  RESEARCH TOURNAMENT     (threshold gate -> MULTIPLE survivors;
      ↓                           rejected ideas preserved with reasons)
PHASE 4  EXPERIMENT DESIGN       (LLM selects from a SAFE optimiser registry)
      ↓
         REAL SIMULATION         (seeded baseline-vs-proposed trajectories)
      ↓
PHASE 5  MEASURED RESULTS        (iterations, loss, convergence rate, % gain)
      ↓
PHASE 6  DEEP ANALYSIS           (LLM reasoning grounded ONLY in real numbers)
      ↓
         HYPOTHESIS REFINEMENT   (H1 -> H1-R1, evidence-informed)
      ↓
FINAL FINDINGS + REPORT + PLOTS  (saved per-run, never overwritten)
```

## Key guarantees

- **No fabricated evidence.** Every number comes from executable NumPy
  simulations (`showcase/optimizers.py`). The LLM never generates code; it only
  picks/tunes parameters inside a validated registry.
- **Multiple survivors.** The tournament keeps *every* hypothesis above the
  score threshold; rejected ones keep their scores and rejection reasons.
- **Failure isolation.** One broken experiment, malformed LLM reply, or missing
  API key marks that stage failed and continues.
- **Traceability.** Full state is persisted to JSON each run under
  `runs/<date>_<seq>/` (`research.json`, `hypotheses.json`, `experiments.json`,
  `results.json`, `report.md`, `plots/`).
- **Security.** The API key is read from `.env`/environment only, never printed,
  never stored in results. `.gitignore` excludes `.env` and `runs/`.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Showcase entry point |
| `test_groq.py` | Groq connectivity + diagnostics |
| `showcase/config.py` | Central configuration (env + tuned defaults) |
| `showcase/records.py` | Structured hypothesis/experiment/run dataclasses |
| `showcase/llm_service.py` | Single reusable Groq client: robust JSON parsing, bounded retries, friendly error taxonomy |
| `showcase/generation.py` | Hypothesis generation, critique, tournament, experiment design |
| `showcase/optimizers.py` | Safe optimiser registry + controlled problems |
| `showcase/experiment_engine.py` | Baseline vs proposed execution + aggregation |
| `showcase/analysis.py` | Evidence-grounded analysis + refinement |
| `showcase/report.py` | Terminal presentation, Markdown report, matplotlib plots |
| `showcase/storage.py` | Unique per-run storage |
| `tests/test_showcases.py` | Offline unit + end-to-end tests |

## Reproducibility

Seeds, dimensions, iteration budgets, tolerance, run counts and the full config
snapshot are stored in every run's JSON. Re-running with identical flags
reproduces identical measurements.
