"""
AARL — Autonomous AI Research Laboratory
=========================================
Top-level entry point for the research showcase.

Usage:
    python main.py                     # run a full showcase cycle
    python main.py --hypotheses 8      # tune the number of hypotheses
    python main.py --runs 3            # experimental repeats
    python main.py --self-test         # run the built-in test suite
    python main.py --generate-paper    # generate a research paper from outputs

The LLM (Groq) plans and analyses; all experimental numbers come from real,
seeded simulations. If no GROQ_API_KEY is configured the pipeline transparently
uses a deterministic planning fallback (clearly labelled in the outputs).
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from typing import List

# -- logging: concise terminal logging, full details only in run JSON ---------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname).1s] %(name)s: %(message)s",
)
log = logging.getLogger("aarl.main")

_DESCRIPTION = (
    "AARL research-showcase: generate competing hypotheses, critique them, keep "
    "multiple survivors, test each with real simulations, analyse the evidence "
    "with an LLM, and produce a traceable report.\n\n"
    "Additional options for research paper generation:"
    "\n  --generate-paper              generate a research paper from existing outputs"
    "\n  --paper-format markdown|latex  paper output format (default: markdown)"
    "\n  --paper-filename NAME         custom filename for the paper"
    "\n  --paper-output-dir DIR        output directory for the paper"
    "\n  --paper-run-dir DIR           specific run directory to use"
    "\n\n"
    "Parallel processing (all pipelines fan out over the CPU cores):"
    "\n  --workers N | --workers=N     worker cap (0 = auto = all cores)"
    "\n  --parallel-mode thread|process  pool type (default: thread)"
    "\n  --parallel-batch N            batch size for very large job lists"
    "\n  --no-parallel                 force single-threaded execution"
    "\n  --processing parallel|sequential  processing style (default: parallel)"
    "\n  --normal | --one-by-one       alias of --processing sequential"
    "\n  --parallel                    force parallel fan-out (default)"
    "\n  --llm-workers N               concurrency cap for LLM-backed fan-out"
)


def _print_llm_status() -> None:
    from showcase.config import get_groq_api_key, get_groq_model

    key_set = bool(get_groq_api_key())
    model = get_groq_model()
    print("=" * 60)
    print("  LLM CONFIGURATION")
    print("=" * 60)
    print(f"  Groq API key : {'DETECTED' if key_set else 'NOT SET'}")
    print(f"  Groq model   : {model}")
    print("=" * 60)
    if not key_set:
        print("\n  NOTE: no GROQ_API_KEY. Planning stages will use a clearly\n"
              "        labelled deterministic fallback; experiments remain real.\n"
              "        Add GROQ_API_KEY=... to a .env file to enable the LLM.\n")


def run_showcase(args: argparse.Namespace) -> int:
    from showcase.config import ResearchConfig
    from showcase.pipeline import run_research_cycle, persist_and_report
    from showcase.report import banner

    _print_llm_status()
    print(banner())

    parallel_status = _apply_parallel_args(args)
    print(f"Parallel Processing: {parallel_status}\n")

    config = ResearchConfig(
        research_question=(args.question or ResearchConfig().research_question),
        num_hypotheses=max(2, int(args.hypotheses)),
        num_runs=max(1, int(args.runs)),
        dimensionality=max(4, int(args.dimensions)),
        max_iterations=max(500, int(args.max_iterations)),
        random_seed=int(args.seed),
        run_root=args.runs_dir,
        parallel=not bool(getattr(args, "no_parallel", False)),
        max_workers=int(getattr(args, "workers", 0) or 0),
    )

    print(f"RESEARCH PROBLEM\n{'-' * 76}\n{config.research_question}\n")

    run = run_research_cycle(config)
    result = persist_and_report(run, config)

    # ---- polished terminal presentation -------------------------------
    print(result["block"])

    print("\nRUN ARTIFACTS")
    print("-" * 76)
    for kind, path in _flatten(result["written"]):
        print(f"  {kind:<22s} {path}")
    print()
    return 0


def _flatten(written: dict) -> List[tuple]:
    out = []
    for k, v in written.items():
        if isinstance(v, list):
            for p in v:
                out.append((k, p))
        else:
            out.append((k, v))
    return out


def run_selftest() -> int:
    """Run the bundled test suite via unittest discovery."""
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _apply_parallel_args(args: argparse.Namespace) -> str:
    """
    Export the parallelism CLI flags as environment variables and report the
    resolved setup.

    The shared ParallelExecutor reads its configuration from the environment on
    first use, so setting the variables here (before any pipeline module fans
    out work) configures the whole run, including the paper generator.

        --workers N | --workers=N   worker cap (0 = auto = cpu_count)
        --parallel-mode MODE        thread (default) or process
        --parallel-batch N          batch size for very large job lists
        --processing STYLE          parallel (default) or sequential one-by-one
        --normal / --one-by-one     alias of --processing sequential
        --parallel                  force parallel fan-out (default)
        --llm-workers N             concurrency cap for LLM-backed fan-out
        --no-parallel               force single-threaded execution
    """
    if getattr(args, "workers", None) is not None:
        os.environ["AARL_MAX_WORKERS"] = str(max(0, int(args.workers)))
    if getattr(args, "parallel_mode", None):
        os.environ["AARL_PARALLEL_MODE"] = str(args.parallel_mode).lower()
    if getattr(args, "parallel_batch", None):
        os.environ["AARL_PARALLEL_BATCH"] = str(max(0, int(args.parallel_batch)))
    if getattr(args, "llm_workers", None) is not None:
        os.environ["AARL_LLM_WORKERS"] = str(max(1, int(args.llm_workers)))

    style = getattr(args, "processing", None)
    if getattr(args, "normal", False):
        style = "sequential"
    if getattr(args, "parallel", False):
        style = "parallel"
    if style:
        os.environ["AARL_PROCESSING"] = str(style).lower()

    if getattr(args, "no_parallel", False):
        os.environ["AARL_PARALLEL"] = "0"
    else:
        os.environ.setdefault("AARL_PARALLEL", "1")

    try:
        from Engine.Question_Engine.ParallelExecutor import describe_parallelism

        return describe_parallelism()
    except Exception:  # pragma: no cover - pipeline not importable
        return "unavailable"


def run_paper_generator(args: argparse.Namespace) -> int:
    """Run the research paper generator standalone."""
    from Engine.Question_Engine.ResearchPaperGenerator import generate_research_paper

    parallel_status = _apply_parallel_args(args)

    print("=" * 70)
    print("AARL Research Paper Generator")
    print("=" * 70)
    print(f"Root Directory: {args.root_dir}")
    print(f"Output Format: {args.paper_format}")
    print(f"Output Directory: {args.paper_output_dir or os.path.join(args.root_dir, 'Research_Papers')}")
    print(f"Parallel Processing: {parallel_status}")
    print("-" * 70)

    try:
        paper_path = generate_research_paper(
            root_dir=args.root_dir,
            output_dir=args.paper_output_dir,
            run_dir=args.paper_run_dir,
            format=args.paper_format,
            filename=args.paper_filename,
            max_workers=getattr(args, "workers", None),
        )
        print("[OK] Research paper generated successfully!")
        print(f"[FILE] Paper location: {paper_path}")

        # Show paper contents
        print("")
        print("Paper Contents:")
        print("   - Title Page with problem statement")
        print("   - Abstract with key findings")
        print("   - Table of Contents")
        print("   - Introduction (motivation, problem statement, objectives)")
        print("   - Literature Review / State of the Art")
        print("   - Research Methodology (hypothesis generation, evaluation, validation)")
        print("   - Results and Analysis")
        print("   - Discussion with implications")
        print("   - Conclusion with future work")
        print("   - References")
        print("   - Appendices with detailed data")
        print("   - FINAL IDEA SUMMARY (comprehensive synthesis)")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"[FAIL] Error generating paper: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="AARL",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--question", default=None, help="research problem (default: gradient descent convergence)")
    parser.add_argument("--hypotheses", type=int, default=8, help="number of hypotheses to generate")
    parser.add_argument("--runs", type=int, default=3, help="experiment runs per method")
    parser.add_argument("--dimensions", type=int, default=12, help="problem dimensionality")
    parser.add_argument("--max-iterations", type=int, default=4000, help="iteration budget per trajectory")
    parser.add_argument("--seed", type=int, default=11, help="random seed for reproducibility")
    parser.add_argument("--runs-dir", default="runs", help="directory that holds research runs")
    parser.add_argument("--self-test", action="store_true", help="run the bundled test suite and exit")

    # Research paper generator arguments
    parser.add_argument("--generate-paper", action="store_true", help="generate a comprehensive research paper from existing outputs")
    parser.add_argument("--root-dir", default=".", help="root directory of AARL project (for paper generation)")
    parser.add_argument("--paper-output-dir", default=None, help="output directory for the research paper")
    parser.add_argument("--paper-format", choices=["markdown", "latex"], default="markdown", help="paper output format")
    parser.add_argument("--paper-filename", default=None, help="custom filename for the research paper")
    parser.add_argument("--paper-run-dir", default=None, help="specific run directory to use for paper generation")
    parser.add_argument("--workers", type=int, default=None, metavar="N", help="parallel worker cap (0 = auto = all cores); --workers=N also accepted")
    parser.add_argument("--no-parallel", action="store_true", help="disable parallel fan-out and run single-threaded")
    parser.add_argument("--parallel-mode", choices=["thread", "process"], default=None, help="parallel pool type (default: thread)")
    parser.add_argument("--parallel-batch", type=int, default=None, metavar="N", help="parallel batch size for very large job lists")
    parser.add_argument("--processing", choices=["parallel", "sequential"], default=None, help="processing style: parallel fan-out (default) or sequential one-by-one")
    parser.add_argument("--normal", "--one-by-one", dest="normal", action="store_true", help="alias of --processing sequential (one-by-one processing)")
    parser.add_argument("--parallel", action="store_true", help="force parallel fan-out (default)")
    parser.add_argument("--llm-workers", type=int, default=None, metavar="N", help="concurrency cap for LLM-backed fan-out (env: AARL_LLM_WORKERS)")

    args = parser.parse_args(argv)

    # Handle paper generation mode
    if args.generate_paper:
        return run_paper_generator(args)

    if args.self_test:
        try:
            return run_selftest()
        except Exception as exc:  # noqa: BLE001
            log.error("Self-test failed: %s", exc)
            return 1

    try:
        return run_showcase(args)
    except KeyboardInterrupt:
        print("\n[AARL] interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.error("Research cycle failed: %s", exc)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
