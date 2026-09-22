"""
AARL Showcase — Presentation & Report Generation
=================================================
- Polished, research-oriented terminal output (colour-free, ASCII-safe).
- A full Markdown research report stored with the run.
- Lightweight matplotlib visualisations (loss vs iteration, improvement bars).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .records import Hypothesis, ResearchRun

log = logging.getLogger("aarl.report")

_SLINE = "-" * 76
_GAP = "\n" + "-" * 76 + "\n"


def banner() -> str:
    return (
        "\n"
        "====================================================================\n"
        "       AARL - AUTONOMOUS AI RESEARCH LABORATORY\n"
        "       (research-showcase build)\n"
        "====================================================================\n"
    )


def section(title: str) -> str:
    return (f"\n------------------------------------------------------------\n"
            f"{title}\n------------------------------------------------------------\n")


def format_improvement(pct: Optional[float]) -> str:
    if pct is None:
        return "n/a (no clean comparison)"
    if pct > 0:
        return f"+{pct:.1f}%"
    return f"{pct:.1f}%"


def render_hypotheses(generated: List[Hypothesis]) -> str:
    lines = [section("PHASE 1 - HYPOTHESIS GENERATION"),
             f"Generated: {len(generated)} hypotheses"]
    for h in generated:
        lines.append(f"{h.hypothesis_id:<4s} {h.title}")
    return "\n".join(lines)


def render_tournament(hyps: List[Hypothesis]) -> str:
    lines = [section("PHASE 2 - CRITICAL EVALUATION")]
    for h in hyps:
        status = "PASS" if h.passed else "REJECTED"
        lines.append(
            f"{h.hypothesis_id:<4s} {h.evaluation.overall_score:6.1f}/100  "
            f"{status:<8s} {h.title}"
        )
        if not h.passed and h.rejection_reason:
            lines.append(f"       reason: {h.rejection_reason}")

    survivors = [h for h in hyps if h.passed]
    rejected = [h for h in hyps if not h.passed]
    lines.append(section("PHASE 3 - RESEARCH TOURNAMENT"))
    lines.append("Survivors (multiple hypotheses continue to experimentation):")
    for h in survivors:
        lines.append(f"   {h.hypothesis_id}  {h.title}")
    if rejected:
        lines.append("Rejected (kept for research history):")
        for h in rejected:
            lines.append(f"   {h.hypothesis_id}  {h.title}")
    return "\n".join(lines)
def render_experiments(hyps: List[Hypothesis]) -> str:
    lines = [section("PHASE 4 - EXPERIMENTATION (REAL SIMULATION)")]
    for h in hyps:
        if not h.passed:
            continue
        method = h.experiment_spec.proposed_method or "?"
        lines.append(f"[{h.hypothesis_id}] Running baseline vs {method} ...")
        if h.experiment.status == "SUCCESS":
            imp = format_improvement(h.experiment.improvement_pct)
            lines.append(
                f"[{h.hypothesis_id}] COMPLETE  "
                f"baseline={h.experiment.baseline.get('mean_iterations')} iters, "
                f"proposed={h.experiment.proposed.get('mean_iterations')} iters, "
                f"improvement={imp}"
            )
        else:
            lines.append(f"[{h.hypothesis_id}] {h.experiment.status} - {h.experiment.message}")
    return "\n".join(lines)


def render_results(hyps: List[Hypothesis]) -> str:
    lines = [section("PHASE 5 - EXPERIMENTAL RESULTS")]
    for h in hyps:
        if not h.passed:
            continue
        if h.experiment.status != "SUCCESS":
            lines.append(
                f"{h.hypothesis_id}  {h.experiment.status}: "
                f"{h.experiment.error or h.experiment.message}"
            )
            continue
        b = h.experiment.baseline
        p = h.experiment.proposed
        lines.extend([
            f"{h.hypothesis_id}  {h.title}",
            f"   Baseline : conv_rate={b.get('convergence_rate')}, "
            f"mean_iters={b.get('mean_iterations')}, final_loss={b.get('mean_final_loss')}",
            f"   Proposed : conv_rate={p.get('convergence_rate')}, "
            f"mean_iters={p.get('mean_iterations')}, final_loss={p.get('mean_final_loss')}",
            f"   Improvement: {format_improvement(h.experiment.improvement_pct)}",
        ])
    return "\n".join(lines)


def render_analysis(hyps: List[Hypothesis]) -> str:
    lines = [section("PHASE 6 - DEEP RESEARCH ANALYSIS")]
    for h in hyps:
        if not h.passed:
            continue
        lines.append(f"{h.hypothesis_id}")
        lines.append(f"   Mechanism: {h.mechanism}")
        lines.append(f"   Evidence strength: {h.analysis.evidence_strength}")
        lines.append(f"   Conclusion: {h.analysis.conclusion}")
    return "\n".join(lines)


def render_findings(run: ResearchRun) -> str:
    lines = [section("FINAL RESEARCH FINDINGS")]
    findings = run.findings or {}
    lines.append(findings.get("summary", "No findings summary generated."))
    best = findings.get("best_hypothesis") or {}
    if best:
        lines.append(
            f"Best hypothesis: {best.get('hypothesis_id')} - {best.get('title')} "
            f"({format_improvement(best.get('improvement_pct'))})"
        )
    return "\n".join(lines)


def build_terminal_block(run: ResearchRun) -> str:
    """Closing presentation block (phases 1-4 are streamed live by the pipeline)."""
    blocks = [
        render_results(run.hypotheses),
        render_analysis(run.hypotheses),
        render_findings(run),
        "\n====================================================================\n"
        f"  RUN COMPLETE - full history saved under {run.config.get('run_dir', '(not saved)')}\n"
        "====================================================================\n",
    ]
    return "\n".join(blocks)
# ---------------------------------------------------------------------------
# Markdown research report
# ---------------------------------------------------------------------------
def build_markdown_report(run: ResearchRun) -> str:
    """Generate the full, structured, traceable research report (Markdown)."""
    cfg = run.config or {}
    findings = run.findings or {}
    passed = [h for h in run.hypotheses if h.passed]
    rejected = [h for h in run.hypotheses if not h.passed]
    lines = [
        "# AARL Research Report",
        "",
        f"- **Run ID**: {run.run_id}",
        f"- **Created**: {run.created_at}",
        f"- **Domain**: {run.domain}",
        f"- **LLM available**: {run.llm_available} ({run.llm_model})",
        "",
        "## 1. Research Problem",
        "",
        run.research_question,
        "",
        "## 2. Research Objective",
        "",
        "Generate competing hypotheses, critically evaluate them, keep multiple "
        "survivors, test each survivor with real experiments, and produce a "
        "traceable research report.",
        "",
        f"Reproducibility: seed **{cfg.get('random_seed')}**, "
        f"{cfg.get('num_runs')} runs, {cfg.get('dimensionality')} dims, "
        f"{cfg.get('max_iterations')} max iterations, "
        f"convergence tolerance {cfg.get('convergence_tol')}.",
        "",
        "## 3. Hypotheses Generated",
        "",
    ]
    for h in run.hypotheses:
        lines.append(f"### {h.hypothesis_id} - {h.title}")
        lines.append("")
        lines.append(f"- **Description**: {h.description}")
        lines.append(f"- **Research question**: {h.research_question}")
        lines.append(f"- **Reasoning**: {h.reasoning}")
        lines.append(f"- **Mechanism**: {h.mechanism}")
        lines.append(f"- **Assumptions**: {', '.join(h.assumptions) if h.assumptions else 'none'}")
        lines.append(f"- **Source**: {h.source}")
        lines.append("")

    lines.append("## 4. Critical Evaluation")
    lines.append("")
    lines.append("| ID | Novelty | Feasibility | Reasoning | Overall | Verdict |")
    lines.append("|----|---------|-------------|-----------|---------|---------|")
    for h in run.hypotheses:
        ev = h.evaluation
        verdict = "PASS" if h.passed else "REJECTED"
        lines.append(
            f"| {h.hypothesis_id} | {ev.novelty_score:.1f} | "
            f"{ev.feasibility_score:.1f} | {ev.reasoning_score:.1f} | "
            f"{ev.overall_score:.1f} | {verdict} |"
        )
    lines.append("")

    lines.append("## 5. Tournament Results")
    lines.append("")
    lines.append(
        f"Pass threshold: **{cfg.get('tournament_pass_threshold', 50)}/100**. "
        "Multiple survivors continue to experimentation."
    )
    lines.append("Rejected hypotheses are retained with their scores and reasons.")
    lines.append("")
    for h in rejected:
        lines.append(
            f"- **{h.hypothesis_id}** rejected: {h.rejection_reason or h.evaluation.reviewer_reasoning}"
        )
    lines.append("")

    lines.append("## 6. Surviving Hypotheses")
    lines.append("")
    for h in passed:
        lines.append(f"- **{h.hypothesis_id}**: {h.title}")
    lines.append("")
    lines.append("## 7. Experimental Methodology")
    lines.append("")
    lines.append(
        "Each surviving hypothesis was mapped to a **safe, predefined optimiser** "
        "from a fixed registry (the LLM selects/tunes parameters; it never "
        "generates code). Every experiment ran the **baseline SGD** and the "
        "**proposed method** on an identical controlled problem with identical "
        "seeds, starting points and iteration budgets."
    )
    lines.append("")
    lines.append("**Baseline**: SGD, learning rate 0.1.")
    lines.append("")
    lines.append("## 8. Baseline")
    lines.append("")
    first = next((h for h in passed if h.experiment.status == "SUCCESS"), None)
    if first:
        b = first.experiment.baseline
        lines.extend([
            f"- Convergence rate: {b.get('convergence_rate')}",
            f"- Mean iterations: {b.get('mean_iterations')}",
            f"- Mean final loss: {b.get('mean_final_loss')}",
            "",
        ])
    else:
        lines.extend(["_Baseline did not produce measurable data._", ""])

    lines.append("## 9. Experimental Results")
    lines.append("")
    lines.append("| ID | Method | Status | Conv (B/P) | Mean iters (B/P) | Loss (B/P) | Improv. |")
    lines.append("|----|--------|--------|------------|------------------|------------|---------|")
    for h in passed:
        exp = h.experiment
        b, p = exp.baseline, exp.proposed
        lines.append(
            f"| {h.hypothesis_id} | {h.experiment_spec.proposed_method} | "
            f"{exp.status} | {b.get('convergence_rate')} / {p.get('convergence_rate')} | "
            f"{b.get('mean_iterations')} / {p.get('mean_iterations')} | "
            f"{b.get('mean_final_loss')} / {p.get('mean_final_loss')} | "
            f"{format_improvement(exp.improvement_pct)} |"
        )
    lines.append("")
    lines.append("Failed / unsupported experiments (if any):")
    lines.append("")
    for h in passed:
        if h.experiment.status != "SUCCESS":
            lines.append(
                f"- **{h.hypothesis_id}** {h.experiment.status}: "
                f"{h.experiment.error or h.experiment.message}"
            )
    lines.append("")

    lines.append("## 10. Comparative Analysis")
    lines.append("")
    improved = [
        h for h in passed
        if h.experiment.improvement_pct is not None and h.experiment.improvement_pct > 0
    ]
    worsened = [
        h for h in passed
        if h.experiment.improvement_pct is not None and h.experiment.improvement_pct < 0
    ]
    if improved:
        lines.append("Hypotheses supported by measured data:")
        for h in improved:
            lines.append(
                f"- **{h.hypothesis_id}** ({h.experiment_spec.proposed_method}): "
                f"{format_improvement(h.experiment.improvement_pct)} mean iterations"
            )
    else:
        lines.append("_No hypothesis showed a measured improvement on this problem._")
    lines.append("")
    if worsened:
        lines.append("Hypotheses contradicted by measured data:")
        for h in worsened:
            lines.append(
                f"- **{h.hypothesis_id}** ({h.experiment_spec.proposed_method}): "
                f"{format_improvement(h.experiment.improvement_pct)} mean iterations"
            )
        lines.append("")

    lines.append("## 11. Deep Research Analysis")
    lines.append("")
    for h in passed:
        if h.experiment.status != "SUCCESS":
            continue
        lines.extend([
            f"### {h.hypothesis_id} analysis",
            "",
            f"- **Interpretation**: {h.analysis.interpretation}",
            f"- **Strengths**: {', '.join(h.analysis.strengths) if h.analysis.strengths else 'none'}",
            f"- **Weaknesses**: {', '.join(h.analysis.weaknesses) if h.analysis.weaknesses else 'none'}",
            f"- **Limitations**: {', '.join(h.analysis.limitations) if h.analysis.limitations else 'none'}",
            f"- **Improvements suggested**: {', '.join(h.analysis.improvements) if h.analysis.improvements else 'none'}",
            f"- **Evidence strength**: {h.analysis.evidence_strength}",
            f"- **Conclusion**: {h.analysis.conclusion}",
            "",
        ])

    lines.append("## 12. Refined Hypotheses")
    lines.append("")
    refined = [r for h in run.hypotheses for r in h.refinements]
    if refined:
        for r in refined:
            lines.extend([
                f"### {r['refined_id']} (refinement of {r['hypothesis_id']})",
                "",
                f"- **Title**: {r.get('title', '')}",
                f"- **Description**: {r.get('description', '')}",
                f"- **Mechanism**: {r.get('mechanism', '')}",
                f"- **Change summary**: {r.get('change_summary', '')}",
                f"- **Expected improvement**: {r.get('expected_improvement', '')}",
                f"- **Source**: {r.get('source', '')}",
                "",
            ])
    else:
        lines.extend(["_No hypotheses were refined._", ""])

    lines.append("## 13. Final Findings")
    lines.append("")
    lines.append(findings.get("summary", "_No summary generated._"))
    lines.append("")
    lines.append("## 14. Limitations")
    lines.append("")
    for lim in findings.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")
    lines.append("## 15. Future Research")
    lines.append("")
    for fut in findings.get("future_research", []):
        lines.append(f"- {fut}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated automatically by the AARL research laboratory. Every "
                 "experimental number traces back to the stored experiment records "
                 "in this run directory.*")
    lines.append("")
    return "\n".join(lines)
# ---------------------------------------------------------------------------
# Visualisation (lightweight, headless-safe)
# ---------------------------------------------------------------------------
def generate_plots(run: ResearchRun, plots_dir: str) -> List[str]:
    """Generate simple, scientifically meaningful plots. Returns plot file paths."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    created: List[str] = []
    success = [h for h in run.hypotheses if h.passed and h.experiment.status == "SUCCESS"]

    if not success:
        return created

    # ---- 1) Loss-vs-iteration curve per successful hypothesis -------------
    n = len(success)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, h in enumerate(success):
        ax = axes[i]
        b = h.experiment.baseline_loss_curve
        p = h.experiment.proposed_loss_curve
        if b:
            ax.plot(range(len(b)), b, label="Baseline SGD", color="#1f77b4")
        if p:
            ax.plot(range(len(p)), p, label=h.experiment_spec.proposed_method, color="#d62728")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss (log)")
        ax.set_title(f"{h.hypothesis_id}: {h.title[:40]}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "loss_convergence.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    created.append(p1)

    # ---- 2) Improvement bar chart ---------------------------------------------
    labels = []
    vals = []
    for h in success:
        labels.append(h.hypothesis_id)
        vals.append(h.experiment.improvement_pct if h.experiment.improvement_pct is not None else 0.0)
    if success:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
        ax.bar(labels, vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Improvement in iterations (%)")
        ax.set_title("Baseline vs Proposed — mean-iteration improvement")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plots_dir, "improvement.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        created.append(path)

    return created