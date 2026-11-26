"""Stage 7: Honest Scorecard and Final Report Generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from forgeai.judge import wilson_confidence_interval
from forgeai.schemas import (
    AgentDefinition,
    EvalCase,
    GroundabilityReport,
    IterationLogEntry,
    JudgeResult,
    RequirementsDoc,
)


def render_scorecard_md(scorecard_rows: list[dict[str, Any]]) -> str:
    """Render 07_scorecard.md comparing dev and held-out metrics per iteration."""
    lines = [
        "# ForgeAI Evaluation Scorecard",
        "",
        "| Iter | Split | Total | Pass Rate | Answerable | Out-of-Scope | Edge | Retrieval Hit | Grounded | Correct Refusal | Wrongful Refusal |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scorecard_rows:
        it = row["iteration"]
        for split_name in ("dev", "heldout"):
            m = row.get(split_name, {})
            if not m or m.get("total", 0) == 0:
                continue
            lines.append(
                f"| {it} | {split_name} | {m['total']} | {m['pass_rate']:.1%} | "
                f"{m['answerable_pass_rate']:.1%} | {m['out_of_scope_pass_rate']:.1%} | "
                f"{m['edge_pass_rate']:.1%} | {m['retrieval_hit_rate']:.1%} | "
                f"{m['groundedness_rate']:.1%} | {m['correct_refusal_rate']:.1%} | "
                f"{m['wrongful_refusal_rate']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def generate_final_report(
    brief: str,
    requirements: RequirementsDoc,
    groundability: GroundabilityReport,
    final_agent: AgentDefinition,
    eval_cases: list[EvalCase],
    final_judgements: list[JudgeResult],
    iteration_logs: list[IterationLogEntry],
    scorecard_rows: list[dict[str, Any]],
    sanity_baselines: dict[str, dict[str, float]] | None = None,
    calibration_stats: dict[str, Any] | None = None,
    total_tokens: int = 0,
    total_latency_s: float = 0.0,
    run_dir: Optional[Path] = None,
) -> tuple[str, str]:
    """Generate 07_scorecard.md and 08_final_report.md with full candour."""
    scorecard_md = render_scorecard_md(scorecard_rows)

    latest = scorecard_rows[-1] if scorecard_rows else {}
    dev_m = latest.get("dev", {})
    held_m = latest.get("heldout", {})
    dev_rate = float(dev_m.get("pass_rate", 0.0))
    held_rate = float(held_m.get("pass_rate", 0.0))
    held_ci = wilson_confidence_interval(
        int(held_m.get("passed", 0)), int(held_m.get("total", 0))
    )
    dev_ci = wilson_confidence_interval(
        int(dev_m.get("passed", 0)), int(dev_m.get("total", 0))
    )

    lines: list[str] = [
        f"# ForgeAI Final Report: `{final_agent.name}`",
        "",
        "## 1. Brief, Requirements, and Groundability Flags",
        f"**Brief:** {brief.strip()}",
        "",
        f"- **Scope:** {requirements.scope}",
        f"- **Audience:** {requirements.audience}",
        f"- **Tone:** {requirements.tone}",
        f"- **Must answer:** {', '.join(requirements.must_answer) or 'None'}",
        f"- **Must refuse:** {', '.join(requirements.must_refuse) or 'None'}",
        f"- **Derived tool flags:** `needs_tabular_lookup={requirements.needs_tabular_lookup}`, "
        f"`needs_numeric={requirements.needs_numeric}`, "
        f"`needs_external_data={requirements.needs_external_data}`, "
        f"`needs_current_date={requirements.needs_current_date}`",
        "",
        "### Groundability Assessment",
        f"- Supported: {groundability.supported_count} | Partial: {groundability.partial_count} | "
        f"Unsupported: {groundability.unsupported_count} ({groundability.unsupported_fraction:.1%})",
        "",
    ]
    for flag in groundability.claims:
        lines.append(
            f"- **[{flag.status.upper()}]** `{flag.ask}` "
            f"(dense={flag.max_similarity:.2f}, bm25={flag.max_bm25:.2f}) -> {flag.recommended_action}"
        )

    lines.extend(
        [
            "",
            "## 2. Final Agent Definition & Tool Choices",
            f"- **Version:** `v{final_agent.version}`",
            f"- **Model:** role=`{final_agent.model.role}`, temp=`{final_agent.model.temperature}`, max_tokens=`{final_agent.model.max_tokens}`",
            f"- **Retrieval:** `chunk_size={final_agent.retrieval.chunk_size}`, "
            f"`chunk_overlap={final_agent.retrieval.chunk_overlap}`, "
            f"`top_k={final_agent.retrieval.top_k}`, "
            f"`hybrid_alpha={final_agent.retrieval.hybrid_alpha}`, "
            f"`min_score={final_agent.retrieval.min_score}`, "
            f"`query_rewrite={final_agent.retrieval.query_rewrite}`",
            f"- **Source Documents:** {', '.join(final_agent.source_documents)}",
            f"- **Attached Tools:** {', '.join(final_agent.tools) if final_agent.tools else 'None (zero tools attached)'}",
        ]
    )
    for tname, just in final_agent.tool_justifications.items():
        lines.append(f"  - `{tname}` justification: \"{just}\"")
    if final_agent.removed_tools:
        lines.append("- **Tools Rejected or Removed by Validation/Ablation:**")
        for tname, reason in final_agent.removed_tools.items():
            lines.append(f"  - `{tname}`: {reason}")

    lines.extend(["", "## 3. Scorecard per Iteration", "", scorecard_md.strip()])

    if held_m.get("total", 0) > 0 and (dev_rate - held_rate) > 0.10:
        lines.append(
            f"\n> **Honesty Flag (Overfitting Warning):** Held-out pass rate ({held_rate:.1%}) "
            f"is more than 10 percentage points lower than dev pass rate ({dev_rate:.1%}), "
            "indicating partial overfitting of patches to the dev split."
        )

    lines.extend(["", "## 4. Iteration Log Summary", ""])
    for entry in iteration_logs:
        status_str = "ACCEPTED" if entry.accepted else "REVERTED"
        patch_str = (
            f"`{entry.patch.path}`" if entry.patch is not None else "baseline"
        )
        lines.append(
            f"- **Iter {entry.iteration} [{status_str}]** (`{entry.operator_name}`, {patch_str}): "
            f"{entry.rationale} | dev: {entry.dev_score_before:.1%} -> {entry.dev_score_after:.1%}, "
            f"heldout: {entry.heldout_score:.1%}, tokens: {entry.tokens}, latency: {entry.latency:.2f}s"
        )
        if entry.rejection_reason:
            lines.append(f"  - Rejection reason: {entry.rejection_reason}")

    lines.extend(["", "## 5. What Still Fails and Why", ""])
    case_map = {c.id: c for c in eval_cases}
    remaining_failures = [j for j in final_judgements if not j.passed]
    if not remaining_failures:
        lines.append(
            "No eval cases failed in the final run; however, note the small eval sample size and LLM-generated cases below."
        )
    else:
        for j in remaining_failures:
            c = case_map.get(j.case_id)
            q_preview = c.question if c else j.case_id
            lines.append(
                f"- **`{j.case_id}` ({j.split}, {j.case_type}/{j.subtype}) [{j.failure_category}]**: "
                f"\"{q_preview}\" — {j.root_cause}"
            )

    lines.extend(
        [
            "",
            "## 6. Known Limitations of the Evaluation",
            f"- **Synthetic / LLM-generated eval set:** {len(eval_cases)} cases generated from the source documents; may miss phrasing variations or complex multi-turn user intents.",
            f"- **Small sample size:** Dev pass rate {dev_rate:.1%} (95% Wilson CI: [{dev_ci[0]:.1%}, {dev_ci[1]:.1%}]); "
            f"Held-out pass rate {held_rate:.1%} (95% Wilson CI: [{held_ci[0]:.1%}, {held_ci[1]:.1%}]).",
            "- **Generator / judge leakage risk:** Although the judge uses a different model family than the agent, factory and judge share the Gemini family unless overridden.",
            f"- **Total run cost:** {total_tokens} tokens, {total_latency_s:.1f}s LLM latency.",
        ]
    )
    if calibration_stats:
        lines.append(
            f"- **Judge-versus-human calibration agreement:** {calibration_stats.get('agreement_rate', 0):.1%} "
            f"({calibration_stats.get('agreements', 0)}/{calibration_stats.get('samples', 0)} hand-labeled outputs)."
        )
    else:
        lines.append(
            "- **Judge-versus-human calibration agreement:** Not yet measured on this run (run `forge calibrate` with 15 hand-labeled samples)."
        )

    if sanity_baselines:
        lines.extend(["", "### Sanity Baseline Discrimination Scores", ""])
        lines.append("| Baseline | Overall Pass | Answerable | Out-of-Scope | Groundedness |")
        lines.append("|:---|---:|---:|---:|---:|")
        for bname, bm in sanity_baselines.items():
            lines.append(
                f"| `{bname}` | {bm['pass_rate']:.1%} | {bm['answerable_pass_rate']:.1%} | "
                f"{bm['out_of_scope_pass_rate']:.1%} | {bm['groundedness_rate']:.1%} |"
            )

    lines.extend(
        [
            "",
            "## 7. Do Not Use This Agent For",
            "- Decisions with legal, medical, financial, or safety-critical liability without human verification.",
            "- Questions outside the provided source documents or topics explicitly moved to `must_refuse` by the groundability guard.",
            "- Real-time external facts not covered by an attached tool.",
        ]
    )
    for flag in groundability.claims:
        if flag.status == "unsupported":
            lines.append(f"- `{flag.ask}` (unsupported by current document set).")

    report_md = "\n".join(lines) + "\n"

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "07_scorecard.md").write_text(scorecard_md, encoding="utf-8")
        (run_dir / "08_final_report.md").write_text(report_md, encoding="utf-8")

    return scorecard_md, report_md
