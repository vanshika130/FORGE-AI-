"""Stage 6 Orchestration: Critic-Optimizer Loop with revert-on-regression and tool ablation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Optional

import yaml

from forgeai.critic import classify_all_failures
from forgeai.ingest import ingest_documents
from forgeai.judge import compute_split_metrics, judge_all
from forgeai.llm import BudgetExceededError, LLMClient
from forgeai.optimizer import apply_patch, propose_patch_for_category
from forgeai.retrieval import EmbeddingBackend, HybridIndex
from forgeai.runner import AgentRunner
from forgeai.schemas import (
    AgentDefinition,
    ChunkRecord,
    EvalCase,
    EvalSet,
    IterationLogEntry,
    JudgeResult,
    PatchOperation,
    RunnerCaseResult,
)


def _evaluate_split(
    agent: AgentDefinition,
    cases: list[EvalCase],
    docs_dir: Path,
    llm: LLMClient,
    embedder: EmbeddingBackend,
    doc_hash: str,
) -> tuple[list[RunnerCaseResult], list[JudgeResult], dict[str, int], Optional[str]]:
    """Ingest/index according to agent's retrieval params, run cases, judge, and classify."""
    _, chunks = ingest_documents(
        docs_dir,
        chunk_size=agent.retrieval.chunk_size,
        chunk_overlap=agent.retrieval.chunk_overlap,
        source_documents=agent.source_documents or None,
    )
    chunk_key = f"{agent.retrieval.chunk_size}_{agent.retrieval.chunk_overlap}"
    index = HybridIndex(
        chunks,
        embedder=embedder,
        doc_hash=doc_hash,
        chunk_params_key=chunk_key,
    )
    runner = AgentRunner(agent=agent, index=index, docs_dir=docs_dir, llm=llm)
    results = [runner.run_case(c) for c in cases]
    judgements = judge_all(cases, results, llm)
    judgements, counts, dominant = classify_all_failures(cases, results, judgements)
    return results, judgements, counts, dominant


def _save_iter_artifacts(
    iter_dir: Path,
    agent: Optional[AgentDefinition],
    patch: Optional[PatchOperation],
    results: list[RunnerCaseResult],
    judgements: list[JudgeResult],
) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    if agent is not None:
        (iter_dir / "agent.yaml").write_text(
            yaml.safe_dump(agent.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    if patch is not None:
        (iter_dir / "patch.json").write_text(
            patch.model_dump_json(indent=2), encoding="utf-8"
        )
    (iter_dir / "results.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in results], indent=2),
        encoding="utf-8",
    )
    (iter_dir / "judge.json").write_text(
        json.dumps([j.model_dump(mode="json", by_alias=True) for j in judgements], indent=2),
        encoding="utf-8",
    )
    failures = [j.model_dump(mode="json", by_alias=True) for j in judgements if not j.passed]
    (iter_dir / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )


def run_tool_ablation(
    agent: AgentDefinition,
    dev_cases: list[EvalCase],
    dev_results: list[RunnerCaseResult],
    dev_score_before: float,
    docs_dir: Path,
    llm: LLMClient,
    embedder: EmbeddingBackend,
    doc_hash: str,
) -> AgentDefinition:
    """Layer 3 check: remove any attached tool that was never called and doesn't hurt score."""
    if not agent.tools:
        return agent

    called_tools: set[str] = set()
    for res in dev_results:
        for tc in res.tool_calls_made:
            called_tools.add(tc.split("(")[0].strip())

    current = agent
    for tool_name in list(agent.tools):
        if tool_name in called_tools:
            continue
        # Test removing unused tool on dev split
        data = current.model_dump(mode="json")
        data["tools"] = [t for t in data["tools"] if t != tool_name]
        data["tool_justifications"].pop(tool_name, None)
        data["removed_tools"][tool_name] = (
            "Removed by ablation check: tool was never called and score did not drop without it."
        )
        candidate = AgentDefinition.model_validate(data)
        res_abl, jud_abl, _, _ = _evaluate_split(
            candidate, dev_cases, docs_dir, llm, embedder, doc_hash
        )
        metrics_abl = compute_split_metrics(dev_cases, res_abl, jud_abl, split="dev")
        if metrics_abl.pass_rate >= dev_score_before:
            current = candidate
    return current


def run_optimization_loop(
    initial_agent: AgentDefinition,
    eval_set: EvalSet,
    docs_dir: Path,
    llm: LLMClient,
    embedder: EmbeddingBackend,
    doc_hash: str,
    max_iterations: int = 3,
    max_prompt_lines: int = 12,
    run_dir: Optional[Path] = None,
) -> tuple[
    AgentDefinition,
    list[IterationLogEntry],
    list[dict],
]:
    """Run baseline iter_0 + up to max_iterations targeted patches with revert guard."""
    dev_cases = [c for c in eval_set.cases if c.split == "dev"]
    heldout_cases = [c for c in eval_set.cases if c.split == "heldout"]
    all_cases = eval_set.cases

    current_agent = initial_agent
    iteration_logs: list[IterationLogEntry] = []
    scorecard_rows: list[dict] = []

    # Iteration 0: Baseline evaluation on full eval set
    t0 = time.monotonic()
    tok0 = llm.total_tokens
    res0, jud0, counts0, dominant0 = _evaluate_split(
        current_agent, all_cases, docs_dir, llm, embedder, doc_hash
    )
    dev_m0 = compute_split_metrics(all_cases, res0, jud0, split="dev")
    held_m0 = compute_split_metrics(all_cases, res0, jud0, split="heldout")
    if run_dir is not None:
        _save_iter_artifacts(run_dir / "iter_0", current_agent, None, res0, jud0)

    # Layer 3 tool ablation on baseline dev results
    dev_res0 = [r for r in res0 if any(c.id == r.case_id for c in dev_cases)]
    current_agent = run_tool_ablation(
        current_agent,
        dev_cases,
        dev_res0,
        dev_m0.pass_rate,
        docs_dir,
        llm,
        embedder,
        doc_hash,
    )

    scorecard_rows.append(
        {
            "iteration": 0,
            "dev": dev_m0.model_dump(),
            "heldout": held_m0.model_dump(),
        }
    )

    current_dev_res = [r for r in res0 if any(c.id == r.case_id for c in dev_cases)]
    current_dev_jud = [j for j in jud0 if any(c.id == j.case_id for c in dev_cases)]
    current_dev_metrics = dev_m0
    current_held_metrics = held_m0
    current_counts = counts0
    current_dominant = dominant0
    tried_operators: set[str] = set()

    iteration_logs.append(
        IterationLogEntry(
            iteration=0,
            failures_by_category=counts0,
            patch=None,
            operator_name="baseline",
            rationale="Initial agent_v0 baseline evaluation.",
            failures_targeted=[],
            failures_fixed=[],
            dev_score_before=dev_m0.pass_rate,
            dev_score_after=dev_m0.pass_rate,
            heldout_score=held_m0.pass_rate,
            accepted=True,
            tokens=llm.total_tokens - tok0,
            latency=round(time.monotonic() - t0, 2),
        )
    )

    # Iterations 1..max_iterations
    for it in range(1, max_iterations + 1):
        if current_dev_metrics.pass_rate >= 1.0 or not current_dominant:
            break
        try:
            llm.check_budget()
        except BudgetExceededError:
            break

        it_t0 = time.monotonic()
        it_tok0 = llm.total_tokens

        failing_ids = [j.case_id for j in current_dev_jud if not j.passed]
        targeted_cases = [
            c
            for c in dev_cases
            if any(
                j.case_id == c.id and not j.passed and j.failure_category == current_dominant
                for j in current_dev_jud
            )
        ]
        if not targeted_cases:
            targeted_cases = [c for c in dev_cases if c.id in failing_ids]

        _, chunks_cur = ingest_documents(
            docs_dir,
            chunk_size=current_agent.retrieval.chunk_size,
            chunk_overlap=current_agent.retrieval.chunk_overlap,
            source_documents=current_agent.source_documents or None,
        )
        index_cur = HybridIndex(
            chunks_cur,
            embedder=embedder,
            doc_hash=doc_hash,
            chunk_params_key=f"{current_agent.retrieval.chunk_size}_{current_agent.retrieval.chunk_overlap}",
        )

        proposal = propose_patch_for_category(
            category=current_dominant,
            agent=current_agent,
            failing_cases=targeted_cases,
            failing_results=current_dev_res,
            failing_judgements=current_dev_jud,
            index=index_cur,
            llm=llm,
            tried_operators=tried_operators,
            max_prompt_lines=max_prompt_lines,
        )
        if proposal is None:
            break

        op_name, patch, rationale = proposal
        tried_operators.add(op_name)
        candidate_agent = apply_patch(current_agent, patch)

        # Re-run dev set first
        cand_dev_res, cand_dev_jud, cand_counts, cand_dominant = _evaluate_split(
            candidate_agent, dev_cases, docs_dir, llm, embedder, doc_hash
        )
        cand_dev_metrics = compute_split_metrics(
            dev_cases, cand_dev_res, cand_dev_jud, split="dev"
        )

        # Guard: accept ONLY if dev score improves AND correct refusals don't drop by >1 case
        prev_ref_ok = round(
            current_dev_metrics.correct_refusal_rate
            * sum(1 for c in dev_cases if c.expected_status == "refused")
        )
        cand_ref_ok = round(
            cand_dev_metrics.correct_refusal_rate
            * sum(1 for c in dev_cases if c.expected_status == "refused")
        )
        refusal_regressed_badly = (prev_ref_ok - cand_ref_ok) > 1
        improved = cand_dev_metrics.pass_rate > current_dev_metrics.pass_rate
        accepted = bool(improved and not refusal_regressed_badly)

        rejection_reason = ""
        if not improved:
            rejection_reason = (
                f"Reverted: dev pass_rate did not improve "
                f"({current_dev_metrics.pass_rate:.2%} -> {cand_dev_metrics.pass_rate:.2%})."
            )
        elif refusal_regressed_badly:
            rejection_reason = (
                f"Reverted: correct refusals dropped by {prev_ref_ok - cand_ref_ok} cases."
            )

        if accepted:
            prev_fail_set = {j.case_id for j in current_dev_jud if not j.passed}
            new_pass_set = {j.case_id for j in cand_dev_jud if j.passed}
            fixed_ids = sorted(prev_fail_set & new_pass_set)
            # Also evaluate held-out split to track generalization honestly
            if heldout_cases:
                cand_held_res, cand_held_jud, _, _ = _evaluate_split(
                    candidate_agent, heldout_cases, docs_dir, llm, embedder, doc_hash
                )
                cand_held_metrics = compute_split_metrics(
                    heldout_cases, cand_held_res, cand_held_jud, split="heldout"
                )
            else:
                cand_held_res, cand_held_jud = [], []
                cand_held_metrics = current_held_metrics

            if run_dir is not None:
                _save_iter_artifacts(
                    run_dir / f"iter_{it}",
                    candidate_agent,
                    patch,
                    cand_dev_res + cand_held_res,
                    cand_dev_jud + cand_held_jud,
                )

            current_agent = candidate_agent
            current_dev_res = cand_dev_res
            current_dev_jud = cand_dev_jud
            current_dev_metrics = cand_dev_metrics
            current_held_metrics = cand_held_metrics
            current_counts = cand_counts
            current_dominant = cand_dominant
            scorecard_rows.append(
                {
                    "iteration": it,
                    "dev": current_dev_metrics.model_dump(),
                    "heldout": current_held_metrics.model_dump(),
                }
            )
        else:
            fixed_ids = []
            if run_dir is not None:
                _save_iter_artifacts(
                    run_dir / f"iter_{it}",
                    candidate_agent,
                    patch,
                    cand_dev_res,
                    cand_dev_jud,
                )

        iteration_logs.append(
            IterationLogEntry(
                iteration=it,
                failures_by_category=cand_counts if accepted else current_counts,
                patch=patch,
                operator_name=op_name,
                rationale=rationale,
                failures_targeted=[c.id for c in targeted_cases],
                failures_fixed=fixed_ids,
                dev_score_before=dev_m0.pass_rate
                if it == 1
                else iteration_logs[-1].dev_score_after,
                dev_score_after=cand_dev_metrics.pass_rate if accepted else current_dev_metrics.pass_rate,
                heldout_score=current_held_metrics.pass_rate,
                accepted=accepted,
                rejection_reason=rejection_reason,
                tokens=llm.total_tokens - it_tok0,
                latency=round(time.monotonic() - it_t0, 2),
            )
        )

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "06_iteration_log.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in iteration_logs], indent=2),
            encoding="utf-8",
        )
        (run_dir / "agent_final.yaml").write_text(
            yaml.safe_dump(
                current_agent.model_dump(mode="json"), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )

    return current_agent, iteration_logs, scorecard_rows
