"""Stage 6 Optimizer: Targeted JSON patch operators on agent.yaml (never full regeneration)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from forgeai.llm import LLMClient
from forgeai.prompt_loader import render_prompt
from forgeai.retrieval import HybridIndex
from forgeai.schemas import (
    AgentDefinition,
    EvalCase,
    JudgeResult,
    PatchOperation,
    PromptPatchProposal,
    RunnerCaseResult,
)


def apply_patch(agent: AgentDefinition, patch: PatchOperation) -> AgentDefinition:
    """Apply a dotted-path JSON patch to a copy of AgentDefinition."""
    data = agent.model_dump(mode="json")
    parts = patch.path.split(".")
    target: Any = data
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = deepcopy(patch.new)
    data["version"] = int(data.get("version", 0)) + 1
    return AgentDefinition.model_validate(data)


def _measure_retrieval_hit_rate(
    cases: list[EvalCase], index: HybridIndex, agent: AgentDefinition
) -> float:
    ans_cases = [c for c in cases if c.expected_status == "answered"]
    if not ans_cases:
        return 1.0
    hits = 0
    for c in ans_cases:
        retrieved = index.retrieve_with_config(c.question, agent.retrieval)
        blob = " ".join(r.chunk.text.lower() for r in retrieved)
        if any(f.lower() in blob for f in c.key_facts) or any(
            r.chunk.doc in c.expected_sources for r in retrieved
        ):
            hits += 1
    return hits / len(ans_cases)


def propose_patch_for_category(
    category: str,
    agent: AgentDefinition,
    failing_cases: list[EvalCase],
    failing_results: list[RunnerCaseResult],
    failing_judgements: list[JudgeResult],
    index: HybridIndex,
    llm: LLMClient,
    tried_operators: set[str],
    max_prompt_lines: int = 12,
) -> Optional[tuple[str, PatchOperation, str]]:
    """Select an untried targeted patch operator for the dominant failure category."""
    if category == "RETRIEVAL_MISS":
        candidates: list[tuple[str, PatchOperation, str]] = []
        if "ret_raise_topk" not in tried_operators and agent.retrieval.top_k < 8:
            new_k = min(8, agent.retrieval.top_k + 2)
            candidates.append(
                (
                    "ret_raise_topk",
                    PatchOperation(
                        path="retrieval.top_k", old=agent.retrieval.top_k, new=new_k
                    ),
                    f"Raise retrieval.top_k from {agent.retrieval.top_k} to {new_k} to capture missed passages.",
                )
            )
        if "ret_lower_min_score" not in tried_operators and agent.retrieval.min_score > 0.08:
            new_ms = round(max(0.05, agent.retrieval.min_score - 0.10), 2)
            candidates.append(
                (
                    "ret_lower_min_score",
                    PatchOperation(
                        path="retrieval.min_score",
                        old=agent.retrieval.min_score,
                        new=new_ms,
                    ),
                    f"Lower retrieval.min_score from {agent.retrieval.min_score} to {new_ms} so borderline chunks pass threshold.",
                )
            )
        if "ret_shift_alpha_bm25" not in tried_operators:
            new_alpha = 0.30 if agent.retrieval.hybrid_alpha >= 0.45 else 0.70
            candidates.append(
                (
                    "ret_shift_alpha_bm25",
                    PatchOperation(
                        path="retrieval.hybrid_alpha",
                        old=agent.retrieval.hybrid_alpha,
                        new=new_alpha,
                    ),
                    f"Shift retrieval.hybrid_alpha from {agent.retrieval.hybrid_alpha} to {new_alpha} to rebalance BM25 vs dense.",
                )
            )
        if "ret_query_rewrite" not in tried_operators and not agent.retrieval.query_rewrite:
            candidates.append(
                (
                    "ret_query_rewrite",
                    PatchOperation(path="retrieval.query_rewrite", old=False, new=True),
                    "Enable query_rewrite to improve lexical/semantic alignment during retrieval.",
                )
            )
        # Zero-LLM verification: pick the parameter change that maximizes retrieval hit rate
        if candidates:
            best = candidates[0]
            best_hit = -1.0
            for op_name, patch, rationale in candidates:
                cand_agent = apply_patch(agent, patch)
                hit_rate = _measure_retrieval_hit_rate(failing_cases, index, cand_agent)
                if hit_rate > best_hit:
                    best_hit = hit_rate
                    best = (op_name, patch, rationale)
            return best

    if category == "HALLUCINATION":
        if "hal_strict_prompt" not in tried_operators:
            addition = (
                "\n- STRICT GROUNDING GUARD: Never infer, extrapolate, or use outside knowledge. "
                "If any detail or topic is missing from the retrieved context, return status='refused' "
                "and state that the documents do not cover it. Every factual sentence must cite an exact quote in sources."
            )
            marker = "## Grounding Rules"
            old_prompt = agent.system_prompt
            if marker in old_prompt:
                new_prompt = old_prompt.replace(marker, marker + addition, 1)
            else:
                new_prompt = old_prompt + "\n" + addition
            return (
                "hal_strict_prompt",
                PatchOperation(path="system_prompt", old=old_prompt, new=new_prompt),
                "Inject stricter grounding & refusal guard into Grounding Rules section.",
            )
        if "hal_raise_min_score" not in tried_operators and agent.retrieval.min_score < 0.35:
            new_ms = round(min(0.40, agent.retrieval.min_score + 0.08), 2)
            return (
                "hal_raise_min_score",
                PatchOperation(
                    path="retrieval.min_score",
                    old=agent.retrieval.min_score,
                    new=new_ms,
                ),
                f"Raise retrieval.min_score to {new_ms} to suppress noisy low-relevance chunks.",
            )
        if "hal_zero_temp" not in tried_operators and agent.model.temperature > 0.0:
            return (
                "hal_zero_temp",
                PatchOperation(
                    path="model.temperature", old=agent.model.temperature, new=0.0
                ),
                "Lower agent temperature to 0.0 to reduce speculative completions.",
            )

    if category == "WRONG_REFUSAL":
        if "wr_loosen_prompt" not in tried_operators:
            addition = (
                "\n- IN-SCOPE RULE: Only refuse when the question is clearly out-of-scope or unsupported. "
                "If the retrieved passages contain relevant facts or partial policy rules, answer using those verbatim facts "
                "and note any caveats instead of refusing."
            )
            marker = "## Refusal Rules"
            old_prompt = agent.system_prompt
            if marker in old_prompt:
                new_prompt = old_prompt.replace(marker, marker + addition, 1)
            else:
                new_prompt = old_prompt + "\n" + addition
            return (
                "wr_loosen_prompt",
                PatchOperation(path="system_prompt", old=old_prompt, new=new_prompt),
                "Clarify Refusal Rules so supported borderline questions are answered from context.",
            )
        if "wr_lower_min_score" not in tried_operators and agent.retrieval.min_score > 0.08:
            new_ms = round(max(0.05, agent.retrieval.min_score - 0.08), 2)
            return (
                "wr_lower_min_score",
                PatchOperation(
                    path="retrieval.min_score",
                    old=agent.retrieval.min_score,
                    new=new_ms,
                ),
                f"Lower retrieval.min_score to {new_ms} so relevant context reaches the agent.",
            )

    # PROMPT_DEFECT or fallback: ask factory model for minimal diff to one section
    if "prompt_targeted_diff" not in tried_operators:
        examples_text = []
        res_map = {r.case_id: r for r in failing_results}
        jud_map = {j.case_id: j for j in failing_judgements}
        for c in failing_cases[:3]:
            r = res_map.get(c.id)
            j = jud_map.get(c.id)
            examples_text.append(
                f"- Question: {c.question}\n"
                f"  Expected status: {c.expected_status}, Key facts: {c.key_facts}\n"
                f"  Agent status: {r.response.status if r else '?'}, Answer: {(r.response.answer[:200] if r else '')}\n"
                f"  Failure: {j.root_cause if j else ''}"
            )
        prompt = render_prompt(
            "patch_prompt",
            system_prompt=agent.system_prompt,
            failure_category=category,
            failing_examples="\n".join(examples_text),
        )
        try:
            proposal = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Propose a minimal, targeted edit to ONE section of the system prompt. Return JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model_role="factory",
                json_schema=PromptPatchProposal,
                temperature=0.0,
            )
            assert isinstance(proposal, PromptPatchProposal)
            old_snip = proposal.old_snippet.strip()
            new_snip = proposal.new_snippet.strip()
            if (
                old_snip
                and old_snip in agent.system_prompt
                and len(new_snip.splitlines()) <= max_prompt_lines
                and old_snip != new_snip
            ):
                new_prompt = agent.system_prompt.replace(old_snip, new_snip, 1)
                return (
                    "prompt_targeted_diff",
                    PatchOperation(
                        path="system_prompt", old=agent.system_prompt, new=new_prompt
                    ),
                    proposal.rationale or f"Minimal edit to {proposal.section_name}.",
                )
        except Exception:
            pass

    return None
