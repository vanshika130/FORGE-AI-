"""Stage 6 Critic: Deterministic rule-based failure classifier with severity ranking."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from forgeai.schemas import EvalCase, JudgeResult, RunnerCaseResult

SEVERITY_ORDER = {
    "HALLUCINATION": 4,
    "RETRIEVAL_MISS": 3,
    "WRONG_REFUSAL": 2,
    "PROMPT_DEFECT": 1,
}


def classify_failure(
    case: EvalCase, result: RunnerCaseResult, judgement: JudgeResult
) -> tuple[str, str]:
    """Classify a failing case deterministically and emit a root-cause sentence."""
    if judgement.passed:
        return "", ""

    actual_status = result.response.status
    expected = case.expected_status

    if expected == "answered" and not judgement.retrieval_hit:
        return (
            "RETRIEVAL_MISS",
            f"Expected facts from {case.expected_sources or ['docs']} were not retrieved in top_k={len(result.retrieved_chunk_ids)} chunks.",
        )

    if expected == "answered" and actual_status == "refused":
        return (
            "WRONG_REFUSAL",
            "Relevant context was retrieved, but the agent refused an in-scope answerable question.",
        )

    if expected == "refused" and actual_status == "answered":
        return (
            "HALLUCINATION",
            f"Missed refusal on {case.subtype} out-of-scope question; agent answered instead of refusing.",
        )

    if not judgement.grounded:
        claims_str = "; ".join(judgement.unsupported_claims[:2]) or "unverified claim"
        return (
            "HALLUCINATION",
            f"Answer contained ungrounded claim(s) not supported verbatim in context: {claims_str}.",
        )

    if judgement.fact_recall < 0.8:
        return (
            "PROMPT_DEFECT",
            f"Retrieval hit succeeded and claims were grounded, but fact_recall was {judgement.fact_recall:.2f} (< 0.80).",
        )

    if not judgement.citation_valid:
        return (
            "PROMPT_DEFECT",
            "Answer omitted valid verbatim document citations matching the retrieved context.",
        )

    return (
        "PROMPT_DEFECT",
        f"Status or output contract mismatch (expected {expected}, got {actual_status}).",
    )


def classify_all_failures(
    cases: list[EvalCase],
    results: list[RunnerCaseResult],
    judgements: list[JudgeResult],
) -> tuple[list[JudgeResult], dict[str, int], Optional[str]]:
    """Populate failure_category and root_cause on JudgeResults and pick dominant category."""
    case_map = {c.id: c for c in cases}
    res_map = {r.case_id: r for r in results}
    counts: Counter[str] = Counter()

    for j in judgements:
        if j.passed:
            j.failure_category = ""
            j.root_cause = ""
            continue
        cat, cause = classify_failure(case_map[j.case_id], res_map[j.case_id], j)
        j.failure_category = cat
        j.root_cause = cause
        if cat:
            counts[cat] += 1

    if not counts:
        return judgements, {}, None

    dominant = max(
        counts.keys(),
        key=lambda cat: (counts[cat], SEVERITY_ORDER.get(cat, 0)),
    )
    return judgements, dict(counts), dominant
