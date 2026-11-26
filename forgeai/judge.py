"""Stage 5: Hybrid Deterministic + Bias-Defended LLM Judge."""

from __future__ import annotations

from difflib import SequenceMatcher
import math
import re
from typing import Iterable, Optional

from forgeai.evalgen import normalize_fact_text
from forgeai.llm import LLMClient
from forgeai.prompt_loader import render_prompt
from forgeai.schemas import (
    AtomicClaimsOutput,
    ClaimEvaluation,
    ClaimSupportVerdict,
    EvalCase,
    JudgeResult,
    RunnerCaseResult,
    SplitMetrics,
)

_REFUSAL_PHRASES = (
    "i cannot answer",
    "i can't answer",
    "outside my scope",
    "outside the scope",
    "documents do not cover",
    "not mentioned in the",
    "not covered in the",
    "cannot provide",
    "unable to answer",
    "no information available",
)


def _fuzzy_ratio(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(a, b))
    except ImportError:
        a_n = a.strip().lower()
        b_n = b.strip().lower()
        if not a_n or not b_n:
            return 0.0
        if a_n in b_n:
            return 100.0
        return SequenceMatcher(None, a_n, b_n).ratio() * 100.0


def check_status_correct(case: EvalCase, result: RunnerCaseResult) -> bool:
    """Deterministic status check plus refusal-phrase cross-check."""
    actual = result.response.status
    expected = case.expected_status
    ans_low = result.response.answer.lower()
    looks_like_refusal = any(p in ans_low for p in _REFUSAL_PHRASES)

    if case.subtype == "ambiguous" and expected == "needs_clarification":
        # Plan allows needs_clarification OR an answer covering the cases if not refused
        if actual == "needs_clarification":
            return True
        if actual == "answered" and not looks_like_refusal:
            return True
        return False

    if actual != expected:
        return False

    # Cross-check: if model marked status='answered' while actually refusing in prose
    if expected == "answered" and actual == "answered" and looks_like_refusal and len(ans_low) < 180:
        return False
    return True


def check_retrieval_hit(case: EvalCase, result: RunnerCaseResult) -> bool:
    """Check retrieval hit via verbatim evidence/key-fact text OR chunk/doc match."""
    if case.expected_status != "answered":
        return True
    ctx_norm = normalize_fact_text(result.retrieved_context)
    if not ctx_norm:
        return False

    # Prefer key-fact / verbatim evidence matching so re-chunking never invalidates evals
    if case.key_facts:
        for fact in case.key_facts:
            nf = normalize_fact_text(fact)
            if nf and nf in ctx_norm:
                return True

    if any(cid in result.retrieved_chunk_ids for cid in case.gold_chunk_ids):
        return True
    if case.expected_sources and any(
        src in result.retrieved_docs for src in case.expected_sources
    ):
        return True
    return False


def compute_fact_recall(key_facts: list[str], answer: str) -> float:
    """Fraction of key facts present in answer via normalized match or fuzzy fallback."""
    if not key_facts:
        return 1.0
    norm_ans = normalize_fact_text(answer)
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", norm_ans))
    hits = 0
    for fact in key_facts:
        nf = normalize_fact_text(fact)
        if not nf:
            continue
        fact_nums = re.findall(r"\b\d+(?:\.\d+)?\b", nf)
        nums_ok = all(n in ans_nums for n in fact_nums)
        if nf in norm_ans and nums_ok:
            hits += 1
        elif nums_ok and _fuzzy_ratio(nf, norm_ans) >= 85.0:
            hits += 1
    return round(hits / len(key_facts), 4)


def check_citations_valid(result: RunnerCaseResult) -> bool:
    """Every cited quote must exist in retrieved context and reference a retrieved doc."""
    if result.response.status != "answered":
        return True
    if not result.response.sources:
        # Allow tool-backed answers (e.g. calculator/weather/date) if tools were called
        return bool(result.tool_calls_made)

    ctx_norm = re.sub(r"\s+", " ", result.retrieved_context.lower())
    retrieved_docs_lower = {d.lower() for d in result.retrieved_docs}
    for src in result.response.sources:
        doc_ok = (
            src.document.lower() in retrieved_docs_lower
            or src.document.lower().startswith("tool:")
            or any(src.document.lower() in d for d in retrieved_docs_lower)
        )
        if not doc_ok:
            return False
        quote_norm = re.sub(r"\s+", " ", src.quote.strip().lower())
        if len(quote_norm) < 4:
            return False
        if quote_norm not in ctx_norm and _fuzzy_ratio(quote_norm, ctx_norm) < 82.0:
            return False
    return True


def _verify_quote_in_context(quote: str, context: str) -> bool:
    q = re.sub(r"\s+", " ", quote.strip().lower())
    c = re.sub(r"\s+", " ", context.lower())
    if len(q) < 6:
        return False
    return (q in c) or (_fuzzy_ratio(q, c) >= 85.0)


def evaluate_groundedness(
    answer: str, retrieved_context: str, llm: LLMClient
) -> tuple[bool, list[str], list[ClaimEvaluation]]:
    """Extract atomic claims and verify each against retrieved_context with verbatim guard."""
    if not answer.strip():
        return False, ["Empty answer"], []
    if not retrieved_context.strip():
        return False, ["No retrieved context available to support claims"], []

    extract_prompt = render_prompt(
        "judge_grounded",
        mode="extract_claims",
        answer=answer,
        context=retrieved_context,
        claim="",
    )
    try:
        claims_out = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Extract concise atomic factual claims from the answer. Return JSON.",
                },
                {"role": "user", "content": extract_prompt},
            ],
            model_role="judge",
            json_schema=AtomicClaimsOutput,
            temperature=0.0,
        )
        assert isinstance(claims_out, AtomicClaimsOutput)
        claims = [c.strip() for c in claims_out.claims if c.strip()][:8]
    except Exception:
        claims = [answer.strip()]

    if not claims:
        return True, [], []

    evaluations: list[ClaimEvaluation] = []
    unsupported: list[str] = []

    for claim in claims:
        verdict_prompt = render_prompt(
            "judge_grounded",
            mode="verify_claim",
            answer=answer,
            context=retrieved_context,
            claim=claim,
        )
        try:
            verdict = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Check ONLY whether the claim is explicitly supported by the context. "
                            "Longer answers are NOT better. Return JSON."
                        ),
                    },
                    {"role": "user", "content": verdict_prompt},
                ],
                model_role="judge",
                json_schema=ClaimSupportVerdict,
                temperature=0.0,
            )
            assert isinstance(verdict, ClaimSupportVerdict)
        except Exception:
            evaluations.append(
                ClaimEvaluation(
                    claim=claim,
                    llm_verdict="NO",
                    supporting_sentence="",
                    verified_in_context=False,
                    supported=False,
                )
            )
            unsupported.append(claim)
            continue

        # Anti-hallucination guard: override YES -> NO if supporting sentence isn't in context
        quote_ok = _verify_quote_in_context(verdict.supporting_sentence, retrieved_context)
        is_supported = verdict.verdict == "YES" and quote_ok
        if not is_supported:
            unsupported.append(claim)
        evaluations.append(
            ClaimEvaluation(
                claim=claim,
                llm_verdict=verdict.verdict,
                supporting_sentence=verdict.supporting_sentence,
                verified_in_context=quote_ok,
                supported=is_supported,
            )
        )

    return len(unsupported) == 0, unsupported, evaluations


def judge_case(
    case: EvalCase, result: RunnerCaseResult, llm: LLMClient
) -> JudgeResult:
    """Judge a single eval case: deterministic gates first, LLM groundedness where needed."""
    status_ok = check_status_correct(case, result)
    ret_hit = check_retrieval_hit(case, result)

    if result.response.status == "answered":
        fact_rec = compute_fact_recall(case.key_facts, result.response.answer)
        cit_ok = check_citations_valid(result)
        grounded, unsup_claims, claim_evals = evaluate_groundedness(
            result.response.answer, result.retrieved_context, llm
        )
    else:
        fact_rec = 1.0 if case.expected_status != "answered" else 0.0
        cit_ok = True
        grounded = True
        unsup_claims = []
        claim_evals = []

    if case.expected_status == "answered":
        passed = bool(status_ok and (fact_rec >= 0.8) and grounded and cit_ok)
    else:
        passed = bool(status_ok)

    return JudgeResult(
        case_id=case.id,
        case_type=case.type,
        subtype=case.subtype,
        split=case.split,
        retrieval_hit=ret_hit,
        status_correct=status_ok,
        fact_recall=fact_rec,
        grounded=grounded,
        unsupported_claims=unsup_claims,
        claim_evaluations=claim_evals,
        citation_valid=cit_ok,
        **{"pass": passed},
    )


def judge_all(
    cases: list[EvalCase],
    results: list[RunnerCaseResult],
    llm: LLMClient,
) -> list[JudgeResult]:
    """Judge all cases matched by case_id."""
    res_map = {r.case_id: r for r in results}
    out: list[JudgeResult] = []
    for case in cases:
        res = res_map[case.id]
        out.append(judge_case(case, res, llm))
    return out


def compute_split_metrics(
    cases: list[EvalCase],
    results: list[RunnerCaseResult],
    judgements: list[JudgeResult],
    split: Optional[str] = None,
) -> SplitMetrics:
    """Compute aggregate scorecard metrics for a split ('dev', 'heldout', or all)."""
    case_map = {c.id: c for c in cases}
    res_map = {r.case_id: r for r in results}
    filtered = [
        j for j in judgements if split is None or case_map[j.case_id].split == split
    ]
    total = len(filtered)
    if total == 0:
        return SplitMetrics()

    passed = sum(1 for j in filtered if j.passed)

    def _rate(items: Iterable[JudgeResult], pred) -> float:
        lst = list(items)
        return round(sum(1 for x in lst if pred(x)) / len(lst), 4) if lst else 0.0

    ans_items = [j for j in filtered if case_map[j.case_id].type == "answerable"]
    oos_items = [j for j in filtered if case_map[j.case_id].type == "out_of_scope"]
    edge_items = [j for j in filtered if case_map[j.case_id].type == "edge"]

    should_answer = [
        j for j in filtered if case_map[j.case_id].expected_status == "answered"
    ]
    should_refuse = [
        j for j in filtered if case_map[j.case_id].expected_status == "refused"
    ]

    correct_refusals = sum(
        1
        for j in should_refuse
        if res_map[j.case_id].response.status == "refused"
    )
    wrongful_refusals = sum(
        1
        for j in should_answer
        if res_map[j.case_id].response.status == "refused"
    )

    return SplitMetrics(
        total=total,
        passed=passed,
        pass_rate=round(passed / total, 4),
        answerable_pass_rate=_rate(ans_items, lambda j: j.passed),
        out_of_scope_pass_rate=_rate(oos_items, lambda j: j.passed),
        edge_pass_rate=_rate(edge_items, lambda j: j.passed),
        retrieval_hit_rate=_rate(should_answer, lambda j: j.retrieval_hit),
        groundedness_rate=_rate(should_answer, lambda j: j.grounded),
        correct_refusal_rate=round(correct_refusals / len(should_refuse), 4)
        if should_refuse
        else 1.0,
        wrongful_refusal_rate=round(wrongful_refusals / len(should_answer), 4)
        if should_answer
        else 0.0,
    )


def wilson_confidence_interval(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """95% Wilson score confidence interval for small-sample pass rates."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (
        z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    ) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def calibrate_judge_agreement(
    labeled_samples: list[tuple[str, str, bool]], llm: LLMClient
) -> dict[str, float | int]:
    """Run judge groundedness over hand-labeled (answer, context, human_grounded) tuples."""
    if not labeled_samples:
        return {"samples": 0, "agreements": 0, "agreement_rate": 0.0}
    agreements = 0
    for answer, context, human_grounded in labeled_samples:
        pred_grounded, _, _ = evaluate_groundedness(answer, context, llm)
        if pred_grounded == human_grounded:
            agreements += 1
    return {
        "samples": len(labeled_samples),
        "agreements": agreements,
        "agreement_rate": round(agreements / len(labeled_samples), 4),
    }
