"""Stage 4: Grounded Eval Generator with deterministic quality gates and dev/heldout split."""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Optional

from forgeai.llm import LLMClient
from forgeai.prompt_loader import get_prompt_versions, render_prompt
from forgeai.retrieval import HybridIndex
from forgeai.schemas import (
    AnswerabilityCheckOutput,
    ChunkRecord,
    EvalCase,
    EvalSet,
    GeneratedEvalBatch,
    GeneratedMultiEvalDraft,
    GeneratedSingleEvalDraft,
    RequirementsDoc,
)

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "twenty-one": "21",
    "twenty one": "21",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "ninety": "90",
}


_SORTED_NUMBER_WORDS = sorted(
    _NUMBER_WORDS.items(), key=lambda kv: len(kv[0]), reverse=True
)


def normalize_fact_text(text: str) -> str:
    """Normalize punctuation, whitespace, and basic number words for fact matching."""
    low = text.lower()
    for word, digit in _SORTED_NUMBER_WORDS:
        low = re.sub(rf"\b{re.escape(word)}\b", digit, low)
    low = re.sub(r"[^\w\s.%$€£-]", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def verify_key_facts_in_text(key_facts: list[str], source_text: str) -> bool:
    """Gate 1: Every key fact must appear in the source chunk (numbers matched exactly)."""
    if not key_facts:
        return False
    norm_source = normalize_fact_text(source_text)
    source_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", norm_source))
    for fact in key_facts:
        norm_fact = normalize_fact_text(fact)
        if not norm_fact:
            return False
        fact_nums = re.findall(r"\b\d+(?:\.\d+)?\b", norm_fact)
        if any(num not in source_nums for num in fact_nums):
            return False
        if norm_fact in norm_source:
            continue
        # Allow token containment when all meaningful words + exact numbers match
        tokens = [t for t in norm_fact.split() if len(t) >= 2]
        if not tokens:
            return False
        if not all(tok in norm_source for tok in tokens):
            return False
    return True


def _stratified_sample_chunks(
    chunks: list[ChunkRecord], count: int, rng: random.Random
) -> list[ChunkRecord]:
    by_doc: dict[str, list[ChunkRecord]] = {}
    for ch in chunks:
        # Prefer substantive chunks (>=80 chars)
        if len(ch.text.strip()) >= 80:
            by_doc.setdefault(ch.doc, []).append(ch)
    if not by_doc:
        for ch in chunks:
            by_doc.setdefault(ch.doc, []).append(ch)
    if not by_doc:
        return []

    doc_names = list(by_doc.keys())
    for name in doc_names:
        rng.shuffle(by_doc[name])

    selected: list[ChunkRecord] = []
    idx = 0
    while len(selected) < count:
        progressed = False
        for name in doc_names:
            pool = by_doc[name]
            if idx < len(pool):
                selected.append(pool[idx])
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
        idx += 1
    return selected


def _find_cross_doc_pairs(
    chunks: list[ChunkRecord], index: HybridIndex, count: int
) -> list[tuple[ChunkRecord, ChunkRecord]]:
    docs = sorted({c.doc for c in chunks if not c.is_csv_row})
    if len(docs) < 2:
        return []
    pairs: list[tuple[ChunkRecord, ChunkRecord]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for ch in chunks:
        if ch.is_csv_row or len(ch.text.strip()) < 80:
            continue
        neighbours = index.retrieve(ch.text[:250], top_k=8, alpha=0.7, min_score=0.0)
        for nb in neighbours:
            if nb.chunk.doc != ch.doc and not nb.chunk.is_csv_row:
                key = tuple(sorted((ch.chunk_id, nb.chunk.chunk_id)))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    pairs.append((ch, nb.chunk))
                    break
        if len(pairs) >= count * 2:
            break
    return pairs[: count * 2]


def _dedup_cases(cases: list[EvalCase], index: HybridIndex, threshold: float = 0.90) -> list[EvalCase]:
    kept: list[EvalCase] = []
    for case in cases:
        is_dup = False
        for existing in kept:
            if case.question.strip().lower() == existing.question.strip().lower():
                is_dup = True
                break
            sim = index.cosine_similarity_texts(case.question, existing.question)
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(case)
    return kept


def _assign_stratified_splits(
    cases: list[EvalCase], dev_ratio: float, rng: random.Random
) -> list[EvalCase]:
    by_type: dict[str, list[EvalCase]] = {}
    for c in cases:
        by_type.setdefault(c.type, []).append(c)

    result: list[EvalCase] = []
    for c_type, group in by_type.items():
        rng.shuffle(group)
        n_dev = max(1, round(len(group) * dev_ratio)) if len(group) > 1 else len(group)
        if len(group) >= 3 and n_dev == len(group):
            n_dev = len(group) - 1
        for idx, item in enumerate(group):
            item.split = "dev" if idx < n_dev else "heldout"
            result.append(item)
    return result


def generate_eval_set(
    requirements: RequirementsDoc,
    chunks: list[ChunkRecord],
    index: HybridIndex,
    llm: LLMClient,
    target_total: int = 30,
    dev_ratio: float = 0.7,
    seed: int = 42,
    run_dir: Optional[Path] = None,
) -> EvalSet:
    """Generate a grounded, quality-gated evaluation set split into dev and heldout."""
    rng = random.Random(seed)
    cases: list[EvalCase] = []

    n_single = max(8, round(target_total * 0.40))  # 12 for 30
    n_multi = max(2, round(target_total * 0.13))   # 4
    n_ambig = max(2, round(target_total * 0.10))   # 3
    n_false = max(2, round(target_total * 0.07))   # 2
    n_adj = max(3, round(target_total * 0.13))     # 4
    n_off = max(3, round(target_total * 0.17))     # 5

    # 1. Single-document answerable cases with Gate 1 & Gate 2
    candidate_chunks = _stratified_sample_chunks(chunks, count=n_single * 2, rng=rng)
    single_count = 0
    for ch in candidate_chunks:
        if single_count >= n_single:
            break
        prompt = render_prompt(
            "eval_answerable",
            doc=ch.doc,
            chunk_id=ch.chunk_id,
            chunk_text=ch.text,
        )
        try:
            draft = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Create one grounded question and 1-3 verbatim key facts from the chunk. Return JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model_role="factory",
                json_schema=GeneratedSingleEvalDraft,
                temperature=0.0,
            )
            assert isinstance(draft, GeneratedSingleEvalDraft)
        except Exception:
            continue

        # Gate 1: Key facts must exist in source chunk
        if not verify_key_facts_in_text(draft.key_facts, ch.text):
            continue

        # Gate 2: Answerability check using ONLY the source chunk
        try:
            ans_check = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question using ONLY the provided passage. Return JSON.",
                    },
                    {
                        "role": "user",
                        "content": f"Passage:\n{ch.text}\n\nQuestion: {draft.question}",
                    },
                ],
                model_role="factory",
                json_schema=AnswerabilityCheckOutput,
                temperature=0.0,
            )
            assert isinstance(ans_check, AnswerabilityCheckOutput)
            if not ans_check.answerable or not verify_key_facts_in_text(
                draft.key_facts[:1], ans_check.answer
            ):
                continue
        except Exception:
            continue

        single_count += 1
        cases.append(
            EvalCase(
                id=f"ans_{single_count:03d}",
                type="answerable",
                subtype="single_doc",
                question=draft.question.strip(),
                expected_status="answered",
                expected_sources=[ch.doc],
                key_facts=draft.key_facts[:3],
                gold_chunk_ids=[ch.chunk_id],
                gold_evidence_texts=[ch.text],
            )
        )

    # 2. Multi-document edge cases
    pairs = _find_cross_doc_pairs(chunks, index, count=n_multi)
    multi_count = 0
    for ch_a, ch_b in pairs:
        if multi_count >= n_multi:
            break
        prompt = render_prompt(
            "eval_edge",
            edge_kind="multi_doc",
            context=(
                f"Document A ({ch_a.doc}, {ch_a.chunk_id}):\n{ch_a.text}\n\n"
                f"Document B ({ch_b.doc}, {ch_b.chunk_id}):\n{ch_b.text}"
            ),
            count=1,
            scope=requirements.scope,
        )
        try:
            draft = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Generate one multi-document question requiring facts from BOTH passages. Return JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model_role="factory",
                json_schema=GeneratedMultiEvalDraft,
                temperature=0.0,
            )
            assert isinstance(draft, GeneratedMultiEvalDraft)
        except Exception:
            continue

        combined = f"{ch_a.text}\n{ch_b.text}"
        if not verify_key_facts_in_text(draft.key_facts, combined):
            continue

        multi_count += 1
        cases.append(
            EvalCase(
                id=f"edge_multi_{multi_count:03d}",
                type="edge",
                subtype="multi_doc",
                question=draft.question.strip(),
                expected_status="answered",
                expected_sources=sorted({ch_a.doc, ch_b.doc}),
                key_facts=draft.key_facts[:3],
                gold_chunk_ids=[ch_a.chunk_id, ch_b.chunk_id],
                gold_evidence_texts=[ch_a.text, ch_b.text],
            )
        )

    # 3. Ambiguous & false-premise edge cases
    sample_context = "\n\n".join(
        f"[{c.chunk_id}] {c.text}" for c in _stratified_sample_chunks(chunks, 6, rng)
    )
    for subtype, cnt, default_status in (
        ("ambiguous", n_ambig, "needs_clarification"),
        ("false_premise", n_false, "refused"),
    ):
        prompt = render_prompt(
            "eval_edge",
            edge_kind=subtype,
            context=sample_context,
            count=cnt,
            scope=requirements.scope,
        )
        try:
            batch = llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate {cnt} {subtype} edge evaluation questions. Return JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model_role="factory",
                json_schema=GeneratedEvalBatch,
                temperature=0.2,
            )
            assert isinstance(batch, GeneratedEvalBatch)
            for idx, item in enumerate(batch.items[:cnt], start=1):
                cases.append(
                    EvalCase(
                        id=f"edge_{subtype}_{idx:03d}",
                        type="edge",
                        subtype=subtype,  # type: ignore[arg-type]
                        question=item.question.strip(),
                        expected_status=item.expected_status or default_status,  # type: ignore[arg-type]
                        expected_sources=[],
                        key_facts=item.key_facts,
                    )
                )
        except Exception:
            pass

    # 4. Out-of-scope: adjacent unsupported + off-topic & must_refuse (with Gate 3)
    refuse_list = list(requirements.must_refuse)
    oos_prompt = render_prompt(
        "eval_oos",
        scope=requirements.scope,
        must_refuse=", ".join(refuse_list) or "legal advice, personal data",
        doc_summary=sample_context[:1500],
        adjacent_count=n_adj + 2,
        off_topic_count=n_off + 2,
    )
    try:
        oos_batch = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Generate out-of-scope evaluation questions. Return JSON.",
                },
                {"role": "user", "content": oos_prompt},
            ],
            model_role="factory",
            json_schema=GeneratedEvalBatch,
            temperature=0.2,
        )
        assert isinstance(oos_batch, GeneratedEvalBatch)
        adj_added = 0
        off_added = 0
        for item in oos_batch.items:
            q = item.question.strip()
            if not q:
                continue
            # Gate 3: Unanswerability check against top-5 retrieved chunks
            top5 = index.retrieve(q, top_k=5, alpha=0.5, min_score=0.0)
            max_dense = max((r.dense_score for r in top5), default=0.0)
            if max_dense > 0.72:
                try:
                    passage_blob = "\n\n".join(r.chunk.text for r in top5)
                    chk = llm.chat(
                        messages=[
                            {
                                "role": "system",
                                "content": "Determine if the passages actually answer the question. Return JSON.",
                            },
                            {
                                "role": "user",
                                "content": f"Passages:\n{passage_blob}\n\nQuestion: {q}",
                            },
                        ],
                        model_role="factory",
                        json_schema=AnswerabilityCheckOutput,
                        temperature=0.0,
                    )
                    if isinstance(chk, AnswerabilityCheckOutput) and chk.answerable:
                        continue
                except Exception:
                    pass

            if adj_added < n_adj:
                adj_added += 1
                cases.append(
                    EvalCase(
                        id=f"oos_adj_{adj_added:03d}",
                        type="out_of_scope",
                        subtype="adjacent_unsupported",
                        question=q,
                        expected_status="refused",
                    )
                )
            elif off_added < n_off:
                off_added += 1
                cases.append(
                    EvalCase(
                        id=f"oos_off_{off_added:03d}",
                        type="out_of_scope",
                        subtype="off_topic",
                        question=q,
                        expected_status="refused",
                    )
                )
    except Exception:
        pass

    # Gate 4: Embedding deduplication (> 0.90 cosine similarity)
    cases = _dedup_cases(cases, index, threshold=0.90)

    # Gate 5: Stratified 70/30 dev/heldout split
    cases = _assign_stratified_splits(cases, dev_ratio=dev_ratio, rng=rng)

    prompt_versions = get_prompt_versions(["eval_answerable", "eval_edge", "eval_oos"])
    eval_set = EvalSet(seed=seed, prompt_versions=prompt_versions, cases=cases)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "05_eval_set.json").write_text(
            eval_set.model_dump_json(indent=2), encoding="utf-8"
        )

    return eval_set
