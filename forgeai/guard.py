"""Stage 2: Groundability Guard against document gaps and over-ambitious briefs."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Optional

from forgeai.llm import LLMClient
from forgeai.prompt_loader import render_prompt
from forgeai.retrieval import HybridIndex
from forgeai.schemas import (
    EntailmentAssessment,
    GroundabilityClaimsBatch,
    GroundabilityReport,
    GroundingFlag,
    RequirementsDoc,
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_verbatim_in_chunks(quote: str, chunk_texts: list[str]) -> bool:
    q = _normalize_ws(quote)
    if len(q) < 8:
        return False
    for text in chunk_texts:
        if q in _normalize_ws(text):
            return True
    return False


def _external_tool_covering_claim(claim: str, requirements: RequirementsDoc) -> Optional[str]:
    low = claim.lower()
    if any(w in low for w in ("weather", "forecast", "temperature", "wind", "precipitation")):
        if requirements.needs_external_data:
            return "http_weather"
    if any(w in low for w in ("today's date", "current date")):
        return "current_date"
    return None


def run_groundability_guard(
    brief: str,
    requirements: RequirementsDoc,
    index: HybridIndex,
    llm: LLMClient,
    min_similarity: float = 0.22,
    stop_unsupported_fraction: float = 0.50,
    run_dir: Optional[Path] = None,
) -> tuple[GroundabilityReport, RequirementsDoc]:
    """Decompose brief into claims, verify retrieval + entailment, and flag gaps."""
    claims_prompt = render_prompt("guard_claims", brief=brief.strip())
    claims_batch = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "Decompose the agent brief into short atomic capability claims. Return JSON.",
            },
            {"role": "user", "content": claims_prompt},
        ],
        model_role="factory",
        json_schema=GroundabilityClaimsBatch,
        temperature=0.0,
    )
    assert isinstance(claims_batch, GroundabilityClaimsBatch)

    raw_claims = [c.strip()[:200] for c in claims_batch.claims if isinstance(c, str) and c.strip()]
    if not raw_claims:
        raw_claims = [brief.strip()[:200]]

    flags: list[GroundingFlag] = []
    supported_count = 0
    partial_count = 0
    unsupported_count = 0
    unsupported_for_stop = 0

    for claim in raw_claims:
        tool_cover = _external_tool_covering_claim(claim, requirements)
        if tool_cover == "http_weather":
            flags.append(
                GroundingFlag(
                    ask=claim,
                    status="supported",
                    evidence=f"Covered legitimately by registered external tool '{tool_cover}'.",
                    max_similarity=1.0,
                    max_bm25=1.0,
                    covered_by_tool=tool_cover,
                    recommended_action=f"Attach tool {tool_cover}.",
                )
            )
            supported_count += 1
            continue

        retrieved = index.retrieve(claim, top_k=5, alpha=0.5, min_score=0.0)
        max_dense = max((r.dense_score for r in retrieved), default=0.0)
        max_bm25 = max((r.bm25_score for r in retrieved), default=0.0)
        chunk_texts = [r.chunk.text for r in retrieved]
        passages_block = "\n\n".join(
            f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in retrieved
        ) or "(No passages retrieved)"

        entail_prompt = render_prompt(
            "guard_entailment",
            claim=claim,
            passages=passages_block,
        )
        assessment = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Evaluate whether retrieved passages support the capability claim. Return JSON.",
                },
                {"role": "user", "content": entail_prompt},
            ],
            model_role="factory",
            json_schema=EntailmentAssessment,
            temperature=0.0,
        )
        assert isinstance(assessment, EntailmentAssessment)

        llm_status = assessment.status.upper()
        evidence = assessment.evidence.strip()

        # Deterministic checks override optimistic LLM verdicts.
        if not retrieved or (max_dense < min_similarity and max_bm25 < 1.5):
            final_status = "unsupported"
            action = "Move to must_refuse; documents do not cover this capability."
        elif llm_status == "UNSUPPORTED":
            final_status = "unsupported"
            action = "Move to must_refuse; documents do not cover this capability."
        elif llm_status == "SUPPORTED":
            if _is_verbatim_in_chunks(evidence, chunk_texts):
                final_status = "supported"
                action = "Keep in scope."
            else:
                final_status = "partial"
                action = (
                    "Downgraded from SUPPORTED to PARTIAL because quoted evidence was not "
                    "found verbatim; add scoped-down caveat to system prompt."
                )
        else:
            final_status = "partial"
            action = "Add scoped-down caveat to system prompt."

        if final_status == "supported":
            supported_count += 1
        elif final_status == "partial":
            partial_count += 1
        else:
            unsupported_count += 1
            unsupported_for_stop += 1
            refusal_note = f"{claim} (documents do not cover this)"
            if refusal_note not in requirements.must_refuse:
                requirements.must_refuse.append(refusal_note)

        flags.append(
            GroundingFlag(
                ask=claim,
                status=final_status,
                evidence=evidence,
                max_similarity=round(max_dense, 4),
                max_bm25=round(max_bm25, 4),
                covered_by_tool=tool_cover,
                recommended_action=action,
            )
        )

    total = len(flags)
    unsupported_fraction = (unsupported_for_stop / total) if total else 0.0
    feasible = True
    stop_reason = ""
    if total >= 2 and unsupported_fraction > stop_unsupported_fraction:
        feasible = False
        stop_reason = (
            f"Document set cannot support this brief: {unsupported_for_stop}/{total} "
            f"({unsupported_fraction:.0%}) capability claims are unsupported by the provided documents."
        )
    elif total == 1 and unsupported_for_stop == 1:
        feasible = False
        stop_reason = (
            "Document set cannot support this brief: the sole capability claim is unsupported."
        )

    report = GroundabilityReport(
        claims=flags,
        supported_count=supported_count,
        partial_count=partial_count,
        unsupported_count=unsupported_count,
        unsupported_fraction=round(unsupported_fraction, 4),
        feasible=feasible,
        stop_reason=stop_reason,
    )

    requirements.grounding_flags = flags

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "03_groundability_report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "02_requirements.json").write_text(
            requirements.model_dump_json(indent=2), encoding="utf-8"
        )

    return report, requirements
