"""Stage 1: Intake Interviewer & Requirements Writer."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Callable, Optional

from forgeai.ingest import format_digest_for_prompt
from forgeai.llm import LLMClient
from forgeai.prompt_loader import render_prompt
from forgeai.schemas import (
    ClarificationItem,
    DocumentDigest,
    IntakeQuestion,
    IntakeQuestionBatch,
    RequirementsDoc,
)

MAX_CLARIFYING_QUESTIONS = 5


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def filter_redundant_questions(
    questions: list[IntakeQuestion], brief: str
) -> list[IntakeQuestion]:
    """Enforce <=5 cap and drop questions whose answer is already stated in the brief."""
    brief_norm = _normalize(brief)
    seen_questions: set[str] = set()
    kept: list[IntakeQuestion] = []

    for q in questions:
        q_norm = _normalize(q.question)
        if not q_norm or q_norm in seen_questions:
            continue
        # Guard: if the model quoted a brief snippet as "missing" that actually already
        # specifies the answer, or if the default answer phrasing is verbatim in the brief, skip.
        quote_norm = _normalize(q.missing_quote_from_brief)
        ans_norm = _normalize(q.default_answer)
        if quote_norm and quote_norm in brief_norm:
            ans_tokens = [t for t in re.findall(r"[a-z0-9]+", ans_norm) if len(t) > 2]
            if ans_tokens and sum(1 for t in ans_tokens if t in quote_norm) >= max(
                1, len(ans_tokens) // 2
            ):
                continue
        if ans_norm and len(ans_norm) >= 10 and ans_norm in brief_norm:
            continue

        seen_questions.add(q_norm)
        kept.append(q)
        if len(kept) >= MAX_CLARIFYING_QUESTIONS:
            break

    return kept[:MAX_CLARIFYING_QUESTIONS]


def _slugify_agent_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return slug or "custom-assistant"


def run_intake(
    brief: str,
    digest: DocumentDigest,
    llm: LLMClient,
    auto: bool = True,
    run_dir: Optional[Path] = None,
    input_fn: Callable[[str], str] = input,
    human_checkpoint: bool = False,
) -> tuple[list[ClarificationItem], RequirementsDoc]:
    """Run Stage 1 intake interview and generate validated RequirementsDoc."""
    digest_text = format_digest_for_prompt(digest)

    q_prompt = render_prompt(
        "intake_questions",
        brief=brief.strip(),
        digest=digest_text,
    )
    batch = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "You are a precise requirements interviewer. Return valid JSON only.",
            },
            {"role": "user", "content": q_prompt},
        ],
        model_role="factory",
        json_schema=IntakeQuestionBatch,
        temperature=0.0,
    )
    assert isinstance(batch, IntakeQuestionBatch)

    filtered_questions = filter_redundant_questions(batch.questions, brief)
    clarifications: list[ClarificationItem] = []
    assumptions: list[str] = []

    for idx, q in enumerate(filtered_questions, start=1):
        if auto:
            answer = q.default_answer.strip() or "Use standard professional defaults."
            clarifications.append(
                ClarificationItem(q=q.question, a=answer, is_default_assumption=True)
            )
            assumptions.append(f"Q: {q.question} -> Assumed default: {answer}")
        else:
            print(f"\n[Intake Q{idx}/{len(filtered_questions)}] {q.question}")
            print(f"  (Default: {q.default_answer})")
            user_ans = input_fn("  Your answer (Enter for default): ").strip()
            if not user_ans:
                clarifications.append(
                    ClarificationItem(
                        q=q.question, a=q.default_answer, is_default_assumption=True
                    )
                )
                assumptions.append(
                    f"Q: {q.question} -> Assumed default: {q.default_answer}"
                )
            else:
                clarifications.append(
                    ClarificationItem(
                        q=q.question, a=user_ans, is_default_assumption=False
                    )
                )

    clar_text = (
        "\n".join(f"- Q: {c.q}\n  A: {c.a}" for c in clarifications)
        or "(No clarifying questions needed)"
    )

    req_prompt = render_prompt(
        "requirements_writer",
        brief=brief.strip(),
        digest=digest_text,
        clarifications=clar_text,
    )
    requirements = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "You convert a brief, doc digest, and clarifications into requirements JSON.",
            },
            {"role": "user", "content": req_prompt},
        ],
        model_role="factory",
        json_schema=RequirementsDoc,
        temperature=0.0,
    )
    assert isinstance(requirements, RequirementsDoc)

    # Normalize & inject deterministic clarifications/assumptions + cross-check tool flags.
    requirements.agent_name = _slugify_agent_name(requirements.agent_name)
    requirements.clarifications = clarifications
    for a in assumptions:
        if a not in requirements.assumptions:
            requirements.assumptions.append(a)

    combined_text = (
        brief + " " + requirements.scope + " " + " ".join(requirements.must_answer)
    ).lower()
    if digest.has_csv and any(
        kw in combined_text
        for kw in ("csv", "table", "part", "catalogue", "catalog", "lookup", "price", "row", "stock", "spares")
    ):
        requirements.needs_tabular_lookup = True
    if any(
        kw in combined_text
        for kw in ("total", "sum", "calculate", "cost", "estimate", "price", "quote", "arithmetic", "convert")
    ):
        requirements.needs_numeric = True
    if any(kw in combined_text for kw in ("weather", "forecast", "temperature", "wind", "rain")):
        requirements.needs_external_data = True
    if any(
        kw in combined_text
        for kw in ("today", "current date", "deadline", "as of", "validity period", "lead time", "lead-time")
    ):
        requirements.needs_current_date = True

    if human_checkpoint and not auto:
        print("\n=== Requirements Summary ===")
        print(requirements.model_dump_json(indent=2))
        edit_note = input_fn(
            "\nPress Enter to accept, or type extra must_refuse topics (comma-separated): "
        ).strip()
        if edit_note:
            for item in edit_note.split(","):
                cleaned = item.strip()
                if cleaned and cleaned not in requirements.must_refuse:
                    requirements.must_refuse.append(cleaned)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        dialogue_payload = {
            "auto_mode": auto,
            "questions_asked": len(clarifications),
            "max_allowed": MAX_CLARIFYING_QUESTIONS,
            "dialogue": [c.model_dump() for c in clarifications],
        }
        (run_dir / "01_intake_dialogue.json").write_text(
            json.dumps(dialogue_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "02_requirements.json").write_text(
            requirements.model_dump_json(indent=2), encoding="utf-8"
        )

    return clarifications, requirements
