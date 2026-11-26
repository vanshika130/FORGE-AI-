"""End-to-end deterministic tests for Stages 1-7 guards, classification, and patch revert."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.critic import classify_all_failures, classify_failure
from forgeai.evalgen import verify_key_facts_in_text
from forgeai.generator import generate_agent
from forgeai.guard import run_groundability_guard
from forgeai.ingest import compute_docs_hash, ingest_documents
from forgeai.intake import filter_redundant_questions, run_intake
from forgeai.judge import check_citations_valid, compute_fact_recall, judge_case
from forgeai.llm import LLMClient
from forgeai.loop import run_optimization_loop
from forgeai.optimizer import apply_patch
from forgeai.retrieval import EmbeddingBackend, HybridIndex
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    EvalCase,
    EvalSet,
    IntakeQuestion,
    PatchOperation,
    RequirementsDoc,
    RunnerCaseResult,
    SourceCitation,
)


def test_intake_caps_to_five_and_dedupes_answered_brief() -> None:
    brief = "I need an assistant in English with a formal tone for shore staff."
    qs = [
        IntakeQuestion(
            question="What language should it use?",
            reason="Language check",
            default_answer="English",
            missing_quote_from_brief="in English",
        ),
        *[
            IntakeQuestion(
                question=f"Extra clarifying question {i}?",
                reason="Needed",
                default_answer=f"Default {i}",
            )
            for i in range(1, 8)
        ],
    ]
    kept = filter_redundant_questions(qs, brief)
    assert len(kept) == 5
    assert all("What language" not in q.question for q in kept)


def test_key_fact_verification_and_recall() -> None:
    chunk = "Full-time shore staff receive 21 working days of annual leave after 6 months."
    assert verify_key_facts_in_text(["21 working days", "after 6 months"], chunk)
    assert not verify_key_facts_in_text(["25 working days"], chunk)
    recall = compute_fact_recall(
        ["21 working days", "after 6 months"],
        "You receive twenty-one working days of leave after 6 months.",
    )
    assert recall == 1.0


def test_groundability_guard_flags_unsupported_and_halts_overambitious(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "Shore staff receive 21 working days of annual leave and 10 paid sick days.",
        encoding="utf-8",
    )
    _, chunks = ingest_documents(tmp_path)
    index = HybridIndex(chunks, embedder=EmbeddingBackend(force_hashed=True))

    def mock_llm(_target, messages, _temp, _expect_json):
        user_txt = messages[-1]["content"]
        if "Groundability Claim Decomposer" in user_txt:
            return json.dumps(
                {
                    "claims": [
                        "answers annual leave questions",
                        "calculates individual payroll tax withholdings",
                        "gives personalized immigration visa legal advice",
                    ]
                }
            )
        if "answers annual leave questions" in user_txt:
            return json.dumps(
                {
                    "status": "SUPPORTED",
                    "evidence": "Shore staff receive 21 working days of annual leave",
                    "reason": "Covered in leave.md",
                }
            )
        return json.dumps({"status": "UNSUPPORTED", "evidence": "", "reason": "Not in docs"})

    llm = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=mock_llm)
    req = RequirementsDoc(
        agent_name="hr-test",
        scope="HR + payroll + visa",
        audience="staff",
        tone="polite",
    )
    report, updated_req = run_groundability_guard(
        "answers annual leave, calculates payroll tax, gives visa advice",
        req,
        index,
        llm,
        min_similarity=0.10,
        stop_unsupported_fraction=0.50,
    )
    assert report.unsupported_count == 2
    assert report.feasible is False
    assert any("payroll tax" in r for r in updated_req.must_refuse)


def test_judge_deterministic_gates_and_failure_classification(tmp_path: Path) -> None:
    # Judge says YES with fake quote -> deterministic gate overrides to NO -> HALLUCINATION
    def mock_judge(_target, messages, _temp, _expect_json):
        content = messages[-1]["content"]
        if "extract_claims" in content:
            return json.dumps({"claims": ["Staff get 30 days of annual leave."]})
        return json.dumps(
            {
                "verdict": "YES",
                "supporting_sentence": "Fabricated sentence not in context.",
            }
        )

    llm = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=mock_judge)
    case = EvalCase(
        id="ans_001",
        type="answerable",
        subtype="single_doc",
        question="How many annual leave days?",
        expected_status="answered",
        expected_sources=["leave.md"],
        key_facts=["21 working days"],
        gold_chunk_ids=["leave.md#0"],
    )
    result = RunnerCaseResult(
        case_id="ans_001",
        question=case.question,
        response=AgentResponse(
            status="answered",
            answer="Staff get 30 days of annual leave.",
            sources=[SourceCitation(document="leave.md", quote="21 working days")],
            confidence="high",
        ),
        retrieved_chunk_ids=["leave.md#0"],
        retrieved_docs=["leave.md"],
        retrieved_context="Shore staff receive 21 working days of annual leave.",
    )
    judgement = judge_case(case, result, llm)
    assert judgement.grounded is False
    assert judgement.passed is False
    cat, _ = classify_failure(case, result, judgement)
    assert cat == "HALLUCINATION"


def test_optimizer_accepts_improvement_and_reverts_regression(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "leave.md").write_text(
        "Annual leave is 21 working days per year. Carry-over is capped at 5 days.",
        encoding="utf-8",
    )

    def mock_loop(_target, messages, _temp, _expect_json):
        sys_txt = messages[0]["content"]
        usr_txt = messages[-1]["content"]
        if "Mode: extract_claims" in usr_txt:
            return json.dumps({"claims": ["Annual leave is 21 working days."]})
        if "Mode: verify_claim" in usr_txt:
            return json.dumps(
                {
                    "verdict": "YES",
                    "supporting_sentence": "Annual leave is 21 working days per year.",
                }
            )
        # Agent behaviour depends on whether stricter grounding guard is present
        if "STRICT GROUNDING GUARD" in sys_txt:
            if "horoscope" in usr_txt.lower():
                return json.dumps(
                    {
                        "status": "refused",
                        "answer": "I cannot answer horoscope questions.",
                        "sources": [],
                        "confidence": "high",
                    }
                )
            return json.dumps(
                {
                    "status": "answered",
                    "answer": "Annual leave is 21 working days per year.",
                    "sources": [
                        {
                            "document": "leave.md",
                            "quote": "Annual leave is 21 working days per year.",
                        }
                    ],
                    "confidence": "high",
                }
            )
        # Baseline agent hallucinates / fails refusal on horoscope
        if "horoscope" in usr_txt.lower():
            return json.dumps(
                {
                    "status": "answered",
                    "answer": "Your horoscope says good luck today.",
                    "sources": [],
                    "confidence": "medium",
                }
            )
        return json.dumps(
            {
                "status": "answered",
                "answer": "Annual leave is 21 working days per year.",
                "sources": [
                    {
                        "document": "leave.md",
                        "quote": "Annual leave is 21 working days per year.",
                    }
                ],
                "confidence": "high",
            }
        )

    llm = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=mock_loop)
    embedder = EmbeddingBackend(force_hashed=True)
    doc_hash = compute_docs_hash(docs_dir)

    agent_v0 = AgentDefinition(
        name="hr-test",
        version=0,
        system_prompt="## Role and Scope\nHR helper\n\n## Grounding Rules\nStay grounded.\n\n## Refusal Rules\nRefuse off-topic.",
        source_documents=["leave.md"],
    )
    eval_set = EvalSet(
        seed=42,
        cases=[
            EvalCase(
                id="ans_1",
                type="answerable",
                subtype="single_doc",
                question="How many annual leave days?",
                expected_status="answered",
                expected_sources=["leave.md"],
                key_facts=["21 working days"],
                split="dev",
            ),
            EvalCase(
                id="oos_1",
                type="out_of_scope",
                subtype="off_topic",
                question="What is my horoscope today?",
                expected_status="refused",
                split="dev",
            ),
            EvalCase(
                id="oos_2",
                type="out_of_scope",
                subtype="off_topic",
                question="Tell me today's horoscope please.",
                expected_status="refused",
                split="heldout",
            ),
        ],
    )
    final_agent, logs, scorecard = run_optimization_loop(
        initial_agent=agent_v0,
        eval_set=eval_set,
        docs_dir=docs_dir,
        llm=llm,
        embedder=embedder,
        doc_hash=doc_hash,
        max_iterations=2,
        run_dir=tmp_path / "run",
    )
    assert logs[0].dev_score_before == 0.5
    assert logs[1].accepted is True
    assert logs[1].dev_score_after == 1.0
    assert final_agent.version == 1
    assert (tmp_path / "run" / "agent_final.yaml").exists()


def test_optimizer_reverts_when_no_improvement(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "leave.md").write_text(
        "Annual leave is 21 working days per year.",
        encoding="utf-8",
    )

    def mock_no_gain(_target, messages, _temp, _expect_json):
        usr_txt = messages[-1]["content"]
        if "Mode: extract_claims" in usr_txt:
            return json.dumps({"claims": ["Annual leave is 21 working days."]})
        if "Mode: verify_claim" in usr_txt:
            return json.dumps(
                {
                    "verdict": "YES",
                    "supporting_sentence": "Annual leave is 21 working days per year.",
                }
            )
        if "horoscope" in usr_txt.lower():
            # Always fails refusal even after prompt patch -> patch must be reverted
            return json.dumps(
                {
                    "status": "answered",
                    "answer": "Good luck today.",
                    "sources": [],
                    "confidence": "medium",
                }
            )
        return json.dumps(
            {
                "status": "answered",
                "answer": "Annual leave is 21 working days per year.",
                "sources": [
                    {
                        "document": "leave.md",
                        "quote": "Annual leave is 21 working days per year.",
                    }
                ],
                "confidence": "high",
            }
        )

    llm = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=mock_no_gain)
    embedder = EmbeddingBackend(force_hashed=True)
    doc_hash = compute_docs_hash(docs_dir)
    agent_v0 = AgentDefinition(
        name="hr-revert",
        version=0,
        system_prompt="## Role and Scope\nHR\n\n## Grounding Rules\nBe grounded.",
        source_documents=["leave.md"],
    )
    eval_set = EvalSet(
        seed=42,
        cases=[
            EvalCase(
                id="ans_1",
                type="answerable",
                subtype="single_doc",
                question="How many annual leave days?",
                expected_status="answered",
                expected_sources=["leave.md"],
                key_facts=["21 working days"],
                split="dev",
            ),
            EvalCase(
                id="oos_1",
                type="out_of_scope",
                subtype="off_topic",
                question="What is my horoscope?",
                expected_status="refused",
                split="dev",
            ),
        ],
    )
    final_agent, logs, _ = run_optimization_loop(
        initial_agent=agent_v0,
        eval_set=eval_set,
        docs_dir=docs_dir,
        llm=llm,
        embedder=embedder,
        doc_hash=doc_hash,
        max_iterations=1,
        run_dir=tmp_path / "run",
    )
    assert logs[1].accepted is False
    assert "Reverted" in logs[1].rejection_reason
    assert final_agent.version == 0
