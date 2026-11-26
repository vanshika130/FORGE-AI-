"""Tests verifying generated demo PDF/CSV assets, report generation, baselines, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.cli import cmd_check
from forgeai.ingest import ingest_documents
from forgeai.llm import LLMClient
from forgeai.report import generate_final_report
from forgeai.retrieval import EmbeddingBackend, HybridIndex
from forgeai.runner import run_sanity_baselines
from forgeai.schemas import (
    AgentDefinition,
    EvalCase,
    EvalSet,
    GroundabilityReport,
    GroundingFlag,
    IterationLogEntry,
    JudgeResult,
    RequirementsDoc,
)


def test_demo_assets_ingest_cleanly() -> None:
    hr_digest, hr_chunks = ingest_documents(Path("demos/hr_policy/docs"))
    assert len(hr_digest.documents) == 3
    assert hr_digest.has_csv is False
    assert len(hr_chunks) >= 6
    assert all(d.num_chars > 100 for d in hr_digest.documents)

    ship_digest, ship_chunks = ingest_documents(Path("demos/ship_spares/docs"))
    assert len(ship_digest.documents) == 3
    assert ship_digest.has_csv is True
    assert any(d.file_name == "spares_catalogue.csv" and len(d.csv_columns) == 7 for d in ship_digest.documents)
    assert len(ship_chunks) >= 200


def test_sanity_baselines_and_honest_report(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text(
        "Annual leave is 21 working days per year after 6 months probation.",
        encoding="utf-8",
    )
    _, chunks = ingest_documents(docs_dir)
    index = HybridIndex(chunks, embedder=EmbeddingBackend(force_hashed=True))

    def mock_judge(_target, messages, _temp, _expect_json):
        content = messages[-1]["content"]
        if "extract_claims" in content:
            return json.dumps({"claims": ["Annual leave is 21 working days."]})
        return json.dumps(
            {
                "verdict": "YES",
                "supporting_sentence": "Annual leave is 21 working days per year after 6 months probation.",
            }
        )

    llm = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=mock_judge)
    eval_set = EvalSet(
        seed=42,
        cases=[
            EvalCase(
                id="ans_1",
                type="answerable",
                subtype="single_doc",
                question="How many annual leave days?",
                expected_status="answered",
                expected_sources=["policy.md"],
                key_facts=["21 working days"],
                split="dev",
            ),
            EvalCase(
                id="oos_1",
                type="out_of_scope",
                subtype="off_topic",
                question="What is the stock price today?",
                expected_status="refused",
                split="heldout",
            ),
        ],
    )
    baselines = run_sanity_baselines(eval_set, index, llm)
    assert baselines["always_refuse"]["out_of_scope_pass_rate"] == 1.0
    assert baselines["always_refuse"]["answerable_pass_rate"] == 0.0
    assert baselines["always_answer"]["out_of_scope_pass_rate"] == 0.0

    req = RequirementsDoc(
        agent_name="hr-demo",
        scope="HR questions",
        audience="Shore staff",
        tone="Clear",
        must_answer=["leave"],
        must_refuse=["payroll"],
    )
    gb = GroundabilityReport(
        claims=[GroundingFlag(ask="payroll", status="unsupported", recommended_action="refuse")],
        unsupported_count=1,
    )
    agent = AgentDefinition(name="hr-demo", version=1, system_prompt="Prompt", source_documents=["policy.md"])
    juds = [
        JudgeResult(
            case_id="ans_1",
            case_type="answerable",
            subtype="single_doc",
            split="dev",
            retrieval_hit=True,
            status_correct=True,
            fact_recall=1.0,
            grounded=True,
            citation_valid=True,
            **{"pass": True},
        ),
        JudgeResult(
            case_id="oos_1",
            case_type="out_of_scope",
            subtype="off_topic",
            split="heldout",
            retrieval_hit=True,
            status_correct=False,
            fact_recall=1.0,
            grounded=True,
            citation_valid=True,
            failure_category="HALLUCINATION",
            root_cause="Missed refusal on off-topic question.",
            **{"pass": False},
        ),
    ]
    logs = [
        IterationLogEntry(
            iteration=0,
            dev_score_before=1.0,
            dev_score_after=1.0,
            heldout_score=0.0,
            accepted=True,
            rationale="Baseline",
        )
    ]
    scorecard = [
        {
            "iteration": 0,
            "dev": {
                "total": 1,
                "passed": 1,
                "pass_rate": 1.0,
                "answerable_pass_rate": 1.0,
                "out_of_scope_pass_rate": 0.0,
                "edge_pass_rate": 0.0,
                "retrieval_hit_rate": 1.0,
                "groundedness_rate": 1.0,
                "correct_refusal_rate": 1.0,
                "wrongful_refusal_rate": 0.0,
            },
            "heldout": {
                "total": 1,
                "passed": 0,
                "pass_rate": 0.0,
                "answerable_pass_rate": 0.0,
                "out_of_scope_pass_rate": 0.0,
                "edge_pass_rate": 0.0,
                "retrieval_hit_rate": 1.0,
                "groundedness_rate": 1.0,
                "correct_refusal_rate": 0.0,
                "wrongful_refusal_rate": 0.0,
            },
        }
    ]
    _, report_md = generate_final_report(
        "HR assistant",
        req,
        gb,
        agent,
        eval_set.cases,
        juds,
        logs,
        scorecard,
        sanity_baselines=baselines,
        run_dir=tmp_path / "run",
    )
    assert "What Still Fails and Why" in report_md
    assert "Honesty Flag (Overfitting Warning)" in report_md
    assert "Sanity Baseline Discrimination Scores" in report_md
    assert (tmp_path / "run" / "08_final_report.md").exists()


def test_cli_check_command() -> None:
    assert cmd_check() == 0
