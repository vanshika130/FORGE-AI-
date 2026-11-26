"""Tests across 5 unseen-domain throwaway briefs (Section 8 robustness & tool attachment)."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.generator import generate_agent
from forgeai.guard import run_groundability_guard
from forgeai.ingest import ingest_documents
from forgeai.intake import run_intake
from forgeai.llm import LLMClient
from forgeai.retrieval import EmbeddingBackend, HybridIndex

DOMAINS = [
    (
        "it_helpdesk",
        "I need an internal IT helpdesk assistant for password resets and VPN setup.",
        "helpdesk.md",
        "VPN setup requires FortiClient 7.2 and corporate SSO. Password resets expire after 15 minutes.",
        [],
    ),
    (
        "recipe_book",
        "I need a kitchen assistant answering sourdough baking recipes and oven temperatures.",
        "recipes.md",
        "Bake sourdough at 230C covered for 20 minutes, then 210C uncovered for 22 minutes.",
        [],
    ),
    (
        "product_manual",
        "I need an espresso machine manual assistant that helps calculate descaling solution totals.",
        "manual.md",
        "Mix 50 ml descaling solution per 500 ml water tank capacity and run the rinse cycle twice.",
        ["calculator"],
    ),
    (
        "legal_faq",
        "I need an assistant summarizing our NDA confidentiality FAQ without giving legal advice.",
        "nda_faq.md",
        "Standard mutual NDAs run for 3 years from signing and exclude publicly available information.",
        [],
    ),
    (
        "weather_logistics",
        "I need a port logistics assistant checking berth rules and live weather forecasts at port locations.",
        "port_rules.md",
        "Container crane operations pause when sustained wind exceeds 22 knots at berth.",
        ["http_weather"],
    ),
]


def test_five_unseen_domains_pipeline_and_tool_selection(tmp_path: Path) -> None:
    embedder = EmbeddingBackend(force_hashed=True)

    for name, brief, doc_name, doc_text, expected_tools in DOMAINS:
        ddir = tmp_path / name / "docs"
        ddir.mkdir(parents=True)
        (ddir / doc_name).write_text(doc_text, encoding="utf-8")

        def make_mock(d_name: str, d_brief: str, d_text: str, exp_tools: list[str]):
            def _mock(_target, messages, _temp, _expect_json):
                usr = messages[-1]["content"]
                if "Stage 1 Intake Interviewer" in usr:
                    return json.dumps({"questions": []})
                if "Stage 1 Requirements Writer" in usr:
                    return json.dumps(
                        {
                            "agent_name": d_name.replace("_", "-"),
                            "scope": d_brief,
                            "audience": "staff",
                            "tone": "clear",
                            "must_answer": [d_name],
                            "must_refuse": ["unrelated topics"],
                            "needs_numeric": "calculator" in exp_tools,
                            "needs_external_data": "http_weather" in exp_tools,
                            "needs_tabular_lookup": False,
                            "needs_current_date": False,
                        }
                    )
                if "Groundability Claim Decomposer" in usr:
                    return json.dumps({"claims": [d_brief]})
                if "Groundability Entailment Checker" in usr:
                    return json.dumps(
                        {
                            "status": "SUPPORTED",
                            "evidence": d_text[:60],
                            "reason": "Covered in doc",
                        }
                    )
                if "Stage 3 Agent Generator" in usr:
                    proposals = []
                    for t in exp_tools:
                        quote = (
                            "calculate descaling solution totals"
                            if t == "calculator"
                            else "live weather forecasts"
                        )
                        proposals.append(
                            {
                                "name": t,
                                "justification_quote": quote,
                                "reason": "Requested in brief",
                            }
                        )
                    return json.dumps(
                        {
                            "tool_proposals": proposals,
                            "role_and_scope_section": f"Answer questions for {d_name}.",
                            "audience_and_tone_section": "Clear and concise.",
                            "grounding_rules_section": "Cite retrieved text.",
                            "refusal_rules_section": "Refuse unsupported queries.",
                            "tool_use_rules_section": "Use attached tools only when needed.",
                        }
                    )
                return "{}"

            return _mock

        llm = LLMClient(
            use_cache=False,
            cache_dir=tmp_path / name / "cache",
            mock_handler=make_mock(name, brief, doc_text, expected_tools),
        )
        digest, chunks = ingest_documents(ddir)
        index = HybridIndex(chunks, embedder=embedder)
        _, req = run_intake(brief, digest, llm, auto=True)
        gb, req = run_groundability_guard(
            brief, req, index, llm, min_similarity=0.05
        )
        assert gb.feasible is True
        agent_v0 = generate_agent(brief, req, gb, digest, llm)
        assert agent_v0.tools == expected_tools
