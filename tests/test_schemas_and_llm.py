"""Tests for Pydantic schemas, LLMClient JSON repair, caching, and role isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.llm import LLMClient, LLMError
from forgeai.schemas import AgentResponse, IntakeQuestionBatch


def test_llm_client_json_repair_succeeds(tmp_path: Path) -> None:
    calls = {"n": 0}

    def mock_handler(target, messages, temperature, expect_json):
        calls["n"] += 1
        if calls["n"] == 1:
            return "{not valid json"
        return json.dumps(
            {
                "status": "answered",
                "answer": "Annual leave is 21 days.",
                "sources": [{"document": "leave.pdf", "quote": "21 days"}],
                "confidence": "high",
            }
        )

    client = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=mock_handler)
    res = client.chat(
        messages=[{"role": "user", "content": "How many days?"}],
        model_role="factory",
        json_schema=AgentResponse,
    )
    assert isinstance(res, AgentResponse)
    assert res.status == "answered"
    assert calls["n"] == 2


def test_llm_client_rejects_after_max_repairs(tmp_path: Path) -> None:
    def always_bad(_target, _messages, _temp, _expect_json):
        return '{"status": "invalid_status"}'

    client = LLMClient(use_cache=False, cache_dir=tmp_path, mock_handler=always_bad)
    with pytest.raises(LLMError):
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model_role="factory",
            json_schema=AgentResponse,
        )


def test_llm_client_cache_and_judge_family_isolation(tmp_path: Path) -> None:
    calls = {"n": 0}

    def mock_handler(_target, _messages, _temp, _expect_json):
        calls["n"] += 1
        return json.dumps({"questions": []})

    client = LLMClient(use_cache=True, cache_dir=tmp_path, mock_handler=mock_handler)
    r1 = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        model_role="factory",
        json_schema=IntakeQuestionBatch,
    )
    r2 = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        model_role="factory",
        json_schema=IntakeQuestionBatch,
    )
    assert isinstance(r1, IntakeQuestionBatch) and isinstance(r2, IntakeQuestionBatch)
    assert calls["n"] == 1
    assert client.cache_hits == 1

    agent_targets = client.get_role_targets("agent")
    judge_targets = client.get_role_targets("judge")
    assert agent_targets[0].family != judge_targets[0].family
