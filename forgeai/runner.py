"""Stage 8: Generic Agent Runner + Section 9 Sanity Baselines."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Optional

from pydantic import ValidationError
import yaml

from forgeai.ingest import compute_docs_hash, ingest_documents
from forgeai.llm import LLMClient, _extract_json_block
from forgeai.retrieval import EmbeddingBackend, HybridIndex, RetrievedChunk
from forgeai.schemas import (
    AgentDefinition,
    AgentResponse,
    EvalCase,
    EvalSet,
    RunnerCaseResult,
    SourceCitation,
    ToolCallRequest,
)
from forgeai.tools.registry import execute_tool

MAX_TOOL_STEPS = 3
MAX_HISTORY_TURNS = 4


def load_agent_definition(yaml_path: str | Path) -> AgentDefinition:
    """Load and Pydantic-validate an agent YAML file."""
    path = Path(yaml_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AgentDefinition.model_validate(data)


class AgentRunner:
    """Generic executor that runs any declarative AgentDefinition."""

    def __init__(
        self,
        agent: AgentDefinition,
        index: HybridIndex,
        docs_dir: str | Path,
        llm: LLMClient,
    ) -> None:
        self.agent = agent
        self.index = index
        self.docs_dir = Path(docs_dir)
        self.llm = llm

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        docs_dir: str | Path | None = None,
        llm: LLMClient | None = None,
        embedder: EmbeddingBackend | None = None,
    ) -> "AgentRunner":
        agent = load_agent_definition(yaml_path)
        resolved_docs = Path(docs_dir) if docs_dir else Path(yaml_path).resolve().parent / "docs"
        _, chunks = ingest_documents(
            resolved_docs,
            chunk_size=agent.retrieval.chunk_size,
            chunk_overlap=agent.retrieval.chunk_overlap,
            source_documents=agent.source_documents or None,
        )
        doc_hash = compute_docs_hash(resolved_docs)
        chunk_key = f"{agent.retrieval.chunk_size}_{agent.retrieval.chunk_overlap}"
        index = HybridIndex(
            chunks,
            embedder=embedder or EmbeddingBackend(),
            doc_hash=doc_hash,
            chunk_params_key=chunk_key,
        )
        return cls(agent=agent, index=index, docs_dir=resolved_docs, llm=llm or LLMClient())

    def _rewrite_query(self, question: str) -> str:
        if not self.agent.retrieval.query_rewrite:
            return question
        try:
            rewritten = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the user question into concise search keywords. Return ONLY the rewritten query.",
                    },
                    {"role": "user", "content": question},
                ],
                model_role=self.agent.model.role,
                temperature=0.0,
                max_tokens=80,
            )
            assert isinstance(rewritten, str)
            cleaned = rewritten.strip()
            return cleaned if cleaned else question
        except Exception:
            return question

    @staticmethod
    def _format_context(retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "(No passages met the retrieval threshold.)"
        parts = []
        for r in retrieved:
            parts.append(
                f"[{r.chunk.doc} | {r.chunk.chunk_id} | score={r.score:.3f}]\n{r.chunk.text}"
            )
        return "\n\n".join(parts)

    def _parse_tool_or_response(
        self, raw_text: str
    ) -> tuple[Optional[ToolCallRequest], Optional[AgentResponse], Optional[str]]:
        extracted = _extract_json_block(raw_text)
        try:
            obj = json.loads(extracted)
        except json.JSONDecodeError as exc:
            return None, None, f"Invalid JSON: {exc}"

        if isinstance(obj, dict) and "tool" in obj and "status" not in obj:
            try:
                return ToolCallRequest.model_validate(obj), None, None
            except ValidationError as exc:
                return None, None, f"Invalid tool call schema: {exc}"

        try:
            return None, AgentResponse.model_validate(obj), None
        except ValidationError as exc:
            return None, None, f"Invalid AgentResponse schema: {exc}"

    def run_turn(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> RunnerCaseResult:
        """Execute a single user turn with retrieval, up to 3 tool calls, and 1 JSON repair."""
        t0 = time.monotonic()
        search_query = self._rewrite_query(question)
        retrieved = self.index.retrieve_with_config(search_query, self.agent.retrieval)
        context_text = self._format_context(retrieved)
        retrieved_chunk_ids = [r.chunk.chunk_id for r in retrieved]
        retrieved_docs = list(dict.fromkeys(r.chunk.doc for r in retrieved))

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.agent.system_prompt}
        ]
        if history:
            # Keep last 4 turns (up to 8 user/assistant messages)
            for msg in history[-(MAX_HISTORY_TURNS * 2) :]:
                if msg.get("role") in {"user", "assistant"}:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        user_payload = (
            f"Retrieved Context:\n{context_text}\n\n"
            f"User Question: {question}\n\n"
            "Respond ONLY with valid JSON matching the Output Contract (or a tool call JSON if an attached tool is required)."
        )
        messages.append({"role": "user", "content": user_payload})

        tool_calls_made: list[str] = []
        extra_tool_context: list[str] = []
        final_response: Optional[AgentResponse] = None
        last_error: Optional[str] = None

        for step in range(MAX_TOOL_STEPS + 1):
            raw = self.llm.chat(
                messages=messages,
                model_role=self.agent.model.role,
                temperature=self.agent.model.temperature,
                max_tokens=self.agent.model.max_tokens,
            )
            assert isinstance(raw, str)
            tool_req, resp_obj, err = self._parse_tool_or_response(raw)

            if err is not None:
                # One repair retry for small open models
                repair_msgs = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your output failed JSON validation ({err}). "
                            "Return ONLY valid JSON with keys: status, answer, sources, confidence."
                        ),
                    },
                ]
                repaired = self.llm.chat(
                    messages=repair_msgs,
                    model_role=self.agent.model.role,
                    temperature=0.0,
                    max_tokens=self.agent.model.max_tokens,
                )
                assert isinstance(repaired, str)
                tool_req, resp_obj, err2 = self._parse_tool_or_response(repaired)
                if err2 is not None:
                    last_error = err2
                    break

            if resp_obj is not None:
                final_response = resp_obj
                break

            if tool_req is not None:
                if step >= MAX_TOOL_STEPS:
                    last_error = "Exceeded max tool steps."
                    break
                if tool_req.tool not in self.agent.tools:
                    tool_out: dict[str, Any] = {
                        "ok": False,
                        "error": f"Tool '{tool_req.tool}' is not attached to this agent.",
                    }
                else:
                    tool_out = execute_tool(
                        tool_req.tool,
                        tool_req.args,
                        context={"docs_dir": str(self.docs_dir)},
                    )
                call_sig = f"{tool_req.tool}({json.dumps(tool_req.args)})"
                tool_calls_made.append(call_sig)
                tool_snippet = f"[Tool {tool_req.tool} output]\n{json.dumps(tool_out, ensure_ascii=False)}"
                extra_tool_context.append(tool_snippet)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool Result:\n{tool_snippet}\n\n"
                            "Now produce your final JSON answer matching the Output Contract."
                        ),
                    }
                )

        if final_response is None:
            final_response = AgentResponse(
                status="refused",
                answer=self.agent.refusal_policy.refusal_message,
                sources=[],
                confidence="low",
            )

        full_context = context_text
        if extra_tool_context:
            full_context = full_context + "\n\n" + "\n\n".join(extra_tool_context)

        return RunnerCaseResult(
            case_id="interactive",
            question=question,
            response=final_response,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_docs=retrieved_docs,
            retrieved_context=full_context,
            tool_calls_made=tool_calls_made,
            latency_s=round(time.monotonic() - t0, 3),
            error=last_error,
        )

    def run_case(self, case: EvalCase) -> RunnerCaseResult:
        res = self.run_turn(case.question)
        res.case_id = case.id
        return res


def run_sanity_baselines(
    eval_set: EvalSet,
    index: HybridIndex,
    llm: LLMClient,
) -> dict[str, dict[str, float]]:
    """Section 9 factory self-test: 4 deterministic baseline agents to prove eval discrimination."""
    from forgeai.judge import compute_split_metrics, judge_all

    cases = eval_set.cases
    if not cases:
        return {}

    def _run_mode(mode: str) -> dict[str, float]:
        results: list[RunnerCaseResult] = []
        for c in cases:
            if mode == "no_retrieval":
                results.append(
                    RunnerCaseResult(
                        case_id=c.id,
                        question=c.question,
                        response=AgentResponse(
                            status="refused",
                            answer="No context retrieved.",
                            sources=[],
                            confidence="low",
                        ),
                        retrieved_context="",
                    )
                )
            elif mode == "echo_context":
                ret = index.retrieve(c.question, top_k=3, alpha=0.5, min_score=0.0)
                ctx = "\n\n".join(r.chunk.text for r in ret)
                first_doc = ret[0].chunk.doc if ret else ""
                first_quote = ret[0].chunk.text[:120] if ret else ""
                results.append(
                    RunnerCaseResult(
                        case_id=c.id,
                        question=c.question,
                        response=AgentResponse(
                            status="answered",
                            answer=first_quote or "No text.",
                            sources=[SourceCitation(document=first_doc, quote=first_quote)]
                            if first_doc
                            else [],
                            confidence="medium",
                        ),
                        retrieved_chunk_ids=[r.chunk.chunk_id for r in ret],
                        retrieved_docs=list({r.chunk.doc for r in ret}),
                        retrieved_context=ctx,
                    )
                )
            elif mode == "always_refuse":
                results.append(
                    RunnerCaseResult(
                        case_id=c.id,
                        question=c.question,
                        response=AgentResponse(
                            status="refused",
                            answer="I refuse all questions.",
                            sources=[],
                            confidence="high",
                        ),
                        retrieved_context="",
                    )
                )
            elif mode == "always_answer":
                ret = index.retrieve(c.question, top_k=3, alpha=0.5, min_score=0.0)
                ctx = "\n\n".join(r.chunk.text for r in ret)
                results.append(
                    RunnerCaseResult(
                        case_id=c.id,
                        question=c.question,
                        response=AgentResponse(
                            status="answered",
                            answer="Yes, this is supported and guaranteed in full.",
                            sources=[],
                            confidence="high",
                        ),
                        retrieved_chunk_ids=[r.chunk.chunk_id for r in ret],
                        retrieved_docs=list({r.chunk.doc for r in ret}),
                        retrieved_context=ctx,
                    )
                )
        judgements = judge_all(cases, results, llm)
        m = compute_split_metrics(cases, results, judgements, split=None)
        return {
            "pass_rate": m.pass_rate,
            "answerable_pass_rate": m.answerable_pass_rate,
            "out_of_scope_pass_rate": m.out_of_scope_pass_rate,
            "groundedness_rate": m.groundedness_rate,
        }

    return {
        "no_retrieval": _run_mode("no_retrieval"),
        "echo_context": _run_mode("echo_context"),
        "always_refuse": _run_mode("always_refuse"),
        "always_answer": _run_mode("always_answer"),
    }
