"""Stage 3: Declarative Agent Generator with strict tool validation and prompt template."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import yaml

from forgeai.ingest import format_digest_for_prompt
from forgeai.llm import LLMClient
from forgeai.prompt_loader import render_prompt
from forgeai.schemas import (
    AgentDefinition,
    AgentGenerationDraft,
    DocumentDigest,
    GroundabilityReport,
    ModelConfig,
    OutputContractConfig,
    RefusalPolicy,
    RequirementsDoc,
    RetrievalConfig,
)
from forgeai.tools.registry import (
    TOOL_REGISTRY,
    check_tool_precondition,
    format_registry_for_prompt,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _quote_is_valid(
    quote: str, brief: str, requirements: RequirementsDoc, digest_text: str
) -> bool:
    q = _normalize(quote)
    if len(q) < 4:
        return False
    corpus = _normalize(
        " ".join(
            [
                brief,
                requirements.scope,
                " ".join(requirements.must_answer),
                " ".join(c.a for c in requirements.clarifications),
                digest_text,
            ]
        )
    )
    if q in corpus:
        return True
    # Allow matching if majority of content tokens (>=3 chars) appear in the source corpus
    tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) >= 3]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in corpus)
    return (hits / len(tokens)) >= 0.75


def choose_initial_retrieval_params(digest: DocumentDigest) -> RetrievalConfig:
    """Choose starting retrieval parameters from document heuristics."""
    non_csv_docs = [d for d in digest.documents if d.file_type not in {"csv", "xlsx", "xls"}]
    if not non_csv_docs and digest.has_csv:
        return RetrievalConfig(
            chunk_size=500,
            chunk_overlap=80,
            top_k=5,
            hybrid_alpha=0.5,
            min_score=0.20,
            query_rewrite=False,
        )
    avg_chars = (
        sum(d.num_chars for d in non_csv_docs) / len(non_csv_docs)
        if non_csv_docs
        else 2000
    )
    if avg_chars >= 8000:
        return RetrievalConfig(
            chunk_size=900,
            chunk_overlap=100,
            top_k=5,
            hybrid_alpha=0.5,
            min_score=0.22,
            query_rewrite=False,
        )
    return RetrievalConfig(
        chunk_size=500,
        chunk_overlap=80,
        top_k=4,
        hybrid_alpha=0.5,
        min_score=0.22,
        query_rewrite=False,
    )


def assemble_system_prompt(
    draft: AgentGenerationDraft,
    requirements: RequirementsDoc,
    groundability: GroundabilityReport,
    attached_tools: list[str],
) -> str:
    """Assemble the mandatory sectioned system prompt template."""
    partial_notes = [
        f"- Partial coverage only for '{f.ask}': answer only what the retrieved text explicitly states and note limitations."
        for f in groundability.claims
        if f.status == "partial"
    ]
    unsupported_notes = [
        f"- {f.ask} (documents do not cover this)"
        for f in groundability.claims
        if f.status == "unsupported"
    ]
    refuse_lines = list(dict.fromkeys(requirements.must_refuse + unsupported_notes))
    refuse_block = "\n".join(f"- {r}" for r in refuse_lines) or "- Any request outside the provided documents."

    if attached_tools:
        tool_specs = "\n".join(
            f"- {name}: {TOOL_REGISTRY[name].purpose} Args: {TOOL_REGISTRY[name].args_schema}"
            for name in attached_tools
        )
        tool_section = (
            f"{draft.tool_use_rules_section.strip()}\n\n"
            f"Available tools (ONLY use these):\n{tool_specs}\n\n"
            'To invoke a tool before giving your final answer, respond ONLY with JSON:\n'
            '{"tool": "<tool_name>", "args": {<arguments>}}\n'
            "After receiving the tool output, produce your final JSON answer."
        ).strip()
    else:
        tool_section = "No external tools are attached. Answer strictly using the retrieved context passages."

    example_json = draft.worked_example_json.strip()
    if not example_json or not example_json.startswith("{"):
        example_json = (
            '{"status": "answered", '
            '"answer": "Standard annual leave is 21 working days per year.", '
            '"sources": [{"document": "leave_policy.pdf", "quote": "Full-time staff receive 21 working days of annual leave."}], '
            '"confidence": "high"}'
        )

    partial_block = ("\n\nScoped-down coverage notes:\n" + "\n".join(partial_notes)) if partial_notes else ""

    sections = [
        f"## Role and Scope\n{draft.role_and_scope_section.strip()}{partial_block}",
        f"## Audience and Tone\n{draft.audience_and_tone_section.strip()}",
        (
            "## Grounding Rules\n"
            f"{draft.grounding_rules_section.strip()}\n"
            "- Answer ONLY from the retrieved document passages or verified tool outputs.\n"
            "- Always include verbatim quotes and exact filenames in `sources`.\n"
            "- If retrieved context does not contain the answer, set `status` to `refused` (if out of scope/unsupported) or `needs_clarification` (if ambiguous)."
        ),
        (
            "## Refusal Rules\n"
            f"{draft.refusal_rules_section.strip()}\n"
            "You MUST refuse the following topics and NEVER fabricate answers for them:\n"
            f"{refuse_block}"
        ),
        f"## Tool-Use Rules\n{tool_section}",
        (
            "## Output Contract\n"
            "Always respond with a single valid JSON object with keys: "
            "`status` ('answered' | 'refused' | 'needs_clarification'), "
            "`answer` (string), "
            "`sources` (list of {\"document\": string, \"quote\": string}), "
            "`confidence` ('high' | 'medium' | 'low').\n"
            f"Worked example:\n{example_json}"
        ),
    ]
    return "\n\n".join(sections)


def generate_agent(
    brief: str,
    requirements: RequirementsDoc,
    groundability: GroundabilityReport,
    digest: DocumentDigest,
    llm: LLMClient,
    run_dir: Optional[Path] = None,
) -> AgentDefinition:
    """Generate initial declarative AgentDefinition (agent_v0.yaml) with deterministic validation."""
    digest_text = format_digest_for_prompt(digest)
    registry_text = format_registry_for_prompt()

    gen_prompt = render_prompt(
        "agent_generator",
        brief=brief.strip(),
        requirements=requirements.model_dump_json(indent=2),
        groundability=groundability.model_dump_json(indent=2),
        digest=digest_text,
        tool_registry=registry_text,
    )
    draft = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "You design minimal, strictly grounded declarative agents. Return valid JSON.",
            },
            {"role": "user", "content": gen_prompt},
        ],
        model_role="factory",
        json_schema=AgentGenerationDraft,
        temperature=0.0,
        max_tokens=1600,
    )
    assert isinstance(draft, AgentGenerationDraft)

    # Layer 1 & 2 deterministic tool validation
    attached_tools: list[str] = []
    tool_justifications: dict[str, str] = {}
    removed_tools: dict[str, str] = {}

    for proposal in draft.tool_proposals:
        name = proposal.name.strip()
        if name in attached_tools:
            continue
        if not _quote_is_valid(proposal.justification_quote, brief, requirements, digest_text):
            removed_tools[name] = (
                f"Rejected at generation: justification quote '{proposal.justification_quote}' "
                "was not verified in brief or documents."
            )
            continue
        ok, reason = check_tool_precondition(name, requirements, digest)
        if not ok:
            removed_tools[name] = f"Rejected at generation: {reason}"
            continue
        attached_tools.append(name)
        tool_justifications[name] = proposal.justification_quote.strip()

    system_prompt = assemble_system_prompt(draft, requirements, groundability, attached_tools)
    retrieval_cfg = choose_initial_retrieval_params(digest)

    agent_def = AgentDefinition(
        name=requirements.agent_name,
        version=0,
        system_prompt=system_prompt,
        retrieval=retrieval_cfg,
        tools=attached_tools,
        tool_justifications=tool_justifications,
        removed_tools=removed_tools,
        model=ModelConfig(role="agent", temperature=0.1, max_tokens=600),
        output_contract=OutputContractConfig(),
        refusal_policy=RefusalPolicy(
            refuse_topics=list(requirements.must_refuse),
            refusal_message=(
                "I cannot answer that request because it falls outside the provided "
                "documents and my allowed scope."
            ),
        ),
        source_documents=[d.file_name for d in digest.documents],
    )

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        yaml_text = yaml.safe_dump(
            agent_def.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        (run_dir / "04_agent_v0.yaml").write_text(yaml_text, encoding="utf-8")

    return agent_def
