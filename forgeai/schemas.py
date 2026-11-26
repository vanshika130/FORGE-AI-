"""Pydantic schemas for every ForgeAI artifact and LLM output."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentInfo(BaseModel):
    """Summary metadata for a single ingested source document."""

    file_name: str
    file_type: str
    size_bytes: int
    num_chars: int
    num_chunks: int = 0
    summary: str
    top_headings: list[str] = Field(default_factory=list)
    csv_columns: list[str] = Field(default_factory=list)
    csv_sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    warning: Optional[str] = None


class DocumentDigest(BaseModel):
    """Digest of all documents in a run's docs folder."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    has_csv: bool = False


class ChunkRecord(BaseModel):
    """A retrievable chunk with stable identifier."""

    doc: str
    page: int = 1
    chunk_id: str
    text: str
    is_csv_row: bool = False


class IntakeQuestion(BaseModel):
    """A single clarifying question generated during Stage 1."""

    question: str
    reason: str
    default_answer: str
    missing_quote_from_brief: str = ""


class IntakeQuestionBatch(BaseModel):
    """LLM wrapper schema for Stage 1 clarifying questions."""

    questions: list[IntakeQuestion] = Field(default_factory=list)


class ClarificationItem(BaseModel):
    """Question-answer pair recorded during intake."""

    q: str
    a: str
    is_default_assumption: bool = False


class GroundingFlag(BaseModel):
    """Groundability assessment for one atomic capability claim from the brief."""

    ask: str
    status: Literal["supported", "partial", "unsupported"]
    evidence: str = ""
    max_similarity: float = 0.0
    max_bm25: float = 0.0
    covered_by_tool: Optional[str] = None
    recommended_action: str = ""


class GroundabilityClaimsBatch(BaseModel):
    """LLM output schema for decomposing a brief into atomic capability claims."""

    claims: list[str] = Field(default_factory=list)


class EntailmentAssessment(BaseModel):
    """LLM output schema for Stage 2 claim entailment check."""

    status: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    evidence: str = ""
    reason: str = ""


class GroundabilityReport(BaseModel):
    """Output artifact 03_groundability_report.json."""

    claims: list[GroundingFlag] = Field(default_factory=list)
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    unsupported_fraction: float = 0.0
    feasible: bool = True
    stop_reason: str = ""


class RequirementsDoc(BaseModel):
    """Output artifact 02_requirements.json."""

    agent_name: str
    scope: str
    audience: str
    tone: str
    language: str = "English"
    escalation_behaviour: str = "Advise the user to contact the appropriate team."
    must_answer: list[str] = Field(default_factory=list)
    must_refuse: list[str] = Field(default_factory=list)
    clarifications: list[ClarificationItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    grounding_flags: list[GroundingFlag] = Field(default_factory=list)
    needs_numeric: bool = False
    needs_external_data: bool = False
    needs_tabular_lookup: bool = False
    needs_current_date: bool = False


class RetrievalConfig(BaseModel):
    """Retrieval parameters stored inside agent YAML."""

    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    hybrid_alpha: float = 0.5
    min_score: float = 0.25
    query_rewrite: bool = False


class ModelConfig(BaseModel):
    """Agent LLM parameters stored inside agent YAML."""

    role: str = "agent"
    temperature: float = 0.1
    max_tokens: int = 600


class OutputContractConfig(BaseModel):
    """Structured response contract for the generated agent."""

    format: str = "json"
    fields: dict[str, str] = Field(
        default_factory=lambda: {
            "status": "answered | refused | needs_clarification",
            "answer": "string",
            "sources": "list of {document, quote}",
            "confidence": "high | medium | low",
        }
    )


class RefusalPolicy(BaseModel):
    """Refusal behaviour configuration for the generated agent."""

    refuse_topics: list[str] = Field(default_factory=list)
    refusal_message: str = (
        "I cannot answer that request because it falls outside my supported documents and scope."
    )


class ToolProposal(BaseModel):
    """A tool proposed by the generator LLM with a verbatim justification quote."""

    name: str
    justification_quote: str
    reason: str = ""


class AgentGenerationDraft(BaseModel):
    """LLM draft output before deterministic validation and template assembly."""

    tool_proposals: list[ToolProposal] = Field(default_factory=list)
    role_and_scope_section: str
    audience_and_tone_section: str
    grounding_rules_section: str
    refusal_rules_section: str
    tool_use_rules_section: str = ""
    worked_example_json: str = ""


class AgentDefinition(BaseModel):
    """Declarative YAML definition for a generated agent."""

    name: str
    version: int = 0
    system_prompt: str
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: list[str] = Field(default_factory=list)
    tool_justifications: dict[str, str] = Field(default_factory=dict)
    removed_tools: dict[str, str] = Field(default_factory=dict)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output_contract: OutputContractConfig = Field(default_factory=OutputContractConfig)
    refusal_policy: RefusalPolicy = Field(default_factory=RefusalPolicy)
    source_documents: list[str] = Field(default_factory=list)


class SourceCitation(BaseModel):
    """A document quote citation returned by the agent."""

    document: str
    quote: str


class AgentResponse(BaseModel):
    """Runtime JSON output contract returned by runner.py."""

    status: Literal["answered", "refused", "needs_clarification"]
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class ToolCallRequest(BaseModel):
    """Simple JSON tool-calling protocol for small open models."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    """Single evaluation case stored in 05_eval_set.json."""

    id: str
    type: Literal["answerable", "out_of_scope", "edge"]
    subtype: Literal[
        "single_doc",
        "multi_doc",
        "ambiguous",
        "false_premise",
        "adjacent_unsupported",
        "off_topic",
        "must_refuse",
    ]
    question: str
    expected_status: Literal["answered", "refused", "needs_clarification"]
    expected_sources: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_evidence_texts: list[str] = Field(default_factory=list)
    split: Literal["dev", "heldout"] = "dev"


class EvalSet(BaseModel):
    """Complete evaluation set artifact 05_eval_set.json."""

    seed: int = 42
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    cases: list[EvalCase] = Field(default_factory=list)


class GeneratedSingleEvalDraft(BaseModel):
    """LLM draft for a single-chunk answerable eval case."""

    question: str
    key_facts: list[str] = Field(default_factory=list)


class GeneratedMultiEvalDraft(BaseModel):
    """LLM draft for a multi-document eval case."""

    question: str
    key_facts: list[str] = Field(default_factory=list)


class GeneratedEdgeEvalDraft(BaseModel):
    """LLM draft for an edge or out-of-scope eval case."""

    question: str
    expected_status: Literal["answered", "refused", "needs_clarification"] = "refused"
    key_facts: list[str] = Field(default_factory=list)


class GeneratedEvalBatch(BaseModel):
    """Batch of edge or out-of-scope eval drafts."""

    items: list[GeneratedEdgeEvalDraft] = Field(default_factory=list)


class AnswerabilityCheckOutput(BaseModel):
    """LLM response when checking if a chunk answers a generated question."""

    answerable: bool
    answer: str = ""


class AtomicClaimsOutput(BaseModel):
    """Atomic claims extracted from an agent answer by the judge."""

    claims: list[str] = Field(default_factory=list)


class ClaimSupportVerdict(BaseModel):
    """LLM judge verdict for one atomic claim against retrieved context."""

    verdict: Literal["YES", "NO"]
    supporting_sentence: str = ""


class ClaimEvaluation(BaseModel):
    """Combined LLM + deterministic claim verification result."""

    claim: str
    llm_verdict: Literal["YES", "NO"]
    supporting_sentence: str = ""
    verified_in_context: bool = False
    supported: bool = False


class RunnerCaseResult(BaseModel):
    """Execution record for a single eval case run through runner.py."""

    case_id: str
    question: str
    response: AgentResponse
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_docs: list[str] = Field(default_factory=list)
    retrieved_context: str = ""
    tool_calls_made: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    error: Optional[str] = None


class JudgeResult(BaseModel):
    """Evaluation verdict for one case."""

    model_config = ConfigDict(populate_by_name=True)

    case_id: str
    case_type: str = ""
    subtype: str = ""
    split: str = "dev"
    retrieval_hit: bool
    status_correct: bool
    fact_recall: float
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    claim_evaluations: list[ClaimEvaluation] = Field(default_factory=list)
    citation_valid: bool
    passed: bool = Field(alias="pass")
    failure_category: str = ""
    root_cause: str = ""


class PatchOperation(BaseModel):
    """Targeted patch applied to agent.yaml."""

    path: str
    old: Any = None
    new: Any = None


class PromptPatchProposal(BaseModel):
    """LLM proposal for a minimal edit to one system prompt section."""

    section_name: str
    old_snippet: str
    new_snippet: str
    rationale: str = ""


class IterationLogEntry(BaseModel):
    """One iteration of the Stage 6 critic-optimizer loop."""

    iteration: int
    failures_by_category: dict[str, int] = Field(default_factory=dict)
    patch: Optional[PatchOperation] = None
    operator_name: str = ""
    rationale: str = ""
    failures_targeted: list[str] = Field(default_factory=list)
    failures_fixed: list[str] = Field(default_factory=list)
    dev_score_before: float
    dev_score_after: float
    heldout_score: float
    accepted: bool
    rejection_reason: str = ""
    tokens: int = 0
    latency: float = 0.0


class SplitMetrics(BaseModel):
    """Aggregated scorecard metrics for a split (dev or heldout)."""

    total: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    answerable_pass_rate: float = 0.0
    out_of_scope_pass_rate: float = 0.0
    edge_pass_rate: float = 0.0
    retrieval_hit_rate: float = 0.0
    groundedness_rate: float = 0.0
    correct_refusal_rate: float = 0.0
    wrongful_refusal_rate: float = 0.0
