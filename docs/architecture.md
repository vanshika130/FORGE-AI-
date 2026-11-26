# ForgeAI Architecture & Design Decisions

## Pipeline Overview

```mermaid
flowchart TD
    A["brief.txt + docs/"] --> B["Stage 1: Intake Interviewer (<=5 Qs, --auto defaults)"]
    B --> C["02_requirements.json"]
    C --> D["Stage 2: Groundability Guard (claims + similarity + verbatim entailment)"]
    D -->|">50% unsupported"| STOP["Halt & Report Document Gaps"]
    D -->|"Feasible"| E["Stage 3: Agent Generator + Tool Precondition Validator"]
    E --> F["04_agent_v0.yaml"]
    F --> G["Stage 4: Eval Generator (30 cases, 4 quality gates, 70/30 dev/heldout)"]
    G --> H["Stage 8: Generic Runner"]
    H --> I["Stage 5: Hybrid Judge (deterministic gates + bias-defended LLM groundedness)"]
    I --> J["Stage 6: Critic + Targeted Patch Optimizer (max 3 iters, revert guard)"]
    J -->|"Improved & no refusal regression"| H
    J --> K["Stage 7: Honest Scorecard & Final Report + agent_final.yaml"]
```

## Key Design Decisions
1. **Untrusted LLM Outputs**: Every LLM call in Intake, Guard, Generator, EvalGen, Runner, and Judge passes through Pydantic schema validation with up to 2 JSON repair retries and deterministic cross-checks.
2. **Bias-Defended Judge**:
   - **Self-preference bias**: Judge model (`gemini` / `mistral` / `qwen`) is enforced to use a different model family from the agent (`llama`).
   - **Position bias**: Pointwise case scoring only (no pairwise A/B order bias).
   - **Verbosity bias**: Claim-level verification ignores prose length; longer answers are never rewarded.
   - **Judge hallucination**: Every `YES` claim verdict must include a verbatim supporting sentence verified in `retrieved_context`, or it is deterministically overridden to `NO`.
3. **Re-chunking Safe Retrieval Eval**: `05_eval_set.json` stores `key_facts` and `gold_evidence_texts` alongside `gold_chunk_ids`, so `retrieval_hit` remains valid even when the optimizer modifies `chunk_size` or `chunk_overlap`.
4. **Targeted Patching (No Regeneration)**: Each optimizer step modifies a single dotted field or prompt snippet via `PatchOperation`, accepts only on strict `dev` pass-rate improvement without refusal regression, and logs reverted patches transparently.
