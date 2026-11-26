# ForgeAI — The Agent That Builds Agents (Zero-Cost Stack)

ForgeAI is a nine-stage automated agent factory built on a zero-cost stack. Given a one-paragraph `brief.txt` and a `docs/` folder (`PDF`, `CSV`, `XLSX`, `TXT`, `MD`, `DOCX`), it interviews for missing requirements, checks whether the documents actually support the brief, generates a declarative YAML agent with strictly justified tools, builds a quality-gated evaluation suite (`dev` + `heldout`), iteratively patches failure modes with a revert-on-regression guard, and outputs an honest scorecard and final report plus a generic chat runner.

---

## Project Status at a Glance

| Area | Status | Details |
| :--- | :--- | :--- |
| **9-Stage Pipeline & Tools** | Done | All modules, schemas, deterministic guards, and 4 tools implemented |
| **Committed Prompts & Calibration** | Done | All 11 `prompts/*.md` files + 15-sample `calibration/judge_calibration_15.json` |
| **Demo Input Datasets** | Done | Demo A (`hr_policy/` 3 PDFs + over-ambitious brief) & Demo B (`ship_spares/` 200-row CSV + 2 PDFs) |
| **CLI, `.venv`, & Streamlit UI** | Done | `.venv\Scripts\forge.exe` (`check`, `calibrate`, `build`, `chat`, `eval`, `regress`) & `app/streamlit_app.py` |
| **Offline Automated Tests** | Done | **20/20 tests passing** via `.\.venv\Scripts\python.exe -m pytest -v` |
| **Free API Keys (`.env`)** | **To Do** | Add `GROQ_API_KEY` and `GEMINI_API_KEY` (or start Ollama) |
| **Live Demo Runs (`runs/`)** | **To Do** | Run `.\scripts\run_all_demos.ps1` once `.env` keys are added |
| **Unseen-Brief Timed Dry Run** | **To Do** | Run one `--auto --no-cache` build on an unseen brief under ~15–20 min |

---

## What Has Been Built (Completed Work)

### 1. Provider Abstraction & Core Schemas
- [`forgeai/llm.py`](forgeai/llm.py): `LLMClient` supporting `agent`, `factory`, and `judge` roles mapped in [`config.yaml`](config.yaml):
  - Provider fallback chain: **Groq → Google AI Studio (Gemini) → OpenRouter (`:free`) → local Ollama**.
  - Enforces a **different model family** between `judge` and `agent` to prevent self-preference bias.
  - Exponential backoff retries on HTTP `429`/`5xx`, per-provider token-bucket rate limiting, SHA-256 disk caching (disableable via `--no-cache` or `FORGEAI_NO_CACHE=1`), token/latency/budget counters, and Pydantic JSON validation with up to 2 automatic repair retries.
- [`forgeai/schemas.py`](forgeai/schemas.py): Pydantic models for every pipeline artifact (`DocumentDigest`, `RequirementsDoc`, `GroundabilityReport`, `AgentDefinition`, `AgentResponse`, `EvalSet`, `JudgeResult`, `PatchOperation`, `IterationLogEntry`, `SplitMetrics`).

### 2. Document Ingestion, Hybrid Retrieval, & Tool Registry
- [`forgeai/ingest.py`](forgeai/ingest.py): Parses PDFs (`pypdf` with `pdfplumber` table fallback), CSV/XLSX (multi-encoding fallback: UTF-8, UTF-8-sig, cp1252, latin-1), TXT, MD, and DOCX. Flags empty/scanned PDFs, caps oversized files (`MAX_CHUNKS_PER_DOC = 400`), produces `DocumentDigest`, and chunks recursively by headings/paragraphs/sentences with stable `doc#index` IDs.
- [`forgeai/retrieval.py`](forgeai/retrieval.py): `HybridIndex` combining dense cosine similarity (FAISS/NumPy with `sentence-transformers` `all-MiniLM-L6-v2` → Ollama `nomic-embed-text` → offline `hashed_ngrams` fallback) and BM25 (`rank_bm25` + small-corpus smoothed `SimpleBM25`) using `score = alpha * dense_norm + (1 - alpha) * bm25_norm`. Caches embeddings on disk by document hash and chunk parameters.
- [`forgeai/tools/`](forgeai/tools/):
  - [`registry.py`](forgeai/tools/registry.py): Tool specs and deterministic precondition validator (`check_tool_precondition`).
  - [`csv_lookup.py`](forgeai/tools/csv_lookup.py): Filters/searches CSV/XLSX tables by column/value or free-text query.
  - [`calculator.py`](forgeai/tools/calculator.py): Safe arithmetic evaluator using Python `ast` (never `eval`).
  - [`http_weather.py`](forgeai/tools/http_weather.py): Keyless Open-Meteo geocoding and forecast lookup.
  - [`current_date.py`](forgeai/tools/current_date.py): Current UTC date and weekday lookup.

### 3. Nine-Stage Factory Pipeline & Committed Prompts
All generation, judge, and patch prompts are committed under [`prompts/`](prompts/):
- **Stage 1 — Intake Interviewer** ([`forgeai/intake.py`](forgeai/intake.py), `intake_questions.md`, `requirements_writer.md`): Asks at most 5 clarifying questions, deduplicates questions already answered in the brief, supports `--auto` default assumptions and interactive `--human-checkpoint`, and outputs `01_intake_dialogue.json` + `02_requirements.json`.
- **Stage 2 — Groundability Guard** ([`forgeai/guard.py`](forgeai/guard.py), `guard_claims.md`, `guard_entailment.md`): Decomposes the brief into atomic capability claims, checks retrieval similarity + LLM entailment, overrides optimistic verdicts if similarity is below threshold or quoted evidence is not verbatim in retrieved chunks, injects unsupported claims into `must_refuse`, and halts cleanly if >50% of claims are unsupported (outputting `03_groundability_report.json` and a stop report).
- **Stage 3 — Agent Generator** ([`forgeai/generator.py`](forgeai/generator.py), `agent_generator.md`): Proposes tools with verbatim justification quotes, filters tools through deterministic precondition checks, picks heuristic retrieval parameters, and assembles `04_agent_v0.yaml` using a mandatory 6-section system prompt template.
- **Stage 4 — Eval Generator** ([`forgeai/evalgen.py`](forgeai/evalgen.py), `eval_answerable.md`, `eval_edge.md`, `eval_oos.md`): Generates answerable (`single_doc`), edge (`multi_doc`, `ambiguous`, `false_premise`), and out-of-scope (`adjacent_unsupported`, `off_topic`/`must_refuse`) cases enforced by 4 quality gates (key-fact verbatim check, single-chunk answerability check, out-of-scope unanswerability check, and cosine `> 0.90` deduplication), stratified into 70% `dev` / 30% `heldout` in `05_eval_set.json` with `gold_evidence_texts` so re-chunking never breaks retrieval-hit tracking.
- **Stage 5 — Hybrid Judge** ([`forgeai/judge.py`](forgeai/judge.py), `judge_grounded.md`): Evaluates deterministic gates first (`status_correct`, `retrieval_hit`, `fact_recall` with number-word normalization + fuzzy fallback, `citation_valid`) and uses the cross-family LLM judge strictly for claim-level groundedness (`YES` verdicts overridden to `NO` unless supporting sentence exists verbatim in context). Includes [`calibration/judge_calibration_15.json`](calibration/judge_calibration_15.json) for judge-vs-human agreement reporting.
- **Stage 6 — Critic-Optimizer Loop** ([`forgeai/critic.py`](forgeai/critic.py), [`forgeai/optimizer.py`](forgeai/optimizer.py), [`forgeai/loop.py`](forgeai/loop.py), `critic_classify.md`, `patch_prompt.md`): Classifies failures (`HALLUCINATION`, `RETRIEVAL_MISS`, `WRONG_REFUSAL`, `PROMPT_DEFECT`), applies single targeted JSON patches on `agent.yaml` (up to 3 iterations), runs Layer 3 tool ablation, accepts patches only on strict `dev` improvement without refusal regression, and logs accepted/reverted patches to `06_iteration_log.json` and `agent_final.yaml`.
- **Stage 7 — Honest Reporting** ([`forgeai/report.py`](forgeai/report.py)): Generates `07_scorecard.md` and `08_final_report.md` with per-iteration `dev`/`heldout` metrics, 95% Wilson confidence intervals, overfitting warnings, remaining failure root causes, calibration agreement, sanity baseline discrimination scores (`no_retrieval`, `echo_context`, `always_refuse`, `always_answer`), and a "Do Not Use This Agent For" section.
- **Stage 8 — Generic Runner, CLI, & Optional UI** ([`forgeai/runner.py`](forgeai/runner.py), [`forgeai/cli.py`](forgeai/cli.py), [`app/streamlit_app.py`](app/streamlit_app.py)): Executes any `agent_final.yaml` with JSON output-contract repair and up to 3 JSON tool-call steps (`{"tool": "...", "args": {...}}`), keeping the last 4 conversation turns.

### 4. Demo Input Datasets & Automated Tests
- **Demo A (HR Policy Assistant, no tools)**: [`demos/hr_policy/brief.txt`](demos/hr_policy/brief.txt), [`demos/hr_policy/brief_overambitious.txt`](demos/hr_policy/brief_overambitious.txt), and 3 synthesized PDFs in `demos/hr_policy/docs/`.
- **Demo B (Ship-Spares Procurement Assistant, `csv_lookup` + `calculator`)**: [`demos/ship_spares/brief.txt`](demos/ship_spares/brief.txt), 200-row [`spares_catalogue.csv`](demos/ship_spares/docs/spares_catalogue.csv), and 2 supplier terms PDFs in `demos/ship_spares/docs/`.
- **Demo Synthesis Prompts**: Documented in [`demos/SYNTHESIS_PROMPTS.md`](demos/SYNTHESIS_PROMPTS.md) and generated via [`demos/generate_demo_assets.py`](demos/generate_demo_assets.py).
- **Test Suite (`tests/`)**: 20 passing unit & mocked pipeline tests across [`test_schemas_and_llm.py`](tests/test_schemas_and_llm.py), [`test_ingest_and_retrieval.py`](tests/test_ingest_and_retrieval.py), [`test_tools.py`](tests/test_tools.py), [`test_stages_pipeline.py`](tests/test_stages_pipeline.py), [`test_demos_and_cli.py`](tests/test_demos_and_cli.py), and [`test_unseen_domains.py`](tests/test_unseen_domains.py) (5 unseen domains: IT helpdesk, recipe book, product manual, NDA legal FAQ, weather logistics).

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the Mermaid pipeline diagram and design decisions.

```text
brief.txt + docs/
  -> [1] Intake Interviewer (<=5 Qs, --auto defaults) -> 02_requirements.json
  -> [2] Groundability Guard (claims + similarity + verbatim entailment) -> 03_groundability_report.json
  -> [3] Agent Generator (template prompt + 2-layer tool validation) -> 04_agent_v0.yaml
  -> [4] Eval Generator (30 cases, 4 quality gates, 70/30 dev/heldout) -> 05_eval_set.json
  -> [5] Hybrid Judge (deterministic checks + cross-family claim groundedness)
  -> [6] Critic-Optimizer (targeted patches, max 3 iters, revert guard) -> 06_iteration_log.json + agent_final.yaml
  -> [7] Honest Report & Scorecard -> 07_scorecard.md + 08_final_report.md
  -> [8] Generic Runner -> forge chat runs/<name>/agent_final.yaml
```

---

## Setup & Free API Keys

A project virtual environment (`.venv`) is already set up in this workspace with `forgeai` installed. If setting up fresh on another machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Get Free API Keys (No Credit Card Required)
1. **Groq (Agent model — `llama-3.1-8b-instant`)**: Sign up at <https://console.groq.com>, create an API key, and set `GROQ_API_KEY`.
2. **Google AI Studio (Factory & Judge model — `gemini-2.0-flash`)**: Sign up at <https://aistudio.google.com>, create an API key, and set `GEMINI_API_KEY`.
3. **Optional OpenRouter (`:free` models)**: Create a key at <https://openrouter.ai> and set `OPENROUTER_API_KEY`.
4. **Optional Offline Fallback (Ollama)**: Install Ollama and run `ollama pull llama3.2:3b` and `ollama pull qwen2.5:3b`.

---

## What Needs To Be Done Before Submission

Follow these exact steps in PowerShell from the repository root:

### Step 1: Configure `.env` and Verify Providers
```powershell
Copy-Item .env.example .env
# Open .env and paste GROQ_API_KEY and GEMINI_API_KEY (never commit .env)
.\.venv\Scripts\forge.exe check
```

### Step 2: Generate Both Required Demo Run Folders (`runs/`)
Run the automated script to build Demo A (`hr_policy`), Demo A's over-ambitious variant (`hr_overambitious`), and Demo B (`ship_spares`):
```powershell
.\scripts\run_all_demos.ps1
```
*(Equivalent manual commands:)*
```powershell
.\.venv\Scripts\forge.exe build --brief demos/hr_policy/brief.txt --docs demos/hr_policy/docs --run-name hr_policy --auto
.\.venv\Scripts\forge.exe build --brief demos/hr_policy/brief_overambitious.txt --docs demos/hr_policy/docs --run-name hr_overambitious --auto
.\.venv\Scripts\forge.exe build --brief demos/ship_spares/brief.txt --docs demos/ship_spares/docs --run-name ship_spares --auto
```

### Step 3: Inspect the Generated Artifacts
Verify that `runs/hr_policy/` and `runs/ship_spares/` each contain:
- `01_intake_dialogue.json`, `02_requirements.json`, `03_groundability_report.json`
- `04_agent_v0.yaml`, `05_eval_set.json`, `iter_0/`, `iter_1/`...
- `06_iteration_log.json`, `07_scorecard.md`, `08_final_report.md`, `agent_final.yaml`

Check specifically that:
- `hr_policy` attaches **0 tools**, while `hr_overambitious` triggers the groundability guard.
- `ship_spares` attaches `csv_lookup` and `calculator` (and **not** `http_weather`).
- `08_final_report.md` includes the **judge-versus-human calibration agreement**, **sanity baseline table**, and **remaining failure causes**.

### Step 4: Test Interactive Chat & Unseen-Brief Robustness
```powershell
# Interactive chat with a generated agent
.\.venv\Scripts\forge.exe chat runs/hr_policy/agent_final.yaml
.\.venv\Scripts\forge.exe chat runs/ship_spares/agent_final.yaml

# Timed unseen-brief check (disables LLM response cache)
.\.venv\Scripts\forge.exe build --brief path\to\unseen_brief.txt --docs path\to\unseen_docs --run-name unseen_test --auto --no-cache

# Optional Streamlit UI
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py

# Re-verify unit tests before committing
.\.venv\Scripts\python.exe -m pytest -v
```

### Step 5: Commit Everything Except `.env` and Caches
Confirm `.gitignore` excludes `.env` and `.forgeai_cache/`, then commit `forgeai/`, `prompts/`, `calibration/`, `demos/`, `runs/`, `tests/`, `docs/`, `app/`, `scripts/`, `config.yaml`, and `README.md`.

---

## CLI Reference

| Command | Purpose |
| :--- | :--- |
| `forge check` | Check configured API keys, Ollama availability, and embedding backend |
| `forge calibrate` | Run judge-vs-human groundedness check over `calibration/judge_calibration_15.json` |
| `forge build --brief <file> --docs <dir> [--auto] [--no-cache]` | Run full 9-stage agent factory |
| `forge chat <agent_final.yaml> [--docs <dir>]` | Chat interactively with any generated agent (last 4 turns kept) |
| `forge eval <agent.yaml> --eval-set <05_eval_set.json>` | Score an agent YAML on an eval set |
| `forge regress --old-evals <json> --docs <new_docs> --agent <yaml>` | Re-run existing evals against updated documents |

---

## Known Limitations
- **Synthetic eval sets**: Evaluation cases are generated by an LLM from the source documents and may not reflect every phrasing variation of live users.
- **Small eval sample size**: A 30-case split (~21 `dev` / ~9 `heldout`) carries wide Wilson confidence intervals.
- **Free-tier rate limits**: Running uncached builds across 3 optimization iterations depends on Groq/Gemini free-tier RPM and daily token quotas; `config.yaml` enforces pacing and multi-provider fallback.
- **JSON tool-call protocol**: Uses an explicit JSON tool message (`{"tool": ..., "args": {...}}`) instead of native provider function-calling so small open models behave consistently across providers.
