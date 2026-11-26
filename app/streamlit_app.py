"""Optional Streamlit UI (Stretch Goal #4): Build and Chat tabs."""

from __future__ import annotations

import io
import os
from pathlib import Path
import zipfile

import streamlit as st

from forgeai.critic import classify_all_failures
from forgeai.evalgen import generate_eval_set
from forgeai.generator import generate_agent
from forgeai.guard import run_groundability_guard
from forgeai.ingest import compute_docs_hash, ingest_documents
from forgeai.intake import run_intake
from forgeai.judge import judge_all
from forgeai.llm import LLMClient, load_config
from forgeai.loop import run_optimization_loop
from forgeai.report import generate_final_report
from forgeai.retrieval import EmbeddingBackend, HybridIndex
from forgeai.runner import AgentRunner, run_sanity_baselines

st.set_page_config(page_title="ForgeAI — The Agent That Builds Agents", layout="wide")

# Load Streamlit Cloud / HF Spaces secrets into env if configured
try:
    for key_name in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL"):
        if key_name in st.secrets and not os.getenv(key_name):
            os.environ[key_name] = str(st.secrets[key_name])
except Exception:
    pass


def _zip_run_folder(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(run_dir))
    return buf.getvalue()


st.title("ForgeAI: The Agent That Builds Agents")

with st.sidebar:
    st.header("Provider Status")
    cfg = load_config()
    probe_client = LLMClient(config=cfg)
    providers = probe_client.check_providers()
    for pname, ready in providers.items():
        st.write(f"- `{pname}`: {'READY' if ready else 'not configured'}")
    if not any(providers.values()):
        st.warning(
            "No LLM keys detected. Add `GROQ_API_KEY` and `GEMINI_API_KEY` to `.env` "
            "(or Streamlit secrets) to build or chat."
        )

tab_build, tab_chat = st.tabs(["Build Agent", "Chat with Agent"])

with tab_build:
    st.subheader("Upload Documents & Enter Agent Brief")
    brief_input = st.text_area(
        "One-paragraph brief",
        value="I need an assistant that answers HR policy questions for shore staff.",
        height=100,
    )
    uploaded_files = st.file_uploader(
        "Upload source documents (PDF, CSV, XLSX, TXT, MD — max 10 files)",
        accept_multiple_files=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        run_name = st.text_input("Run name", value="streamlit_run")
    with col2:
        eval_count = st.number_input("Eval cases", min_value=20, max_value=40, value=24, step=2)
    with col3:
        max_iters = st.number_input("Max optimization iterations", min_value=1, max_value=3, value=3)

    if st.button("Build Agent (--auto)") and brief_input.strip() and uploaded_files:
        if not any(providers.values()):
            st.error("Configure at least one LLM provider key in `.env` before building.")
        elif len(uploaded_files) > 10:
            st.error("Upload limit is 10 files to protect free-tier quota.")
        else:
            run_dir = Path("runs") / run_name
            docs_dir = run_dir / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "brief.txt").write_text(brief_input.strip() + "\n", encoding="utf-8")
            for uf in uploaded_files:
                (docs_dir / uf.name).write_bytes(uf.getvalue())

            llm = LLMClient(config=cfg)
            embedder = EmbeddingBackend()
            with st.status("Running 9-stage ForgeAI pipeline...", expanded=True) as status:
                st.write("1. Ingesting documents...")
                digest, chunks = ingest_documents(docs_dir)
                doc_hash = compute_docs_hash(docs_dir)
                index = HybridIndex(chunks, embedder=embedder, doc_hash=doc_hash, chunk_params_key="500_80")

                st.write("2. Stage 1 Intake...")
                _, req = run_intake(brief_input, digest, llm, auto=True, run_dir=run_dir)

                st.write("3. Stage 2 Groundability Guard...")
                gb, req = run_groundability_guard(brief_input, req, index, llm, run_dir=run_dir)
                if not gb.feasible:
                    status.update(label="Stopped by Groundability Guard", state="error")
                    st.error(gb.stop_reason)
                else:
                    st.write("4. Stage 3 Agent Generator...")
                    v0 = generate_agent(brief_input, req, gb, digest, llm, run_dir=run_dir)

                    st.write("5. Stage 4 Eval Generator...")
                    ev = generate_eval_set(
                        req, chunks, index, llm, target_total=int(eval_count), run_dir=run_dir
                    )

                    st.write("6. Stage 5 & 6 Critic-Optimizer Loop...")
                    final_agent, logs, scorecard = run_optimization_loop(
                        v0,
                        ev,
                        docs_dir,
                        llm,
                        embedder,
                        doc_hash,
                        max_iterations=int(max_iters),
                        run_dir=run_dir,
                    )
                    _, f_chunks = ingest_documents(
                        docs_dir,
                        chunk_size=final_agent.retrieval.chunk_size,
                        chunk_overlap=final_agent.retrieval.chunk_overlap,
                        source_documents=final_agent.source_documents or None,
                    )
                    f_index = HybridIndex(
                        f_chunks,
                        embedder=embedder,
                        doc_hash=doc_hash,
                        chunk_params_key=f"{final_agent.retrieval.chunk_size}_{final_agent.retrieval.chunk_overlap}",
                    )
                    final_runner = AgentRunner(final_agent, f_index, docs_dir, llm)
                    f_res = [final_runner.run_case(c) for c in ev.cases]
                    f_jud = judge_all(ev.cases, f_res, llm)
                    f_jud, _, _ = classify_all_failures(ev.cases, f_res, f_jud)
                    baselines = run_sanity_baselines(ev, f_index, llm)
                    _, report_md = generate_final_report(
                        brief_input,
                        req,
                        gb,
                        final_agent,
                        ev.cases,
                        f_jud,
                        logs,
                        scorecard,
                        sanity_baselines=baselines,
                        total_tokens=llm.total_tokens,
                        total_latency_s=llm.total_latency_s,
                        run_dir=run_dir,
                    )
                    status.update(label="Build complete!", state="complete")
                    st.download_button(
                        "Download All Run Artifacts (ZIP)",
                        data=_zip_run_folder(run_dir),
                        file_name=f"{run_name}_artifacts.zip",
                        mime="application/zip",
                    )
                    st.markdown(report_md)

with tab_chat:
    st.subheader("Run Any Declarative Agent Definition")
    discovered_yamls = sorted(str(p) for p in Path("runs").glob("*/agent_final.yaml")) if Path("runs").exists() else []
    selected_yaml = (
        st.selectbox("Discovered runs", options=discovered_yamls)
        if discovered_yamls
        else ""
    )
    yaml_path = st.text_input(
        "Path to agent_final.yaml",
        value=selected_yaml or "runs/hr_policy/agent_final.yaml",
    )
    docs_path = st.text_input("Path to docs folder (optional)", value="")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    q = st.chat_input("Ask your generated agent a question...")
    if q:
        if not Path(yaml_path).exists():
            st.error(f"Agent YAML not found: {yaml_path}")
        else:
            with st.chat_message("user"):
                st.write(q)
            runner = AgentRunner.from_yaml(yaml_path, docs_dir=docs_path or None)
            res = runner.run_turn(q, history=st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": q})
            st.session_state.chat_history.append({"role": "assistant", "content": res.response.answer})
            with st.chat_message("assistant"):
                st.caption(
                    f"Status: `{res.response.status}` | Confidence: `{res.response.confidence}`"
                    + (f" | Tools: `{', '.join(res.tool_calls_made)}`" if res.tool_calls_made else "")
                )
                st.write(res.response.answer)
                if res.response.sources:
                    with st.expander("Cited Sources"):
                        st.json([s.model_dump() for s in res.response.sources])
