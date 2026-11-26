# ForgeAI CLI: forge build / chat / eval / regress / calibrate / check

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Optional

from forgeai.critic import classify_all_failures
from forgeai.evalgen import generate_eval_set
from forgeai.generator import generate_agent
from forgeai.guard import run_groundability_guard
from forgeai.ingest import compute_docs_hash, ingest_documents
from forgeai.intake import run_intake
from forgeai.judge import calibrate_judge_agreement, compute_split_metrics, judge_all
from forgeai.llm import BudgetExceededError, LLMClient, LLMError, load_config
from forgeai.loop import run_optimization_loop
from forgeai.report import generate_final_report
from forgeai.retrieval import EmbeddingBackend, HybridIndex
from forgeai.runner import AgentRunner, run_sanity_baselines
from forgeai.schemas import EvalSet


def _echo(msg: str, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    print(msg, file=stream)


def _run_calibration_if_available(
    llm: LLMClient, samples_path: Path | None = None
) -> Optional[dict]:
    path = samples_path or Path("calibration/judge_calibration_15.json")
    if not path.exists():
        return None
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
        tuples = [
            (item["answer"], item["context"], bool(item["human_grounded"]))
            for item in items
        ]
        return calibrate_judge_agreement(tuples, llm)
    except Exception:
        return None


def cmd_check() -> int:
    """Verify configured API keys and local provider/embedding availability."""
    llm = LLMClient()
    status = llm.check_providers()
    embedder = EmbeddingBackend()
    _echo("Provider status:")
    for name, ok in status.items():
        _echo(f"  - {name}: {'READY' if ok else 'not configured / unreachable'}")
    _echo(f"Embedding backend: {embedder.backend_name}")
    return 0


def cmd_calibrate(samples: Path = Path("calibration/judge_calibration_15.json")) -> int:
    """Run judge-vs-human groundedness calibration over 15 hand-labeled samples."""
    if not samples.exists():
        _echo(f"ERROR: Calibration file not found: {samples}", err=True)
        return 1
    llm = LLMClient()
    if not any(llm.check_providers().values()):
        _echo("ERROR: No LLM providers configured for calibration.", err=True)
        return 1
    stats = _run_calibration_if_available(llm, samples)
    _echo(json.dumps(stats, indent=2))
    return 0


def cmd_build(
    brief: Path,
    docs: Path,
    run_name: Optional[str] = None,
    auto: bool = False,
    no_cache: bool = False,
    max_iters: int = 3,
    eval_count: int = 30,
    human_checkpoint: bool = False,
    with_baselines: bool = True,
) -> int:
    """Run the full 9-stage ForgeAI factory pipeline on a brief and docs folder."""
    if not brief.exists():
        _echo(f"ERROR: Brief file not found: {brief}", err=True)
        return 1
    if not docs.is_dir():
        _echo(f"ERROR: Docs directory not found: {docs}", err=True)
        return 1

    cfg = load_config()
    budgets = cfg.get("budgets", {})
    slug = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    brief_text = brief.read_text(encoding="utf-8").strip()
    (run_dir / "brief.txt").write_text(brief_text + "\n", encoding="utf-8")

    run_docs_dir = run_dir / "docs"
    if docs.resolve() != run_docs_dir.resolve():
        if run_docs_dir.exists():
            shutil.rmtree(run_docs_dir)
        shutil.copytree(docs, run_docs_dir)

    llm = LLMClient(config=cfg, use_cache=(False if no_cache else None))
    providers = llm.check_providers()
    if not any(providers.values()):
        _echo(
            "ERROR: No LLM providers configured. Set GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, or start Ollama.",
            err=True,
        )
        return 1

    embedder = EmbeddingBackend(
        model_name=cfg.get("embeddings", {}).get("model", "all-MiniLM-L6-v2"),
        ollama_model=cfg.get("embeddings", {}).get("ollama_fallback_model", "nomic-embed-text"),
    )

    try:
        _echo(f"[1/7] Ingesting documents from {run_docs_dir}...")
        digest, chunks = ingest_documents(run_docs_dir)
        if not digest.documents:
            _echo("ERROR: No supported documents found in docs folder.", err=True)
            return 1
        for w in digest.warnings:
            _echo(f"  Warning: {w}")
        doc_hash = compute_docs_hash(run_docs_dir)
        index = HybridIndex(chunks, embedder=embedder, doc_hash=doc_hash, chunk_params_key="500_80")

        _echo("[2/7] Stage 1: Intake Interviewer...")
        _, requirements = run_intake(
            brief=brief_text,
            digest=digest,
            llm=llm,
            auto=auto,
            run_dir=run_dir,
            human_checkpoint=human_checkpoint,
        )

        _echo("[3/7] Stage 2: Groundability Guard...")
        groundability, requirements = run_groundability_guard(
            brief=brief_text,
            requirements=requirements,
            index=index,
            llm=llm,
            min_similarity=float(budgets.get("groundability_min_similarity", 0.22)),
            stop_unsupported_fraction=float(
                budgets.get("groundability_stop_unsupported_fraction", 0.50)
            ),
            run_dir=run_dir,
        )
        if not groundability.feasible:
            (run_dir / "08_final_report.md").write_text(
                f"# ForgeAI Groundability Stop Report: `{requirements.agent_name}`\n\n"
                f"**Brief:** {brief_text}\n\n"
                f"**Result:** STOPPED BY GROUNDABILITY GUARD\n\n"
                f"**Reason:** {groundability.stop_reason}\n\n"
                "## Claims\n"
                + "\n".join(
                    f"- **[{c.status.upper()}]** `{c.ask}` (dense={c.max_similarity:.2f}) -> {c.recommended_action}"
                    for c in groundability.claims
                )
                + "\n",
                encoding="utf-8",
            )
            _echo(f"STOPPED BY GROUNDABILITY GUARD: {groundability.stop_reason}", err=True)
            _echo(f"Partial artifacts and stop report saved to {run_dir}")
            return 2

        _echo("[4/7] Stage 3: Generating agent_v0.yaml...")
        agent_v0 = generate_agent(
            brief=brief_text,
            requirements=requirements,
            groundability=groundability,
            digest=digest,
            llm=llm,
            run_dir=run_dir,
        )
        _echo(f"  Attached tools: {agent_v0.tools or 'None'}")

        _echo(f"[5/7] Stage 4: Generating {eval_count}-case quality-gated eval set...")
        eval_set = generate_eval_set(
            requirements=requirements,
            chunks=chunks,
            index=index,
            llm=llm,
            target_total=eval_count,
            dev_ratio=float(budgets.get("dev_split_ratio", 0.7)),
            run_dir=run_dir,
        )
        _echo(f"  Generated {len(eval_set.cases)} eval cases.")

        _echo("[6/7] Stage 5 & 6: Running critic-optimizer loop...")
        final_agent, iter_logs, scorecard_rows = run_optimization_loop(
            initial_agent=agent_v0,
            eval_set=eval_set,
            docs_dir=run_docs_dir,
            llm=llm,
            embedder=embedder,
            doc_hash=doc_hash,
            max_iterations=max_iters,
            max_prompt_lines=int(budgets.get("max_prompt_patch_lines", 12)),
            run_dir=run_dir,
        )

        _, final_chunks = ingest_documents(
            run_docs_dir,
            chunk_size=final_agent.retrieval.chunk_size,
            chunk_overlap=final_agent.retrieval.chunk_overlap,
            source_documents=final_agent.source_documents or None,
        )
        final_index = HybridIndex(
            final_chunks,
            embedder=embedder,
            doc_hash=doc_hash,
            chunk_params_key=f"{final_agent.retrieval.chunk_size}_{final_agent.retrieval.chunk_overlap}",
        )
        runner = AgentRunner(agent=final_agent, index=final_index, docs_dir=run_docs_dir, llm=llm)
        final_results = [runner.run_case(c) for c in eval_set.cases]
        final_judgements = judge_all(eval_set.cases, final_results, llm)
        final_judgements, _, _ = classify_all_failures(
            eval_set.cases, final_results, final_judgements
        )

        baselines = run_sanity_baselines(eval_set, final_index, llm) if with_baselines else None
        calibration_stats = _run_calibration_if_available(llm)

        _echo("[7/7] Stage 7: Writing scorecard & honest final report...")
        generate_final_report(
            brief=brief_text,
            requirements=requirements,
            groundability=groundability,
            final_agent=final_agent,
            eval_cases=eval_set.cases,
            final_judgements=final_judgements,
            iteration_logs=iter_logs,
            scorecard_rows=scorecard_rows,
            sanity_baselines=baselines,
            calibration_stats=calibration_stats,
            total_tokens=llm.total_tokens,
            total_latency_s=llm.total_latency_s,
            run_dir=run_dir,
        )
        _echo(f"Done! Artifacts saved in {run_dir}")
        _echo(f"Chat with your agent: forge chat {run_dir / 'agent_final.yaml'}")
        return 0
    except (LLMError, BudgetExceededError) as exc:
        _echo(f"Pipeline halted cleanly with partial artifacts in {run_dir}: {exc}", err=True)
        return 1
    except Exception as exc:
        _echo(f"Unexpected error (partial artifacts in {run_dir}): {exc}", err=True)
        return 1


def cmd_chat(agent_yaml: Path, docs: Optional[Path] = None) -> int:
    """Interactive CLI chat runner loading any declarative agent YAML."""
    runner = AgentRunner.from_yaml(agent_yaml, docs_dir=docs)
    _echo(
        f"Loaded agent '{runner.agent.name}' (v{runner.agent.version}). "
        f"Tools: {runner.agent.tools or 'None'}"
    )
    _echo("Type 'exit' or 'quit' to stop.\n")

    history: list[dict[str, str]] = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        res = runner.run_turn(user_input, history=history)
        _echo(
            f"\n[{res.response.status.upper()} | confidence={res.response.confidence}] "
            f"{res.response.answer}"
        )
        if res.response.sources:
            _echo("Sources:")
            for s in res.response.sources:
                _echo(f"  - {s.document}: \"{s.quote}\"")
        if res.tool_calls_made:
            _echo(f"Tools used: {', '.join(res.tool_calls_made)}")
        _echo("")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": res.response.answer})
    return 0


def cmd_eval(agent_yaml: Path, eval_set_path: Path, docs: Optional[Path] = None) -> int:
    """Evaluate a declarative agent YAML against a saved eval set."""
    eval_set = EvalSet.model_validate_json(eval_set_path.read_text(encoding="utf-8"))
    runner = AgentRunner.from_yaml(agent_yaml, docs_dir=docs)
    results = [runner.run_case(c) for c in eval_set.cases]
    judgements = judge_all(eval_set.cases, results, runner.llm)
    dev_m = compute_split_metrics(eval_set.cases, results, judgements, split="dev")
    held_m = compute_split_metrics(eval_set.cases, results, judgements, split="heldout")
    _echo(f"Dev pass rate: {dev_m.pass_rate:.1%} ({dev_m.passed}/{dev_m.total})")
    _echo(f"Held-out pass rate: {held_m.pass_rate:.1%} ({held_m.passed}/{held_m.total})")
    return 0


def cmd_regress(old_evals: Path, docs: Path, agent_yaml: Path) -> int:
    """Stretch goal: rebuild index on new_docs and diff pass/fail against old eval set."""
    eval_set = EvalSet.model_validate_json(old_evals.read_text(encoding="utf-8"))
    runner = AgentRunner.from_yaml(agent_yaml, docs_dir=docs)
    results = [runner.run_case(c) for c in eval_set.cases]
    judgements = judge_all(eval_set.cases, results, runner.llm)
    all_m = compute_split_metrics(eval_set.cases, results, judgements, split=None)
    _echo(f"Regression pass rate on {docs}: {all_m.pass_rate:.1%} ({all_m.passed}/{all_m.total})")
    for j in judgements:
        mark = "PASS" if j.passed else f"FAIL ({j.failure_category})"
        _echo(f"  {j.case_id}: {mark}")
    return 0


try:
    import typer

    app = typer.Typer(
        name="forge",
        help="ForgeAI: The Agent That Builds Agents (Zero-Cost Stack).",
        add_completion=False,
    )

    @app.command("check")
    def _typer_check() -> None:
        raise typer.Exit(code=cmd_check())

    @app.command("calibrate")
    def _typer_calibrate(
        samples: Path = typer.Option(
            Path("calibration/judge_calibration_15.json"), "--samples", exists=True
        ),
    ) -> None:
        raise typer.Exit(code=cmd_calibrate(samples))

    @app.command("build")
    def _typer_build(
        brief: Path = typer.Option(..., "--brief", exists=True),
        docs: Path = typer.Option(..., "--docs", exists=True, file_okay=False),
        run_name: Optional[str] = typer.Option(None, "--run-name"),
        auto: bool = typer.Option(False, "--auto"),
        no_cache: bool = typer.Option(False, "--no-cache"),
        max_iters: int = typer.Option(3, "--max-iters"),
        eval_count: int = typer.Option(30, "--eval-count"),
        human_checkpoint: bool = typer.Option(False, "--human-checkpoint"),
        with_baselines: bool = typer.Option(True, "--with-baselines/--no-baselines"),
    ) -> None:
        raise typer.Exit(
            code=cmd_build(
                brief, docs, run_name, auto, no_cache, max_iters, eval_count, human_checkpoint, with_baselines
            )
        )

    @app.command("chat")
    def _typer_chat(
        agent_yaml: Path = typer.Argument(..., exists=True),
        docs: Optional[Path] = typer.Option(None, "--docs"),
    ) -> None:
        raise typer.Exit(code=cmd_chat(agent_yaml, docs))

    @app.command("eval")
    def _typer_eval(
        agent_yaml: Path = typer.Argument(..., exists=True),
        eval_set_path: Path = typer.Option(..., "--eval-set", exists=True),
        docs: Optional[Path] = typer.Option(None, "--docs"),
    ) -> None:
        raise typer.Exit(code=cmd_eval(agent_yaml, eval_set_path, docs))

    @app.command("regress")
    def _typer_regress(
        old_evals: Path = typer.Option(..., "--old-evals", exists=True),
        docs: Path = typer.Option(..., "--docs", exists=True, file_okay=False),
        agent_yaml: Path = typer.Option(..., "--agent", exists=True),
    ) -> None:
        raise typer.Exit(code=cmd_regress(old_evals, docs, agent_yaml))

except ImportError:

    def app() -> None:
        """Argparse fallback CLI when typer is not installed."""
        parser = argparse.ArgumentParser(prog="forge", description="ForgeAI CLI")
        sub = parser.add_subparsers(dest="command", required=True)
        sub.add_parser("check")

        p_cal = sub.add_parser("calibrate")
        p_cal.add_argument(
            "--samples", type=Path, default=Path("calibration/judge_calibration_15.json")
        )

        p_build = sub.add_parser("build")
        p_build.add_argument("--brief", type=Path, required=True)
        p_build.add_argument("--docs", type=Path, required=True)
        p_build.add_argument("--run-name", type=str, default=None)
        p_build.add_argument("--auto", action="store_true")
        p_build.add_argument("--no-cache", action="store_true")
        p_build.add_argument("--max-iters", type=int, default=3)
        p_build.add_argument("--eval-count", type=int, default=30)
        p_build.add_argument("--human-checkpoint", action="store_true")
        p_build.add_argument("--no-baselines", action="store_true")

        p_chat = sub.add_parser("chat")
        p_chat.add_argument("agent_yaml", type=Path)
        p_chat.add_argument("--docs", type=Path, default=None)

        p_eval = sub.add_parser("eval")
        p_eval.add_argument("agent_yaml", type=Path)
        p_eval.add_argument("--eval-set", type=Path, required=True)
        p_eval.add_argument("--docs", type=Path, default=None)

        p_reg = sub.add_parser("regress")
        p_reg.add_argument("--old-evals", type=Path, required=True)
        p_reg.add_argument("--docs", type=Path, required=True)
        p_reg.add_argument("--agent", type=Path, required=True)

        args = parser.parse_args()
        if args.command == "check":
            raise SystemExit(cmd_check())
        if args.command == "calibrate":
            raise SystemExit(cmd_calibrate(args.samples))
        if args.command == "build":
            raise SystemExit(
                cmd_build(
                    args.brief,
                    args.docs,
                    args.run_name,
                    args.auto,
                    args.no_cache,
                    args.max_iters,
                    args.eval_count,
                    args.human_checkpoint,
                    not args.no_baselines,
                )
            )
        if args.command == "chat":
            raise SystemExit(cmd_chat(args.agent_yaml, args.docs))
        if args.command == "eval":
            raise SystemExit(cmd_eval(args.agent_yaml, args.eval_set, args.docs))
        if args.command == "regress":
            raise SystemExit(cmd_regress(args.old_evals, args.docs, args.agent))


if __name__ == "__main__":
    app()
