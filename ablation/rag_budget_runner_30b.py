"""Run RAG budget ablations on the sampled benchmark topics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from .common import (
    CONSTRUCTION_ROOT,
    DATA_ROOT,
    OUTPUTS_ROOT,
    PREDICTION_ROOT,
    PREDICTION_PY_ROOT,
    YEAR_SLUG,
    import_module_from_path,
    read_json,
    utc_now_tag,
)

ENV_FILES = (
    Path(__file__).resolve().parent / ".env",
    PREDICTION_PY_ROOT / ".env",
)


def _load_runtime_env() -> None:
    """Load ablation and prediction env files before CLI defaults are resolved."""
    for env_path in ENV_FILES:
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_runtime_env()

FAMILY_CONFIG = {
    "qwen3_8b_rag": {
        "module_file": PREDICTION_PY_ROOT / "qwen3_8B_rag.py",
        "module_name": "ablation_qwen3_8b_rag",
        "default_model_path": os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen3-8B"),
    },
    "qwen3_30b_awq_rag": {
        "module_file": PREDICTION_PY_ROOT / "qwen3_30B_rag.py",
        "module_name": "ablation_qwen3_30b_rag",
        "default_model_path": os.getenv("QWEN_MODEL_PATH", "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"),
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RAG budget runner."""
    parser = argparse.ArgumentParser(description="Run or reuse RAG top-k variants for the ablation benchmark.")
    parser.add_argument("--family", choices=sorted(FAMILY_CONFIG), required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 30, 50])
    parser.add_argument("--benchmark-json", type=Path, default=DATA_ROOT / "topic_benchmark_50.json")
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--reuse-existing-k30", action="store_true", default=True)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--vllm-host", default=os.getenv("QWEN_VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--vllm-port", type=int, default=int(os.getenv("QWEN_VLLM_PORT", "6004")))
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-vllm-start", action="store_true")
    parser.add_argument("--s2-max-total", type=int, default=10000)
    parser.add_argument("--s2-page-size", type=int, default=100)
    parser.add_argument(
        "--s2-api-key",
        default=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY", ""),
    )
    parser.add_argument("--s2-max-retries", type=int, default=5)
    parser.add_argument("--s2-retry-backoff", type=float, default=1.5)
    return parser.parse_args()


def load_benchmark_topics(benchmark_json: Path) -> list[dict]:
    """Load the benchmark topic list."""
    payload = read_json(benchmark_json)
    return payload["topics"]


def load_display_context(domain_slug: str, topic_slug: str) -> tuple[str, str]:
    """Load human-readable domain and topic names from GT metadata."""
    meta = read_json(
        CONSTRUCTION_ROOT
        / domain_slug
        / topic_slug
        / "problem"
        / "result_latest.json"
    )["metadata"]
    return meta["domain"], meta["mainframe_topic"]


def copy_existing_baseline(source_model: str, variant_model: str, domain_slug: str, topic_slug: str, direction: str) -> bool:
    """Copy existing baseline outputs into the ablation output tree if available."""
    source_dir = PREDICTION_ROOT / source_model / domain_slug / topic_slug / direction / YEAR_SLUG
    if not source_dir.exists():
        return False
    target_dir = OUTPUTS_ROOT / variant_model / domain_slug / topic_slug / direction / YEAR_SLUG
    target_dir.mkdir(parents=True, exist_ok=True)

    response_text = (source_dir / "response_latest.txt").read_text(encoding="utf-8")
    payload = read_json(source_dir / "signals_latest.json")
    payload["ablation"] = {
        "source": "reused_existing",
        "budget_kind": "rag_top_k",
        "budget_value": 30,
        "variant_model": variant_model,
        "copied_from_model": source_model,
    }
    (target_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")
    (target_dir / f"response_{utc_now_tag()}.txt").write_text(response_text, encoding="utf-8")
    (target_dir / "signals_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (target_dir / f"signals_{utc_now_tag()}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def save_generated_result(
    variant_model: str,
    domain_slug: str,
    topic_slug: str,
    direction: str,
    response_text: str,
    signals: list[str],
    retrieved_metadata: list[dict],
    extra_metadata: dict,
) -> Path:
    """Save a generated prediction result to the ablation output tree."""
    result_dir = OUTPUTS_ROOT / variant_model / domain_slug / topic_slug / direction / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_tag()
    payload = {
        "domain": extra_metadata["domain_display"],
        "space": direction,
        "mainframe_topic": extra_metadata["topic_display"],
        "year_range": extra_metadata["year_range"],
        "timestamp": timestamp,
        "signals": signals,
        "retrieved_papers": retrieved_metadata,
        "ablation": {
            "budget_kind": "rag_top_k",
            "budget_value": extra_metadata["rag_top_k"],
            "variant_model": variant_model,
            "source": "generated",
            "source_family": extra_metadata["source_family"],
        },
    }
    (result_dir / f"response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")
    (result_dir / f"signals_{timestamp}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (result_dir / "signals_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_dir


def run_family(args: argparse.Namespace) -> None:
    """Run all requested top-k variants for one RAG family."""
    cfg = FAMILY_CONFIG[args.family]
    module = import_module_from_path(cfg["module_name"], cfg["module_file"])
    topics = load_benchmark_topics(args.benchmark_json)
    qwen_model = args.qwen_model or cfg["default_model_path"]
    base_url = f"http://{args.vllm_host}:{args.vllm_port}"

    if not args.skip_vllm_start:
        module.start_vllm_server(
            model=qwen_model,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_mem=args.gpu_memory_utilization,
        )

    cutoff_year = module.year_range_to_cutoff(module.YEAR_RANGE)
    for top_k in args.top_k:
        variant_model = f"{args.family}_k{top_k}"
        for item in topics:
            domain_slug = item["domain"]
            topic_slug = item["topic"]
            domain_display, topic_display = load_display_context(domain_slug, topic_slug)
            for direction in ("problem", "solution"):
                result_dir = OUTPUTS_ROOT / variant_model / domain_slug / topic_slug / direction / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Exists: {result_dir}")
                    continue

                if top_k == 30 and args.reuse_existing_k30:
                    reused = copy_existing_baseline(args.family, variant_model, domain_slug, topic_slug, direction)
                    if reused:
                        print(f"[reuse] {variant_model}/{domain_slug}/{topic_slug}/{direction}")
                        continue

                print(f"[run] {variant_model}/{domain_slug}/{topic_slug}/{direction}")
                evidence_block, retrieved_metadata = module.run_rag_retrieval(
                    retrieval_queries=[topic_display],
                    cutoff_year=cutoff_year,
                    s2_max_total=args.s2_max_total,
                    s2_page_size=args.s2_page_size,
                    s2_api_key=args.s2_api_key,
                    s2_max_retries=args.s2_max_retries,
                    s2_retry_backoff=args.s2_retry_backoff,
                    rag_top_k=top_k,
                    space=direction,
                    mainframe_topic=topic_display,
                    domain=domain_display,
                )

                base_prompt = module.build_prompt(direction, domain_display, topic_display)
                suffix = "\n\nUse this evidence when generating the weak signals."
                evidence_papers = evidence_block.split("\n\n") if evidence_block else [""]
                while True:
                    cur_evidence = "\n\n".join(evidence_papers)
                    augmented_prompt = (
                        f"{base_prompt}\n\nRetrieved paper evidence (already time-cutoff filtered):\n"
                        f"{cur_evidence}{suffix}"
                    )
                    estimated_tokens = len(augmented_prompt) // 3
                    if estimated_tokens <= args.max_model_len or len(evidence_papers) <= 1:
                        break
                    evidence_papers.pop()

                response_text = module.generate_with_qwen(
                    base_url=base_url,
                    model=qwen_model,
                    user_prompt=augmented_prompt,
                    temperature=args.temperature,
                )
                signals = module.extract_candidate_signals(response_text)
                save_generated_result(
                    variant_model=variant_model,
                    domain_slug=domain_slug,
                    topic_slug=topic_slug,
                    direction=direction,
                    response_text=response_text,
                    signals=signals,
                    retrieved_metadata=retrieved_metadata,
                    extra_metadata={
                        "domain_display": domain_display,
                        "topic_display": topic_display,
                        "year_range": module.YEAR_RANGE,
                        "rag_top_k": top_k,
                        "source_family": args.family,
                    },
                )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_family(args)


if __name__ == "__main__":
    main()
