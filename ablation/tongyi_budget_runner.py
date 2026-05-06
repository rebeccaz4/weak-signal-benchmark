"""Run Tongyi google_scholar-budget ablations on the labeleds benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from .common import CONSTRUCTION_ROOT, DATA_ROOT, OUTPUTS_ROOT, PREDICTION_PY_ROOT, YEAR_SLUG, import_module_from_path, read_json, utc_now_tag

DEFAULT_TONGYI_BENCHMARK_JSON = DATA_ROOT / "topic_benchmark_tongyi_20.json"
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
DEFAULT_TONGYI_VLLM_ENV = os.getenv("TONGYI_VLLM_CONDA_ENV", "weak-signal")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Tongyi budget runner."""
    parser = argparse.ArgumentParser(description="Run Tongyi google_scholar tool-budget variants for the ablation benchmark.")
    parser.add_argument("--benchmark-json", type=Path, default=DEFAULT_TONGYI_BENCHMARK_JSON)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--tongyi-dir", type=Path, required=True)
    parser.add_argument("--google-scholar-budget", nargs="+", type=int, default=[0, 1, 3, 8])
    parser.add_argument("--tongyi-model", default=os.getenv("TONGYI_MODEL_PATH", "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"))
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=int(os.getenv("TONGYI_VLLM_PORT", "6001")))
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--skip-vllm-start", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=1.1)
    parser.add_argument("--s2-api-key", default=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY", ""))
    return parser.parse_args()


def load_benchmark_topics(benchmark_json: Path) -> list[dict]:
    """Load the benchmark topic list."""
    return read_json(benchmark_json)["topics"]


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


def patch_tongyi_repo_for_google_scholar_budget(tongyi_dir: Path) -> None:
    """Add dynamic google_scholar budget enforcement to the patched Tongyi agent."""
    agent_file = tongyi_dir / "inference" / "react_agent.py"
    src = agent_file.read_text(encoding="utf-8")
    if "STEPB6_GS_BUDGET_PATCH" in src:
        return

    old_budget_block = "MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))\n"
    new_budget_block = (
        "MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))\n"
        "GOOGLE_SCHOLAR_TOOL_BUDGET = int(os.getenv('GOOGLE_SCHOLAR_TOOL_BUDGET', '-1'))\n"
        "# STEPB6_GS_BUDGET_PATCH\n"
    )
    if old_budget_block not in src:
        raise RuntimeError(f"Could not locate MAX_LLM_CALL_PER_RUN block in {agent_file}")
    src = src.replace(old_budget_block, new_budget_block, 1)

    old_prompt_block = (
        "        system_prompt = SYSTEM_PROMPT\n"
        "        cur_date = today_date()\n"
        "        system_prompt = system_prompt + str(cur_date)\n"
    )
    new_prompt_block = (
        "        system_prompt = SYSTEM_PROMPT\n"
        "        cur_date = today_date()\n"
        "        self.google_scholar_budget = GOOGLE_SCHOLAR_TOOL_BUDGET\n"
        "        self.google_scholar_call_count = 0\n"
        "        budget_note = ''\n"
        "        if self.google_scholar_budget >= 0:\n"
        "            budget_note = (\n"
        "                f\"\\n\\nTool budget constraint: google_scholar may be called at most \"\n"
        "                f\"{self.google_scholar_budget} times in this run.\"\n"
        "            )\n"
        "        system_prompt = system_prompt + str(cur_date) + budget_note\n"
    )
    if old_prompt_block not in src:
        raise RuntimeError(f"Could not locate system prompt block in {agent_file}")
    src = src.replace(old_prompt_block, new_prompt_block, 1)

    old_tool_block = (
        "            if \"python\" in tool_name.lower():\n"
        "                result = TOOL_MAP['PythonInterpreter'].call(tool_args)\n"
        "            elif tool_name == \"parse_file\":\n"
        "                params = {\"files\": tool_args[\"files\"]}\n"
        "                \n"
        "                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path=\"./eval_data/file_corpus\"))\n"
        "                result = raw_result\n"
        "\n"
        "                if not isinstance(raw_result, str):\n"
        "                    result = str(raw_result)\n"
        "            else:\n"
        "                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)\n"
        "                result = raw_result\n"
    )
    new_tool_block = (
        "            if \"python\" in tool_name.lower():\n"
        "                result = TOOL_MAP['PythonInterpreter'].call(tool_args)\n"
        "            elif tool_name == \"parse_file\":\n"
        "                params = {\"files\": tool_args[\"files\"]}\n"
        "                \n"
        "                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path=\"./eval_data/file_corpus\"))\n"
        "                result = raw_result\n"
        "\n"
        "                if not isinstance(raw_result, str):\n"
        "                    result = str(raw_result)\n"
        "            elif tool_name == \"google_scholar\":\n"
        "                if self.google_scholar_budget >= 0 and self.google_scholar_call_count >= self.google_scholar_budget:\n"
        "                    return (\n"
        "                        f\"Error: google_scholar tool budget exhausted. \"\n"
        "                        f\"Allowed={self.google_scholar_budget}, used={self.google_scholar_call_count}.\"\n"
        "                    )\n"
        "                self.google_scholar_call_count += 1\n"
        "                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)\n"
        "                result = raw_result\n"
        "            else:\n"
        "                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)\n"
        "                result = raw_result\n"
    )
    if old_tool_block not in src:
        raise RuntimeError(f"Could not locate custom_call_tool block in {agent_file}")
    src = src.replace(old_tool_block, new_tool_block, 1)
    agent_file.write_text(src, encoding="utf-8")


def start_vllm_server_with_python3(
    model: str,
    host: str,
    port: int,
    tensor_parallel: int = 2,
    max_model_len: int = 32768,
    gpu_mem: float = 0.9,
) -> subprocess.Popen | None:
    """Start vLLM for Tongyi from a conda environment that has vllm installed."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((host, port)) == 0:
            print(f"vLLM already running on {host}:{port}. Reusing.")
            return None

    log_path = Path(f"tongyi_vllm_server_{port}.log")
    cmd = [
        "conda", "run", "-n", DEFAULT_TONGYI_VLLM_ENV, "python",
        "-m", "vllm.entrypoints.openai.api_server",
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enforce-eager",
    ]

    print(f"Starting vLLM on {host}:{port} with model {model} via conda env {DEFAULT_TONGYI_VLLM_ENV} ...")
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"Spawned vLLM pid={proc.pid}, log={log_path}")

    import requests as http_requests

    max_wait = 1200
    interval = 5
    waited = 0
    while waited < max_wait:
        time.sleep(interval)
        waited += interval
        try:
            response = http_requests.get(f"http://{host}:{port}/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"vLLM ready after {waited}s")
                return proc
        except Exception:
            print(f"  Waiting for vLLM ... ({waited}s elapsed)", flush=True)

    print(f"[warn] vLLM may not be ready after {max_wait}s. Check {log_path}")
    return proc


def extract_tool_usage(result_row: dict) -> dict:
    """Parse google_scholar call/query counts from Tongyi messages."""
    google_scholar_calls = 0
    google_scholar_queries = 0
    budget_exhausted = False
    for message in result_row.get("messages", []):
        content = (message or {}).get("content") or ""
        if "budget exhausted" in content:
            budget_exhausted = True
        for match in TOOL_CALL_PATTERN.findall(content):
            try:
                payload = json.loads(match)
            except json.JSONDecodeError:
                continue
            if payload.get("name") != "google_scholar":
                continue
            google_scholar_calls += 1
            query = (payload.get("arguments") or {}).get("query", [])
            if isinstance(query, list):
                google_scholar_queries += len(query)
            elif query:
                google_scholar_queries += 1
    return {
        "google_scholar_calls": google_scholar_calls,
        "google_scholar_queries": google_scholar_queries,
        "budget_exhausted": budget_exhausted,
    }


def run_tongyi_inference_with_budget(
    infer_dir: Path,
    model: str,
    prompt: str,
    output_base: Path,
    cutoff_year: int,
    s2_api_key: str,
    google_scholar_budget: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
) -> dict:
    """Run Tongyi with a caller-controlled google_scholar tool-call budget."""
    output_base = output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run_tag = utc_now_tag()
    dataset_name = f"tongyi_input_{run_tag}.json"
    dataset_path = output_base / dataset_name
    dataset_path.write_text(json.dumps([{"question": prompt, "answer": ""}], ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["S2_CUTOFF_YEAR"] = str(cutoff_year)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GOOGLE_SCHOLAR_TOOL_BUDGET"] = str(google_scholar_budget)
    if s2_api_key:
        env["SEMANTIC_SCHOLAR_API_KEY"] = s2_api_key
        env["S2_API_KEY"] = s2_api_key

    runner_output = output_base / "tongyi_runner_outputs"
    runner_output.mkdir(parents=True, exist_ok=True)
    run_script = infer_dir / "run_multi_react.py"
    cmd = [
        "python3", "-B", str(run_script),
        "--model", model,
        "--output", str(runner_output),
        "--dataset", dataset_name,
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--presence_penalty", str(presence_penalty),
        "--max_workers", "1",
        "--roll_out_count", "1",
        "--total_splits", "1",
        "--worker_split", "1",
    ]
    res = subprocess.run(cmd, cwd=str(output_base), env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "Tongyi run_multi_react.py failed.\n"
            f"stdout:\n{chr(10).join(res.stdout.splitlines()[-40:])}\n"
            f"stderr:\n{chr(10).join(res.stderr.splitlines()[-40:])}"
        )

    model_leaf = os.path.basename(model.rstrip("/"))
    iter1_path = runner_output / f"{model_leaf}_sglang" / dataset_name / "iter1.jsonl"
    if not iter1_path.exists():
        candidates = sorted(runner_output.glob("**/iter1.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("iter1.jsonl not found under Tongyi runner output directory.")
        iter1_path = candidates[0]
    rows = [line for line in iter1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows found in {iter1_path}")
    return json.loads(rows[-1])


def save_generated_result(
    variant_model: str,
    domain_slug: str,
    topic_slug: str,
    direction: str,
    response_text: str,
    signals: list[str],
    domain_display: str,
    topic_display: str,
    year_range: str,
    google_scholar_budget: int,
    tool_usage: dict,
) -> Path:
    """Save a generated Tongyi result into the ablation output tree."""
    result_dir = OUTPUTS_ROOT / variant_model / domain_slug / topic_slug / direction / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_tag()
    payload = {
        "space": direction,
        "domain": domain_display,
        "mainframe_topic": topic_display,
        "year_range": year_range,
        "timestamp": timestamp,
        "signals": signals,
        "ablation": {
            "budget_kind": "google_scholar_tool_call_budget",
            "budget_value": google_scholar_budget,
            "variant_model": variant_model,
            "source": "generated",
            "source_family": "tongyi",
            "tool_usage": tool_usage,
        },
    }
    (result_dir / f"response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")
    (result_dir / f"signals_{timestamp}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (result_dir / "signals_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_dir


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    module = import_module_from_path("ablation_tongyi_eval", PREDICTION_PY_ROOT / "Tongyi_eval.py")
    module.patch_tongyi_repo(args.tongyi_dir)
    patch_tongyi_repo_for_google_scholar_budget(args.tongyi_dir)
    if not args.skip_vllm_start:
        start_vllm_server_with_python3(
            model=args.tongyi_model,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_mem=args.gpu_memory_utilization,
        )

    topics = load_benchmark_topics(args.benchmark_json)
    infer_dir = args.tongyi_dir / "inference"
    cutoff_year = module.year_range_to_cutoff(module.YEAR_RANGE)
    for google_scholar_budget in args.google_scholar_budget:
        variant_model = f"tongyi_gsb{google_scholar_budget}"
        for item in topics:
            domain_slug = item["domain"]
            topic_slug = item["topic"]
            domain_display, topic_display = load_display_context(domain_slug, topic_slug)
            for direction in ("problem", "solution"):
                result_dir = OUTPUTS_ROOT / variant_model / domain_slug / topic_slug / direction / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Exists: {result_dir}")
                    continue

                prompt = module.build_prompt(direction, domain_display, topic_display)
                run_output_base = result_dir / "tongyi_runs"
                print(f"[run] {variant_model}/{domain_slug}/{topic_slug}/{direction}")
                result_row = run_tongyi_inference_with_budget(
                    infer_dir=infer_dir,
                    model=args.tongyi_model,
                    prompt=prompt,
                    output_base=run_output_base,
                    cutoff_year=cutoff_year,
                    s2_api_key=args.s2_api_key,
                    google_scholar_budget=google_scholar_budget,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                )
                response_text = (result_row.get("prediction") or "").strip()
                signals = module.extract_candidate_signals(response_text)
                tool_usage = extract_tool_usage(result_row)
                save_generated_result(
                    variant_model=variant_model,
                    domain_slug=domain_slug,
                    topic_slug=topic_slug,
                    direction=direction,
                    response_text=response_text,
                    signals=signals,
                    domain_display=domain_display,
                    topic_display=topic_display,
                    year_range=module.YEAR_RANGE,
                    google_scholar_budget=google_scholar_budget,
                    tool_usage=tool_usage,
                )


if __name__ == "__main__":
    main()
