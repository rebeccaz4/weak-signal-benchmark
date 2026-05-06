"""Watch a Tongyi runner and its vLLM backend, restart both when vLLM dies.

Run this script on the same node as the target vLLM and Tongyi job.
It is intentionally independent from the runner code so it does not mutate
existing experiment outputs or interfere with unrelated jobs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch Tongyi runner + vLLM and restart when vLLM dies.")
    parser.add_argument("--health-url", default="http://127.0.0.1:6001/v1/models")
    parser.add_argument("--check-interval", type=int, default=20)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--restart-backoff", type=int, default=10)
    parser.add_argument("--log-dir", type=Path, default=Path("ablation/results/watchdog_logs"))
    parser.add_argument("--vllm-cmd", required=True, help="Shell command that starts vLLM.")
    parser.add_argument("--runner-cmd", required=True, help="Shell command that starts the Tongyi python runner.")
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(handle: TextIO, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def is_healthy(url: str) -> bool:
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def start_process(command: str, log_path: Path) -> subprocess.Popen:
    log_handle = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        ["bash", "-lc", command],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return


def wait_until_healthy(url: str, timeout: int, interval: int, log_handle: TextIO) -> bool:
    waited = 0
    while waited < timeout:
        if is_healthy(url):
            log_line(log_handle, f"vLLM became healthy after {waited}s")
            return True
        time.sleep(interval)
        waited += interval
        log_line(log_handle, f"Waiting for vLLM health endpoint: {waited}s elapsed")
    return False


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    session_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    watchdog_log = args.log_dir / f"watchdog_{session_tag}.log"
    vllm_log = args.log_dir / f"vllm_{session_tag}.log"
    runner_log = args.log_dir / f"runner_{session_tag}.log"

    with watchdog_log.open("a", encoding="utf-8") as log_handle:
        log_line(log_handle, f"Health URL: {args.health_url}")
        log_line(log_handle, f"vLLM command: {args.vllm_cmd}")
        log_line(log_handle, f"Runner command: {args.runner_cmd}")
        log_line(log_handle, f"vLLM log: {vllm_log}")
        log_line(log_handle, f"Runner log: {runner_log}")

        vllm_proc: subprocess.Popen | None = None
        runner_proc: subprocess.Popen | None = None

        while True:
            healthy = is_healthy(args.health_url)

            if not healthy:
                log_line(log_handle, "vLLM health check failed")
                terminate_process(runner_proc)
                runner_proc = None
                terminate_process(vllm_proc)
                vllm_proc = None
                time.sleep(args.restart_backoff)

                log_line(log_handle, "Restarting vLLM")
                vllm_proc = start_process(args.vllm_cmd, vllm_log)
                if not wait_until_healthy(args.health_url, args.startup_timeout, args.check_interval, log_handle):
                    log_line(log_handle, "vLLM failed to become healthy before timeout; retrying")
                    terminate_process(vllm_proc)
                    vllm_proc = None
                    continue

                log_line(log_handle, "Starting runner")
                runner_proc = start_process(args.runner_cmd, runner_log)
            elif runner_proc is None:
                log_line(log_handle, "vLLM healthy but runner missing; starting runner")
                runner_proc = start_process(args.runner_cmd, runner_log)

            if runner_proc is not None:
                exit_code = runner_proc.poll()
                if exit_code is not None:
                    if exit_code == 0:
                        log_line(log_handle, "Runner exited successfully; watchdog stops here")
                        return
                    log_line(log_handle, f"Runner exited with code {exit_code}; it will be restarted")
                    runner_proc = None
                    time.sleep(args.restart_backoff)
                    continue

            time.sleep(args.check_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
