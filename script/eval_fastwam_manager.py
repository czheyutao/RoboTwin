"""
脚本作用：在 RoboTwin 主仓库中批量调度 FastWAM 策略评测，支持单任务或全任务、多 GPU 并发。
命令示例：
python script/eval_fastwam_manager.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --task-name click_alarmclock \
  --eval-num-episodes 1 \
  --num-gpus 1 \
  --max-tasks-per-gpu 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
SINGLE_ENTRY = ROBOTWIN_ROOT / "script" / "eval_fastwam_single.py"
EVAL_STEP_LIMIT_FILE = ROBOTWIN_ROOT / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")

    seen: set[str] = set()
    tasks: list[str] = []
    for task in task_map.keys():
        if task in seen:
            continue
        seen.add(task)
        tasks.append(str(task))
    return tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _phase_task_config(phase: str) -> str:
    if phase == "clean":
        return "demo_clean"
    if phase == "random":
        return "demo_randomized"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboTwin tasks with FastWAM policy.")
    parser.add_argument("--ckpt", required=True, help="FastWAM checkpoint path.")
    parser.add_argument("--dataset-stats-path", required=True, help="FastWAM dataset stats JSON path.")
    parser.add_argument("--task-name", default=None, help="Run one task. If omitted, run all tasks.")
    parser.add_argument("--phase", default="both", choices=["both", "clean", "random"])
    parser.add_argument("--instruction-type", default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--eval-num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sim-task", default="robotwin_uncond_3cam_384_1e-4")
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--timing-enabled", action="store_true")
    parser.add_argument(
        "--skip-get-obs-within-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str
    process: subprocess.Popen[str]


def main() -> int:
    args = _parse_args()
    if args.eval_num_episodes <= 0:
        raise ValueError(f"`eval-num-episodes` must be > 0, got: {args.eval_num_episodes}")
    if args.num_gpus <= 0:
        raise ValueError(f"`num-gpus` must be > 0, got: {args.num_gpus}")
    if args.max_tasks_per_gpu <= 0:
        raise ValueError(f"`max-tasks-per-gpu` must be > 0, got: {args.max_tasks_per_gpu}")

    ckpt_path = _resolve_path(args.ckpt, base=ROBOTWIN_ROOT)
    dataset_stats_path = _resolve_path(args.dataset_stats_path, base=ROBOTWIN_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not dataset_stats_path.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {dataset_stats_path}")

    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        run_output_dir = ROBOTWIN_ROOT / "evaluate_results" / "fastwam" / ckpt_tag / run_ts
    else:
        run_output_dir = _resolve_path(args.output_dir, base=ROBOTWIN_ROOT)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    tasks = [args.task_name] if args.task_name else _load_all_tasks()
    phases = ["clean", "random"] if args.phase == "both" else [args.phase]
    gpu_ids = list(range(args.num_gpus))

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []
    next_phase_idx: dict[str, int] = {task: 0 for task in tasks}

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_output_dir = run_output_dir / task_name
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            "--ckpt",
            str(ckpt_path),
            "--dataset-stats-path",
            str(dataset_stats_path),
            "--task-name",
            task_name,
            "--task-config",
            _phase_task_config(phase),
            "--instruction-type",
            args.instruction_type,
            "--eval-num-episodes",
            str(args.eval_num_episodes),
            "--gpu-id",
            str(gpu_id),
            "--seed",
            str(args.seed),
            "--eval-output-dir",
            str(task_output_dir),
            "--sim-task",
            args.sim_task,
            "--mixed-precision",
            args.mixed_precision,
            "--device",
            args.device,
            "--replan-steps",
            str(args.replan_steps),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--text-cfg-scale",
            str(args.text_cfg_scale),
            "--negative-prompt",
            args.negative_prompt,
            "--rand-device",
            args.rand_device,
        ]
        if args.action_horizon is not None:
            cmd.extend(["--action-horizon", str(args.action_horizon)])
        if args.sigma_shift is not None:
            cmd.extend(["--sigma-shift", str(args.sigma_shift)])
        if args.tiled:
            cmd.append("--tiled")
        if args.timing_enabled:
            cmd.append("--timing-enabled")
        if not args.skip_get_obs_within_replan:
            cmd.append("--no-skip-get-obs-within-replan")
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(f"launch task={task_name} phase={phase} gpu={gpu_id} cmd={' '.join(cmd)}")
        process = subprocess.Popen(cmd, cwd=str(ROBOTWIN_ROOT), text=True)
        return RunningState(task_name=task_name, gpu_id=gpu_id, phase=phase, process=process)

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        return sum(
            1
            for state in running_states
            if state.gpu_id == gpu_id and state.process.poll() is None
        )

    def try_launch_pending(gpu_id: int) -> None:
        while pending_tasks and gpu_running_count(gpu_id) < args.max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            phase = phases[next_phase_idx[task_name]]
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase=phase))

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow([task, task_rates[task]["clean"], task_rates[task]["random"]])
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} phases={phases} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={args.max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while running_states:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            next_phase_idx[state.task_name] += 1
            if next_phase_idx[state.task_name] < len(phases):
                next_phase = phases[next_phase_idx[state.task_name]]
                running_states.append(
                    launch_phase(task_name=state.task_name, gpu_id=gpu_id, phase=next_phase)
                )
                continue

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
