"""
脚本作用：从指定任务开始续跑 FastWAM/RoboTwin 挑战子集评测，默认从 open_microwave 开始，使用 4 张 GPU 并行评测剩余 7 个任务，每个任务默认 50 条，按 RoboTwin 原始 check_success 成功率评分，并保存低帧更新率评测视频。
执行命令示例：
python script/eval_fastwam_challenge_resume_parallel.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --eval-num-episodes 50 \
  --gpu-ids 0,1,2,3 \
  --start-task open_microwave

执行命令示例：
python script/eval_fastwam_challenge_resume_parallel.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --eval-num-episodes 50 \
  --gpu-ids 0,1,2,3 \
  --start-task open_microwave \
  --dry-run

执行命令示例：
python script/eval_fastwam_challenge_resume_parallel.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_trained/step_010000.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_trained/dataset_stats.json \
  --eval-num-episodes 10 \
  --gpu-ids 0,1,2,3 \
  --tasks move_stapler_pad,place_fan,handover_mic,open_microwave,place_can_basket,place_dual_shoes,move_can_pot,stack_blocks_three,blocks_ranking_rgb,blocks_ranking_size \
  --video-dit-num-layers 16 \
  --action-dit-num-layers 16 \
  --no-skip-get-obs-within-replan
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROBOTWIN_ROOT / "third_party" / "FastWAM"
SINGLE_ENTRY = ROBOTWIN_ROOT / "script" / "eval_fastwam_single.py"

DEFAULT_CKPT = "third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
DEFAULT_DATASET_STATS = (
    "third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
)

CHALLENGE_TASKS = [
    "move_stapler_pad",
    "place_fan",
    "handover_mic",
    "open_microwave",
    "place_can_basket",
    "place_dual_shoes",
    "move_can_pot",
    "stack_blocks_three",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
]

SCORING_RULE = "RoboTwin 原始 check_success 二值成功率"


def _resolve_path(path_str: str) -> Path:
    raw_path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if raw_path.is_absolute():
        return raw_path.resolve()

    rel = Path(str(raw_path).removeprefix("./"))
    candidates = [
        ROBOTWIN_ROOT / raw_path,
        ROBOTWIN_ROOT / rel,
        FASTWAM_ROOT / raw_path,
        FASTWAM_ROOT / rel,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    tried = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"路径不存在，已尝试：\n{tried}")


def _resolve_output_dir(path_str: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (ROBOTWIN_ROOT / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _result_filename(task_config: str) -> str:
    if task_config == "demo_clean":
        return "_result_clean.txt"
    if task_config == "demo_randomized":
        return "_result_random.txt"
    raise ValueError(f"不支持的 task_config：{task_config}")


def _parse_last_float(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"结果文件不存在：{result_file}")
    last_value: float | None = None
    for line in result_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"无法从结果文件解析成功率：{result_file}")
    return last_value


def _parse_gpu_ids(raw_gpu_ids: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in raw_gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("`--gpu-ids` 至少需要一张 GPU，例如：0,1,2,3")
    return gpu_ids


def _parse_tasks(raw_tasks: str | None, start_task: str) -> list[str]:
    if raw_tasks:
        tasks = [item.strip() for item in raw_tasks.split(",") if item.strip()]
    else:
        if start_task not in CHALLENGE_TASKS:
            raise ValueError(f"`--start-task` 不在挑战任务列表中：{start_task}")
        tasks = CHALLENGE_TASKS[CHALLENGE_TASKS.index(start_task):]

    unknown = [task for task in tasks if task not in CHALLENGE_TASKS]
    if unknown:
        raise ValueError(f"发现未知任务：{unknown}")
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel resume FastWAM challenge subset evaluation.")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="FastWAM checkpoint path.")
    parser.add_argument("--dataset-stats-path", default=DEFAULT_DATASET_STATS, help="FastWAM dataset stats JSON path.")
    parser.add_argument("--eval-num-episodes", type=int, default=50)
    parser.add_argument("--gpu-ids", default="0,1,2,3", help="Comma separated GPU ids.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-task", default="open_microwave")
    parser.add_argument("--tasks", default=None, help="Optional comma separated task override.")
    parser.add_argument("--sim-task", default="robotwin_uncond_3cam_384_1e-4")
    parser.add_argument("--instruction-type", default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--task-config", default="demo_randomized", choices=["demo_clean", "demo_randomized"])
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--video-dit-num-layers", type=int, default=None)
    parser.add_argument("--action-dit-num-layers", type=int, default=None)
    parser.add_argument(
        "--skip-get-obs-within-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="为 True 时使用低帧率评测视频；为 False 时保存完整渲染视频。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_single_cmd(
    args: argparse.Namespace,
    *,
    task_name: str,
    gpu_id: int,
    ckpt: Path,
    dataset_stats: Path,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SINGLE_ENTRY),
        "--ckpt",
        str(ckpt),
        "--dataset-stats-path",
        str(dataset_stats),
        "--task-name",
        task_name,
        "--task-config",
        args.task_config,
        "--instruction-type",
        args.instruction_type,
        "--eval-num-episodes",
        str(args.eval_num_episodes),
        "--gpu-id",
        str(gpu_id),
        "--seed",
        str(args.seed),
        "--eval-output-dir",
        str(output_dir / task_name),
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
        "--eval-video-log",
    ]
    if args.skip_get_obs_within_replan:
        cmd.append("--skip-get-obs-within-replan")
    else:
        cmd.append("--no-skip-get-obs-within-replan")
    if args.video_dit_num_layers is not None:
        cmd.extend(["--video-dit-num-layers", str(args.video_dit_num_layers)])
    if args.action_dit_num_layers is not None:
        cmd.extend(["--action-dit-num-layers", str(args.action_dit_num_layers)])
    return cmd


def _write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    summary_csv = output_dir / "challenge_parallel_summary.csv"
    summary_json = output_dir / "challenge_parallel_summary.json"
    failed_file = output_dir / "failed_tasks.txt"

    valid_rates = [record["success_rate"] for record in records if record["success_rate"] is not None]
    overall_success_rate = float(sum(valid_rates) / len(valid_rates)) if valid_rates else None

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "gpu_id", "success_rate", "status", "result_file", "log_file", "worker_log"])
        for record in records:
            writer.writerow([
                record["task_name"],
                record["gpu_id"],
                record["success_rate"],
                record["status"],
                record["result_file"],
                record["log_file"],
                record["worker_log"],
            ])
        writer.writerow(["__overall__", "", overall_success_rate, "", "", "", ""])

    payload = {
        "overall_success_rate": overall_success_rate,
        "scoring_rule": SCORING_RULE,
        "tasks": records,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with failed_file.open("w", encoding="utf-8") as f:
        for record in records:
            if record["status"] != "ok":
                f.write(f"{record['task_name']},{record['status']},{record['error']}\n")


def main() -> int:
    args = _parse_args()
    if args.eval_num_episodes <= 0:
        raise ValueError(f"`--eval-num-episodes` 必须大于 0，当前为：{args.eval_num_episodes}")

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    tasks = _parse_tasks(args.tasks, args.start_task)
    ckpt = _resolve_path(args.ckpt)
    dataset_stats = _resolve_path(args.dataset_stats_path)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = _resolve_output_dir(args.output_dir)
    else:
        output_dir = (
            ROBOTWIN_ROOT
            / "evaluate_results"
            / "fastwam_challenge_subset_parallel"
            / _resolve_ckpt_tag(ckpt)
            / run_ts
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = output_dir / "challenge_parallel_manager.log"
    records: list[dict[str, Any]] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(
        f"start tasks={tasks} eval_num_episodes={args.eval_num_episodes} "
        f"gpu_ids={gpu_ids} output_dir={output_dir}"
    )

    has_failure = False
    for batch_start in range(0, len(tasks), len(gpu_ids)):
        batch_tasks = tasks[batch_start:batch_start + len(gpu_ids)]
        batch_no = batch_start // len(gpu_ids) + 1
        log(f"start batch={batch_no} tasks={batch_tasks}")
        running: list[dict[str, Any]] = []

        for gpu_id, task_name in zip(gpu_ids, batch_tasks):
            task_output_dir = output_dir / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            worker_log = task_output_dir / f"parallel_worker_{task_name}_{run_ts}.log"
            cmd = _build_single_cmd(
                args,
                task_name=task_name,
                gpu_id=gpu_id,
                ckpt=ckpt,
                dataset_stats=dataset_stats,
                output_dir=output_dir,
            )
            log(f"launch batch={batch_no} task={task_name} gpu={gpu_id} cmd={' '.join(shlex.quote(part) for part in cmd)}")

            record = {
                "task_name": task_name,
                "gpu_id": gpu_id,
                "success_rate": None,
                "status": "pending",
                "error": "",
                "result_file": str(task_output_dir / _result_filename(args.task_config)),
                "log_file": "",
                "worker_log": str(worker_log),
                "eval_num_episodes": args.eval_num_episodes,
            }

            if args.dry_run:
                record["status"] = "dry_run"
                records.append(record)
                continue

            worker_log_f = worker_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(ROBOTWIN_ROOT),
                text=True,
                stdout=worker_log_f,
                stderr=subprocess.STDOUT,
            )
            running.append({"process": process, "log_file_handle": worker_log_f, "record": record, "task_output_dir": task_output_dir})

        if args.dry_run:
            _write_summary(output_dir, records)
            continue

        for item in running:
            process: subprocess.Popen[str] = item["process"]
            record = item["record"]
            task_output_dir: Path = item["task_output_dir"]
            worker_log_f = item["log_file_handle"]
            return_code = process.wait()
            worker_log_f.close()
            task_name = record["task_name"]
            log(f"finished batch={batch_no} task={task_name} gpu={record['gpu_id']} return_code={return_code}")

            logs = sorted(task_output_dir.glob(f"eval_{task_name}_{args.task_config}_*.log"))
            record["log_file"] = str(logs[-1]) if logs else ""
            if return_code != 0:
                record["status"] = "process_failed"
                record["error"] = f"return_code={return_code}"
                records.append(record)
                has_failure = True
                _write_summary(output_dir, records)
                continue

            try:
                score = _parse_last_float(task_output_dir / _result_filename(args.task_config))
                record["success_rate"] = score
                record["status"] = "ok"
                log(f"score task={task_name} success_rate={score:.4f}")
            except Exception as exc:
                record["status"] = "result_parse_failed"
                record["error"] = repr(exc)
                has_failure = True
                log(f"parse failed task={task_name} error={repr(exc)}")

            records.append(record)
            _write_summary(output_dir, records)

    _write_summary(output_dir, records)
    log("summary saved")
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
