"""
脚本作用：在 RoboTwin 主仓库中调用 FastWAM 策略，执行单个 RoboTwin 任务评测。
命令示例：
python script/eval_fastwam_single.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --task-name click_alarmclock \
  --task-config demo_randomized \
  --eval-num-episodes 1 \
  --gpu-id 0 \
  --no-eval-video-log

命令示例：
python script/eval_fastwam_single.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_trained/step_010000.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_trained/dataset_stats.json \
  --task-name place_fan \
  --task-config demo_randomized \
  --eval-num-episodes 50 \
  --gpu-id 0 \
  --video-dit-num-layers 16 \
  --action-dit-num-layers 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROBOTWIN_ROOT / "third_party" / "FastWAM"
POLICY_NAME = "fastwam_policy"


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


def _ensure_policy_symlink() -> Path:
    policy_root = ROBOTWIN_ROOT / "policy"
    policy_source_dir = FASTWAM_ROOT / "experiments" / "robotwin" / POLICY_NAME
    policy_target = policy_root / POLICY_NAME

    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"FastWAM policy directory not found: {policy_source_dir}")

    source_resolved = policy_source_dir.resolve()
    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RoboTwin task with FastWAM policy.")
    parser.add_argument("--ckpt", required=True, help="FastWAM checkpoint path.")
    parser.add_argument("--dataset-stats-path", required=True, help="FastWAM dataset stats JSON path.")
    parser.add_argument("--task-name", required=True, help="RoboTwin task name.")
    parser.add_argument(
        "--task-config",
        default="demo_randomized",
        choices=["demo_clean", "demo_randomized"],
        help="RoboTwin task config.",
    )
    parser.add_argument("--instruction-type", default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--eval-num-episodes", type=int, default=100)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-output-dir", default=None)
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
    parser.add_argument("--video-dit-num-layers", type=int, default=None)
    parser.add_argument("--action-dit-num-layers", type=int, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--timing-enabled", action="store_true")
    parser.add_argument("--score-mode", action="store_true", help="Use task-specific challenge scoring when available.")
    parser.add_argument(
        "--eval-video-log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override task config eval_video_log.",
    )
    parser.add_argument(
        "--skip-get-obs-within-replan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    ckpt_path = _resolve_path(args.ckpt, base=ROBOTWIN_ROOT)
    dataset_stats_path = _resolve_path(args.dataset_stats_path, base=ROBOTWIN_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not dataset_stats_path.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {dataset_stats_path}")

    sim_cfg_path = FASTWAM_ROOT / "configs" / "sim_robotwin.yaml"
    deploy_config_path = Path("policy") / POLICY_NAME / "deploy_policy.yml"
    if not sim_cfg_path.exists():
        raise FileNotFoundError(f"FastWAM sim config not found: {sim_cfg_path}")

    _ensure_policy_symlink()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.eval_output_dir is None:
        ckpt_tag = _resolve_ckpt_tag(ckpt_path)
        eval_output_dir = (
            ROBOTWIN_ROOT
            / "evaluate_results"
            / "fastwam"
            / ckpt_tag
            / run_ts
            / args.task_name
        )
    else:
        eval_output_dir = _resolve_path(args.eval_output_dir, base=ROBOTWIN_ROOT)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / f"eval_{args.task_name}_{args.task_config}_{run_ts}.log"

    overrides: list[str] = []
    _append_override(overrides, "task_name", args.task_name)
    _append_override(overrides, "task_config", args.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", args.seed)
    _append_override(overrides, "policy_name", POLICY_NAME)
    _append_override(overrides, "instruction_type", args.instruction_type)
    _append_override(overrides, "eval_num_episodes", args.eval_num_episodes)
    _append_override(overrides, "eval_output_dir", str(eval_output_dir))
    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", args.sim_task)
    _append_override(overrides, "mixed_precision", args.mixed_precision)
    _append_override(overrides, "device", args.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", args.action_horizon)
    _append_override(overrides, "replan_steps", args.replan_steps)
    _append_override(overrides, "num_inference_steps", args.num_inference_steps)
    _append_override(overrides, "sigma_shift", args.sigma_shift)
    _append_override(overrides, "text_cfg_scale", args.text_cfg_scale)
    _append_override(overrides, "negative_prompt", args.negative_prompt)
    _append_override(overrides, "rand_device", args.rand_device)
    _append_override(overrides, "model_video_dit_num_layers", args.video_dit_num_layers)
    _append_override(overrides, "model_action_dit_num_layers", args.action_dit_num_layers)
    _append_override(overrides, "tiled", args.tiled)
    _append_override(overrides, "timing_enabled", args.timing_enabled)
    _append_override(overrides, "skip_get_obs_within_replan", args.skip_get_obs_within_replan)
    _append_override(overrides, "score_mode", args.score_mode)
    _append_override(overrides, "eval_video_log", args.eval_video_log)

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        str(deploy_config_path),
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = (
            f"{conda_prefix}/lib"
            if not env.get("LD_LIBRARY_PATH")
            else f"{conda_prefix}/lib:{env['LD_LIBRARY_PATH']}"
        )
    nvidia_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if nvidia_icd.exists() and not env.get("VK_ICD_FILENAMES"):
        env["VK_ICD_FILENAMES"] = str(nvidia_icd)

    with log_file.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROBOTWIN_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"FastWAM evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
