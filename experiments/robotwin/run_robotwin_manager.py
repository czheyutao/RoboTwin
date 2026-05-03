"""
脚本作用：兼容 FastWAM 原版 Hydra 风格命令，并转发到 RoboTwin 主仓库中已适配的多卡 FastWAM 评测管理器。
命令示例：
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8

命令示例：
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.eval_num_episodes=1 \
  MULTIRUN.num_gpus=1 \
  DRY_RUN=true
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROBOTWIN_ROOT = Path(__file__).resolve().parents[2]
FASTWAM_ROOT = ROBOTWIN_ROOT / "third_party" / "FastWAM"
MANAGER_ENTRY = ROBOTWIN_ROOT / "script" / "eval_fastwam_manager.py"

DEFAULT_PHASE = "random"
DEFAULT_INSTRUCTION_TYPE = "unseen"
DEFAULT_SKIP_GET_OBS_WITHIN_REPLAN = False
DEFAULT_MAX_TASKS_PER_GPU = 1


def _strip_quotes(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _strip_quotes(str(value)).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔值：{value}")


def _parse_overrides(argv: list[str]) -> tuple[dict[str, str], bool]:
    overrides: dict[str, str] = {}
    dry_run = False
    for raw_arg in argv:
        if raw_arg == "--dry-run":
            dry_run = True
            continue
        if "=" not in raw_arg:
            raise ValueError(f"只支持 key=value 风格参数，收到：{raw_arg}")
        key, value = raw_arg.split("=", 1)
        key = key.lstrip("+").strip()
        value = _strip_quotes(value)
        if key in {"DRY_RUN", "dry_run"}:
            dry_run = _parse_bool(value)
            continue
        overrides[key] = value
    return overrides, dry_run


def _resolve_existing_path(path_str: str, *, label: str) -> Path:
    raw_path = Path(os.path.expanduser(os.path.expandvars(_strip_quotes(path_str))))
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        rel = Path(str(raw_path).removeprefix("./"))
        candidates.extend(
            [
                (ROBOTWIN_ROOT / raw_path),
                (ROBOTWIN_ROOT / rel),
                (FASTWAM_ROOT / raw_path),
                (FASTWAM_ROOT / rel),
            ]
        )

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
        if resolved.exists():
            return resolved

    tried = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(f"{label} 不存在，已尝试：\n{tried}")


def _append_if_present(cmd: list[str], overrides: dict[str, str], key: str, flag: str) -> None:
    if key in overrides:
        cmd.extend([flag, overrides[key]])


def _phase_from_task_config(task_config: str) -> str:
    value = _strip_quotes(task_config)
    if value == "demo_clean":
        return "clean"
    if value == "demo_randomized":
        return "random"
    raise ValueError(f"不支持的 EVALUATION.task_config：{task_config}")


def _build_manager_cmd(overrides: dict[str, str]) -> list[str]:
    if not MANAGER_ENTRY.exists():
        raise FileNotFoundError(f"找不到内部评测管理器：{MANAGER_ENTRY}")

    ckpt = overrides.get("ckpt")
    if ckpt is None:
        raise ValueError("缺少必需参数：ckpt=...")
    dataset_stats = overrides.get("EVALUATION.dataset_stats_path")
    if dataset_stats is None:
        raise ValueError("缺少必需参数：EVALUATION.dataset_stats_path=...")

    ckpt_path = _resolve_existing_path(ckpt, label="ckpt")
    dataset_stats_path = _resolve_existing_path(dataset_stats, label="dataset stats")

    sim_task = overrides.get("task", "robotwin_uncond_3cam_384_1e-4")
    phase = overrides.get("EVALUATION.phase", DEFAULT_PHASE)
    if "EVALUATION.task_config" in overrides:
        phase = _phase_from_task_config(overrides["EVALUATION.task_config"])

    cmd = [
        sys.executable,
        str(MANAGER_ENTRY),
        "--ckpt",
        str(ckpt_path),
        "--dataset-stats-path",
        str(dataset_stats_path),
        "--sim-task",
        sim_task,
        "--phase",
        phase,
        "--instruction-type",
        overrides.get("EVALUATION.instruction_type", DEFAULT_INSTRUCTION_TYPE),
        "--num-gpus",
        overrides.get("MULTIRUN.num_gpus", "1"),
        "--max-tasks-per-gpu",
        overrides.get("MULTIRUN.max_tasks_per_gpu", str(DEFAULT_MAX_TASKS_PER_GPU)),
    ]

    if "EVALUATION.task_name" in overrides:
        cmd.extend(["--task-name", overrides["EVALUATION.task_name"]])
    if "EVALUATION.output_dir" in overrides:
        cmd.extend(["--output-dir", overrides["EVALUATION.output_dir"]])

    _append_if_present(cmd, overrides, "EVALUATION.eval_num_episodes", "--eval-num-episodes")
    _append_if_present(cmd, overrides, "EVALUATION.seed", "--seed")
    _append_if_present(cmd, overrides, "mixed_precision", "--mixed-precision")
    _append_if_present(cmd, overrides, "EVALUATION.mixed_precision", "--mixed-precision")
    _append_if_present(cmd, overrides, "EVALUATION.device", "--device")
    _append_if_present(cmd, overrides, "EVALUATION.action_horizon", "--action-horizon")
    _append_if_present(cmd, overrides, "EVALUATION.replan_steps", "--replan-steps")
    _append_if_present(cmd, overrides, "EVALUATION.num_inference_steps", "--num-inference-steps")
    _append_if_present(cmd, overrides, "EVALUATION.sigma_shift", "--sigma-shift")
    _append_if_present(cmd, overrides, "EVALUATION.text_cfg_scale", "--text-cfg-scale")
    _append_if_present(cmd, overrides, "EVALUATION.negative_prompt", "--negative-prompt")
    _append_if_present(cmd, overrides, "EVALUATION.rand_device", "--rand-device")

    if _parse_bool(overrides.get("EVALUATION.tiled", "false")):
        cmd.append("--tiled")
    if _parse_bool(overrides.get("EVALUATION.timing_enabled", "false")):
        cmd.append("--timing-enabled")

    skip_obs = _parse_bool(
        overrides.get(
            "EVALUATION.skip_get_obs_within_replan",
            str(DEFAULT_SKIP_GET_OBS_WITHIN_REPLAN),
        )
    )
    if not skip_obs:
        cmd.append("--no-skip-get-obs-within-replan")

    known_keys = {
        "task",
        "ckpt",
        "mixed_precision",
        "EVALUATION.dataset_stats_path",
        "EVALUATION.phase",
        "EVALUATION.task_config",
        "EVALUATION.instruction_type",
        "EVALUATION.task_name",
        "EVALUATION.output_dir",
        "EVALUATION.eval_num_episodes",
        "EVALUATION.seed",
        "EVALUATION.mixed_precision",
        "EVALUATION.device",
        "EVALUATION.action_horizon",
        "EVALUATION.replan_steps",
        "EVALUATION.num_inference_steps",
        "EVALUATION.sigma_shift",
        "EVALUATION.text_cfg_scale",
        "EVALUATION.negative_prompt",
        "EVALUATION.rand_device",
        "EVALUATION.tiled",
        "EVALUATION.timing_enabled",
        "EVALUATION.skip_get_obs_within_replan",
        "MULTIRUN.num_gpus",
        "MULTIRUN.max_tasks_per_gpu",
    }
    ignored_keys = sorted(set(overrides) - known_keys)
    if ignored_keys:
        print(f"[WARN] 以下参数当前兼容层未使用：{', '.join(ignored_keys)}", file=sys.stderr)

    return cmd


def main(argv: list[str] | None = None) -> int:
    overrides, dry_run = _parse_overrides(sys.argv[1:] if argv is None else argv)
    cmd = _build_manager_cmd(overrides)
    print("[INFO] 转发到内部多卡评测命令：")
    print(" ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(ROBOTWIN_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
