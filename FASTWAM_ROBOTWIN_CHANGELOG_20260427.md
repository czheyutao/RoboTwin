# FastWAM 集成 RoboTwin 改动日志

记录时间：2026-04-27

## 目标

在只有 CLI 的服务器上，将 FastWAM 集成到 RoboTwin，并跑通 RoboTwin 上的 FastWAM eval。

最终验证结果：

- 任务：`click_alarmclock`
- 配置：`demo_randomized`
- episode 数：`1`
- checkpoint：`third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`
- dataset stats：`third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json`
- 结果：`1/1` 成功，成功率 `100%`
- 结果文件：`evaluate_results/fastwam/robotwin_uncond_3cam_384/20260427_164757/click_alarmclock/_result_random.txt`
- eval 日志：`evaluate_results/fastwam/robotwin_uncond_3cam_384/20260427_164757/click_alarmclock/eval_click_alarmclock_demo_randomized_20260427_164757.log`

## 仓库接入改动

1. 添加 FastWAM 子模块。
   - 新增：`.gitmodules`
   - 新增：`third_party/FastWAM`
   - 固定代码来源：`yuantianyuan01/FastWAM`
   - 本次使用的 FastWAM 提交：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`

2. 添加 RoboTwin policy 软链接。
   - 新增：`policy/fastwam_policy`
   - 指向：`third_party/FastWAM/experiments/robotwin/fastwam_policy`
   - 作用：让 RoboTwin 原有的 `policy_name=fastwam_policy` 机制可以直接导入 FastWAM policy。

## 主仓库代码改动

### `script/eval_policy.py`

为兼容 FastWAM eval 做了以下调整：

- 支持通过命令行 overrides 传入 FastWAM 所需参数。
- 支持自定义 `eval_output_dir`。
- 支持通过 `eval_video_log=false` 关闭评测视频保存。
- 支持 `score_mode=true`，按任务环境提供的 `get_eval_score()` 输出挑战得分。
- 根据 `task_config` 输出 `_result_clean.txt` 或 `_result_random.txt`。
- 保留 RoboTwin 原有 eval 流程，同时让 FastWAM policy 能在同一入口下运行。

### `script/eval_fastwam_challenge_subset.py`

新增挑战子集顺序评测脚本。

主要功能：

- 固定顺序评测 `move_stapler_pad`、`place_fan`、`handover_mic`、`open_microwave`、`place_can_basket`、`place_dual_shoes`、`move_can_pot`、`stack_blocks_three`、`blocks_ranking_rgb`、`blocks_ranking_size`。
- 默认每个任务执行 `50` 条。
- 默认不传 `--score-mode`，沿用 RoboTwin 原始 `check_success()` 二值成功率评分。
- 默认传入 `--eval-video-log` 和 `--skip-get-obs-within-replan`，保存低帧更新率评测视频，避免完整逐步 RGB 渲染。
- 每个任务顺序调用 `script/eval_fastwam_single.py`，并汇总原始 `success_rate` 到 `challenge_summary.csv` 和 `challenge_summary.json`。

### `envs/_base_task.py`

为 eval 视频保存增加真实三相机拼接。

主要功能：

- 新增默认 `get_eval_score()`，二值任务沿用 `check_success()` 得分。
- 评测视频写帧时不再只写 `head_camera`。
- 当 `left_camera` 和 `right_camera` 存在时，生成真实三相机画布：
  - 上半部分居中显示 `head_camera`。
  - 下半部分左侧显示 `left_camera`。
  - 下半部分右侧显示 `right_camera`。
- 对默认 D435 配置，实际写入帧尺寸为 `640x480`，与 `script/eval_policy.py` 中的 ffmpeg `video_size` 一致。
- 修复此前 ffmpeg 以 `640x480` 读取 `320x240` 单相机字节流导致的 8 宫格假拼接问题。

### `envs/stack_blocks_three.py`

新增挑战得分规则：

- 完整三块堆叠成功：`1.0`
- 完成第一个堆叠：`0.4`
- 其他：`0.0`

### `envs/blocks_ranking_rgb.py` 与 `envs/blocks_ranking_size.py`

新增挑战得分规则：

- 第一个物块摆放成功：`0.2`
- 第二个物块摆放成功：`0.6`
- 第三个物块摆放成功或完整成功：`1.0`
- 其他：`0.0`

### `script/eval_fastwam_single.py`

新增单任务 FastWAM eval 入口脚本。

主要功能：

- 检查 checkpoint 和 dataset stats 是否存在。
- 自动确认 `policy/fastwam_policy` 软链接。
- 组装并调用 `script/eval_policy.py`。
- 为每次运行创建独立结果目录和日志文件。
- 设置 `CUDA_VISIBLE_DEVICES`，便于指定 GPU。
- 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，缓解 PyTorch CUDA 显存碎片问题。
- 设置 `CUDA_MODULE_LOADING=LAZY`，减少 CUDA 模块预加载显存占用。
- 支持 `--score-mode`，用于挑战细则得分。
- 支持 `--no-eval-video-log`，用于关闭完整视频保存。

### `script/eval_fastwam_manager.py`

新增批量 FastWAM eval 调度脚本。

主要功能：

- 支持单任务或全部任务。
- 支持 `clean`、`random` 或两种 phase 都跑。
- 支持多 GPU 并发调度。
- 支持限制每张 GPU 上并发任务数。
- 自动汇总成功率到 `summary.csv` 和 `summary.json`。
- 失败任务写入 `failed_tasks.txt`。

### `experiments/robotwin/run_robotwin_manager.py`

新增 FastWAM 原版 Hydra 风格命令兼容入口。

主要功能：

- 支持直接运行 `python experiments/robotwin/run_robotwin_manager.py task=... ckpt=... EVALUATION.dataset_stats_path=... MULTIRUN.num_gpus=...`。
- 自动转发到主仓库已适配的 `script/eval_fastwam_manager.py`。
- 自动将原版 FastWAM 命令里的 `./checkpoints/fastwam_release/...` 解析到 `third_party/FastWAM/checkpoints/fastwam_release/...`。
- 默认使用 `EVALUATION.instruction_type=unseen`、`demo_randomized` 对应的 `random` phase。
- 默认设置 `EVALUATION.skip_get_obs_within_replan=false`，用于保存更完整的视频帧。
- 默认设置 `MULTIRUN.max_tasks_per_gpu=1`，避免 24GB GPU 上同卡并发导致 OOM。
- 支持 `DRY_RUN=true` 或 `--dry-run` 只打印内部转发命令，便于启动前检查参数。

## FastWAM 子模块代码改动

### `third_party/FastWAM/experiments/robotwin/fastwam_policy/deploy_policy.py`

为在 24GB RTX 3090 上跑通 eval，新增 eval-only 组件 offload 逻辑。

背景：

- 原始 FastWAM policy 会把 text encoder、VAE、video DiT、ActionDiT 一次性放到同一张 GPU。
- 3090 24GB 在加载 ActionDiT 或推理时搬入 text encoder 会触发 CUDA OOM。
- 实际报错位置包括：
  - 初始化 ActionDiT 时分配额外显存失败。
  - 第一次动作推理时将 text encoder 搬上 GPU 失败。

本次改动：

- 在 CUDA eval 时，先在 CPU 上构建 FastWAM 模型并加载 checkpoint。
- checkpoint 加载后，只将 MoT 核心和 proprio encoder 放到目标 GPU。
- VAE 和 text encoder 默认放在 CPU。
- 文本 prompt 第一次使用时：
  - 临时把 MoT 核心挪到 CPU。
  - 把 text encoder 搬到 GPU 计算 `context/context_mask`。
  - 计算完成后将 text encoder 挪回 CPU。
  - 缓存 CPU 版 `context/context_mask`，后续同一 prompt 复用缓存。
- 图片 latent 编码时：
  - 临时把 MoT 核心挪到 CPU。
  - 把 VAE 搬到 GPU 编码首帧。
  - 编码完成后将 VAE 挪回 CPU。
  - 再把 MoT 核心放回 GPU 继续动作推理。
- 推理时优先传入缓存后的 `context/context_mask`，避免每次都重新加载 text encoder。

这个改动牺牲了一部分速度，但可以让 24GB 单卡跑通 FastWAM 在 RoboTwin 上的 eval。

## 下载和缓存内容

FastWAM 发布权重：

- `third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`
- `third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json`

Wan / DiffSynth 运行依赖缓存：

- `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`
- `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors`
- `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/*`

RoboTwin assets：

- 已下载并解压到 `assets/`。

## 环境处理记录

使用 conda 环境：

- 环境名：`fastwam_robotwin`
- Python：`3.10`
- PyTorch：`2.7.1+cu128`
- torchvision：`0.22.1+cu128`

额外处理：

- 安装 FastWAM editable 包。
- 安装 RoboTwin 依赖时避免降级 PyTorch / torchvision / huggingface_hub。
- 编译安装 PyTorch3D。
- 安装 curobo。
- 安装 `setuptools<81`，解决 `sapien` 缺少 `pkg_resources` 的兼容问题。
- 安装 conda 版 `vulkan-tools` 和 `libvulkan-loader`。
- patch 过本环境下的 `sapien` 和 `mplib` site-packages，用于让 RoboTwin 渲染和规划流程跑通。

已验证：

- `python script/test_render.py` 返回 `Render Well`。
- FastWAM policy 可以 import。
- `script/eval_fastwam_single.py`、`script/eval_fastwam_manager.py`、`experiments/robotwin/run_robotwin_manager.py`、`deploy_policy.py` 语法检查通过。
- `experiments/robotwin/run_robotwin_manager.py` 的 8 卡命令 dry-run 转发参数正确。

## 已跑通命令

激活环境：

```bash
source /data/hyt/anaconda3/etc/profile.d/conda.sh
conda activate fastwam_robotwin
```

单任务 smoke eval：

```bash
python script/eval_fastwam_single.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --task-name click_alarmclock \
  --task-config demo_randomized \
  --eval-num-episodes 1 \
  --gpu-id 0
```

批量 eval 示例：

```bash
python script/eval_fastwam_manager.py \
  --ckpt third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --dataset-stats-path third_party/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --phase random \
  --eval-num-episodes 10 \
  --num-gpus 8 \
  --max-tasks-per-gpu 1
```

兼容 FastWAM 原版写法的批量 eval 示例：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

## 注意事项

- CLI 服务器可以跑，不需要 Linux 虚拟机或可视化桌面。
- 当前 24GB 单卡依赖 eval-only offload 才能稳定加载和推理。
- offload 会增加每次重规划时的 CPU/GPU 搬运开销，因此完整全任务 eval 会较慢。
- 建议 `max-tasks-per-gpu=1`，避免多个 FastWAM eval 同时占用同一张 24GB GPU。
- `moviepy/pillow` 和 `scikit-image/scipy` 仍可能有 `pip check` 版本提示；本次保持 FastWAM 与 RoboTwin 当前可运行组合，没有强行调整。
