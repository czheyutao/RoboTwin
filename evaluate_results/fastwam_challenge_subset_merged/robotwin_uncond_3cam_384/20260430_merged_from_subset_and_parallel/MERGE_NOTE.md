# FastWAM RoboTwin Challenge Subset 合并结果

- 合并时间: 2026-04-30
- 前 3 个任务来源: `/data/hyt/RoboTwin/evaluate_results/fastwam_challenge_subset/robotwin_uncond_3cam_384/20260428_213615`
- 后 7 个任务来源: `/data/hyt/RoboTwin/evaluate_results/fastwam_challenge_subset_parallel/robotwin_uncond_3cam_384/20260429_104543`
- overall_success_rate: `0.8340`

| task_name | success_rate | source |
|---|---:|---|
| blocks_ranking_rgb | 1.00 | parallel_last7 |
| blocks_ranking_size | 0.96 | parallel_last7 |
| handover_mic | 1.00 | subset_first3 |
| move_can_pot | 0.98 | parallel_last7 |
| move_stapler_pad | 0.70 | subset_first3 |
| open_microwave | 0.28 | parallel_last7 |
| place_can_basket | 0.62 | parallel_last7 |
| place_dual_shoes | 0.90 | parallel_last7 |
| place_fan | 0.96 | subset_first3 |
| stack_blocks_three | 0.94 | parallel_last7 |


我刚查了下，现在采集已经推进不少了，**还在跑的只剩 4 个 randomized 任务**：

| 任务 | 当前进度 |
|---|---:|
| `stack_blocks_three/demo_randomized` | `460 / 500` |
| `move_can_pot/demo_randomized` | `338 / 500` |
| `blocks_ranking_rgb/demo_randomized` | `98 / 500` |
| `blocks_ranking_size/demo_randomized` | `83 / 500` |

已经完成的现在是：

- 全部 `demo_clean`：**10/10 完成**
- `demo_randomized` 已完成：
  - `handover_mic`
  - `move_stapler_pad`
  - `open_microwave`
  - `place_can_basket`
  - `place_dual_shoes`
  - `place_fan`

也就是说现在总体是：

- `demo_clean`：`10/10`
- `demo_randomized`：`6/10`
- 全部任务对：`16/20`