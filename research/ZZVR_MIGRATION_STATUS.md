# ZZVR Migration Status

更新时间：`2026-08-27`

## 目标

bl_tunnel 即将关机；将当前 M001 / E028B sparse matching 主线迁移到 zzvr，保证可以在 zzvr 上接手继续
E-038A 或重新启动 formal。

## zzvr 接手路径

- 迁移根目录：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated`
- 研究规格：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated/research`
- 工具脚本：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated/tools`
- 本地可用 staging：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated/tmp`

## 推荐环境

zzvr 上已验证可用：

- `/mnt/sda/zhushihao/multiModel_matching/xoftr_sar/envs/xoftr5090`
  - PyTorch `2.8.0+cu128`
  - CUDA `12.8`
  - `torch.cuda.is_available() == True`
  - `8` GPUs visible
- `/mnt/sda/zhushihao/图像匹配/EfficientLoFTR/.conda/eloftr5090`
  - PyTorch `2.8.0+cu128`
  - CUDA `12.8`
  - `8` GPUs visible

优先建议使用 `xoftr5090` 环境做 E-038A 重启/恢复。

## 已迁移

- `research/`：已同步到 zzvr；
- `tools/`：正在/已尝试同步到 zzvr；
- `tmp/e028b_staging`、`tmp/e029a_staging`：正在/已尝试同步到 zzvr；
- `.codex_tmp/e034a_e028b_recurrent_affine_matcher`：同步到 zzvr；
- `.codex_tmp/e034b_e028b_raw_top1_coarse_selection_diag`：同步到 zzvr；
- `.codex_tmp/e034c_e028b_coarse_inlier_selection_sweep`：同步到 zzvr；
- `.codex_tmp/e034d_e028b_cmr5_failure_attribution`：同步到 zzvr；
- `.codex_tmp/e037a_plain_feature_interaction_coarse`：同步到 zzvr；这是 E-037A 可运行骨架的主要本地来源；
- `.codex_tmp/e038a_strong_matcher_facing_projection`：同步到 zzvr；注意该目录为未完成草稿，工程复查显示
  `py_compile` 失败且仍有 E-037A/cross-attention/E036 遗留，不能直接作为 formal 运行。
- E-038A 冻结规格：
  - `research/METHOD_DELTA.md`
  - `research/experiments/E-038A-E028B-STRONG-MATCHER-FACING-PROJECTION.md`
  - `research/CURRENT_STATE.md`

## 关键 SHA256

- `CURRENT_STATE.md`: `9d4b1b90ed1deb1b4cc4fdc0df88f4a1de6404ff2221fa59dd9375938a91b8bf`
- `METHOD_DELTA.md`: `7fd3f3d1fb3a3892e45e6c1972b69ea0bacb238be6da9467841c1f14db0c33a0`
- `E-038A spec`: `f619dd0310338d60a58bdc220c3c43e3a4bfefe77fd96311344f3ee0a18c95af`
- `METHOD_SPEC.md`: `4891eccfedf347c8d62150defaa646644e001748df6f6bfbd289795b4f8db183`
- `INNOVATION_CONTRACT.yaml`: `7d96cc07351df0ec726d7d74489124b826024a00bf9b3e9922a4591ca8c902d2`

## 缺失/风险

- Codex app 一键 handoff 到 zzvr 失败：`No matching saved project was found on zzvr`。
- 从当前主线程 ssh `bl_tunnel` 失败：`websocket: bad handshake`。
- 因此目前无法从 bl_tunnel 拉取仍在运行/未同步的 E-038A watchdog 中间 checkpoint 或 heavy formal artifacts。
- 当前本地 workspace 没有 E-034C/E-037A/E-038A 的完整 heavy run root；这些正式 run root 原本位于
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/...`。
- zzvr 初查未发现同名 GDROS sparsegeo run 镜像；若 bl_tunnel 无法恢复，E-038A 需要在 zzvr 根据冻结规格重建/重跑。
- E-037A best projection checkpoint/formal artifacts 如果只存在于 bl_tunnel，则目前无法迁移；zzvr 侧可用 E-037A
  代码骨架和 frozen metrics，但 replay best checkpoint 可能需要重训或重新 materialize。

## 当前 E-038A 要点

E-038A 不是新方法扩展，而是调强 E-037A 成功的 matcher-facing projection/norm。

边界：

- E028B checkpoint 只读；
- 不 finetune E028B backbone/adapter；
- 不做 cross-attention / cross-image interaction；
- 不接 GDROS backend / dense GRU；
- E-034C downstream coarse selection / rank-growth / IRLS 必须保持不变；
- 不使用 testing.zip / confirm / OS424；
- outputs before GT metric join。

主对照：

- E-034C：AEPE `8.550`，CMR@2 `8.996%`，CMR@5 `38.783%`，CMR@16 `87.942%`；
- E-037A best projection/norm：AEPE `7.247`，CMR@2 `10.720%`，CMR@5 `45.939%`，
  CMR@16 `92.337%`。

成功判读：

- `PROJECTION_STRONG_GAIN_FOUND`：AEPE < `7.247` 且 CMR@5 > `45.939%`；
- `PROJECTION_MAJOR_GAIN_FOUND`：AEPE <= `6.5` 或 CMR@5 >= `52%`，且另一项不低于 E-037A。

## zzvr 接手建议

1. 进入迁移目录：
   - `cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated`
2. 激活环境：
   - `source /home/zzvr/miniconda3/bin/activate /mnt/sda/zhushihao/multiModel_matching/xoftr_sar/envs/xoftr5090`
3. 读取：
   - `research/CURRENT_STATE.md`
   - `research/METHOD_DELTA.md`
   - `research/experiments/E-038A-E028B-STRONG-MATCHER-FACING-PROJECTION.md`
4. 若 bl_tunnel 无法恢复，则在 zzvr 重建 E-034C/E-037A replay 依赖并重新启动 E-038A formal。
5. 代码参考：
   - E-034C：`.codex_tmp/e034c_e028b_coarse_inlier_selection_sweep`
   - E-037A：`.codex_tmp/e037a_plain_feature_interaction_coarse`
   - E-038A 草稿：`.codex_tmp/e038a_strong_matcher_facing_projection`
