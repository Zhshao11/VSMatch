# E-037A zzvr projection/norm replay status

更新时间：`2026-08-27`

目标：在 zzvr 上复现 E-037A `NO_INTERACTION_TRAINED_PROJECTION_lr125e4`，对齐 bl_tunnel 指标：

- AEPE `7.247`
- CMR@2 `10.720%`
- CMR@5 `45.939%`
- CMR@16 `92.337%`

## 已完成

- zzvr 迁移根目录：`/mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated`
- 已验证环境：
  - `/mnt/sda/zhushihao/multiModel_matching/xoftr_sar/envs/xoftr5090`
  - PyTorch `2.8.0+cu128`
  - CUDA available
  - `8` GPUs visible
- 已同步/构建 E037A package 依赖：
  - `experiments/sparsegeo/e003a`
  - `experiments/sparsegeo/e023a`
  - `experiments/sparsegeo/e025a`
  - `experiments/sparsegeo/e028b`
  - `experiments/sparsegeo/e034c_e028b_coarse_inlier_selection_sweep`
  - `experiments/sparsegeo/e037a_plain_feature_interaction_coarse`
- E037A `py_compile`：PASS。

## 正在进行

- 本地 official archive 正在 rsync 到：
  - `/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros/training.zip`
  - `/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros/testing.zip`
- 本地源：
  - `/Users/shihao/毕业论文/os_dataset/training.zip`
  - `/Users/shihao/毕业论文/os_dataset/testing.zip`
- SHA 已确认：
  - training.zip `2887e7de8cd72290df688de0ce4071f42d7bfdec8507a24716ae2ea15c3f3cf5`
  - testing.zip `e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218`

## 当前阻塞

- E028B safe dual adapter checkpoint `projection_only_safe.pt` 尚未定位到 zzvr 或本地副本。
- E037A frozen 期望 checkpoint SHA：
  - `5f4707e73b68de0e4420ca2589dada5a929231493fa57a7a05994ebe2538e485`
- 没有该 checkpoint 时，无法声称是同一 E037A projection/norm replay。

## 下一步

1. 等 official archive 完成同步；
2. 解压 training.zip 并重建/恢复 E003A split manifest；
3. 定位或恢复 E028B `projection_only_safe.pt`；
4. 用 zzvr `xoftr5090` 环境启动 E037A，只保留 `NO_INTERACTION_TRAINED_PROJECTION_lr125e4`
   作为 projection/norm replay track；
5. 对比 bl_tunnel 指标是否复现。

## 禁止

- 不启动 E038/E039；
- 不 finetune E028B；
- 不接 GDROS backend/dense GRU；
- 不用 testing.zip 做选择或训练；
- 不启动 OS424/confirm/rescue。
