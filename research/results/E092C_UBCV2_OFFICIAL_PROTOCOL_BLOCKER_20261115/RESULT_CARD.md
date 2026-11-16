# E092C UBCv2 官方协议资产阻塞卡

状态：`BLOCKED_MISSING_AUTHOR_TEST_PROTOCOL`。本轮不提交 UBCv2 GPU 评测。

## 已核实资产

- 原始归档：`/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/raw/test_images.zip`
- SHA256：`515559a9f2b67981e86d7bd36170d720658a38724f2309bb3003a0d565960aeb`
- 归档含 RGB/SAR 各 1626 张 TIFF，共 3252 个成员。现有解压目录各有 1625 张 TIFF，另各有 1 个 `.tif.aux.xml` sidecar。
- 本地 `generated/seed901` 是 1625 对的本地仿射生成集；manifest 明确记录 `prescreen: none (all official test pairs kept; paper pre-screened to 1447)`。
- `seed901` manifest SHA256：`599b8bdab2fc32486c4d4f46ee496d065e3649ecdad918a9906f96efa4e0997d`。
- 生成器 SHA256：`53f788cbe6bf8516c90c63ab623575b86168944d27dc68bf32ffd152e57bf3d2`。

## 不能执行官方评测的原因

GDROS 论文的 UBCv2 是从 7170 对经作者筛选后得到 `3517/1437/1447` 的 train/val/test，并使用三组不同随机测试变换汇总 mean±std。现有资产没有提供作者测试 1447 个 ID、三组实际变换/seed、作者 GT/warp 或可复核的筛选与划分映射。官方 README 也没有发布 UBCv2 数据入口或对应权重。

论文 Table III 的主阈值是 `CMR@2/3/5`；合同仍要求保留 `1..5` 全阈值。这个阈值事实不能补足缺失的测试身份。

## 解锁条件

需要作者筛选后的 1447 测试 ID，以及三组作者测试变换与 GT（或足以逐项重建它们的 seed、筛选、warp/crop/interpolation/padding 细节）。若目标是复现论文中 target-trained GDROS 行，还需作者 UBCv2 checkpoint。

在这些文件可核验前，不得把已有 1625 对 `seed901` 数字写成官方 E092C，不得按模型误差删到 1447 对，也不得把单组逐对标准差解释为论文三组标准差。
