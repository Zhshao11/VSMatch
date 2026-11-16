# E092C UBCv2 原始归档计数更正与后续实验身份

本文件更正同目录初版 `RESULT_CARD` 的归档成员计数；保留初版卡作为审计历史，不覆盖原记录。

## 计数更正

- 多模态官方原始包：`/Users/shihao/Downloads/UBCv2_test_images/multi-modal_fine-grained_building_roof_instance_segmentation/test_images.zip`
- 文件大小：`2,405,008,673` bytes
- SHA256：`515559a9f2b67981e86d7bd36170d720658a38724f2309bb3003a0d565960aeb`
- 与 `zzvr:/mnt/sda/zhushihao/VS_Match/datasets/ubcv2/raw/test_images.zip` 完全同哈希。
- ZIP 有 3252 个非目录成员，其中 3250 个 `.tif` 和 2 个 `.xml` sidecar；RGB/SAR 各 1625 张真实 TIFF。因此正确配对数为 1625，不是初版卡误记的 1626。
- 另一个 `fine-grained_building_roof_instance_segmentation/test_images.zip` SHA256 为 `2d16457539de2560d19b729fd7a6bab54de09b2ccb8290e26e651a3276c20dc3`，只含 2273 张单模态光学 TIFF，不用于 RGB-SAR 匹配评测。

## 状态更新

用户确认上述官方原始多模态包已下载并上传。它仍不包含 GDROS 作者筛选后的 1447 ID、三组作者实际变换或作者 GT，因此不能做作者 1447 精确回放；但可基于全部 1625 官方原始配对，按论文 UBCv2 参数范围构造预注册的三 seed 独立复现实验 E092C-R。

E092C-R 必须明确标注为“官方原始 1625 对上的 paper-style 独立泛化复现”，不得冒充论文作者 1447 对或其三 seed 精确结果。M4 固定 OS E075A step80K，GDROS 固定官方 OS 权重；无训练、无按结果筛图。
