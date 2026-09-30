# REVIEW_PACKET：E-032A-E028B-NATIVE-MATCHER-DIAG

只审 E-032A native matcher 设计/诊断。不得回到 E028B-to-GDROS-backend adapter。

## 审查重点

1. Primary track 必须直接消费 E-028B safe dual adapter features，不设计 GDROS backend adapter。
2. 不复用 GDROS dense GRU/backend 作为 primary matcher。
3. 不训练新 descriptor；optional trainable matcher head 只能在 no-train matcher 显示 positive signal 后启用。
4. Candidate/correspondence 必须在读取 truth_flow/GT 前 materialize。
5. 输出必须区分 coarse matching、local refinement、affine consensus 三类瓶颈。

## 必读文件

- `research/CURRENT_STATE.md`
- `research/METHOD_DELTA.md`
- `research/INNOVATION_CONTRACT.yaml`
- `research/experiments/E-032A-E028B-NATIVE-MATCHER-DIAG.md`

## Blocking findings

- `ADAPTER_ROUTE_BLOCKED`：又把 E-028B 接入 GDROS backend/adapter 作为主路线。
- `LEAKAGE_BLOCKED`：truth_flow/GT 在 candidate materialization 前参与匹配。
- `CHECKPOINT_POLLUTION_BLOCKED`：写入/覆盖原 E-028B、DINO 或 GDROS baseline checkpoint。
- `UNINTERPRETABLE_BLOCKED`：不能区分 matching/refinement/consensus 瓶颈。

## 输出

只给结论与会改变方法语义的 findings。若无 blocking finding，写 `APPROVE`。
