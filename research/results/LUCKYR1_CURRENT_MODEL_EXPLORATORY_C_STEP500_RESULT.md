# exploratory_c

Status: `COST_GATE_STOP_NO_SIGNAL`

```json
{
  "claim_scope": "CURRENT_MODEL_EXPLORATORY_ONLY",
  "cost_gate_policy": "continue only if (CMR@5, CMR@2, CMR@1, -AEPE) lexicographically exceeds protected M4",
  "cost_gate_signal": false,
  "evaluation_pairs": 404,
  "evaluations": [
    {
      "checkpoint": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/current_model_exploratory/exploratory_c/checkpoints/exploratory_c_step500.pt",
      "checkpoint_sha256": "365c9d64aa9b41f16af9469c0b5d8a2d82e18f3aa2bac810818bd173564e0b8d",
      "metrics": {
        "aepe": 2.36151146471943,
        "cmr1": 56.43564356435643,
        "cmr10": 94.80198019801979,
        "cmr16": 97.02970297029702,
        "cmr2": 78.96039603960396,
        "cmr5": 88.86138613861387,
        "metric_join": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/current_model_exploratory/exploratory_c/step500_iter24_post_lsr_METRIC_JOIN.csv",
        "metric_scope": "pair-level",
        "pairs": 404,
        "prediction_manifest": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/current_model_exploratory/exploratory_c/predictions/step500_iter24_post_lsr/PREDICTIONS.jsonl",
        "prediction_manifest_sha256": "9dbcf3d2fe0541c6542b0d0251c959fea6176001a2351aa8c7e186d5f8bf72f2",
        "prediction_materialized_before_gt_join": true
      },
      "step": 500
    }
  ],
  "implementation_version": "luckyr1-bc-v3-e074b-evidence-graph-20260915",
  "join_count": 404,
  "last_completed_stage": "STEP500_FULL_404_EVALUATED_COST_GATE",
  "manifest_count": 404,
  "prediction_count": 404,
  "prediction_materialized_before_gt_join": true,
  "protected_m4": {
    "metrics": {
      "aepe": 2.3613,
      "cmr1": 56.683168316831676,
      "cmr10": 94.80198019801979,
      "cmr16": 97.02970297029702,
      "cmr2": 78.96039603960396,
      "cmr5": 88.86138613861387
    },
    "sha256": "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b"
  },
  "protected_m4_modified": false,
  "selected": {
    "checkpoint": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/current_model_exploratory/exploratory_c/checkpoints/exploratory_c_step500.pt",
    "checkpoint_sha256": "365c9d64aa9b41f16af9469c0b5d8a2d82e18f3aa2bac810818bd173564e0b8d",
    "metrics": {
      "aepe": 2.36151146471943,
      "cmr1": 56.43564356435643,
      "cmr10": 94.80198019801979,
      "cmr16": 97.02970297029702,
      "cmr2": 78.96039603960396,
      "cmr5": 88.86138613861387,
      "metric_join": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/current_model_exploratory/exploratory_c/step500_iter24_post_lsr_METRIC_JOIN.csv",
      "metric_scope": "pair-level",
      "pairs": 404,
      "prediction_manifest": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN/current_model_exploratory/exploratory_c/predictions/step500_iter24_post_lsr/PREDICTIONS.jsonl",
      "prediction_manifest_sha256": "9dbcf3d2fe0541c6542b0d0251c959fea6176001a2351aa8c7e186d5f8bf72f2",
      "prediction_materialized_before_gt_join": true
    },
    "step": 500
  },
  "source_job": "20260915_093904_7b216495",
  "status": "COST_GATE_STOP_NO_SIGNAL",
  "task": "exploratory_c",
  "train_pairs": 2011
}
```
