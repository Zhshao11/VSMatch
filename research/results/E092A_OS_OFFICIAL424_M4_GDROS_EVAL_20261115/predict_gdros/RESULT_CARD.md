# E092A predict_gdros

Status: `COMPLETE`

Last completed stage: `424_PREDICTIONS_MATERIALIZED`

```json
{
  "environment": {
    "batch_size": 1,
    "cuda_runtime": "12.8",
    "fp32": true,
    "hostname": "ubuntu",
    "numpy": "2.2.6",
    "pillow": "12.2.0",
    "python": "3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]",
    "seed": 1234,
    "torch": "2.8.0+cu128"
  },
  "ground_truth_loaded": false,
  "last_completed_stage": "424_PREDICTIONS_MATERIALIZED",
  "load_report": {
    "missing_keys": [],
    "state_keys": 318,
    "unexpected_keys": []
  },
  "logical_device": "cuda:0",
  "manifests": {
    "gdros_iter32_post_lsr": {
      "path": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-092A-OS-OFFICIAL424-M4-GDROS-EVAL/predictions/gdros_iter32_post_lsr/PREDICTIONS.jsonl",
      "records": 424,
      "sha256": "d97385275945c0db589f512acc5cf9ae1d4346758c43f97e4074eb41b0c553ec"
    }
  },
  "status": "COMPLETE",
  "task": "predict_gdros",
  "updated_at_unix": 1794709488.2000413
}
```
