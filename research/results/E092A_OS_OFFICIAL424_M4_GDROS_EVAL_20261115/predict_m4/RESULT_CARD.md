# E092A predict_m4

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
    "checkpoint_step": 80000,
    "missing_keys": [],
    "state_keys": 5583,
    "unexpected_keys": []
  },
  "logical_device": "cuda:0",
  "manifests": {
    "m4_iter24_post_lsr": {
      "path": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-092A-OS-OFFICIAL424-M4-GDROS-EVAL/predictions/m4_iter24_post_lsr/PREDICTIONS.jsonl",
      "records": 424,
      "sha256": "8f050a340f65a17365c1ff2f767c421a18836e9bf5423aeb0530f52926a35fe2"
    },
    "m4_iter32_post_lsr": {
      "path": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-092A-OS-OFFICIAL424-M4-GDROS-EVAL/predictions/m4_iter32_post_lsr/PREDICTIONS.jsonl",
      "records": 424,
      "sha256": "c54756570838339d7078fc695a0df3e98c007b139bee2e92befd776c88d6a44a"
    }
  },
  "status": "COMPLETE",
  "task": "predict_m4",
  "updated_at_unix": 1794709959.1885846
}
```
