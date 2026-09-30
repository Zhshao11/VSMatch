# Proposed A Evidence Rebuild Validation

Date: `2026-09-09`

Status: `CPU_IMPLEMENTATION_PASS / GPU_INFERENCE_NOT_RUN`

## Source SHA256

- `global_multimodal_geometry_posterior.py`: `f66c7e0c9ce4dd5824cb1f2f5053f7c59320c2b6e8b9cdadc14f678650f9d242`
- `proposed_a_evidence_bundle.py`: `1ac792cb5a4c4398ff09a0765dfbc7eabfdd0aab9aed3251b2c774f98da1bb88`
- `proposed_a_d1_compare.py`: `ade17c562147791705c46b0dc2c4414be71c0352ea90c0d38105ffdac5c8dcfd`
- `proposed_a_gdros_frozen_provider.py`: `ffc254391b41d4d6be3fcf791417a79d62fdecf2fea986a35a0b0240a8f08bda`
- `proposed_a_flow_reassessment.py`: `24ad670a7158589b30807f00434401444fa2432f4332f552ed45b1558969138f`
- `proposed_a_prepare_evidence_view.py`: `f99fdd0d6860823df8a359c0ae94bd1a7a89a65d7d8e2fa671fd8509e7ea4f91`
- geometry CPU test: `6a5395ec1f5487a9d5a05ad938303486a398659e68c41ad21d9495f82dcda287`
- evidence pipeline CPU test: `4c587f2fd789ad9451785ee0383a4a539dbfe1f7a42766f41120391056da3322`

## Commands Actually Run

Local static compilation:

```bash
python3 -m py_compile <all Proposed A Python files>
```

zzvr CPU-only synthetic tests:

```bash
CUDA_VISIBLE_DEVICES="" /mnt/sda/zhushihao/multiModel_matching/xoftr_sar/envs/xoftr5090/bin/python /tmp/test_global_multimodal_geometry_posterior.py
CUDA_VISIBLE_DEVICES="" /mnt/sda/zhushihao/multiModel_matching/xoftr_sar/envs/xoftr5090/bin/python /tmp/test_proposed_a_evidence_pipeline.py
```

Result:

```text
global_multimodal_geometry_posterior CPU tests: PASS
{"status": "PASS", "tests": 4}
```

Manifest schema validation:

```text
proposed_a_d1_os_evidence_manifest.json OS 32 PASS approx_bundle_MiB 214.0
proposed_a_d1_whu_evidence_manifest.json WHU 32 PASS approx_bundle_MiB 214.0
```

The storage estimate covers fp16 source/target F8 features, float32 top-4 edges and raw flow, and masks. Expected total evidence storage is approximately `428 MiB` plus NPZ/JSON overhead. It is an estimate, not measured export usage.

Covered properties:

- affine/identity coordinate units;
- correct versus incorrect evidence scores;
- gradient through fixed-hypothesis scoring;
- fit-region exclusion and held-out-region evaluation;
- incompatible modes are not averaged;
- evidence schema round-trip and validation;
- translation/affine same-evidence D1 comparison;
- GT sidecar separation;
- CPU-equivalent LSR recovery on an exact synthetic affine flow;
- no-GT shared inference signature.

## Real E084 Flow Reassessment Check

Input: first 32 records from the already materialized E084A-A2 step60K WHU validation manifest. This is a limited operator check, not a final performance estimate and not a newly selected subset.

Command shape:

```bash
CUDA_VISIBLE_DEVICES="" .../python proposed_a_flow_reassessment.py \
  --materialized-root <E084A-A2 validation_step60000> \
  --truth-flow-root <E084 WHU validation truth_flow> \
  --max-pairs 32 \
  --output <temporary-json>
```

First attempt: `FAIL`, because the reporting-only RMSE expression used nonexistent `array.square()`. Fixed to `np.square(array)`; no method or metric-selection behavior changed.

Second attempt: `PASS`.

Selected stage summary:

| Stage | Operator | CMR@1 | CMR@2 | CMR@5 | AEPE |
| --- | --- | ---: | ---: | ---: | ---: |
| coarse | raw | 0.000% | 15.625% | 100.000% | 2.862 |
| coarse | post-LSR | 0.000% | 28.125% | 100.000% | 2.484 |
| F4 | raw | 0.000% | 15.625% | 100.000% | 2.853 |
| F4 | post-LSR | 3.125% | 31.250% | 100.000% | 2.475 |
| final/F2 | raw | 0.000% | 15.625% | 100.000% | 2.840 |
| final/F2 | post-LSR | 3.125% | 31.250% | 100.000% | 2.461 |

Derived interpretation limited to these 32 records:

- The raw/post-LSR re-evaluation path is feasible without candidate logits.
- LSR materially improves CMR@2 and AEPE on this subset.
- Fine output provides only a small gain relative to coarse under the same post-LSR operator.
- This subset cannot quantify the full 700-pair gap or serve as checkpoint selection evidence.

## Not Run

- No GPU inference.
- No frozen evidence bundle from real image inference.
- No real OS/WHU D1 candidate-family comparison.
- No training, optimizer, backward, smoke training, or hyperparameter search.
- No testing.zip evaluation or checkpoint selection.
- No existing queue or run modification.
