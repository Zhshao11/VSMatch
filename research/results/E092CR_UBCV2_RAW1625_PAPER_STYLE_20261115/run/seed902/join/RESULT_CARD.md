# E092C-R seed902 join

Status: `COMPLETE`

```json
{
  "last_completed_stage": "CPU_GT_JOIN_UBCV2_RAW1625",
  "main_table": [
    "gdros_os_iter32_post_lsr",
    "os_m4_iter32_post_lsr"
  ],
  "metrics": {
    "gdros_os_iter32_post_lsr": {
      "public_component": {
        "cmr@1": 0.06153846153846154,
        "cmr@10": 39.13846153846154,
        "cmr@16": 61.47692307692307,
        "cmr@2": 0.9230769230769231,
        "cmr@3": 2.4615384615384617,
        "cmr@4": 5.846153846153846,
        "cmr@5": 9.784615384615385,
        "failures_ge_1": 1624,
        "failures_ge_10": 989,
        "failures_ge_16": 626,
        "failures_ge_2": 1610,
        "failures_ge_3": 1585,
        "failures_ge_4": 1530,
        "failures_ge_5": 1466,
        "mean": 22.249709569160753,
        "p95": 79.30497894287109,
        "p99": 126.60249877929687,
        "pairs": 1625,
        "std": 26.19488579131467,
        "success_subset@1": {
          "count": 1,
          "mean": 0.9503150582313538,
          "std": 0.0,
          "variance_official_rmse_field": 0.0
        },
        "success_subset@2": {
          "count": 15,
          "mean": 1.6523168524106344,
          "std": 0.2805863543453921,
          "variance_official_rmse_field": 0.07872870224483794
        },
        "success_subset@3": {
          "count": 40,
          "mean": 2.1829139456152915,
          "std": 0.4862925624931373,
          "variance_official_rmse_field": 0.23648045633614184
        },
        "success_subset@4": {
          "count": 95,
          "mean": 2.9574810172382153,
          "std": 0.760189281027576,
          "variance_official_rmse_field": 0.5778877429892229
        },
        "success_subset@5": {
          "count": 159,
          "mean": 3.57633819782509,
          "std": 0.9747694364224289,
          "variance_official_rmse_field": 0.9501754541832995
        },
        "variance": 686.1720416200193
      },
      "vector_epe": {
        "cmr@1": 0.0,
        "cmr@10": 16.73846153846154,
        "cmr@16": 39.32307692307692,
        "cmr@2": 0.12307692307692308,
        "cmr@3": 0.6153846153846154,
        "cmr@4": 1.7846153846153845,
        "cmr@5": 2.9538461538461536,
        "failures_ge_1": 1625,
        "failures_ge_10": 1353,
        "failures_ge_16": 986,
        "failures_ge_2": 1623,
        "failures_ge_3": 1615,
        "failures_ge_4": 1596,
        "failures_ge_5": 1577,
        "mean": 35.48944986922924,
        "p95": 126.73364410400389,
        "p99": 198.25835998535155,
        "pairs": 1625,
        "std": 41.44226217098487,
        "success_subset@1": {
          "count": 0,
          "mean": null,
          "std": null,
          "variance_official_rmse_field": null
        },
        "success_subset@2": {
          "count": 2,
          "mean": 1.7013640999794006,
          "std": 0.19679099321365356,
          "variance_official_rmse_field": 0.03872669501001624
        },
        "success_subset@3": {
          "count": 10,
          "mean": 2.3960035920143126,
          "std": 0.4023751722600725,
          "variance_official_rmse_field": 0.16190577925132305
        },
        "success_subset@4": {
          "count": 29,
          "mean": 3.134329758841416,
          "std": 0.6437959839229596,
          "variance_official_rmse_field": 0.41447326891533165
        },
        "success_subset@5": {
          "count": 48,
          "mean": 3.711041080454985,
          "std": 0.8865377369362949,
          "variance_official_rmse_field": 0.7859491590121271
        },
        "variance": 1717.4610938486437
      }
    },
    "os_m4_iter24_post_lsr": {
      "public_component": {
        "cmr@1": 46.09230769230769,
        "cmr@10": 85.6,
        "cmr@16": 96.18461538461538,
        "cmr@2": 60.30769230769231,
        "cmr@3": 66.70769230769231,
        "cmr@4": 70.33846153846154,
        "cmr@5": 73.41538461538461,
        "failures_ge_1": 876,
        "failures_ge_10": 234,
        "failures_ge_16": 62,
        "failures_ge_2": 645,
        "failures_ge_3": 541,
        "failures_ge_4": 482,
        "failures_ge_5": 432,
        "mean": 4.145811363518238,
        "p95": 12.923016738891599,
        "p99": 37.395139312744135,
        "pairs": 1625,
        "std": 6.755994545172115,
        "success_subset@1": {
          "count": 749,
          "mean": 0.4478361528709988,
          "std": 0.240233110792788,
          "variance_official_rmse_field": 0.05771194752117995
        },
        "success_subset@2": {
          "count": 980,
          "mean": 0.6694808638901735,
          "std": 0.47071243195620494,
          "variance_official_rmse_field": 0.22157019359812485
        },
        "success_subset@3": {
          "count": 1084,
          "mean": 0.8376389431846207,
          "std": 0.6897168622146319,
          "variance_official_rmse_field": 0.4757093500231976
        },
        "success_subset@4": {
          "count": 1143,
          "mean": 0.9756878955535793,
          "std": 0.8978243926317333,
          "variance_official_rmse_field": 0.8060886400045407
        },
        "success_subset@5": {
          "count": 1193,
          "mean": 1.1245465378611905,
          "std": 1.1324751235229735,
          "variance_official_rmse_field": 1.2824999053983743
        },
        "variance": 45.64346229439537
      },
      "vector_epe": {
        "cmr@1": 35.07692307692308,
        "cmr@10": 76.12307692307692,
        "cmr@16": 86.33846153846154,
        "cmr@2": 52.246153846153845,
        "cmr@3": 59.692307692307686,
        "cmr@4": 64.36923076923077,
        "cmr@5": 67.26153846153846,
        "failures_ge_1": 1055,
        "failures_ge_10": 388,
        "failures_ge_16": 222,
        "failures_ge_2": 776,
        "failures_ge_3": 655,
        "failures_ge_4": 579,
        "failures_ge_5": 532,
        "mean": 6.5114714877742985,
        "p95": 20.397357940673828,
        "p99": 59.97520080566405,
        "pairs": 1625,
        "std": 10.61005106796068,
        "success_subset@1": {
          "count": 570,
          "mean": 0.5395341358556036,
          "std": 0.23149548166601938,
          "variance_official_rmse_field": 0.05359015803178231
        },
        "success_subset@2": {
          "count": 849,
          "mean": 0.8359624673870203,
          "std": 0.4948317548138824,
          "variance_official_rmse_field": 0.24485846557218627
        },
        "success_subset@3": {
          "count": 970,
          "mean": 1.0382877814324247,
          "std": 0.7156065444832681,
          "variance_official_rmse_field": 0.5120927265072835
        },
        "success_subset@4": {
          "count": 1046,
          "mean": 1.2146920852334617,
          "std": 0.9371670093582029,
          "variance_official_rmse_field": 0.8782820034293978
        },
        "success_subset@5": {
          "count": 1093,
          "mean": 1.3537648795112927,
          "std": 1.1288618612155814,
          "variance_official_rmse_field": 1.2743291017071068
        },
        "variance": 112.57318366473355
      }
    },
    "os_m4_iter32_post_lsr": {
      "public_component": {
        "cmr@1": 45.96923076923077,
        "cmr@10": 84.73846153846154,
        "cmr@16": 96.06153846153846,
        "cmr@2": 61.169230769230765,
        "cmr@3": 67.07692307692308,
        "cmr@4": 70.52307692307693,
        "cmr@5": 73.23076923076923,
        "failures_ge_1": 878,
        "failures_ge_10": 248,
        "failures_ge_16": 64,
        "failures_ge_2": 631,
        "failures_ge_3": 535,
        "failures_ge_4": 479,
        "failures_ge_5": 435,
        "mean": 4.2034320709705355,
        "p95": 12.998975944519042,
        "p99": 37.02005813598632,
        "pairs": 1625,
        "std": 6.832092019594629,
        "success_subset@1": {
          "count": 747,
          "mean": 0.46629556167716163,
          "std": 0.2374334313700807,
          "variance_official_rmse_field": 0.05637463433217082
        },
        "success_subset@2": {
          "count": 994,
          "mean": 0.6942332796227764,
          "std": 0.46897784776819745,
          "variance_official_rmse_field": 0.21994022169729058
        },
        "success_subset@3": {
          "count": 1090,
          "mean": 0.8534840479083018,
          "std": 0.68700351104772,
          "variance_official_rmse_field": 0.4719738241918947
        },
        "success_subset@4": {
          "count": 1146,
          "mean": 0.9843028205695576,
          "std": 0.8864434037584277,
          "variance_official_rmse_field": 0.7857819080668269
        },
        "success_subset@5": {
          "count": 1190,
          "mean": 1.1160499999753568,
          "std": 1.1009562091091798,
          "variance_official_rmse_field": 1.212104574376056
        },
        "variance": 46.67748136420862
      },
      "vector_epe": {
        "cmr@1": 34.52307692307692,
        "cmr@10": 75.87692307692308,
        "cmr@16": 85.78461538461538,
        "cmr@2": 52.43076923076922,
        "cmr@3": 60.49230769230769,
        "cmr@4": 64.43076923076923,
        "cmr@5": 67.38461538461539,
        "failures_ge_1": 1064,
        "failures_ge_10": 392,
        "failures_ge_16": 231,
        "failures_ge_2": 773,
        "failures_ge_3": 642,
        "failures_ge_4": 578,
        "failures_ge_5": 530,
        "mean": 6.59988871636299,
        "p95": 20.363276672363273,
        "p99": 59.34839187622069,
        "pairs": 1625,
        "std": 10.7283607988721,
        "success_subset@1": {
          "count": 561,
          "mean": 0.5682879051927364,
          "std": 0.23221323131396218,
          "variance_official_rmse_field": 0.0539229847972717
        },
        "success_subset@2": {
          "count": 852,
          "mean": 0.8660690673132877,
          "std": 0.4832244281197261,
          "variance_official_rmse_field": 0.23350584793163634
        },
        "success_subset@3": {
          "count": 983,
          "mean": 1.0762438376870913,
          "std": 0.707758954833585,
          "variance_official_rmse_field": 0.5009227381471285
        },
        "success_subset@4": {
          "count": 1047,
          "mean": 1.2230002885148883,
          "std": 0.8976026346222706,
          "variance_official_rmse_field": 0.8056904896808416
        },
        "success_subset@5": {
          "count": 1095,
          "mean": 1.3652404152027005,
          "std": 1.1018962151516638,
          "variance_official_rmse_field": 1.2141752689655618
        },
        "variance": 115.09772543077561
      }
    }
  },
  "paper_protocol_scope": "paper-style independent seed; not authors' screened 1447/exact seed",
  "per_pair_csv": "/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-092CR-UBCV2-RAW1625-PAPER-STYLE/seed902/join/PER_PAIR_METRICS.csv",
  "per_pair_csv_sha256": "f2d34e94ce8bb9e2aab9cf20be0c60b4a5c9e887a390042489e0f99c91216e74",
  "prediction_manifests_verified_before_gt_load": true,
  "public_formula_parity": {
    "fixture": {
      "official": 3.5,
      "public": 3.5,
      "status": "PASS",
      "vector": 5.0
    },
    "max_abs_difference": 0.0,
    "numpy_reduction_max_abs_difference": 2.288818359375e-05,
    "status": "PASS",
    "tolerance": 1e-06
  },
  "seed": 902,
  "status": "COMPLETE",
  "supplement_only": "os_m4_iter24_post_lsr",
  "task": "join",
  "updated_at_unix": 1794720022.8076048,
  "whole_image_no_mask": true
}
```
