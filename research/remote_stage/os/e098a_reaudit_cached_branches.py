#!/usr/bin/env python3
"""Re-audit E093A's cached 32-step branch outputs against accepted E096A M4.

Preserves the failed old E092A regression card. This is a new cache audit,
not a silent change to historical acceptance status or a fresh GPU inference.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import e092a_official_eval as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError("Use an empty output directory")
    run = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-093A-OS-BC-FROZEN-ABLATION-AND-PROFILE"
    current = base.ROOT / "GDROS/experiments/sparsegeo/runs/E-096A-GDROS-CLEAN-M4-OS424-20261116"
    preflight = json.loads((run / "preflight/RESULT_CARD.json").read_text())
    predict = json.loads((run / "predict/RESULT_CARD.json").read_text())
    failed = json.loads((run / "join/RESULT_CARD.json").read_text())
    assert preflight["status"] == "PASS" and predict["status"] == "COMPLETE"
    assert preflight["hashes"][str(base.M4_CKPT)] == base.EXPECTED[base.M4_CKPT]
    assert preflight["dataset_provenance"]["all_match_archive"]
    accepted = {r["pair_index"]:r for r in json.loads((current/"PREDICTION_MANIFEST.json").read_text()) if r["variant"]=="m4_clean"}
    indices = base.pairs(include_gt=False)
    manifests, manifest_audit, control = {}, {}, []
    for variant in ("b", "c", "avg", "m4"):
        metadata = predict["manifests"][f"{variant}_iter32_post_lsr"]
        path = Path(metadata["path"])
        assert base.sha256(path) == metadata["sha256"]
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records)==424
        for row, record in zip(indices,records):
            assert row["pair_index"]==record["pair_index"] and row["pair_id"]==record["pair_id"]
            assert not record["uses_truth_flow"] and record["iters"]==32 and record["post_lsr"]
            assert base.sha256(Path(record["prediction"]))==record["sha256"]
            if variant=="m4":
                archived=accepted[row["pair_index"]]
                assert base.sha256(Path(archived["path"]))==archived["sha256"]
                left=np.load(record["prediction"],allow_pickle=False)
                right=np.load(archived["path"],allow_pickle=False)
                delta=float(np.abs(left-right).max())
                control.append({"pair_index":row["pair_index"],"max_abs":delta,"bitwise_equal":bool(np.array_equal(left,right))})
                assert delta==0
        manifests[variant]=records
        manifest_audit[variant]={"path":str(path),"sha256":metadata["sha256"],"verified_predictions":len(records)}
    print(json.dumps({"stage":"ALL_CACHE_HASHES_AND_M4_CONTROL_VERIFIED"}),flush=True)
    metrics={v:{"public_component":[],"vector_epe":[]} for v in manifests}
    metric_rows=[]
    for row in base.pairs(include_gt=True):
        gt=base.read_flo(row["gt"]).astype(np.float32,copy=False)
        result={"pair_index":row["pair_index"],"pair_id":row["pair_id"]}
        for variant,records in manifests.items():
            pred=np.load(records[row["pair_index"]]["prediction"],allow_pickle=False)[0]
            assert pred.shape==gt.shape and np.isfinite(pred).all()
            delta=pred-gt
            values={"public_component":float(np.mean((np.abs(delta[0])+np.abs(delta[1]))/2)),
                    "vector_epe":float(np.mean(np.sqrt(np.square(delta[0])+np.square(delta[1]))))}
            for name,value in values.items():
                metrics[variant][name].append(value)
                result[f"{variant}__{name}"]=value
        metric_rows.append(result)
    with (out/"METRICS.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(metric_rows[0]))
        writer.writeheader();writer.writerows(metric_rows)
    card={"status":"COMPLETE","new_audit":True,"fresh_inference":False,"pairs":424,"iterations":32,
          "checkpoint_sha256":base.EXPECTED[base.M4_CKPT],"manifests":manifest_audit,
          "metrics":{v:{name:base.summarize(values) for name,values in by.items()} for v,by in metrics.items()},
          "m4_vs_accepted_e096a":{"bitwise_equal_pairs":sum(r["bitwise_equal"] for r in control),"max_abs":max(r["max_abs"] for r in control)},
          "historical_join_status":failed["status"],"historical_join_failure_reason":failed.get("failure_reason"),
          "historical_files_preserved":True,"prediction_before_gt_join":True,
          "source_hashes":{str(Path(__file__).resolve()):base.sha256(Path(__file__).resolve()),str(Path(base.__file__).resolve()):base.sha256(Path(base.__file__).resolve())}}
    base.write_json(out/"RESULT_CARD.json",card)
    base.write_json(out/"M4_CONTROL_COMPARISONS.json",control)
    print(json.dumps({"stage":"COMPLETE","metrics":card["metrics"]}),flush=True)


if __name__=="__main__":
    main()
