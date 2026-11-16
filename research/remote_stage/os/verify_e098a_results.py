#!/usr/bin/env python3
"""Recompute E098A cached-branch statistics and profile aggregates offline."""
import argparse
import json
from pathlib import Path
import statistics


def close(left, right, tolerance=1e-8):
    if abs(left-right)>tolerance:
        raise AssertionError((left,right))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    root=parser.parse_args().directory
    cache=json.loads((root/"cache_audit/RESULT_CARD.json").read_text())
    metrics=json.loads((root/"cache_audit/METRICS_ROWS.json").read_text())
    assert cache["status"]=="COMPLETE" and cache["pairs"]==424 and len(metrics)==424
    assert cache["m4_vs_accepted_e096a"]=={"bitwise_equal_pairs":424,"max_abs":0.0}
    assert cache["historical_files_preserved"] and not cache["fresh_inference"]
    for variant in ("b","c","avg","m4"):
        assert cache["manifests"][variant]["verified_predictions"]==424
        for name in ("public_component","vector_epe"):
            errors=[float(r[f"{variant}__{name}"]) for r in metrics]
            expected=cache["metrics"][variant][name]
            close(statistics.fmean(errors),expected["mean"])
            close(statistics.pstdev(errors),expected["std"])
            for threshold in (1,2,3,4,5,10,16):
                close(sum(e<threshold for e in errors)/424*100,expected[f"cmr@{threshold}"])
    for folder in ("profile_fine","profile_coarse"):
        card=json.loads((root/folder/"RESULT_CARD.json").read_text())
        rows=json.loads((root/folder/"ROWS.json").read_text())
        parity=json.loads((root/folder/"PARITY.json").read_text())
        assert card["status"]=="COMPLETE" and len(rows)==len(parity)==120
        assert card["source_hashes_start"]==card["source_hashes_end"]
        assert card["instrumentation_restored"]
        assert card["instrumentation_parity_max_abs"]==0
        for variant in ("m4_clean","gdros"):
            expected=card["summary"][variant]
            for phase,key in (("plain","plain_mean_ms"),("instrumented","instrumented_wall_mean_ms")):
                selected=[r for r in rows if r["phase"]==phase and r["variant"]==variant]
                assert len(selected)==30
                assert len({(r["pair_index"],r["repeat"]) for r in selected})==30
                close(statistics.fmean(r["wall_ms"] for r in selected),expected[key])
            selected=[r for r in rows if r["phase"]=="instrumented" and r["variant"]==variant]
            for record in selected:
                close(sum(record["categories_ms"].values()),record["stream_total_ms"],1e-5)
                assert record["category_calls"]==expected["category_calls_per_pair"]
            close(statistics.fmean(r["stream_total_ms"] for r in selected),expected["instrumented_stream_mean_ms"])
            for key,value in expected["categories_mean_ms"].items():
                close(statistics.fmean(r["categories_ms"].get(key,0) for r in selected),value)
        for record in parity:
            assert record["max_abs"]<=1e-4
            if record["phase"]=="instrumented":
                assert record["max_abs"]==0
    print(json.dumps({"status":"PASS","branch_pairs":424,"cached_branch_predictions_hashed":1696,
                      "profile_records_per_protocol":120,"instrumentation_output_change":0},indent=2))


if __name__=="__main__":
    main()
