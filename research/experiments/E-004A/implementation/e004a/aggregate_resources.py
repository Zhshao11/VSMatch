#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np

def run(args):
    rows=[json.loads(p.read_text()) for p in args.inputs]
    if len(rows)!=5: raise RuntimeError("exactly five fresh processes required")
    if len({str(p.resolve()) for p in args.inputs})!=5 or len({r["process"]["pid"] for r in rows})!=5 or len({r["process"]["run_uuid"] for r in rows})!=5: raise RuntimeError("five distinct fresh process artifacts required")
    p50=np.array([r["end_to_end"]["p50_ms"] for r in rows]); p95=np.array([r["end_to_end"]["p95_ms"] for r in rows]); cv=float(p50.std()/p50.mean())
    gates={"all_workers_pass":all(r["status"]=="PASS" for r in rows),"median_p50_le18":float(np.median(p50))<=18,"median_p95_le25":float(np.median(p95))<=25,"p50_cv_le5pct":cv<=.05,"cold_warm_memory":all(r["cold_peak_mib"]<=284 and r["end_to_end"]["peak_allocated_mib"]<=284 for r in rows)}
    result={"status":"PASS" if all(gates.values()) else "FAIL","process_p50_ms":p50.tolist(),"process_p95_ms":p95.tolist(),"median_p50_ms":float(np.median(p50)),"median_p95_ms":float(np.median(p95)),"p50_cv":cv,"gates":gates,"inputs":[str(p) for p in args.inputs]}; args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");
    if result["status"]!="PASS": raise SystemExit(2)
def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,nargs=5,required=True);p.add_argument("--output",type=Path,required=True);return p.parse_args()
if __name__=="__main__":run(parse_args())
