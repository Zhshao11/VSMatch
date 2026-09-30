#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sparsegeo.e002a.diagnostics import binary_ranking_metrics
from experiments.sparsegeo.e003a.candidate_cache import read_flow
from experiments.sparsegeo.e003a.data_protocol import load_split_manifest,sha256_file
from experiments.sparsegeo.e004a.anchor_heads import PostSearchVerifier, PreSearchRecoverabilityHead
from experiments.sparsegeo.e004a.oracle_gate import evaluate_anchor_set
from experiments.sparsegeo.e004a.proposals import allocate_coverage_constrained_k128, uniform_region_k128
from experiments.sparsegeo.e004a.replacement import build_post_features, replace_same_region_once
from experiments.sparsegeo.e004a.replacement import build_post_features_subset, plan_reserve_attempts, finalize_planned_replacements
from experiments.sparsegeo.e004a.head_cache import validate_head_cache
from experiments.sparsegeo.e004a.spec_binding import CHECKPOINT_SHA256
from experiments.sparsegeo.e004a.proposals import region_counts
from experiments.sparsegeo.e004a.provenance import source_manifest
from experiments.sparsegeo.e004a.proposal_cache import validate_cached_pair
from experiments.sparsegeo.e003a.candidate_cache import _load_image
from experiments.sparsegeo.e004a.spec_binding import SPLIT_MANIFEST_SHA256
from experiments.sparsegeo.e004a.train_anchor_heads import _cache_manifest


def _heads(path,source_sha,cache_sha):
    state=torch.load(path,map_location="cpu",weights_only=True)
    if state.get("source_manifest_sha256")!=source_sha or state.get("cache_source_sha256")!=source_sha or state.get("cache_content_manifest_sha256")!=cache_sha or state.get("split_manifest_sha256")!=SPLIT_MANIFEST_SHA256 or state.get("train_pairs")!=1607 or state.get("dev_pairs")!=404: raise RuntimeError("head checkpoint provenance mismatch")
    pre=PreSearchRecoverabilityHead(); post=PostSearchVerifier(); pre.load_state_dict(state["pre"]); post.load_state_dict(state["post"]); return pre.eval(),post.eval(),state["seed"]


def _summary(rows,name):
    values=[r[name] for r in rows]; valid=[v for v in values if v["valid"]]
    conditions=[v["condition"] for v in values if v["condition"] is not None]
    return {"recoverable_fraction":float(np.mean([v["recoverable_fraction"] for v in values])),"median_recoverable_count":float(np.median([v["recoverable_count"] for v in values])),"aepe":float(np.mean([v["aepe"] for v in valid])),"pair_local_ready_percent":float(100*np.mean([v["pair_local_ready"] for v in values])),"valid_percent":float(100*np.mean([v["valid"] for v in values])),"condition_mean":float(np.mean(conditions)),"condition_max":float(max(conditions))}


def _groups(record,p0l):
    flow=p0l["grid_truth_flow"].reshape(16,16,2); mapped=p0l["grid_xy_pixel"].reshape(16,16,2)+flow; step=511/15
    jx=(mapped[:,1:]-mapped[:,:-1]).mean((0,1))/step; jy=(mapped[1:]-mapped[:-1]).mean((0,1))/step
    rotation=abs(float(np.degrees(np.arctan2(float(jx[1]),float(jx[0]))))); scale=abs(float(np.log2(max(1e-6,abs(float(jx[0]*jy[1]-jx[1]*jy[0]))**.5))))
    displacement=float(torch.linalg.vector_norm(flow,dim=-1).mean()); image=_load_image(Path(record["files"]["opt"]["path"]))[0]; black=float((image.sum(0)==0).float().mean()); texture=float(image.mean(0).std())
    return {"displacement":"lt32" if displacement<32 else "32to64" if displacement<64 else "ge64","rotation":"lt5" if rotation<5 else "5to10" if rotation<10 else "ge10","scale":"lt0.05" if scale<.05 else "0.05to0.10" if scale<.1 else "ge0.10","black_border":"lt5pct" if black<.05 else "5to15pct" if black<.15 else "ge15pct","texture":"low" if texture<30 else "mid" if texture<60 else "high"}


def _group_bootstrap(rows,seed):
    rng=np.random.default_rng(seed); output={}
    for axis in ("displacement","rotation","scale","black_border","texture"):
        output[axis]={}
        for label in sorted({r["groups"][axis] for r in rows}):
            subset=[r for r in rows if r["groups"][axis]==label]; ae=np.array([r["main"]["aepe"] for r in subset if r["main"]["aepe"] is not None]); rf=np.array([r["main"]["recoverable_fraction"] for r in subset]); ids=rng.integers(0,len(subset),(500,len(subset))); aid=rng.integers(0,len(ae),(500,len(ae))) if len(ae) else None
            output[axis][label]={"count":len(subset),"valid_aepe_count":len(ae),"aepe_95_ci":np.quantile(ae[aid].mean(1),[.025,.975]).tolist() if len(ae) else None,"recoverable_fraction_95_ci":np.quantile(rf[ids].mean(1),[.025,.975]).tolist()}
    return output


def evaluate_seed(args,checkpoint):
    pre,post,seed=_heads(checkpoint,args.source_sha,args.cache_manifest_sha); records=load_split_manifest(args.split_manifest,allowed_split="dev"); rows=[]; all_labels=[]; all_scores=[]
    for record in records:
        pid=record["pair_id"]
        c=torch.load(args.cache_root/"candidates"/"dev"/f"{pid}.pt",map_location="cpu",weights_only=True)
        l=torch.load(args.cache_root/"labels"/"dev"/f"{pid}.pt",map_location="cpu",weights_only=True)
        validate_head_cache(c,l,record,"dev",args.source_sha,CHECKPOINT_SHA256)
        with torch.no_grad():
            pre_logits=pre(c["pre_features"]); pre_prob=torch.sigmoid(pre_logits)
        primary,reserves=allocate_coverage_constrained_k128(pre_prob)
        with torch.no_grad():
            primary_post=torch.sigmoid(post(build_post_features_subset(pre_logits[primary],c["candidate_scores"][primary],c["target_xy_token"][primary],c["coarse_top1_xy"][primary])))
        attempts=plan_reserve_attempts(primary,reserves,pre_prob,primary_post,64); reserve_ids=torch.tensor([new for _,new in attempts],dtype=torch.long)
        with torch.no_grad():
            reserve_post=torch.sigmoid(post(build_post_features_subset(pre_logits[reserve_ids],c["candidate_scores"][reserve_ids],c["target_xy_token"][reserve_ids],c["coarse_top1_xy"][reserve_ids]))) if len(attempts) else torch.empty(0)
        final,trace=finalize_planned_replacements(primary,attempts,primary_post,reserve_post)
        p0c=torch.load(args.p0_cache_root/"candidates"/"dev"/f"{pid}.pt",map_location="cpu",weights_only=True); p0l=torch.load(args.p0_cache_root/"labels"/"dev"/f"{pid}.pt",map_location="cpu",weights_only=True)
        validate_cached_pair(p0c,p0l,record,"dev","2e1b64ad31c41a455bebd68b92948c9c97e17a6c02cadab1eedbec16c3bbe801",CHECKPOINT_SHA256)
        flow=read_flow(Path(record["files"]["truth_flow"]["path"])); grid=p0l["grid_xy_pixel"]; truth=p0l["grid_truth_flow"]
        variants={
            "uniform":evaluate_anchor_set(c,l,uniform_region_k128(),flow,grid,truth),
            "no_replacement":evaluate_anchor_set(c,l,primary,flow,grid,truth),
            "main":evaluate_anchor_set(c,l,final,flow,grid,truth),
        }
        p2,r2=allocate_coverage_constrained_k128(pre_prob); a2=plan_reserve_attempts(p2,r2,pre_prob,primary_post,64); f2,t2=finalize_planned_replacements(p2,a2,primary_post,reserve_post)
        counts=region_counts(final); deterministic=torch.equal(primary,p2) and attempts==a2 and torch.equal(final,f2) and trace==t2
        rows.append({"pair_id":pid,**variants,"groups":_groups(record,p0l),"attempts":len(trace),"accepts":sum(t["accepted"] for t in trace),"fine_searches":128+len(trace),"final_ids":final.tolist(),"region_counts":counts.tolist(),"trace":trace,"deterministic":deterministic,"finite":bool(torch.isfinite(pre_prob).all() and torch.isfinite(primary_post).all() and torch.isfinite(reserve_post).all())})
        all_labels.extend(l["recoverable"].tolist()); all_scores.extend(pre_prob.tolist())
    summaries={name:_summary(rows,name) for name in ("uniform","no_replacement","main")}; main=summaries["main"]
    ranking=binary_ranking_metrics(all_labels,all_scores); pred=np.asarray(all_scores)>=.5; lab=np.asarray(all_labels,dtype=bool); ranking.update({"precision_at_0_5":float((pred&lab).sum()/max(1,pred.sum())),"recall_at_0_5":float((pred&lab).sum()/max(1,lab.sum()))})
    gates={"recoverable_fraction_ge50":main["recoverable_fraction"]>=.5,"median_ge64":main["median_recoverable_count"]>=64,"gain_vs_uniform_ge10pp":main["recoverable_fraction"]-summaries["uniform"]["recoverable_fraction"]>=.10,"gain_vs_no_replacement_ge3pp":main["recoverable_fraction"]-summaries["no_replacement"]["recoverable_fraction"]>=.03,"oracle_aepe_le2_5":main["aepe"]<=2.5,"pair_local_ready_ge99":main["pair_local_ready_percent"]>=99,"valid_ge99":main["valid_percent"]>=99,"fine_search_le192":max(r["fine_searches"] for r in rows)<=192,"region_1_to_3":all(min(r["region_counts"])>=1 and max(r["region_counts"])<=3 for r in rows),"final_k128":all(len(r["final_ids"])==128 and len(set(r["final_ids"]))==128 for r in rows),"finite":all(r["finite"] for r in rows),"deterministic":all(r["deterministic"] for r in rows)}
    attempts=sum(r["attempts"] for r in rows); accepts=sum(r["accepts"] for r in rows)
    return {"seed":seed,"status":"PASS" if all(gates.values()) else "FAIL","summaries":summaries,"ranking":ranking,"gates":gates,"recoverable_gain":{"vs_uniform":main["recoverable_fraction"]-summaries["uniform"]["recoverable_fraction"],"vs_no_replacement":main["recoverable_fraction"]-summaries["no_replacement"]["recoverable_fraction"]},"oracle_gap_vs_p0_gt_constrained_aepe":main["aepe"]-2.084816532854987,"replacement":{"attempts":attempts,"accepts":accepts,"accept_rate":accepts/max(1,attempts)},"region_budget_histogram":{str(i):sum(r["region_counts"].count(i) for r in rows) for i in (1,2,3)},"grouped_bootstrap_95_ci":_group_bootstrap(rows,seed),"oracle_label":"selected-anchor top8 GT oracle","rows":rows}


def run(args):
    actual_source=source_manifest(Path(__file__).parent)["sha256"]
    if args.source_sha!=actual_source: raise RuntimeError(f"source SHA mismatch {actual_source}")
    if sha256_file(args.split_manifest)!=SPLIT_MANIFEST_SHA256: raise RuntimeError("frozen split SHA mismatch")
    cache_sha=_cache_manifest(args.cache_root,load_split_manifest(args.split_manifest))
    if args.cache_manifest_sha!=cache_sha: raise RuntimeError("head cache content manifest mismatch")
    results=[evaluate_seed(args,p) for p in args.checkpoints]; seeds=[x["seed"] for x in results]
    if sorted(seeds)!=[20260805,20260806,20260807] or len(set(seeds))!=3: raise RuntimeError("exact three frozen seeds required")
    main=next(x for x in results if x["seed"]==20260805)
    stability=all((x["summaries"]["main"]["recoverable_fraction"]-x["summaries"]["uniform"]["recoverable_fraction"])>0 and (x["summaries"]["main"]["recoverable_fraction"]-x["summaries"]["no_replacement"]["recoverable_fraction"])>0 for x in results)
    result={"status":"PASS" if main["status"]=="PASS" and stability else "FAIL","main_seed":20260805,"three_seed_same_direction":stability,"seeds":[{k:v for k,v in x.items() if k!="rows"} for x in results]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    with args.per_pair_output.open("w") as h:
        for x in results:
            for row in x["rows"]: h.write(json.dumps({"seed":x["seed"],**row},sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"output":str(args.output)}));
    if result["status"]!="PASS": raise SystemExit(2)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--split-manifest",type=Path,required=True); p.add_argument("--cache-root",type=Path,required=True); p.add_argument("--p0-cache-root",type=Path,required=True); p.add_argument("--source-sha",required=True); p.add_argument("--cache-manifest-sha",required=True); p.add_argument("--checkpoints",type=Path,nargs=3,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--per-pair-output",type=Path,required=True); return p.parse_args()


if __name__=="__main__": run(parse_args())
