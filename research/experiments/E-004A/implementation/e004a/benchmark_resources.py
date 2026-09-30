#!/usr/bin/env python3
import argparse,os,uuid
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sparsegeo.e002a.feature import load_reference_gdros, load_slim_feature
from experiments.sparsegeo.e003a.candidate_cache import _load_image
from experiments.sparsegeo.e003a.data_protocol import load_split_manifest
from experiments.sparsegeo.e004a.anchor_heads import PostSearchVerifier, PreSearchRecoverabilityHead
from experiments.sparsegeo.e004a.coarse_probe import compute_coarse_probe_features
from experiments.sparsegeo.e004a.proposal_cache import exact_cosine_top8
from experiments.sparsegeo.e004a.proposals import allocate_coverage_constrained_k128, build_region_proposals
from experiments.sparsegeo.e004a.replacement import build_post_features_subset,plan_reserve_attempts,finalize_planned_replacements
from experiments.sparsegeo.e004a.spec_binding import CHECKPOINT_SHA256,SPLIT_MANIFEST_SHA256
from experiments.sparsegeo.e004a.provenance import source_manifest
from experiments.sparsegeo.e003a.data_protocol import sha256_file


def _post_subset(pre_logits,scores,xy,coarse):
    prob=torch.softmax(scores/.1,dim=-1); entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
    return torch.cat((pre_logits[:,None],scores,(scores[:,0]-scores[:,1])[:,None],entropy[:,None],prob[:,:4].sum(-1,keepdim=True),xy[:,0]/63.0,(xy[:,0]-coarse).abs().amax(-1,keepdim=True)),dim=-1)


@torch.inference_mode()
def acquire(source,target,pre,post):
    proposals=build_region_proposals(); features,coarse,coarse_scores=compute_coarse_probe_features(source,target); logits=pre(features); probability=torch.sigmoid(logits)
    primary,reserves=allocate_coverage_constrained_k128(probability); primary=primary.to(source.device); reserves=reserves.to(source.device)
    live=[]; ps,_,pxy=exact_cosine_top8(source,target,proposals.lattice_indices[primary.cpu()],live_shapes=live,to_cpu=False); pp=torch.sigmoid(post(build_post_features_subset(logits[primary],ps,pxy,coarse[primary])))
    attempts=plan_reserve_attempts(primary.cpu(),reserves.cpu(),probability.cpu(),pp.cpu(),64); reserve_list=[n for _,n in attempts]
    if reserve_list:
        ids=torch.tensor(reserve_list,device=source.device); rs,_,rxy=exact_cosine_top8(source,target,proposals.lattice_indices[ids.cpu()],live_shapes=live,to_cpu=False); rp=torch.sigmoid(post(build_post_features_subset(logits[ids],rs,rxy,coarse[ids])))
    else: rp=torch.empty(0)
    final,trace=finalize_planned_replacements(primary.cpu(),attempts,pp.cpu(),rp.cpu())
    return {"coarse_scores":list(coarse_scores.shape),"coarse_score_elements":coarse_scores.numel(),"fine_scores":[128+len(reserve_list),4096],"fine_score_elements":sum(s["numel"] for s in live),"fine_searches":128+len(reserve_list),"live_shapes":live,"final_k":final.numel(),"trace":trace}


def _measure(fn,warmup=20,measured=200):
    for _ in range(warmup): fn()
    values=[]; torch.cuda.reset_peak_memory_stats(); baseline=torch.cuda.memory_allocated()
    for _ in range(measured):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize(); values.append(1000*(time.perf_counter()-start))
    return {"raw_ms":values,"p50_ms":float(np.percentile(values,50)),"p95_ms":float(np.percentile(values,95)),"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"incremental_peak_mib":(torch.cuda.max_memory_allocated()-baseline)/2**20}


def run(args):
    if sha256_file(args.checkpoint)!=CHECKPOINT_SHA256 or sha256_file(args.split_manifest)!=SPLIT_MANIFEST_SHA256: raise RuntimeError("frozen resource provenance mismatch")
    actual_source=source_manifest(Path(__file__).parent)["sha256"]; state=torch.load(args.heads,map_location="cpu",weights_only=True)
    if state.get("seed")!=20260805 or state.get("source_manifest_sha256")!=actual_source or state.get("split_manifest_sha256")!=SPLIT_MANIFEST_SHA256 or state.get("cache_content_manifest_sha256")!=args.cache_manifest_sha: raise RuntimeError("main-seed head provenance mismatch")
    torch.cuda.reset_peak_memory_stats()
    device=torch.device(args.device); records=load_split_manifest(args.split_manifest,allowed_split="dev"); r=records[0]; optical=_load_image(Path(r["files"]["opt"]["path"])).to(device); sar=_load_image(Path(r["files"]["sar_warped"]["path"])).to(device)
    slim,_,_=load_slim_feature(args.checkpoint,device); slim.requires_grad_(False); pre=PreSearchRecoverabilityHead().to(device).eval(); post=PostSearchVerifier().to(device).eval(); pre.load_state_dict(state["pre"]); post.load_state_dict(state["post"])
    source,target=slim(optical,sar); trace=acquire(source,target,pre,post); cold_peak=torch.cuda.max_memory_allocated()/2**20
    anchor=_measure(lambda:acquire(source,target,pre,post)); end=_measure(lambda:acquire(*slim(optical,sar),pre,post))
    del slim; torch.cuda.empty_cache(); reference,_,_=load_reference_gdros(args.checkpoint,device); reference.requires_grad_(False).eval(); baseline=_measure(lambda:reference(optical,sar,iters=32,test_mode=True))
    result={"status":"PASS","process":{"pid":os.getpid(),"run_uuid":uuid.uuid4().hex},"provenance":{"source_manifest_sha256":actual_source,"checkpoint_sha256":CHECKPOINT_SHA256,"split_manifest_sha256":SPLIT_MANIFEST_SHA256,"cache_content_manifest_sha256":args.cache_manifest_sha,"seed":20260805},"trace":trace,"cold_peak_mib":cold_peak,"anchor":anchor,"end_to_end":end,"gdros":baseline,"speedup":baseline["p50_ms"]/end["p50_ms"]}
    gates={"anchor_p50":anchor["p50_ms"]<=6,"anchor_p95":anchor["p95_ms"]<=10,"anchor_memory":anchor["incremental_peak_mib"]<=96,"end_p50":end["p50_ms"]<=18,"end_p95":end["p95_ms"]<=25,"end_memory":end["peak_allocated_mib"]<=284,"cold_memory":cold_peak<=284,"speedup":result["speedup"]>=5,"fine_budget":trace["fine_searches"]<=192,"final_k128":trace["final_k"]==128,"coarse_exact":trace["coarse_scores"]==[256,256] and trace["coarse_score_elements"]==256*256,"fine_exact":trace["fine_score_elements"]==trace["fine_searches"]*4096,"no_n2":all(s["shape"][1]!=4096 and s["shape"][-1]==1024 for s in trace["live_shapes"])}
    result["gates"]=gates; result["status"]="PASS" if all(gates.values()) else "FAIL"; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(result["status"]);
    if result["status"]!="PASS": raise SystemExit(2)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--heads",type=Path,required=True); p.add_argument("--split-manifest",type=Path,required=True); p.add_argument("--cache-manifest-sha",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); return p.parse_args()


if __name__=="__main__": run(parse_args())
