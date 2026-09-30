#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.sparsegeo.e003a.data_protocol import load_split_manifest
from experiments.sparsegeo.e004a.anchor_heads import PostSearchVerifier, PreSearchRecoverabilityHead
from experiments.sparsegeo.e004a.replacement import build_post_features
from experiments.sparsegeo.e004a.head_cache import validate_head_cache
from experiments.sparsegeo.e004a.spec_binding import CHECKPOINT_SHA256
from experiments.sparsegeo.e004a.provenance import source_manifest
from experiments.sparsegeo.e003a.data_protocol import sha256_file
from experiments.sparsegeo.e004a.spec_binding import SPLIT_MANIFEST_SHA256
import hashlib


def _cache_manifest(cache_root,records):
    rows=[]
    for r in records:
        for kind in ("candidates","labels"):
            p=cache_root/kind/r["split"]/f"{r['pair_id']}.pt"; rows.append((kind,r["split"],r["pair_id"],sha256_file(p)))
    return hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()


def _load(cache_root, records, source_sha):
    candidates, labels = [], []
    for record in records:
        split, pair_id = record["split"], record["pair_id"]
        candidate=torch.load(cache_root/"candidates"/split/f"{pair_id}.pt",map_location="cpu",weights_only=True); label=torch.load(cache_root/"labels"/split/f"{pair_id}.pt",map_location="cpu",weights_only=True)
        validate_head_cache(candidate,label,record,split,source_sha,CHECKPOINT_SHA256); candidates.append(candidate); labels.append(label)
    return candidates, labels


def _fit(model, train_x, train_y, dev_x, dev_y, seed, device):
    generator=torch.Generator().manual_seed(seed); model.to(device)
    positive=float(train_y.sum()); pos_weight=torch.tensor((train_y.numel()-positive)/positive,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best, best_state, patience, history=float("inf"),None,0,[]
    pair_count=train_x.shape[0]
    for epoch in range(50):
        model.train(); order=torch.randperm(pair_count,generator=generator); losses=[]
        for start in range(0,pair_count,32):
            ids=order[start:start+32]; x=train_x[ids].to(device); y=train_y[ids].to(device)
            loss=F.binary_cross_entropy_with_logits(model(x),y,pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            dev_loss=float(F.binary_cross_entropy_with_logits(
                model(dev_x.to(device)),dev_y.to(device),pos_weight=pos_weight).item())
        history.append({"epoch":epoch,"train_bce":float(np.mean(losses)),"dev_bce":dev_loss})
        if dev_loss < best - 1e-8:
            best=dev_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=8: break
    model.load_state_dict(best_state); return model.cpu(),history,float(pos_weight)


def run(args):
    actual_source=source_manifest(Path(__file__).parent)["sha256"]
    if args.source_sha!=actual_source: raise RuntimeError(f"source SHA mismatch {actual_source}")
    if sha256_file(args.split_manifest)!=SPLIT_MANIFEST_SHA256: raise RuntimeError("frozen split SHA mismatch")
    seed=args.seed; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    records=load_split_manifest(args.split_manifest); train=[r for r in records if r["split"]=="train"]; dev=[r for r in records if r["split"]=="dev"]
    train_c,train_l=_load(args.cache_root,train,args.source_sha); dev_c,dev_l=_load(args.cache_root,dev,args.source_sha)
    train_pre=torch.stack([x["pre_features"] for x in train_c]); dev_pre=torch.stack([x["pre_features"] for x in dev_c])
    train_y=torch.stack([x["recoverable"].float() for x in train_l]); dev_y=torch.stack([x["recoverable"].float() for x in dev_l])
    pre,pre_hist,pre_weight=_fit(PreSearchRecoverabilityHead(),train_pre,train_y,dev_pre,dev_y,seed,torch.device(args.device))
    with torch.no_grad():
        train_logits=pre(train_pre); dev_logits=pre(dev_pre)
    train_post=torch.stack([build_post_features(train_logits[i],x["candidate_scores"],x["target_xy_token"],x["coarse_top1_xy"]) for i,x in enumerate(train_c)])
    dev_post=torch.stack([build_post_features(dev_logits[i],x["candidate_scores"],x["target_xy_token"],x["coarse_top1_xy"]) for i,x in enumerate(dev_c)])
    post,post_hist,post_weight=_fit(PostSearchVerifier(),train_post,train_y,dev_post,dev_y,seed+1000,torch.device(args.device))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint={"seed":seed,"pre":pre.state_dict(),"post":post.state_dict(),"pre_pos_weight":pre_weight,"post_pos_weight":post_weight,"source_manifest_sha256":actual_source,"split_manifest_sha256":SPLIT_MANIFEST_SHA256,"cache_source_sha256":args.source_sha,"cache_content_manifest_sha256":_cache_manifest(args.cache_root,records),"train_pairs":1607,"dev_pairs":404}
    torch.save(checkpoint,args.output_dir/f"heads-seed-{seed}.pth")
    result={"status":"PASS","seed":seed,"pre_history":pre_hist,"post_history":post_hist,"pre_parameters":sum(p.numel() for p in pre.parameters()),"post_parameters":sum(p.numel() for p in post.parameters())}
    (args.output_dir/f"train-seed-{seed}.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--split-manifest",type=Path,required=True); p.add_argument("--cache-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--source-sha",required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--device",default="cuda:0"); return p.parse_args()


if __name__=="__main__": run(parse_args())
