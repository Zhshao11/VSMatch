#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
MIG = ROOT / "image_matching_part2_migrated"
TMP = MIG / ".codex_tmp"
RUN_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-074A-ZZVR-F2F3-TWOSTAGE-FUSION"
E073_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-073A-ZZVR-V1V2-FUSION-LONGTRAIN"
F2_CKPT = E073_ROOT / "run-f2-warm120k/checkpoints/F2_step120000.pt"
F3_CKPT = E073_ROOT / "run-f3-warm120k/checkpoints/F3_step120000.pt"
F2_G1 = E073_ROOT / "run-f2-warm120k/formal/G1.json"
F3_G1 = E073_ROOT / "run-f3-warm120k/formal/G1.json"
TEST_ROOT = ROOT / "datasets/extracted/gdros_official_e003a_testing"
TRAIN_FLAT = ROOT / "datasets/extracted/gdros_official_e003a_training"

REFS = {
    "gdros_released": {"aepe": 1.461486, "cmr1": 33.73, "cmr2": 81.60, "cmr5": 99.53},
    "e065": {"aepe": 5.6425, "cmr2": 17.3166, "cmr5": 59.6648, "cmr10": 88.3459, "cmr16": 95.8989},
    "e073_f2": {"aepe": 3.0073599263214237, "cmr1": 22.277227722772277, "cmr2": 60.89108910891089, "cmr5": 89.10891089108911, "cmr10": 93.8118811881188, "cmr16": 96.53465346534654},
    "e073_f3": {"aepe": 2.8285585176015253, "cmr1": 35.64356435643564, "cmr2": 62.62376237623762, "cmr5": 88.36633663366337, "cmr10": 94.3069306930693, "cmr16": 97.52475247524752},
    "e073_f1_avg": {"aepe": 3.632940574226403, "cmr1": 19.05940594059406, "cmr2": 47.77227722772277, "cmr5": 78.96039603960396, "cmr10": 92.57425742574257, "cmr16": 97.52475247524752},
    "e073_f1_nogt": {"aepe": 3.7302960075511793, "cmr1": 12.623762376237623, "cmr2": 43.56435643564357, "cmr5": 78.71287128712872, "cmr10": 93.56435643564357, "cmr16": 97.77227722772277},
    "e073_f1_gt_oracle": {"aepe": 2.810803207762466, "cmr1": 20.2970297029703, "cmr2": 56.43564356435643, "cmr5": 86.63366336633663, "cmr10": 96.53465346534654, "cmr16": 99.00990099009901},
}

def rr(v: str, tag: str = "") -> Path: return RUN_ROOT / f"run-{v.lower()}{('-' + tag.strip('-')) if tag else ''}"
def formal(v: str, tag: str = "") -> Path: return rr(v, tag) / "formal"
def mat_dir(v: str, tag: str = "") -> Path: return formal(v, tag) / "materialized"
def ckpt_dir(v: str, tag: str = "") -> Path: return rr(v, tag) / "checkpoints"
def log_dir(v: str, tag: str = "") -> Path: return rr(v, tag) / "logs"

def js(p: Path, o: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2, sort_keys=True))

def sha(p: Path):
    if not p.exists(): return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def add_paths():
    sys.path.insert(0, str(TMP))
    import e072a_layerwise_frontend_alignment as e072
    import e073a_v1_v2_fusion_longtrain as e073
    e073.add_paths()
    return e072, e073

def data_view(v: str, tag: str):
    r = rr(v, tag) / "data_view/training"; r.mkdir(parents=True, exist_ok=True)
    for n in ["opt", "sar_warped", "truth_flow", "datum"]:
        d = r / n; s = TRAIN_FLAT / n
        if not d.exists(): d.symlink_to(s, target_is_directory=True)
    return rr(v, tag) / "data_view"

def audit(v: str, tag: str):
    out = {
        "status": "PASS",
        "experiment": "E074A",
        "variant": v,
        "run_root": str(rr(v, tag)),
        "fusion_stage2_matrix": {
            "A": "late F2/F3 flow-affine diagnostic: avg, no-GT self-consistency/per-pair chooser, GT oracle chooser upper bound only",
            "B": "learned second-stage per-pixel gate over frozen F2/F3 flow sequences, trained by Lseq+Lgeo threshold-aware metrics",
            "C": "single-student distillation: F3-style student warm-started from F3, distilled toward F2/F3 avg plus GT Lseq+Lgeo",
            "D": "joint/semifrozen F2/F3 dual-branch head fusion: learned scalar/tiny gate over branch outputs, preserving single-branch advantages",
            "E": "multi-threshold objective is tracked via CMR@1/2/5/10/16 and curve-guided continuation",
        },
        "protocol": {
            "baseline_official_e046_to_e073_readonly": True,
            "testing_zip_training": False,
            "no_os424_confirm_rescue": True,
            "materialize_before_gt": True,
            "gpu_policy": "zzvr GPU0-3 only",
        },
        "sources": {
            "f2_checkpoint": str(F2_CKPT), "f2_checkpoint_sha256": sha(F2_CKPT), "f2_g1_sha256": sha(F2_G1),
            "f3_checkpoint": str(F3_CKPT), "f3_checkpoint_sha256": sha(F3_CKPT), "f3_g1_sha256": sha(F3_G1),
            "script": str(Path(__file__)),
        },
        "refs": REFS,
    }
    for n, o in [("P0", out), ("fusion_stage2_matrix", out["fusion_stage2_matrix"]), ("baseline_comparison", REFS)]:
        js(formal(v, tag) / f"{n}.json", o)
    return out

def pair_metrics(vals):
    a = np.asarray(vals, np.float64)
    return {"pairs": int(a.size), "aepe": float(a.mean()) if a.size else None,
            "cmr1": float((a < 1).mean() * 100) if a.size else None,
            "cmr2": float((a < 2).mean() * 100) if a.size else None,
            "cmr5": float((a < 5).mean() * 100) if a.size else None,
            "cmr10": float((a < 10).mean() * 100) if a.size else None,
            "cmr16": float((a < 16).mean() * 100) if a.size else None}

def load_fusion(e073, v: str, ckpt: Path, device):
    e072, _ = add_paths()
    c = e072.cfg()
    m = e073.E073FusionMatcher(e072, c, v, device).to(device)
    payload = torch.load(ckpt, map_location=device)
    miss = m.load_state_dict(payload["model"], strict=False)
    return m, {"checkpoint": str(ckpt), "sha256": sha(ckpt), "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}

def lsr(flow):
    from core.LSRnet.LSmodel import LSR
    return LSR(1, 512, 512, flow, 200000)

class GateNet(nn.Module):
    def __init__(self, in_ch=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
    def forward(self, x): return torch.sigmoid(self.net(x))

class E074Model(nn.Module):
    def __init__(self, e073, variant: str, device, freeze_teachers=True):
        super().__init__()
        self.variant = variant.upper(); self.device = device
        self.f2, self.f2_info = load_fusion(e073, "F2", F2_CKPT, device)
        self.f3, self.f3_info = load_fusion(e073, "F3", F3_CKPT, device)
        if self.variant in ["B", "D"]:
            for p in list(self.f2.parameters()) + list(self.f3.parameters()): p.requires_grad_(False)
            self.gate = GateNet(9 if self.variant == "B" else 5)
            self.scalar = nn.Parameter(torch.tensor(0.5))
        elif self.variant == "C":
            # Single student initialized from stronger F3, teachers frozen for distillation.
            self.student, self.student_info = load_fusion(e073, "F3", F3_CKPT, device)
            for p in list(self.f2.parameters()) + list(self.f3.parameters()): p.requires_grad_(False)
        else:
            raise ValueError(self.variant)

    def teacher_preds(self, a, b, iters):
        with torch.no_grad():
            p2 = self.f2(a, b, iters=iters, return_features=False)
            p3 = self.f3(a, b, iters=iters, return_features=False)
        return p2, p3

    def gate_input(self, p2, p3):
        d = p2 - p3
        l2 = lsr(p2).detach().to(p2.device); l3 = lsr(p3).detach().to(p3.device)
        if self.variant == "B":
            return torch.cat([p2.detach(), p3.detach(), d.detach(), torch.linalg.vector_norm(d.detach(), dim=1, keepdim=True), torch.linalg.vector_norm((p2-l2).detach(), dim=1, keepdim=True), torch.linalg.vector_norm((p3-l3).detach(), dim=1, keepdim=True)], dim=1)
        return torch.cat([p2.detach(), p3.detach(), torch.linalg.vector_norm(d.detach(), dim=1, keepdim=True)], dim=1)

    def forward(self, a, b, iters=12, need_teacher_avg=True):
        """variant C: teacher_preds only feeds last_teacher_avg (E074 distill loss).

        E075A PairFusion training/eval never reads last_teacher_avg, so it calls
        this with need_teacher_avg=False to skip 2 frozen GRU chains per pair.
        """
        if self.variant in ["B", "D"]:
            p2, p3 = self.teacher_preds(a, b, iters)
            outs = []
            aux = {}
            for x2, x3 in zip(p2, p3):
                g = self.gate(self.gate_input(x2, x3))
                s = torch.sigmoid(self.scalar)
                if self.variant == "D":
                    g = 0.5 * g + 0.5 * s
                outs.append((1 - g) * x3 + g * x2)  # gate is F2 weight, F3 is precision prior.
                aux = {"gate_f2_mean": float(g.detach().mean().cpu()), "scalar_f2": float(s.detach().cpu())}
            self.last_aux = aux
            return outs
        if need_teacher_avg:
            p2, p3 = self.teacher_preds(a, b, iters)
        souts = self.student(a, b, iters=iters, return_features=False)
        if need_teacher_avg:
            self.last_teacher_avg = [0.5 * (x2 + x3) for x2, x3 in zip(p2, p3)]
        self.last_aux = {"student": "F3_init", "teacher_avg": bool(need_teacher_avg)}
        return souts

def eval_model(model, ds, device, iters_list, max_pairs, v, tag, name):
    vals = {f"iter{it}_pre_lsr": [] for it in iters_list} | {f"iter{it}_post_lsr": [] for it in iters_list}
    gate_rows = []
    mat = mat_dir(v, tag) / f"{name}_predictions_before_gt.jsonl"; mat.parent.mkdir(parents=True, exist_ok=True); mat.unlink(missing_ok=True)
    with torch.inference_mode():
        for i in range(min(max_pairs, len(ds))):
            file_opt, a, b, gt, _ = ds[i]
            a = a[None].to(device); b = b[None].to(device)
            preds = model(a, b, iters=max(iters_list)); aux = getattr(model, "last_aux", {})
            gate_rows.append(aux)
            rec = {"pair_index": i, "pair_id": Path(file_opt).stem, "uses_truth_flow": False, "predictions_materialized_before_gt": True, "branch_gate": aux}
            for it in iters_list:
                pre = preds[it-1].cpu(); post = lsr(preds[it-1]).cpu()
                for k, fl in [(f"iter{it}_pre_lsr", pre), (f"iter{it}_post_lsr", post)]:
                    e = torch.linalg.vector_norm((fl - gt[None]).float(), dim=1).view(-1)
                    vals[k].append(float(e.mean())); rec[k] = float(e.mean())
            with mat.open("a") as f: f.write(json.dumps(rec, sort_keys=True) + "\n")
            torch.cuda.empty_cache()
    def mean_dict(rows):
        out={}
        for k in sorted({k for r in rows for k in r if isinstance(r.get(k),(int,float))}):
            arr=np.asarray([r[k] for r in rows if k in r],float)
            if arr.size: out[k]=float(arr.mean())
        return out
    return {k: pair_metrics(x) for k, x in vals.items()} | {"branch_gate": mean_dict(gate_rows), "materialized_before_gt": str(mat), "materialized_sha256": sha(mat)}

def train(args):
    e072, e073 = add_paths()
    import core.datasets
    from core.loss import sequence_loss, matrix2flow
    from core.LSRnet.LSmodel import LSR_list
    v=args.variant.upper(); tag=args.run_tag; audit(v, tag)
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    c=e072.cfg()
    tr=core.datasets.opt_sar_train(split="training", dstype="training", root=str(data_view(v, tag)))
    dv=core.datasets.opt_sar_test(split="testing", dstype="testing", root=str(TEST_ROOT))
    model=E074Model(e073, v, device).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    rng=np.random.default_rng(args.seed); curves=[]; evals={}
    milestones=sorted(set([m for m in args.milestones if m<=args.steps]+[args.steps]))
    for phase,steps in [("toy", args.toy_steps), ("full", args.steps)]:
        begin=args.start_step+1 if phase=="full" else 1
        for step in range(begin, steps+1):
            idx=int(rng.integers(0, args.toy_pairs if phase=="toy" else len(tr)))
            a,b,flow,valid=tr[idx]; a=a[None].to(device); b=b[None].to(device); flow=flow[None].to(device); valid=valid[None].to(device)
            preds=model(a,b,iters=args.train_iters)
            lseq,m1=sequence_loss(preds,flow,valid,c); mats=matrix2flow(LSR_list(preds),flow); lgeo,m2=sequence_loss(mats,flow,valid,c)
            loss=args.lambda_seq*lseq + args.lambda_geo*lgeo
            distill=torch.tensor(0.,device=device)
            if v=="C":
                distill=sum(F.smooth_l1_loss(p,t) for p,t in zip(preds,model.last_teacher_avg))/len(preds)
                loss=loss+args.distill_weight*distill
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
            log_every=max(1, steps//10 if phase=="toy" else args.steps//20)
            if step==begin or step%log_every==0:
                row={"variant":v,"phase":phase,"step":step,"loss":float(loss.detach().cpu()),"distill":float(distill.detach().cpu()),"lseq_epe":m1["epe"],"lgeo_epe":m2["epe"],"flow5":float(m1["5px"]*100),"mat5":float(m2["5px"]*100),"aux":getattr(model,"last_aux",{})}
                curves.append(row); print(f"E074A {v} {phase} step={step}/{steps} loss={row['loss']:.3f} flow5={row['flow5']:.2f} mat5={row['mat5']:.2f} aux={row['aux']}", flush=True)
            if phase=="full" and step in milestones:
                cp=ckpt_dir(v,tag)/f"{v}_step{step}.pt"; cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model":model.state_dict(),"variant":v,"step":step,"seed":args.seed,"sources":audit(v,tag)["sources"]},cp)
                evals[str(step)]={"checkpoint":str(cp),"checkpoint_sha256":sha(cp),"dev":eval_model(model.eval(),dv,device,[12,24,32],args.dev_eval_pairs,v,tag,f"{v}_step{step}")}; model.train()
                js(formal(v,tag)/"curve_guided_training_report.json",{"status":"RUNNING","curves":curves,"evals":evals})
                torch.cuda.empty_cache()
    return finalize(v,tag,curves,evals,time.time())

def finalize(v, tag="", curves=None, evals=None, start=None):
    fr=formal(v,tag)
    if evals is None:
        p=fr/"curve_guided_training_report.json"; r=json.loads(p.read_text()) if p.exists() else {}; curves=r.get("curves",[]); evals=r.get("evals",{})
    best=None
    for st,ev in (evals or {}).items():
        for k,val in ev.get("dev",{}).items():
            if isinstance(val,dict) and "aepe" in val:
                row={"variant":v,"step":int(st),"eval_variant":k,**val}
                if best is None or row["aepe"]<best["aepe"]: best=row
    sufficient=bool(evals and any(int(k)>=180000 for k in evals))
    pareto = best and best.get("cmr2",0)>REFS["e073_f3"]["cmr2"] and best.get("cmr5",0)>REFS["e073_f2"]["cmr5"] and best.get("aepe",999)<=REFS["e073_f3"]["aepe"]
    strong = pareto and best.get("cmr2",0)>65 and best.get("cmr5",0)>92 and best.get("aepe",999)<2.7
    status="TRAINING_INSUFFICIENT" if not sufficient else ("STRONG_FUSION_SUCCESS" if strong else ("FUSION_PARETO_SUCCESS" if pareto else ("PARTIAL_FUSION_GAIN" if best and (best.get("cmr2",0)>REFS["e073_f3"]["cmr2"] or best.get("cmr5",0)>REFS["e073_f2"]["cmr5"] or best.get("aepe",999)<REFS["e073_f3"]["aepe"]) else "TRAINING_SUFFICIENT_NO_STAGE2_GAIN")))
    objs={
        "G0":{"status":"PASS" if curves else "PENDING","training_sufficient":sufficient,"toy_signal":curves[:12],"sources":audit(v,tag)["sources"]},
        "G1":{"status":status,"best":best,"refs":REFS},
        "curve_guided_training_report":{"status":status,"curves":curves,"evals":evals},
        "checkpoint_curve_report":{"curves":curves,"evals":evals},
        "branch_gate_report":{k:e.get("dev",{}).get("branch_gate") for k,e in (evals or {}).items()},
        "method_ablation":{"variant":v,"best":best},
        "resource_report":{"gpu_policy":"zzvr GPU0-3","elapsed_seconds":None if start is None else time.time()-start},
        "RUN_MANIFEST":{"experiment":"E074A","variant":v,"status":status,"run_root":str(rr(v,tag)),"formal_root":str(fr),"f2_checkpoint_sha256":sha(F2_CKPT),"f3_checkpoint_sha256":sha(F3_CKPT),"testing_zip_used_for_training":False,"no_os424_confirm_rescue":True,"materialized_before_gt":[str(p) for p in mat_dir(v,tag).glob("*.jsonl")]},
    }
    for n,o in objs.items(): js(fr/f"{n}.json",o)
    (fr/"reviewer_conclusion.txt").write_text(f"E074A {v}: {status}. F2/F3 stage-2 fusion; GT oracle only diagnostic; predictions materialized before GT join.\n",encoding="utf-8")
    (fr/"RESULT_CARD.md").write_text(f"# E074A {v} RESULT_CARD\n\nStatus: `{status}`\n\nBest: `{best}`\n",encoding="utf-8")
    return {"status":status,"best":best,"formal":str(fr)}

def late_diag(args):
    e072,e073=add_paths()
    import core.datasets
    v="A"; tag=args.run_tag; audit(v,tag)
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    f2,_=load_fusion(e073,"F2",F2_CKPT,device); f3,_=load_fusion(e073,"F3",F3_CKPT,device); f2.eval(); f3.eval()
    ds=core.datasets.opt_sar_test(split="testing",dstype="testing",root=str(TEST_ROOT))
    vals={k:[] for k in ["f2","f3","avg","nogt_self_consistency","gt_oracle_chooser"]}
    mat=mat_dir(v,tag)/"A_late_fusion_predictions_before_gt.jsonl"; mat.parent.mkdir(parents=True,exist_ok=True); mat.unlink(missing_ok=True)
    with torch.inference_mode():
        for i in range(min(args.dev_eval_pairs,len(ds))):
            file_opt,a,b,gt,_=ds[i]; a=a[None].to(device); b=b[None].to(device)
            p2=f2(a,b,iters=24,return_features=False)[23]; p3=f3(a,b,iters=24,return_features=False)[23]
            l2=lsr(p2).cpu(); l3=lsr(p3).cpu(); avg=0.5*(l2+l3)
            s2=float(torch.linalg.vector_norm((p2.cpu()-l2).float(),dim=1).mean()); s3=float(torch.linalg.vector_norm((p3.cpu()-l3).float(),dim=1).mean())
            chosen=l2 if s2<=s3 else l3
            rec={"pair_index":i,"pair_id":Path(file_opt).stem,"uses_truth_flow":False,"predictions_materialized_before_gt":True,"self_residual_f2":s2,"self_residual_f3":s3,"nogt_choice":"f2" if s2<=s3 else "f3"}
            e2=torch.linalg.vector_norm((l2-gt[None]).float(),dim=1).view(-1); e3=torch.linalg.vector_norm((l3-gt[None]).float(),dim=1).view(-1)
            ea=torch.linalg.vector_norm((avg-gt[None]).float(),dim=1).view(-1); ec=torch.linalg.vector_norm((chosen-gt[None]).float(),dim=1).view(-1); eo=torch.minimum(e2,e3)
            for name,e in [("f2",e2),("f3",e3),("avg",ea),("nogt_self_consistency",ec),("gt_oracle_chooser",eo)]:
                vals[name].append(float(e.mean())); rec[f"{name}_epe_joined_after_materialization"]=float(e.mean())
            with mat.open("a") as f: f.write(json.dumps(rec,sort_keys=True)+"\n")
            torch.cuda.empty_cache()
    metrics={k:pair_metrics(vv) for k,vv in vals.items()}
    objs={"G0":{"status":"PASS"},"G1":{"status":"DIAGNOSTIC_DONE","metrics":metrics,"refs":REFS},"oracle_vs_nogt_fusion_report":metrics,"resource_report":{"gpu_policy":"zzvr GPU0-3"},"RUN_MANIFEST":{"experiment":"E074A","variant":"A","status":"DIAGNOSTIC_DONE","run_root":str(rr(v,tag)),"formal_root":str(formal(v,tag)),"materialized_before_gt":[str(mat)],"f2_checkpoint_sha256":sha(F2_CKPT),"f3_checkpoint_sha256":sha(F3_CKPT),"testing_zip_used_for_training":False}}
    for n,o in objs.items(): js(formal(v,tag)/f"{n}.json",o)
    (formal(v,tag)/"reviewer_conclusion.txt").write_text("A diagnostic: no-GT chooser plus GT oracle upper bound; oracle not method.\n",encoding="utf-8")
    (formal(v,tag)/"RESULT_CARD.md").write_text(f"# E074A A RESULT_CARD\n\nStatus: `DIAGNOSTIC_DONE`\n\nMetrics: `{metrics}`\n",encoding="utf-8")
    print(json.dumps({"status":"DIAGNOSTIC_DONE","metrics":metrics,"formal":str(formal(v,tag))},indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["train","diag","finalize","p0"],default="train")
    ap.add_argument("--variant",choices=["A","B","C","D"],required=True)
    ap.add_argument("--run-tag",type=str,default="")
    ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--seed",type=int,default=740)
    ap.add_argument("--toy-pairs",type=int,default=8); ap.add_argument("--toy-steps",type=int,default=200)
    ap.add_argument("--start-step",type=int,default=120000); ap.add_argument("--steps",type=int,default=180000); ap.add_argument("--milestones",type=int,nargs="+",default=[140000,160000,180000])
    ap.add_argument("--train-iters",type=int,default=12); ap.add_argument("--dev-eval-pairs",type=int,default=404); ap.add_argument("--lr",type=float,default=3e-5)
    ap.add_argument("--lambda-seq",type=float,default=0.5); ap.add_argument("--lambda-geo",type=float,default=0.5); ap.add_argument("--distill-weight",type=float,default=0.2)
    args=ap.parse_args(); formal(args.variant,args.run_tag).mkdir(parents=True,exist_ok=True); mat_dir(args.variant,args.run_tag).mkdir(parents=True,exist_ok=True); log_dir(args.variant,args.run_tag).mkdir(parents=True,exist_ok=True)
    if args.mode=="p0": print(json.dumps(audit(args.variant,args.run_tag),indent=2))
    elif args.mode=="finalize": print(json.dumps(finalize(args.variant,args.run_tag),indent=2))
    elif args.mode=="diag": late_diag(args)
    else: print(json.dumps(train(args),indent=2))

if __name__=="__main__": main()
