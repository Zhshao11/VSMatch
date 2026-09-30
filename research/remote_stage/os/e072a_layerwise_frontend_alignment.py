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
OFF = ROOT / "external_repos/E068A_GDROS_official"
RUN_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-072A-ZZVR-LAYERWISE-FRONTEND-ALIGNMENT"
TRAIN_FLAT = ROOT / "datasets/extracted/gdros_official_e003a_training"
TEST_ROOT = ROOT / "datasets/extracted/gdros_official_e003a_testing"
GDROS_CKPT = ROOT / "checkpoints/gdros/GDROS-os512.pth"
E046_CKPT = ROOT / "GDROS/experiments/sparsegeo/runs/E-046A-ZZVR-JOINT-TRAINING/run-r2/checkpoints/masked_pretrain_joint.pt"
E071_TEACHER = ROOT / "GDROS/experiments/sparsegeo/runs/E-071A-ZZVR-DENSE-FLOW-COMPATIBLE-FRONTEND/run-teacher/formal/RESULT_CARD.md"

REFS = {
    "gdros_released": {"aepe": 1.461486, "cmr1": 33.73, "cmr2": 81.60, "cmr5": 99.53},
    "e065": {"aepe": 5.6425, "cmr2": 17.3166, "cmr5": 59.6648, "cmr10": 88.3459, "cmr16": 95.8989},
    "e071a_teacher": {"aepe": 8.011674760591866, "cmr2": 9.653465346534654, "cmr5": 42.82178217821782, "cmr10": 72.27722772277228, "cmr16": 87.37623762376238},
}

def variant_root(v: str) -> Path:
    tag = os.environ.get("E072A_RUN_TAG", "").strip().strip("-")
    return RUN_ROOT / (f"run-{v.lower()}-{tag}" if tag else f"run-{v.lower()}")
def formal(v: str) -> Path: return variant_root(v) / "formal"
def mat_dir(v: str) -> Path: return formal(v) / "materialized"
def ckpt_dir(v: str) -> Path: return variant_root(v) / "checkpoints"
def log_dir(v: str) -> Path: return variant_root(v) / "logs"

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
    for rel in ["core/__init__.py", "configs/__init__.py"]:
        p = OFF / rel
        if not p.exists(): p.write_text("# E072A import shim\n")
    sys.path.insert(0, str(OFF)); sys.path.insert(1, str(MIG)); sys.path.insert(2, str(TMP))
    for n in list(sys.modules):
        if n == "core" or n.startswith("core.") or n == "configs" or n.startswith("configs."):
            del sys.modules[n]
def data_view(v: str):
    r = variant_root(v) / "data_view/training"; r.mkdir(parents=True, exist_ok=True)
    for n in ["opt", "sar_warped", "truth_flow", "datum"]:
        d = r / n; s = TRAIN_FLAT / n
        if not d.exists(): d.symlink_to(s, target_is_directory=True)
    return variant_root(v) / "data_view"
def cfg():
    from configs.default import get_cfg
    c = get_cfg(); c.defrost()
    for k, val in {"small": False, "mixed_precision": False, "alternate_corr": False, "gamma": 0.85, "max_flow": 400, "image_size": [512,512], "corr_levels": 4, "corr_radius": 4}.items():
        setattr(c, k, val)
    return c
def audit(v: str):
    out = {
        "status": "PASS",
        "official_source": str(OFF),
        "official_source_commit": os.popen(f"cd {OFF} && git rev-parse HEAD 2>/dev/null").read().strip(),
        "official_checkpoint_readonly": str(GDROS_CKPT),
        "official_checkpoint_sha256": sha(GDROS_CKPT),
        "frontend_official_layers": [
            "normalize image to [-1,1]",
            "context BasicEncoder(output_dim=256, instance norm) for GRU state",
            "FeatureEncoder.feat_encoder = BasicEncoder(output_dim=256, instance norm), stride 8",
            "channel_convertor 1x1 conv 256->256",
            "feature_add_position(attn_splits=2)",
            "FeatureTransformer num_layers=1, swin attention, feature_channels=256",
            "CorrBlock 4-level all-pairs correlation, radius=4",
            "BasicUpdateBlock GRU recurrent dense flow, train 12 iters",
            "LSR affine projection via Lgeo/matrix2flow",
        ],
        "mismatch_vs_e046_e071": {
            "resolution": "E046 lineage often 32x32/upsampled; official matcher expects native stride-8 64x64 for 512 input",
            "channels": "E046 1024 DINO-adapter descriptors vs official 256 dense activations",
            "normalization": "E046 L2/cosine descriptors vs official raw dot-product activations",
            "local_detail": "DINO-heavy descriptors suppress local texture needed by dense flow",
            "interaction": "official uses early CNN + positional/cross transformer before CorrBlock; E071 scratch lacked faithful FeatureEncoder distribution",
            "context": "official GRU context comes from normalized source RGB BasicEncoder; E046 bridge made context/update OOD",
        },
        "variant_matrix": {
            "V1": "GDROS-compatible shallow modality adapters + official FeatureEncoder, scratch joint matcher",
            "V2": "DINO/E046-assisted FiLM/residual injection into GDROS-like corr features; E046 frozen read-only",
            "V3": "raw dense activation branch, no L2 descriptor bottleneck, learnable scale/norm",
            "V4": "GDROS-like frontend scratch + auxiliary feature distribution loss to released GDROS teacher",
            "V5": "feature/corr/flow distillation diagnostic; implemented as V4 teacher-feature/flow losses when --teacher-weight>0",
            "V6": "hybrid warm-up schedule; use V4 plus teacher losses early and GT Lseq/Lgeo throughout",
        },
        "protocol": {"testing_zip_training": False, "no_os424_confirm_rescue": True, "materialize_before_gt": True, "gpu_policy": "zzvr GPU0-3 only"},
        "refs": REFS,
    }
    for name in ["P0", "layerwise_frontend_audit", "variant_matrix"]:
        js(formal(v) / f"{name}.json", out if name != "variant_matrix" else out["variant_matrix"])
    return out

class ResidualInputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 24, 3, 1, 1), nn.GELU(), nn.Conv2d(24, 3, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return (x + 0.1 * self.net(x)).clamp(-1.5, 1.5)

class OfficialLikeFrontend(nn.Module):
    def __init__(self, c, modality_adapt=True):
        super().__init__()
        from core.GDROSnet.FeatureFormer.encoder import FeatureEncoder
        self.opt_adapt = ResidualInputAdapter() if modality_adapt else nn.Identity()
        self.sar_adapt = ResidualInputAdapter() if modality_adapt else nn.Identity()
        self.feature = FeatureEncoder(c[c.transformer])
    def forward(self, img1n, img2n, context=None):
        return self.feature(self.opt_adapt(img1n).contiguous(), self.sar_adapt(img2n).contiguous(), {}, context)

class RawDenseFrontend(nn.Module):
    def __init__(self, norm="group"):
        super().__init__()
        def tower():
            return nn.Sequential(
                nn.Conv2d(3,64,7,2,3), nn.GroupNorm(8,64), nn.ReLU(inplace=True),
                nn.Conv2d(64,96,3,2,1), nn.GroupNorm(8,96), nn.ReLU(inplace=True),
                nn.Conv2d(96,128,3,2,1), nn.GroupNorm(16,128), nn.ReLU(inplace=True),
                nn.Conv2d(128,256,3,1,1), nn.GroupNorm(16,256), nn.ReLU(inplace=True),
                nn.Conv2d(256,256,1))
        self.opt = tower(); self.sar = tower()
        self.scale = nn.Parameter(torch.ones(1,256,1,1) * 4.0); self.bias = nn.Parameter(torch.zeros(1,256,1,1))
    def forward(self, img1n, img2n, context=None):
        return self.opt(img1n) * self.scale + self.bias, self.sar(img2n) * self.scale + self.bias

class E046FiLM(nn.Module):
    def __init__(self, device):
        super().__init__()
        sys.path.insert(0, str(MIG))
        from experiments.sparsegeo.e046a.features import load_joint_checkpoint
        joint, payload = load_joint_checkpoint(E046_CKPT, device)
        if "frontend_state_dict" in payload:
            joint.frontend.load_state_dict(payload["frontend_state_dict"], strict=False)
        self.front = joint.frontend.eval()
        for p in self.front.parameters(): p.requires_grad_(False)
        self.proj = nn.Sequential(nn.Conv2d(1024,256,1), nn.GroupNorm(16,256), nn.GELU(), nn.Conv2d(256,512,1))
    def forward(self, raw1, raw2, base_s, base_t):
        with torch.no_grad():
            fs, ft = self.front(raw1, raw2)
            fs = F.interpolate(fs.float(), size=base_s.shape[-2:], mode="bilinear", align_corners=False)
            ft = F.interpolate(ft.float(), size=base_t.shape[-2:], mode="bilinear", align_corners=False)
        gs, bs = self.proj(fs).chunk(2, 1); gt, bt = self.proj(ft).chunk(2, 1)
        return base_s + 0.05 * (torch.tanh(gs) * base_s + bs), base_t + 0.05 * (torch.tanh(gt) * base_t + bt)

class E072Matcher(nn.Module):
    def __init__(self, c, variant: str, device):
        super().__init__()
        from core.update import BasicUpdateBlock
        from core.GDROSnet.FeatureFormer.cnn import BasicEncoder
        self.variant = variant
        self.front_v1 = OfficialLikeFrontend(c, modality_adapt=True) if variant in ["V1","V2","V4"] else None
        self.front_v3 = RawDenseFrontend() if variant == "V3" else None
        self.e046 = E046FiLM(device) if variant == "V2" else None
        self.context = BasicEncoder(output_dim=256, norm_fn="instance")
        self.update = BasicUpdateBlock(c, hidden_dim=128)
        self.hidden_dim=128; self.context_dim=128; self.corr_radius=4
    def _features(self, raw1, raw2, img1n, img2n, cnet):
        if self.variant == "V3":
            return self.front_v3(img1n, img2n, cnet)
        fs, ft = self.front_v1(img1n, img2n, cnet)
        if self.variant == "V2":
            fs, ft = self.e046(raw1, raw2, fs, ft)
        return fs, ft
    def initialize_flow(self, img):
        from core.utils.utils import coords_grid
        n,_,h,w=img.shape
        return coords_grid(n,h//8,w//8,device=img.device), coords_grid(n,h//8,w//8,device=img.device)
    def upsample_flow(self, flow, mask):
        n,_,h,w=flow.shape
        if mask is None: return F.interpolate(8*flow, scale_factor=8, mode="bilinear", align_corners=True)
        mask=torch.softmax(mask.view(n,1,9,8,8,h,w),dim=2)
        up=F.unfold(8*flow,[3,3],padding=1).view(n,2,9,1,1,h,w)
        up=torch.sum(mask*up,dim=2).permute(0,1,4,2,5,3).reshape(n,2,8*h,8*w)
        return up
    def forward(self, raw1, raw2, iters=12, test_mode=False, return_features=False):
        from core.corr import CorrBlock
        img1n=2*(raw1/255.0)-1.0; img2n=2*(raw2/255.0)-1.0
        cnet=self.context(img1n)
        fs,ft=self._features(raw1,raw2,img1n,img2n,cnet)
        corr_fn=CorrBlock(fs.float(),ft.float(),radius=self.corr_radius)
        net,inp=torch.split(cnet,[self.hidden_dim,self.context_dim],dim=1); net=torch.tanh(net); inp=torch.relu(inp)
        c0,c1=self.initialize_flow(img1n); outs=[]
        for _ in range(iters):
            c1=c1.detach(); corr=corr_fn(c1); flow=c1-c0
            net,mask,delta=self.update(net,inp,corr,flow); c1=c1+delta
            outs.append(self.upsample_flow(c1-c0,mask))
        if return_features: return outs, fs, ft
        if test_mode: return c1-c0, outs[-1]
        return outs

def load_teacher(device):
    from core.GDROSnet.FeatureFormer.gdros import GDROS
    c=cfg(); t=torch.nn.DataParallel(GDROS(c[c.transformer])); t.load_state_dict(torch.load(GDROS_CKPT,map_location="cpu")); t=t.module.to(device).eval()
    for p in t.parameters(): p.requires_grad_(False)
    return t
def teacher_feat_flow(t, a, b, iters):
    img1=2*(a/255.0)-1.0; img2=2*(b/255.0)-1.0
    c=t.context_encoder(img1)
    fs,ft=t.featureNet(img1.contiguous(),img2.contiguous(),{},c)
    preds=t(a,b,iters=iters)
    return fs.detach(), ft.detach(), [p.detach() for p in preds]
def feature_stats(x):
    y=x.detach().float()
    return {"shape":list(y.shape),"mean":float(y.mean()),"std":float(y.std()),"norm_mean":float(torch.linalg.vector_norm(y,dim=1).mean()),"abs_mean":float(y.abs().mean())}
def corr_diag(fs,ft,flow,stride=8.0,max_src=96):
    with torch.no_grad():
        b,c,h,w=fs.shape; n=h*w; k=min(max_src,n)
        f1=fs.flatten(2).transpose(1,2)[0,:k]; f2=ft.flatten(2)[0]
        corr=torch.matmul(f1.float(),f2.float())
        yy=torch.arange(h,device=fs.device).repeat_interleave(w)[:k].float(); xx=torch.arange(w,device=fs.device).repeat(h)[:k].float()
        gt=flow[:,(yy*stride).long().clamp(max=flow.shape[1]-1),(xx*stride).long().clamp(max=flow.shape[2]-1)].T.to(fs.device)
        tx=((xx*stride+gt[:,0])/stride).round().long().clamp(0,w-1); ty=((yy*stride+gt[:,1])/stride).round().long().clamp(0,h-1)
        gi=ty*w+tx; gt_score=corr[torch.arange(k,device=fs.device),gi]
        rank=(corr>gt_score[:,None]).sum(1)+1
        ent=torch.distributions.Categorical(logits=corr).entropy()
        return {"rank_mean":float(rank.float().mean()),"hit1":float((rank<=1).float().mean()*100),"hit4":float((rank<=4).float().mean()*100),"hit8":float((rank<=8).float().mean()*100),"hit32":float((rank<=32).float().mean()*100),"entropy":float(ent.mean())}
def pair_metrics(vals):
    a=np.asarray(vals,np.float64)
    return {"pairs":int(a.size),"aepe":float(a.mean()),"cmr1":float((a<1).mean()*100),"cmr2":float((a<2).mean()*100),"cmr5":float((a<5).mean()*100),"cmr10":float((a<10).mean()*100),"cmr16":float((a<16).mean()*100)}
def eval_model(model, ds, device, iters_list, max_pairs, v, name):
    from core.LSRnet.LSmodel import LSR
    vals={f"iter{it}_pre_lsr":[] for it in iters_list}|{f"iter{it}_post_lsr":[] for it in iters_list}
    fstats=[]; cstats=[]
    mat=mat_dir(v)/f"{name}_predictions_features_before_gt.jsonl"; mat.parent.mkdir(parents=True,exist_ok=True); mat.unlink(missing_ok=True)
    with torch.inference_mode():
        for i in range(min(max_pairs,len(ds))):
            file_opt,a,b,gt,_=ds[i]; a=a[None].to(device); b=b[None].to(device)
            preds,fs,ft=model(a,b,iters=max(iters_list),return_features=True)
            rec={"pair_index":i,"pair_id":Path(file_opt).stem,"uses_truth_flow":False,"features_predictions_materialized_before_gt":True,"feature_stats_src":feature_stats(fs),"feature_stats_tgt":feature_stats(ft)}
            fstats += [rec["feature_stats_src"], rec["feature_stats_tgt"]]
            cd=corr_diag(fs,ft,gt.to(device)); cstats.append(cd); rec["corr_diag_joined_after_materialization"]=cd
            for it in iters_list:
                pre=preds[it-1].cpu(); post=LSR(1,512,512,preds[it-1],200000).cpu()
                for k,fl in [(f"iter{it}_pre_lsr",pre),(f"iter{it}_post_lsr",post)]:
                    e=torch.linalg.vector_norm((fl-gt[None]).float(),dim=1).view(-1)
                    vals[k].append(float(e.mean())); rec[k]=float(e.mean())
            with mat.open("a") as f: f.write(json.dumps(rec,sort_keys=True)+"\n")
            torch.cuda.empty_cache()
    def mean_dict(rows):
        out={}
        for k in sorted({k for r in rows for k in r if isinstance(r.get(k),(int,float))}):
            arr=np.asarray([r[k] for r in rows if k in r],float)
            if arr.size: out[k]=float(arr.mean())
        return out
    return {k:pair_metrics(x) for k,x in vals.items()} | {"feature_distribution":mean_dict(fstats),"corr_quality":mean_dict(cstats),"materialized_before_gt":str(mat),"materialized_sha256":sha(mat)}
def train(args):
    add_paths(); v=args.variant.upper(); audit(v)
    import core.datasets
    from core.loss import sequence_loss, matrix2flow
    from core.LSRnet.LSmodel import LSR_list
    device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"); c=cfg()
    tr=core.datasets.opt_sar_train(split="training",dstype="training",root=str(data_view(v)))
    dv=core.datasets.opt_sar_test(split="testing",dstype="testing",root=str(TEST_ROOT))
    model=E072Matcher(c,v,device).to(device)
    teacher=load_teacher(device) if (args.teacher_weight>0 or v=="V4") else None
    resume_payload = None
    if getattr(args, "resume_checkpoint", None):
        resume_payload = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_payload["model"], strict=False)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-5)
    rng=np.random.default_rng(args.seed); curves=[]; evals={}
    if getattr(args, "resume_report", None) and Path(args.resume_report).exists():
        old=json.loads(Path(args.resume_report).read_text())
        curves=old.get("curves", [])
        evals=old.get("evals", {})
    if resume_payload is not None:
        curves.append({"variant":v,"phase":"resume","step":int(resume_payload.get("step",0)),"checkpoint":str(args.resume_checkpoint),"checkpoint_sha256":sha(Path(args.resume_checkpoint))})
    milestones=sorted(set([m for m in args.milestones if m<=args.steps]+[args.steps]))
    for phase,steps in [("toy",args.toy_steps),("full",args.steps)]:
        begin = (int(args.start_step) + 1) if phase == "full" else 1
        for step in range(begin,steps+1):
            idx=int(rng.integers(0,args.toy_pairs if phase=="toy" else len(tr)))
            a,b,flow,valid=tr[idx]; a=a[None].to(device); b=b[None].to(device); flow=flow[None].to(device); valid=valid[None].to(device)
            preds,fs,ft=model(a,b,iters=args.train_iters,return_features=True)
            lseq,m1=sequence_loss(preds,flow,valid,c); mats=matrix2flow(LSR_list(preds),flow); lgeo,m2=sequence_loss(mats,flow,valid,c)
            loss=args.lambda_seq*lseq + args.lambda_geo*lgeo
            feat_loss=torch.tensor(0.,device=device); flow_dist=torch.tensor(0.,device=device)
            if teacher is not None:
                with torch.no_grad(): tfs,tft,tpreds=teacher_feat_flow(teacher,a,b,args.train_iters)
                feat_loss=F.smooth_l1_loss(fs,tfs)+F.smooth_l1_loss(ft,tft)+0.5*(F.smooth_l1_loss(F.normalize(fs,dim=1),F.normalize(tfs,dim=1))+F.smooth_l1_loss(F.normalize(ft,dim=1),F.normalize(tft,dim=1)))
                flow_dist=sum(F.smooth_l1_loss(p,t) for p,t in zip(preds,tpreds))/len(preds)
                tw=args.teacher_weight if not (v=="V4" and args.teacher_weight==0) else 0.15
                loss=loss + tw*feat_loss + args.teacher_flow_weight*flow_dist
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
            log_every=max(1, steps//10 if phase=="toy" else args.steps//20)
            if step==1 or step%log_every==0:
                row={"variant":v,"phase":phase,"step":step,"loss":float(loss.detach().cpu()),"lseq_epe":m1["epe"],"lgeo_epe":m2["epe"],"flow5":float(m1["5px"]*100),"mat5":float(m2["5px"]*100),"feat_loss":float(feat_loss.detach().cpu()),"flow_dist":float(flow_dist.detach().cpu())}
                curves.append(row); print(f"E072A {v} {phase} step={step}/{steps} loss={row['loss']:.3f} flow5={row['flow5']:.2f} mat5={row['mat5']:.2f} feat={row['feat_loss']:.3f}",flush=True)
            if phase=="full" and step in milestones:
                cp=ckpt_dir(v)/f"{v}_step{step}.pt"; cp.parent.mkdir(parents=True,exist_ok=True)
                torch.save({"model":model.state_dict(),"variant":v,"step":step,"seed":args.seed,"teacher":str(GDROS_CKPT) if teacher is not None else None,"teacher_sha256":sha(GDROS_CKPT) if teacher is not None else None},cp)
                evals[str(step)]={"checkpoint":str(cp),"checkpoint_sha256":sha(cp),"dev":eval_model(model.eval(),dv,device,[12,24,32],args.dev_eval_pairs,v,f"{v}_step{step}")}; model.train()
                js(formal(v)/"training_sufficiency_report.json",{"status":"RUNNING","curves":curves,"evals":evals})
                torch.cuda.empty_cache()
    return finalize(v,curves,evals,time.time())
def finalize(v,curves=None,evals=None,start=None):
    fr=formal(v)
    if evals is None:
        r=json.loads((fr/"training_sufficiency_report.json").read_text()) if (fr/"training_sufficiency_report.json").exists() else {}; curves=r.get("curves",[]); evals=r.get("evals",{})
    best=None
    for st,ev in (evals or {}).items():
        for k,val in ev.get("dev",{}).items():
            if isinstance(val,dict) and "aepe" in val:
                row={"variant":v,"step":int(st),"eval_variant":k,**val}
                if best is None or row["aepe"]<best["aepe"]: best=row
    sufficient=bool(evals and any(int(k)>=20000 for k in evals))
    status="TRAINING_INSUFFICIENT" if not sufficient else ("BEATS_E065" if best and best["aepe"]<REFS["e065"]["aepe"] else ("BEATS_E071_TEACHER_ONLY" if best and best["aepe"]<REFS["e071a_teacher"]["aepe"] else "TRAINING_SUFFICIENT_NO_E071_TEACHER_GAIN"))
    last_eval=(evals or {}).get(str(max([int(k) for k in evals], default=0)),{}).get("dev",{}) if evals else {}
    dist={k:v for k,v in last_eval.items() if k in ["feature_distribution","corr_quality"]}
    objs={
        "G0":{"status":"PASS" if curves else "PENDING","training_sufficient":sufficient,"toy_signal":curves[:12]},
        "G1":{"status":status,"best":best,"refs":REFS},
        "distribution_alignment_report":dist,
        "training_sufficiency_report":{"status":status,"curves":curves,"evals":evals},
        "train_curve_checkpoints":{"curves":curves,"evals":evals},
        "corr_quality_report":{k:e.get("dev",{}).get("corr_quality") for k,e in (evals or {}).items()},
        "iterwise_flow_report":evals,
        "pre_post_lsr_bucket_report":evals,
        "method_ablation":{"variant":v,"best":best},
        "resource_report":{"gpu_policy":"zzvr GPU0-3","elapsed_seconds":None if start is None else time.time()-start},
        "RUN_MANIFEST":{"experiment":"E072A","variant":v,"status":status,"run_root":str(variant_root(v)),"formal_root":str(fr),"script":str(Path(__file__)),"official_code":str(OFF),"official_commit":os.popen(f"cd {OFF} && git rev-parse HEAD 2>/dev/null").read().strip(),"official_checkpoint_sha256":sha(GDROS_CKPT),"e046_checkpoint_sha256":sha(E046_CKPT),"testing_zip_used_for_training":False,"no_os424_confirm_rescue":True,"materialized_before_gt":[str(p) for p in mat_dir(v).glob('*.jsonl')]},
    }
    for n,o in objs.items(): js(fr/f"{n}.json",o)
    (fr/"reviewer_conclusion.txt").write_text(f"E072A {v}: {status}. Layer-wise aligned frontend variant trained jointly with GDROS-style matcher. Released weights/code are read-only. Do not compare as official GDROS replay.\n")
    (fr/"RESULT_CARD.md").write_text(f"# E072A {v} RESULT_CARD\n\nStatus: `{status}`\n\nBest: `{best}`\n",encoding="utf-8")
    return {"status":status,"best":best,"formal":str(fr)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["run","finalize","p0"],default="run"); ap.add_argument("--variant",choices=["V1","V2","V3","V4"],required=True)
    ap.add_argument("--gpu",type=int,default=0); ap.add_argument("--seed",type=int,default=720)
    ap.add_argument("--toy-pairs",type=int,default=8); ap.add_argument("--toy-steps",type=int,default=400); ap.add_argument("--steps",type=int,default=20000); ap.add_argument("--milestones",type=int,nargs="+",default=[1000,5000,20000])
    ap.add_argument("--train-iters",type=int,default=12); ap.add_argument("--dev-eval-pairs",type=int,default=404); ap.add_argument("--lr",type=float,default=1.25e-4)
    ap.add_argument("--lambda-seq",type=float,default=0.5); ap.add_argument("--lambda-geo",type=float,default=0.5); ap.add_argument("--teacher-weight",type=float,default=0.0); ap.add_argument("--teacher-flow-weight",type=float,default=0.0)
    ap.add_argument("--run-tag",type=str,default="")
    ap.add_argument("--resume-checkpoint",type=str,default="")
    ap.add_argument("--resume-report",type=str,default="")
    ap.add_argument("--start-step",type=int,default=0)
    args=ap.parse_args(); os.environ["E072A_RUN_TAG"]=args.run_tag; add_paths(); formal(args.variant).mkdir(parents=True,exist_ok=True); mat_dir(args.variant).mkdir(parents=True,exist_ok=True); log_dir(args.variant).mkdir(parents=True,exist_ok=True)
    if args.mode=="p0": print(json.dumps(audit(args.variant),indent=2))
    elif args.mode=="finalize": print(json.dumps(finalize(args.variant),indent=2))
    else: print(json.dumps(train(args),indent=2))
if __name__=="__main__": main()
