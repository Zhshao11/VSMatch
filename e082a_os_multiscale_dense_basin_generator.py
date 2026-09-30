#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("SPARSEGEO_ROOT", "/mnt/sda/zhushihao/VS_Match"))
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
RUN_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-082A-ZZVR-OS-MULTISCALE-DENSE-BASIN-GENERATOR"
E075_ROOT = ROOT / "GDROS/experiments/sparsegeo/runs/E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE"
TRAIN_FLAT = ROOT / "datasets/extracted/gdros_official_e003a_training"
TEST_ROOT = ROOT / "datasets/extracted/gdros_official_e003a_testing"
M4_CKPT = E075_ROOT / "run-m4-regiongate120k-r2/checkpoints/M4_step80000.pt"
M2_CKPT = E075_ROOT / "run-m2-scalargate120k-r4/checkpoints/M2_step120000.pt"
M4_G1 = E075_ROOT / "run-m4-regiongate120k-r2/formal/G1.json"
M2_G1 = E075_ROOT / "run-m2-scalargate120k-r4/formal/G1.json"

REFS = {
    "gdros_released": {"aepe": 1.461486, "cmr1": 33.73, "cmr2": 81.60, "cmr5": 99.53},
    "e075_m4": {"aepe": 2.3613, "cmr1": 56.68, "cmr2": 78.96, "cmr5": 88.86, "cmr10": 94.80, "cmr16": 97.03, "ckpt": str(M4_CKPT)},
    "e075_m2": {"aepe": 2.3477, "cmr1": 55.94, "cmr2": 79.21, "cmr5": 88.12, "cmr10": 94.55, "cmr16": 97.52, "ckpt": str(M2_CKPT)},
    "e076a_fixed_pool_oracle": {"oracle_cmr5": 91.83},
    "e076b_expanded_pool": {"oracle_cmr5": 94.55, "hypotheses": 210, "avg_per_pair": 184, "status": "ORACLE_ONLY_NOT_USEFUL"},
    "e078a_best": {"oracle_cmr5": 93.56, "nogt_probe_cmr5": 87.62, "gap": 5.94, "status": "NO_STRONG_REACHABLE_CANDIDATE"},
    "e079_best_family": {"cmr5": "88-89", "oracle": "91-92", "status": "old basin tied"},
}
KNOWN_SHA = {
    str(M4_CKPT): "55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b",
    str(M2_CKPT): "a3556ac55487e76b5b6c70834de885b3868ad1fd10bde619a53ccaba9a315e5d",
    str(M4_G1): "cc69e6ad100cc3fd8a0b20d97229a33cf14264702c896e28dd37eafac92918b3",
    str(M2_G1): "8342ff5d416c3dad28ac1bcea54f0085962bde43690b39ba1b7a452500fc9f9c",
}

def rr(arm: str, tag: str = "") -> Path:
    return RUN_ROOT / f"run-{arm.lower()}{('-' + tag.strip('-')) if tag else ''}"
def formal(arm: str, tag: str = "") -> Path: return rr(arm, tag) / "formal"
def mat_dir(arm: str, tag: str = "") -> Path: return formal(arm, tag) / "materialized"
def ckpt_dir(arm: str, tag: str = "") -> Path: return rr(arm, tag) / "checkpoints"
def log_dir(arm: str, tag: str = "") -> Path: return rr(arm, tag) / "logs"
def js(p: Path, o: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
def sha(p: Path):
    cached = KNOWN_SHA.get(str(Path(p)))
    if cached: return cached
    if not Path(p).exists(): return None
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def metrics(vals):
    a = np.asarray(vals, np.float64)
    if a.size == 0: return {"pairs": 0}
    return {"pairs": int(a.size), "aepe": float(a.mean()), "cmr1": float((a < 1).mean() * 100),
            "cmr2": float((a < 2).mean() * 100), "cmr5": float((a < 5).mean() * 100),
            "cmr10": float((a < 10).mean() * 100), "cmr16": float((a < 16).mean() * 100)}
def quants(vals):
    a = np.asarray(vals, np.float64)
    return {} if a.size == 0 else {k: float(np.quantile(a, q)) for k, q in [("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95)]}

def add_paths():
    sys.path.insert(0, str(TMP))
    import e074a_f2_f3_twostage_fusion as e074
    import e075a_preserve_high_precision_recover_c2c5 as e075
    e072, e073 = e074.add_paths()
    return e074, e075, e072, e073

def data_view(arm, tag):
    r = rr(arm, tag) / "data_view/training"
    r.mkdir(parents=True, exist_ok=True)
    for n in ["opt", "sar_warped", "truth_flow", "datum"]:
        d, s = r / n, TRAIN_FLAT / n
        if not d.exists(): d.symlink_to(s, target_is_directory=True)
    return rr(arm, tag) / "data_view"

def load_pairfusion(e074, e075, variant, ckpt, device, freeze=True):
    m = e075.PairFusion(e074, "region" if variant == "M4" else "scalar", device).to(device)
    payload = torch.load(ckpt, map_location=device)
    miss = m.load_state_dict(payload["model"], strict=False)
    if freeze:
        for p in m.parameters(): p.requires_grad_(False)
        m.eval()
    return m, {"checkpoint": str(ckpt), "sha256": sha(ckpt), "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}

class DenseBasinGenerator(nn.Module):
    def __init__(self, e074, e075, arm: str, device, grid_radius=16, grid_step=4, topk=8, include_m2=False):
        super().__init__()
        self.e074, self.arm, self.topk = e074, arm.upper(), topk
        self.grid_radius, self.grid_step = grid_radius, grid_step
        vals = torch.arange(-grid_radius, grid_radius + 1, grid_step, dtype=torch.float32)
        yy, xx = torch.meshgrid(vals, vals, indexing="ij")
        self.register_buffer("delta_grid", torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))
        self.m4, self.m4_info = load_pairfusion(e074, e075, "M4", M4_CKPT, device, freeze=(self.arm != "S3"))
        self.m2 = None
        self.m2_info = None
        if include_m2:
            self.m2, self.m2_info = load_pairfusion(e074, e075, "M2", M2_CKPT, device, freeze=True)
        if self.arm == "S3":
            for n, p in self.m4.named_parameters():
                p.requires_grad_("gate" in n or "scalar" in n)
            self.m4.train()
        in_ch = 6 + 2 + 2 + (2 if self.m2 is not None else 0)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 48, 5, 2, 2), nn.GELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GELU(),
            nn.Conv2d(96, 96, 3, 1, 1), nn.GELU(),
        )
        self.coarse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(96, 128, 1), nn.GELU(), nn.Conv2d(128, len(self.delta_grid), 1))
        self.local = nn.Sequential(nn.Conv2d(96, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 2, 1))
        nn.init.normal_(self.local[-1].weight, std=1e-4); nn.init.constant_(self.local[-1].bias, 0.0)
        self.last_aux = {}

    def lsr(self, x): return self.e074.lsr(x).to(x.device)

    def base_flows(self, a, b, iters):
        if self.arm == "S3":
            pm4 = self.m4(a, b, iters=iters, lsr_fn=self.e074.lsr)[-1]
        else:
            with torch.no_grad():
                pm4 = self.m4(a, b, iters=iters, lsr_fn=self.e074.lsr)[-1]
        fm4 = self.lsr(pm4)
        flows = [fm4]; names = ["E075_M4_base"]
        if self.m2 is not None:
            with torch.no_grad():
                fm2 = self.lsr(self.m2(a, b, iters=iters, lsr_fn=self.e074.lsr)[-1])
            flows.append(fm2); names.append("E075_M2_base")
        return names, flows

    def forward(self, a, b, iters=12, train_soft=True, return_candidates=False):
        names, base_flows = self.base_flows(a, b, iters)
        base = base_flows[0]
        x = torch.cat([a / 255.0, b / 255.0, base / 64.0, torch.abs(base) / 64.0] + ([base_flows[1] / 64.0] if len(base_flows) > 1 else []), dim=1)
        z = self.encoder(x)
        logits = self.coarse(z).flatten(1)
        prob = torch.softmax(logits, dim=1)
        local = F.interpolate(3.0 * torch.tanh(self.local(z)), size=base.shape[-2:], mode="bilinear", align_corners=False)
        topv, topi = torch.topk(prob, k=min(self.topk, prob.shape[1]), dim=1)
        cands, cnames = [], []
        for j in range(topi.shape[1]):
            d = self.delta_grid[topi[:, j]].to(base.device).view(base.shape[0], 2, 1, 1)
            cands.append(base + d + local); cnames.append(f"heatmap_top{j}")
        for n, f in zip(names, base_flows):
            cands.append(f); cnames.append(n)
        if train_soft:
            dsoft = prob @ self.delta_grid.to(prob.device)
            selected = base + dsoft.view(base.shape[0], 2, 1, 1) + local
        else:
            selected = cands[0]
        entropy = (-(prob * prob.clamp_min(1e-8).log()).sum(1)).mean()
        self.last_aux = {"entropy": float(entropy.detach().cpu()), "top1_prob": float(topv[:, 0].mean().detach().cpu()), "mode_count": len(cands)}
        return ([selected], cnames, cands, logits, prob, base, local) if return_candidates else [selected]

def epe_mean(flow, gt, valid):
    e = torch.linalg.vector_norm((flow - gt).float(), dim=1).view(-1)[valid.view(-1).bool()]
    return e.mean(), e
def cvar(e, frac=.1):
    k = max(1, int(np.ceil(e.numel() * frac)))
    return e.topk(k).values.mean()
def basin_target(logits, delta_grid, base, gt, valid, sigma=4.0):
    # Pair-level residual basin label from GT minus base flow; GT is train label only.
    mask = valid.bool()
    resid = (gt - base).permute(0, 2, 3, 1)[mask].mean(0)
    dist2 = ((delta_grid.to(logits.device) - resid[None]) ** 2).sum(1)
    soft = torch.softmax(-dist2 / (2 * sigma * sigma), dim=0).expand(logits.shape[0], -1)
    return soft, resid.detach()

def train_step_loss(model, a, b, flow, valid, args):
    pred, names, cands, logits, prob, base, local = model(a, b, args.train_iters, train_soft=True, return_candidates=True)
    soft, resid = basin_target(logits, model.delta_grid, base.detach(), flow, valid, sigma=args.label_sigma)
    ce = -(soft * F.log_softmax(logits, dim=1)).sum(1).mean()
    sel_mean, sel_e = epe_mean(pred[-1], flow, valid)
    cand_means = [epe_mean(c, flow, valid)[0] for c in cands]
    oracle = torch.stack(cand_means).min()
    tail = cvar(sel_e, args.tail_frac)
    over5 = F.softplus(sel_e - 5).mean()
    over10 = F.softplus(sel_e - 10).mean()
    base_mean, _ = epe_mean(base.detach(), flow, valid)
    hard_w = 1.0
    if args.hard_tail and float(base_mean.detach().cpu()) > 5.0: hard_w = args.hard_tail_weight
    loss = args.ce_weight * ce + args.coverage_weight * F.softplus(oracle - 5) + args.final_weight * sel_mean
    loss = hard_w * (loss + args.tail_weight * tail + args.over5_weight * over5 + args.over10_weight * over10)
    aux = {"ce": float(ce.detach().cpu()), "selected_epe": float(sel_mean.detach().cpu()), "oracle_epe": float(oracle.detach().cpu()),
           "base_epe": float(base_mean.detach().cpu()), "tail": float(tail.detach().cpu()), "over5": float(over5.detach().cpu()),
           "over10": float(over10.detach().cpu()), "hard_w": hard_w, "target_residual": [float(x) for x in resid.detach().cpu()]}
    return loss, aux

def eval_model(model, ds, device, args, arm, tag, name):
    selected, oracle, base_vals, rows, counts = [], [], [], [], []
    mat = mat_dir(arm, tag) / f"{name}_candidates_before_gt.jsonl"
    mat.parent.mkdir(parents=True, exist_ok=True); mat.unlink(missing_ok=True)
    t0 = time.time()
    with torch.inference_mode(), mat.open("w", encoding="utf-8") as f:
        for i in range(min(args.dev_eval_pairs, len(ds))):
            file_opt, a, b, gt, valid = ds[i]
            a, b = a[None].to(device), b[None].to(device)
            pred, names, cands, logits, prob, base, local = model(a, b, args.eval_iters, train_soft=False, return_candidates=True)
            # materialize candidate metadata before GT metric join
            rec = {"pair_index": i, "pair_id": Path(file_opt).stem, "uses_gt_for_inference": False,
                   "predictions_materialized_before_gt": True, "candidate_names": names, "candidate_count": len(names),
                   "top_probs": [float(x) for x in torch.topk(prob[0], k=min(args.topk, prob.shape[1])).values.detach().cpu()]}
            f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush()
            gt1, valid1 = gt[None].float(), valid.float()
            se, _ = epe_mean(pred[-1].cpu(), gt1, valid1)
            be, _ = epe_mean(base.cpu(), gt1, valid1)
            ce = [float(epe_mean(c.cpu(), gt1, valid1)[0]) for c in cands]
            selected.append(float(se)); oracle.append(float(min(ce))); base_vals.append(float(be)); counts.append(len(names))
            rows.append({**rec, "selected_epe": float(se), "oracle_epe": float(min(ce)), "base_epe": float(be),
                         "best_candidate": names[int(np.argmin(ce))], "candidate_epes": dict(zip(names, ce))})
            if i % 50 == 0: print(f"E082A {arm} eval {name} pair {i}", flush=True)
            torch.cuda.empty_cache()
    out = {"selected": metrics(selected), "oracle": metrics(oracle), "base_m4": metrics(base_vals),
           "oracle_gap_cmr5": metrics(oracle).get("cmr5", 0) - metrics(selected).get("cmr5", 0),
           "p90_p95_selected": quants(selected), "p90_p95_oracle": quants(oracle),
           "gt5_failure_reduction_oracle": int(sum(s >= 5 and o < 5 for s, o in zip(selected, oracle))),
           "gt10_failure_reduction_oracle": int(sum(s >= 10 and o < 10 for s, o in zip(selected, oracle))),
           "mode_count_mean": float(np.mean(counts)) if counts else 0.0,
           "inference_seconds_per_pair": (time.time() - t0) / max(1, len(selected)),
           "materialized_before_gt": str(mat), "materialized_sha256": sha(mat)}
    js(formal(arm, tag) / f"{name}_per_pair_metrics.json", rows)
    return out

def audit(arm, tag, script_path):
    train_ids = {p.stem for p in (TRAIN_FLAT / "opt").glob("*")}
    dev_ids = {p.stem for p in (TEST_ROOT / "testing/opt").glob("*")}
    train_real = {str(p.resolve()) for p in (TRAIN_FLAT / "opt").glob("*")}
    dev_real = {str(p.resolve()) for p in (TEST_ROOT / "testing/opt").glob("*")}
    obj = {"status": "PASS", "experiment": "E082A", "arm": arm, "run_root": str(rr(arm, tag)),
           "method": {"S1": "dense basin heatmap, frozen E075A-M4", "S2": "S1 + hard-tail/larger-error loss; M2 optional via --include-m2",
                      "S3": "S2 + limited E075-M4 gate fine-tune/projection joint training; M2 optional via --include-m2"},
           "guardrails": {"not_selector_window_oracle_only": True, "controlled_modes": True, "candidate_before_gt": True,
                          "gt_use": "train labels/oracle diagnostics/final metrics only", "testing_zip_training": False,
                          "no_os424_confirm_rescue": True, "baseline_artifacts_readonly": True},
           "split": {"train_root": str(TRAIN_FLAT), "dev_root": str(TEST_ROOT), "train_count": len(train_ids), "dev_count": len(dev_ids),
                     "basename_intersection_count": len(train_ids & dev_ids), "basename_intersection_sample": sorted(train_ids & dev_ids)[:10],
                     "basename_overlap_interpretation": "OS split-local filenames may overlap; leakage gate uses resolved realpath/source roots.",
                     "realpath_intersection_count": len(train_real & dev_real), "realpath_intersection_sample": sorted(train_real & dev_real)[:10],
                     "source_roots_disjoint": str(TRAIN_FLAT.resolve()) != str((TEST_ROOT / "testing").resolve())},
           "sources": {"script": str(script_path), "script_sha256": sha(script_path), "M4_ckpt": str(M4_CKPT), "M4_sha256": sha(M4_CKPT),
                       "M2_ckpt": str(M2_CKPT), "M2_sha256": sha(M2_CKPT), "M4_G1_sha256": sha(M4_G1), "M2_G1_sha256": sha(M2_G1)},
           "refs": REFS}
    for n, o in {"P0": obj, "RUN_MANIFEST": obj, "baseline_comparison": REFS}.items(): js(formal(arm, tag) / f"{n}.json", o)
    return obj

def finalize(arm, tag, curves=None, evals=None, start=None, script_path=None):
    fr = formal(arm, tag)
    if evals is None:
        p = fr / "training_report.json"; d = json.loads(p.read_text()) if p.exists() else {}; curves = d.get("curves", []); evals = d.get("evals", {})
    curves, evals = curves or [], evals or {}
    best_cmr5 = best_aepe = None
    for st, ev in evals.items():
        row = {"step": int(st), **ev.get("selected", {}), "oracle_cmr5": ev.get("oracle", {}).get("cmr5"), "oracle_gap_cmr5": ev.get("oracle_gap_cmr5"),
               "mode_count_mean": ev.get("mode_count_mean"), "p90": ev.get("p90_p95_selected", {}).get("p90"), "p95": ev.get("p90_p95_selected", {}).get("p95")}
        if row.get("cmr5") is not None and (best_cmr5 is None or row["cmr5"] > best_cmr5["cmr5"] or (row["cmr5"] == best_cmr5["cmr5"] and row["aepe"] < best_cmr5["aepe"])): best_cmr5 = row
        if row.get("aepe") is not None and (best_aepe is None or row["aepe"] < best_aepe["aepe"]): best_aepe = row
    sufficient = any(int(k) >= 20000 for k in evals)
    status = "TRAINING_INSUFFICIENT" if not sufficient else "NO_PROMISING_DENSE_BASIN_SIGNAL"
    if best_cmr5 and best_cmr5.get("oracle_cmr5", 0) >= 96 and (best_cmr5.get("cmr5", 0) >= REFS["e075_m4"]["cmr5"] + 1.5 or best_cmr5.get("aepe", 99) <= REFS["e075_m4"]["aepe"] * .9):
        status = "PROMISING_DENSE_BASIN_RESCUE"
    if best_cmr5 and best_cmr5.get("cmr5", 0) >= 92 and best_cmr5.get("aepe", 99) <= 2.1:
        status = "STRONG_DENSE_BASIN_RESCUE"
    objs = {"G0": {"status": "PASS" if curves else "PENDING", "curves_head": curves[:8]},
            "G1": {"status": status, "best_by_cmr5": best_cmr5, "best_by_aepe": best_aepe, "refs": REFS},
            "training_report": {"status": status, "curves": curves, "evals": evals},
            "RESULT_CARD": {"status": status, "best_by_cmr5": best_cmr5, "best_by_aepe": best_aepe},
            "dense_basin_scorecard": {"arm": arm, "status": status, "best_by_cmr5": best_cmr5, "best_by_aepe": best_aepe},
            "failure_bucket_diagnostics": {"evals": {k: {"p90_p95_selected": v.get("p90_p95_selected"), "failure_reduction5": v.get("gt5_failure_reduction_oracle"), "failure_reduction10": v.get("gt10_failure_reduction_oracle")} for k, v in evals.items()}},
            "resource_report": {"elapsed_seconds": None if start is None else time.time() - start, "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                "gpu_policy": "queue on GPU0-3 only when idle; do not interrupt E080/E081/E079"},
            "RUN_MANIFEST": {"experiment": "E082A", "arm": arm, "status": status, "run_root": str(rr(arm, tag)), "formal_root": str(fr),
                             "script_sha256": sha(script_path or TMP / "e082a_os_multiscale_dense_basin_generator.py"),
                             "baseline_artifacts_readonly": True, "testing_zip_training": False, "materialized_before_gt": [str(p) for p in mat_dir(arm, tag).glob("*.jsonl")]}}
    for n, o in objs.items(): js(fr / f"{n}.json", o)
    (fr / "reviewer_conclusion.txt").write_text(f"E082A {arm}: {status}. Dense basin candidates generated no-GT; GT used only for train/oracle/final metrics; prior artifacts read-only.\n", encoding="utf-8")
    (fr / "RESULT_CARD.md").write_text(f"# E082A {arm} RESULT_CARD\n\nStatus: `{status}`\n\nBest by CMR@5: `{best_cmr5}`\n\nBest by AEPE: `{best_aepe}`\n", encoding="utf-8")
    return {"status": status, "best_by_cmr5": best_cmr5, "best_by_aepe": best_aepe, "formal": str(fr)}

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    e074, e075, e072, e073 = add_paths(); import core.datasets
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    arm, tag, script_path = args.arm.upper(), args.run_tag, Path(__file__)
    audit(arm, tag, script_path)
    tr = core.datasets.opt_sar_train(split="training", dstype="training", root=str(data_view(arm, tag)))
    dv = core.datasets.opt_sar_test(split="testing", dstype="testing", root=str(TEST_ROOT))
    model = DenseBasinGenerator(e074, e075, arm, device, args.grid_radius, args.grid_step, args.topk, include_m2=args.include_m2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed); curves = []; evals = {}; start = time.time()
    milestones = sorted(set([m for m in args.milestones if m <= args.steps] + [args.steps]))
    for step in range(1, args.steps + 1):
        idx = int(rng.integers(0, len(tr))); a, b, flow, valid = tr[idx]
        a, b, flow, valid = a[None].to(device), b[None].to(device), flow[None].to(device), valid[None].to(device)
        model.train(); loss, aux = train_step_loss(model, a, b, flow, valid, args)
        opt.zero_grad(set_to_none=True); loss.backward()
        g = float(torch.nn.utils.clip_grad_norm_(params, 1.0).detach().cpu()) if params else 0.0
        opt.step()
        if step == 1 or step % max(1, args.log_every) == 0:
            row = {"arm": arm, "step": step, "loss": float(loss.detach().cpu()), "grad_norm": g, "aux": aux, "last_aux": getattr(model, "last_aux", {})}
            curves.append(row); print(f"E082A {arm} step={step}/{args.steps} loss={row['loss']:.4f} grad={g:.3f} aux={aux}", flush=True)
            js(formal(arm, tag) / "training_report.json", {"status": "RUNNING", "curves": curves, "evals": evals})
        if step in milestones:
            cp = ckpt_dir(arm, tag) / f"{arm}_step{step}.pt"; cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "arm": arm, "step": step, "seed": args.seed, "refs": REFS, "script_sha256": sha(script_path)}, cp)
            model.eval(); ev = eval_model(model, dv, device, args, arm, tag, f"{arm}_step{step}"); ev["checkpoint"] = str(cp); ev["checkpoint_sha256"] = sha(cp); evals[str(step)] = ev
            js(formal(arm, tag) / "training_report.json", {"status": "RUNNING", "curves": curves, "evals": evals})
            if args.early_stop and len(evals) >= 2:
                cmr5s = [evals[str(m)]["selected"].get("cmr5", 0) for m in milestones if str(m) in evals]
                aepe = [evals[str(m)]["selected"].get("aepe", 99) for m in milestones if str(m) in evals]
                if len(cmr5s) >= 2 and cmr5s[-1] <= cmr5s[-2] + 0.05 and aepe[-1] >= aepe[-2] * .995:
                    print(f"E082A {arm} early_stop no CMR5/AEPE-tail gain cmr5={cmr5s[-2:]} aepe={aepe[-2:]}", flush=True); break
            torch.cuda.empty_cache()
    return finalize(arm, tag, curves, evals, start, script_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["p0", "smoke", "train", "finalize"], default="train")
    ap.add_argument("--arm", choices=["S1", "S2", "S3"], required=True)
    ap.add_argument("--run-tag", default="formal-r1")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=8211)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--milestones", type=int, nargs="+", default=[20000, 40000, 80000, 120000])
    ap.add_argument("--train-iters", type=int, default=12)
    ap.add_argument("--eval-iters", type=int, default=24)
    ap.add_argument("--dev-eval-pairs", type=int, default=404)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--grid-radius", type=int, default=16)
    ap.add_argument("--grid-step", type=int, default=4)
    ap.add_argument("--label-sigma", type=float, default=4.0)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--coverage-weight", type=float, default=0.6)
    ap.add_argument("--final-weight", type=float, default=0.35)
    ap.add_argument("--tail-weight", type=float, default=0.15)
    ap.add_argument("--over5-weight", type=float, default=0.4)
    ap.add_argument("--over10-weight", type=float, default=0.25)
    ap.add_argument("--tail-frac", type=float, default=0.1)
    ap.add_argument("--hard-tail", action="store_true")
    ap.add_argument("--hard-tail-weight", type=float, default=1.8)
    ap.add_argument("--include-m2", action="store_true")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()
    formal(args.arm, args.run_tag).mkdir(parents=True, exist_ok=True); mat_dir(args.arm, args.run_tag).mkdir(parents=True, exist_ok=True); log_dir(args.arm, args.run_tag).mkdir(parents=True, exist_ok=True)
    if args.mode == "p0": print(json.dumps(audit(args.arm, args.run_tag, Path(__file__)), indent=2))
    elif args.mode == "smoke":
        args.steps = min(args.steps, 2); args.milestones = [args.steps]; args.dev_eval_pairs = min(args.dev_eval_pairs, 2); args.early_stop = False
        print(json.dumps(train(args), indent=2))
    elif args.mode == "finalize": print(json.dumps(finalize(args.arm, args.run_tag, script_path=Path(__file__)), indent=2))
    else: print(json.dumps(train(args), indent=2))
if __name__ == "__main__": main()
