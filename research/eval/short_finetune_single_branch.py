#!/usr/bin/env python3
"""Adaptation short-finetune (scheme A) for the two single-branch ablations.

Unlike the zero-training diagnostic bypass, this holds the training budget,
data, seed and trainable-parameter set FIXED across the two configs and only
changes which side the fusion gates are locked to. That is the minimum
controlled comparison that can say something about recoverability.

  v1only : corr_gate_logit=-80 (a=0), fuse_gate=-80 (gs=0)  -> GDROS stream only
  v2only : corr_gate_logit=+80 (a=1), fuse_gate=+80 (gs=1)  -> gates pushed to V2

Frozen: all E072 frontends (b1/b2, including the frozen DINO/E046 injection),
corr_gate_logit / head_gate_logit, and the replaced const fuse_gate.

TRAINABLE (the only parameters that actually receive gradients):
  - PairFusion.gate           (L4 region gate)
  - E074Model(B).gate / scalar
  - E074Model(C).student.context / .update / .fuse_proj
NOTE: E074Model(B).teacher_preds runs under torch.no_grad(), so B's f2/f3
recurrent chains CANNOT be fine-tuned; unfreezing them would only waste
optimizer state. C.student is the only recurrent chain with gradients.

Training data: the same OS training data_view used by the protected M4 run,
same seed (901), same sample order for both configs. Loss mirrors the main
terms of the original E075A-M4 objective (lseq + lgeo + gate/tail/bucket);
the two small regularizers (entropy/balance, weights 0.03/0.05) are omitted
but omitted identically in both configs.

After training, the SAME process evaluates the model with the GDROS official
test.py protocol (424 pairs, iters=32, LSR NPOINT=200000).

Run on zzvr:
  cd /mnt/sda/zhushihao/VS_Match/image_matching_part2_migrated
  CUDA_VISIBLE_DEVICES=5 $PY .codex_tmp/short_finetune_single_branch.py \
    --config v1only --steps 2000 --gpu 0 \
    --out .codex_tmp/short_ft_v1only.json
"""
from __future__ import annotations
import sys
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/mnt/sda/zhushihao/VS_Match"
TMP = ROOT + "/image_matching_part2_migrated/.codex_tmp"
OFFICIAL = ROOT + "/external_repos/E068A_GDROS_official"
sys.path.insert(0, TMP)
sys.path.insert(0, OFFICIAL)

import timm
if not hasattr(timm, "layers"):
    import timm.models.layers as _timm_layers
    sys.modules.setdefault("timm.layers", _timm_layers)

import core.datasets
from core.loss import sequence_loss, matrix2flow
from core.LSRnet.LSmodel import LSR_list
import e074a_f2_f3_twostage_fusion as e074
e072, e073 = e074.add_paths()
import e075a_preserve_high_precision_recover_c2c5 as e075
import eval_os_official_protocol as P
import ablate_backbone_single as AB

M4_CKPT = P.M4_CKPT
TEST_ROOT = P.TEST_ROOT
TRAIN_VIEW = (ROOT + "/GDROS/experiments/sparsegeo/runs/"
              "E-075A-ZZVR-PRESERVE-HIGH-PRECISION-RECOVER-C2C5-AEPE/"
              "run-m4-regiongate120k-r2/data_view")

ZEROSHOT_REF = {
    "v1only": {"AEPE": 7.745715116406876, "CMR@5": 24.764150943396228},
    "v2only": {"AEPE": 41.3298363, "CMR@5": 0.24},
}
BOTH_REF = 1.473827834446231


def configure_trainable(model):
    """Freeze everything, then unfreeze only the parameters with gradients."""
    for p in model.parameters():
        p.requires_grad_(False)

    mods = [model.gate, model.b.gate]
    stu = model.c.student
    mods += [stu.context, stu.update, stu.fuse_proj]
    trainable, seen, groups = [], set(), []
    for tag, mod in zip(("L4_gate", "B_gate", "C_student_context",
                         "C_student_update", "C_student_fuse_proj"), mods):
        n_before = len(trainable)
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                p.requires_grad_(True)
                trainable.append(p)
        groups.append({"group": tag,
                       "n_params": sum(p.numel() for p in trainable[n_before:])})
    if hasattr(model.b, "scalar"):
        model.b.scalar.requires_grad_(True)
        trainable.append(model.b.scalar)
        groups.append({"group": "B_scalar", "n_params": model.b.scalar.numel()})

    for m in model.modules():
        if isinstance(m, e073.E073FusionMatcher):
            m.corr_gate_logit.requires_grad_(False)
            m.head_gate_logit.requires_grad_(False)
    return trainable, groups


def gate_target(pb, pc, gt, temp=0.75):
    g = gt.to(pb.device)
    if g.dim() == pb.dim() - 1:
        g = g[None]
    eb = torch.linalg.vector_norm(
        (e074.lsr(pb).to(pb.device) - g).float(), dim=1)
    ec = torch.linalg.vector_norm(
        (e074.lsr(pc).to(pc.device) - g).float(), dim=1)
    return torch.sigmoid((eb - ec) / temp).detach()[:, None]


@torch.inference_mode()
def eval_official(model, device, max_pairs):
    model.eval()
    ds = core.datasets.opt_sar_test(split="testing", dstype="testing",
                                    root=TEST_ROOT)
    n = min(max_pairs, len(ds))
    col = P.Collector()
    t0 = time.time()
    for i in range(n):
        _, a_img, b_img, gt, _ = ds[i]
        a, b = a_img[None].to(device), b_img[None].to(device)
        preds = model(a, b, iters=P.ITERS, lsr_fn=e074.lsr)
        post = e074.lsr(preds[-1]).cpu()
        col.add_pair(post - gt[None])
        if i % 100 == 0 or i == n - 1:
            print("[eval] %d/%d epe=%.4f %.1fs"
                  % (i + 1, n, col.pairs[-1], time.time() - t0), flush=True)
    return {"pairs": n, "official": col.report(),
            "pair_EPE": [float(x) for x in col.pairs],
            "elapsed_sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["v1only", "v2only"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--train-iters", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--max-pairs", type=int, default=424)
    ap.add_argument("--no-final-eval", action="store_true",
                    help="speed probe only: train, skip the 424-pair eval")
    ap.add_argument("--save-delta", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda:%d" % args.gpu)

    t_load = time.time()
    model = e075.PairFusion(e074, "region", device,
                            c_skip_teachers=True).to(device)
    payload = torch.load(M4_CKPT, map_location=device)
    miss = model.load_state_dict(payload["model"], strict=False)
    print("[m4] missing=%s unexpected=%s c_skip_teachers=%s load=%.1fs"
          % (list(miss.missing_keys), list(miss.unexpected_keys),
             model.c_skip_teachers, time.time() - t_load), flush=True)
    if miss.missing_keys or miss.unexpected_keys:
        raise RuntimeError("checkpoint keys changed; aborting")
    del payload

    state = AB.make_state(model)
    gates = AB.set_config(state, args.config)
    trainable, groups = configure_trainable(model)
    n_params = sum(p.numel() for p in trainable)
    print("[train] config=%s trainable_params=%d groups=%s gates=%s"
          % (args.config, n_params, groups, gates), flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    c = e072.cfg()
    tr = core.datasets.opt_sar_train(split="training", dstype="training",
                                     root=TRAIN_VIEW)
    rng = np.random.default_rng(args.seed)
    print("[data] train_pairs=%d seed=%d train_iters=%d steps=%d lr=%g"
          % (len(tr), args.seed, args.train_iters, args.steps, args.lr),
          flush=True)

    def step_batch():
        idx = int(rng.integers(0, len(tr)))
        a, b, flow, valid = tr[idx]
        return (a[None].to(device), b[None].to(device),
                flow[None].to(device), valid[None].to(device))

    curve = []
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        a, b, flow, valid = step_batch()
        pb = model.b(a, b, iters=args.train_iters)
        pc = model.c(a, b, iters=args.train_iters,
                     need_teacher_avg=False)
        fused, gate = [], None
        for xb, xc in zip(pb, pc):
            feat = model.gate_features(xb, xc, e074.lsr)
            g = model.gate(feat)
            g = F.interpolate(g, size=xb.shape[-2:], mode="bilinear",
                              align_corners=False)
            fused.append((1 - g) * xb + g * xc)
            gate = g
        lseq, m1 = sequence_loss(fused, flow, valid, c)
        mats = matrix2flow(LSR_list(fused), flow)
        lgeo, m2 = sequence_loss(mats, flow, valid, c)
        target = gate_target(pb[-1], pc[-1], flow)
        gate_loss = F.binary_cross_entropy(
            gate.clamp(1e-4, 1 - 1e-4), target)
        epe = torch.linalg.vector_norm(
            (mats[-1] - flow).float(), dim=1)[valid.bool()]
        tail = e075.cvar_loss(epe, frac=0.10)
        bucket = (e075.bucket_weights(epe) * epe).mean()
        loss = lseq + lgeo + 0.35 * gate_loss + 0.20 * tail + 0.12 * bucket

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss.detach().cpu()),
                   "lseq_epe": m1["epe"], "flow5": float(m1["5px"] * 100),
                   "mat5": float(m2["5px"] * 100),
                   "elapsed_sec": round(time.time() - t0, 1)}
            curve.append(row)
            print("[train] step=%d/%d loss=%.4f flow5=%.2f mat5=%.2f %.1fs"
                  % (step, args.steps, row["loss"], row["flow5"],
                     row["mat5"], row["elapsed_sec"]), flush=True)
    train_sec = round(time.time() - t0, 1)

    delta = {}
    if args.save_delta:
        delta = {n: p.detach().cpu()
                 for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"config": args.config, "step": args.steps,
                    "seed": args.seed, "train_iters": args.train_iters,
                    "lr": args.lr, "base_checkpoint": M4_CKPT,
                    "trainable_delta": delta}, args.save_delta)
        print("[ckpt] %s (%d tensors)" % (args.save_delta, len(delta)),
              flush=True)
        del delta

    out = {
        "protocol": "GDROS official test.py, 424 pairs, iters=32",
        "experiment_class": "adaptation_short_finetune_scheme_A",
        "config": args.config,
        "config_definition": AB.CONFIG_DEF[args.config],
        "structural_caveat": AB.STRUCTURAL_CAVEAT,
        "checkpoint": M4_CKPT,
        "pruned_model": {"c_skip_teachers": bool(model.c_skip_teachers)},
        "train": {
            "data_view": TRAIN_VIEW, "seed": args.seed,
            "steps": args.steps, "train_iters": args.train_iters,
            "lr": args.lr, "optimizer": "AdamW(wd=1e-5, clip=1.0)",
            "loss": "lseq + lgeo + 0.35*gate_bce + 0.20*tail_cvar + 0.12*bucket",
            "trainable_params": n_params, "trainable_groups": groups,
            "frozen": "E072 b1/b2 frontends (incl. frozen DINO/E046 injection), "
                      "corr_gate_logit, head_gate_logit, const fuse_gate",
            "gradient_note": "E074Model(B).teacher_preds runs under no_grad, so "
                             "B's f2/f3 chains have no gradient; only C.student "
                             "is a trainable recurrent chain",
            "elapsed_sec": train_sec,
            "sec_per_step": round(train_sec / max(1, args.steps), 4),
            "curve": curve,
        },
        "gate_lock": gates,
        "device": str(device),
        "historical": {"both_AEPE": BOTH_REF,
                       "zeroshot": ZEROSHOT_REF[args.config]},
        "checkpoint_delta": args.save_delta or None,
    }

    if not args.no_final_eval:
        ev = eval_official(model, device, args.max_pairs)
        out["official_after"] = ev
        m = ev["official"]
        print("\n### %s after %d steps" % (args.config, args.steps), flush=True)
        print("AEPE=%.7f CMR@1/2/5=%.2f/%.2f/%.2f RMSE_var=%.4f eval=%.1fs"
              % (m["AEPE"], m["@1px"]["CMR"], m["@2px"]["CMR"],
                 m["@5px"]["CMR"], m["RMSE_official_var"],
                 ev["elapsed_sec"]), flush=True)
        print("delta vs zero-shot: AEPE %+.4f ; vs both %+.4f"
              % (m["AEPE"] - ZEROSHOT_REF[args.config]["AEPE"],
                 m["AEPE"] - BOTH_REF), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("[out] %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
