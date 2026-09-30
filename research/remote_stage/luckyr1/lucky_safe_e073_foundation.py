#!/usr/bin/env python3
"""Split-safe V1xV1 F2/F3 foundation for the LuckyR1 lineage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/mnt/sda/zhushihao/VS_Match")
TMP = ROOT / "image_matching_part2_migrated/.codex_tmp"
PARENT = ROOT / "GDROS/experiments/sparsegeo/runs/LuckyR1-FULL-LINEAGE-RETRAIN"
SEED1 = PARENT / "foundation/F2/checkpoints/F2_step120000.pt"
SEED2 = PARENT / "foundation/F3/checkpoints/F3_step120000.pt"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "pairs": int(array.size),
        "aepe": float(array.mean()),
        **{f"cmr{k}": float((array < k).mean() * 100.0) for k in (1, 2, 5, 10, 16)},
        "tail_ge5": int((array >= 5).sum()),
    }


class SafeE073(nn.Module):
    """E073-style fusion using only two split-safe E072 V1 branches."""

    def __init__(self, e072, cfg, variant: str, device):
        super().__init__()
        from core.GDROSnet.FeatureFormer.cnn import BasicEncoder
        from core.update import BasicUpdateBlock

        self.variant = variant
        self.b1 = self._load_v1(e072, cfg, SEED1, device)
        self.b2 = self._load_v1(e072, cfg, SEED2, device)
        self.context = BasicEncoder(output_dim=256, norm_fn="instance")
        self.update = BasicUpdateBlock(cfg, hidden_dim=128)
        self.context.load_state_dict(self.b1.context.state_dict(), strict=False)
        self.update.load_state_dict(self.b1.update.state_dict(), strict=False)
        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_radius = 4
        self.fuse_proj = nn.Sequential(nn.Conv2d(512, 256, 1), nn.GELU(), nn.Conv2d(256, 256, 1))
        self.fuse_gate = nn.Sequential(nn.Conv2d(512, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
        self.corr_gate_logit = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _load_v1(e072, cfg, path: Path, device):
        model = e072.E072Matcher(cfg, "V1", device).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["model"], strict=True)
        return model

    @staticmethod
    def initialize_flow(image):
        from core.utils.utils import coords_grid

        n, _, h, w = image.shape
        return coords_grid(n, h // 8, w // 8, device=image.device), coords_grid(n, h // 8, w // 8, device=image.device)

    @staticmethod
    def upsample_flow(flow, mask):
        n, _, h, w = flow.shape
        mask = torch.softmax(mask.view(n, 1, 9, 8, 8, h, w), dim=2)
        up = F.unfold(8 * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
        return torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3).reshape(n, 2, 8 * h, 8 * w)

    def features(self, raw1, raw2, image1, image2):
        context1 = self.b1.context(image1)
        f1s, f1t = self.b1._features(raw1, raw2, image1, image2, context1)
        context2 = self.b2.context(image1)
        f2s, f2t = self.b2._features(raw1, raw2, image1, image2, context2)
        return f1s, f1t, f2s, f2t

    def recurrent(self, image, corr_fn, iters: int):
        context = self.context(image)
        net, inp = torch.split(context, [128, 128], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)
        coords0, coords1 = self.initialize_flow(image)
        outputs = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            net, mask, delta = self.update(net, inp, corr, coords1 - coords0)
            coords1 = coords1 + delta
            outputs.append(self.upsample_flow(coords1 - coords0, mask))
        return outputs

    def forward(self, source, target, iters=12):
        from core.corr import CorrBlock

        image1 = 2 * (source / 255.0) - 1.0
        image2 = 2 * (target / 255.0) - 1.0
        f1s, f1t, f2s, f2t = self.features(source, target, image1, image2)
        if self.variant == "F2":
            gate_s = torch.sigmoid(self.fuse_gate(torch.cat([f1s, f2s], dim=1)))
            gate_t = torch.sigmoid(self.fuse_gate(torch.cat([f1t, f2t], dim=1)))
            fused_s = (1 - gate_s) * f1s + gate_s * self.fuse_proj(torch.cat([f1s, f2s], dim=1))
            fused_t = (1 - gate_t) * f1t + gate_t * self.fuse_proj(torch.cat([f1t, f2t], dim=1))
            return self.recurrent(image1, CorrBlock(fused_s.float(), fused_t.float(), radius=4), iters)
        corr1 = CorrBlock(f1s.float(), f1t.float(), radius=4)
        corr2 = CorrBlock(f2s.float(), f2t.float(), radius=4)
        alpha = torch.sigmoid(self.corr_gate_logit)
        return self.recurrent(image1, lambda coords: (1 - alpha) * corr1(coords) + alpha * corr2(coords), iters)


def write_card(run_dir: Path, card: dict) -> None:
    card = dict(card)
    card["updated_at_unix"] = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(card, indent=2, sort_keys=True) + "\n"
    (run_dir / "RESULT_CARD.json").write_text(payload)
    (run_dir / "RESULT_CARD.md").write_text(f"# LuckyR1 split-safe foundation {card.get('variant', 'unknown')}\n\nStatus: `{card.get('status')}`\n\n```json\n{payload}```\n")
    attempt = os.getenv("WATCHDOG_JOB_ID", f"manual-{os.getpid()}")
    attempt_dir = run_dir / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "RESULT_CARD.json").write_text(payload)
    (attempt_dir / "RESULT_CARD.md").write_text(f"# Attempt {attempt}\n\nStatus: `{card.get('status')}`\n\n```json\n{payload}```\n")


def evaluate(model, dataset, device, run_dir: Path, step: int) -> dict:
    from core.LSRnet.LSmodel import LSR

    prediction_dir = run_dir / "predictions" / f"step{step}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prediction_dir / "PREDICTIONS.jsonl"
    manifest_path.unlink(missing_ok=True)
    model.eval()
    with torch.inference_mode(), manifest_path.open("a") as manifest:
        for index in range(len(dataset)):
            pair_id, source, target, _hidden_gt, _valid = dataset[index]
            source, target = source[None].to(device), target[None].to(device)
            flow = LSR(1, 512, 512, model(source, target, 24)[-1], 200000).cpu()
            prediction_path = prediction_dir / f"{index:04d}.pt"
            torch.save({"pair_index": index, "pair_id": Path(pair_id).stem, "flow": flow}, prediction_path)
            manifest.write(json.dumps({"pair_index": index, "pair_id": Path(pair_id).stem, "prediction": str(prediction_path), "uses_truth_flow": False}, sort_keys=True) + "\n")
            manifest.flush()
    materialized_sha = sha256(manifest_path)
    values, rows = [], []
    for index in range(len(dataset)):
        pair_id, _source, _target, gt, _valid = dataset[index]
        prediction_path = prediction_dir / f"{index:04d}.pt"
        flow = torch.load(prediction_path, map_location="cpu")["flow"]
        value = float(torch.linalg.vector_norm((flow - gt[None]).float(), dim=1).mean())
        values.append(value)
        rows.append({"pair_index": index, "pair_id": Path(pair_id).stem, "pair_aepe": value})
    joined_path = run_dir / f"per_pair_dev_step{step}.csv"
    with joined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {**metrics(values), "prediction_manifest": str(manifest_path), "prediction_manifest_sha256": materialized_sha, "metric_join": str(joined_path), "prediction_materialized_before_gt_join": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["F2", "F3"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=120000)
    args = parser.parse_args()
    run_dir = PARENT / "foundation_safe_v1v1" / args.variant
    started = time.time()
    state = {"variant": args.variant, "status": "RUNNING", "last_completed_stage": "START", "step": 0}

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_card(run_dir, state)
    try:
        sys.path.insert(0, str(TMP))
        import e072a_layerwise_frontend_alignment as e072

        e072.add_paths()
        import core.datasets
        from core.LSRnet.LSmodel import LSR_list
        from core.loss import matrix2flow, sequence_loss

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        cfg = e072.cfg()
        model = SafeE073(e072, cfg, args.variant, device).to(device)
        train_set = core.datasets.opt_sar_train(split="training", dstype="training", root=str(PARENT / "data_view"))
        dev_set = core.datasets.opt_sar_test(split="testing", dstype="testing", root=str(PARENT / "data_view"))
        assert len(train_set) == 1810 and len(dev_set) == 201
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
        rng = np.random.default_rng(args.seed)
        progress_path = run_dir / "checkpoint_curve.json"
        resume_pointer = run_dir / "RESUME_LATEST.json"
        curve = json.loads(progress_path.read_text()) if progress_path.exists() else []
        start_step = 0
        latest = None
        if resume_pointer.exists():
            pointed = Path(json.loads(resume_pointer.read_text())["checkpoint"])
            latest = pointed if pointed.is_file() else None
        if latest is None:
            checkpoints = list((run_dir / "checkpoints").glob(f"{args.variant}_step*.pt"))
            latest = max(checkpoints, key=lambda path: int(path.stem.rsplit("step", 1)[1])) if checkpoints else None
        if latest is not None:
            resume = torch.load(latest, map_location=device)
            start_step = int(resume["step"])
            model.load_state_dict(resume["model"], strict=True)
            if "optimizer" in resume:
                optimizer.load_state_dict(resume["optimizer"])
            if "rng_state" in resume:
                rng.bit_generator.state = resume["rng_state"]
        milestones = {s for s in (20000, 40000, 80000, 120000) if s <= args.steps} | {args.steps}

        def checkpoint_payload(step: int) -> dict:
            return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_state": rng.bit_generator.state, "variant": args.variant, "step": step, "seed": args.seed, "sources": state["sources"], "historical_e046_used": False, "architecture": "SafeE073 V1xV1"}

        def update_resume_pointer(checkpoint: Path, step: int) -> None:
            temporary = resume_pointer.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"checkpoint": str(checkpoint), "step": step}, indent=2) + "\n")
            os.replace(temporary, resume_pointer)
        state.update({"last_completed_stage": "SOURCE_AND_SPLIT_AUDIT", "lineage": "two 1810-trained E072 V1 seeds; no E072 V2/E046/historical official checkpoint", "resume_step": start_step, "sources": {"seed1": str(SEED1), "seed1_sha256": sha256(SEED1), "seed2": str(SEED2), "seed2_sha256": sha256(SEED2)}, "scheduler": {"watchdog_job_id": os.getenv("WATCHDOG_JOB_ID"), "assigned_physical_gpus": os.getenv("ASSIGNED_GPUS"), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "logical_device": "cuda:0", "candidate_gpus": None}})
        write_card(run_dir, state)
        if start_step in milestones and not any(row.get("step") == start_step and "dev" in row for row in curve):
            dev = evaluate(model, dev_set, device, run_dir, start_step)
            curve.append({"step": start_step, "dev": dev, "checkpoint": str(latest), "checkpoint_sha256": sha256(latest), "recovered_evaluation": True})
            progress_path.write_text(json.dumps(curve, indent=2) + "\n")
            state.update({"last_completed_stage": f"RECOVERED_CHECKPOINT_{start_step}_EVALUATED", "curve": curve})
            write_card(run_dir, state)
        model.train()
        for step in range(start_step + 1, args.steps + 1):
            index = int(rng.integers(0, len(train_set)))
            source, target, gt, valid = train_set[index]
            source, target = source[None].to(device), target[None].to(device)
            gt, valid = gt[None].to(device), valid[None].to(device)
            predictions = model(source, target, 12)
            loss1, info1 = sequence_loss(predictions, gt, valid, cfg)
            matrices = matrix2flow(LSR_list(predictions), gt)
            loss2, _info2 = sequence_loss(matrices, gt, valid, cfg)
            loss = 0.5 * loss1 + 0.5 * loss2
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            state["step"] = step
            if step == 1 or step % 6000 == 0:
                curve.append({"step": step, "loss": float(loss.detach()), "grad_norm": grad_norm, "epe": info1["epe"]})
                state.update({"last_completed_stage": f"TRAIN_STEP_{step}", "step": step, "curve": curve})
                write_card(run_dir, state)
                if step % 6000 == 0 and step not in milestones:
                    checkpoint_dir = run_dir / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    rolling = checkpoint_dir / f"{args.variant}_resume_latest.pt"
                    temporary = rolling.with_suffix(".pt.tmp")
                    torch.save(checkpoint_payload(step), temporary)
                    os.replace(temporary, rolling)
                    update_resume_pointer(rolling, step)
                    progress_path.write_text(json.dumps(curve, indent=2) + "\n")
                    state.update({"last_completed_stage": f"ROLLING_CHECKPOINT_{step}", "rolling_checkpoint": str(rolling)})
                    write_card(run_dir, state)
            if step in milestones:
                checkpoint = run_dir / "checkpoints" / f"{args.variant}_step{step}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload(step), checkpoint)
                update_resume_pointer(checkpoint, step)
                dev = evaluate(model, dev_set, device, run_dir, step)
                curve.append({"step": step, "dev": dev, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)})
                progress_path.write_text(json.dumps(curve, indent=2) + "\n")
                state.update({"last_completed_stage": f"CHECKPOINT_{step}_EVALUATED", "curve": curve})
                write_card(run_dir, state)
                model.train()
        state.update({"status": "COMPLETE", "last_completed_stage": "SAFE_V1V1_FOUNDATION_COMPLETE", "curve": curve, "elapsed_seconds": time.time() - started})
    except BaseException as error:
        state.update({"status": "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "IMPLEMENTATION_FAILED", "cause": repr(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - started})
    finally:
        write_card(run_dir, state)
    if state["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
