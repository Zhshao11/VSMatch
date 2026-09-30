import unittest
import tempfile,json
from pathlib import Path

import torch

from experiments.sparsegeo.e004a.anchor_heads import PostSearchVerifier, PreSearchRecoverabilityHead
from experiments.sparsegeo.e004a.coarse_probe import compute_coarse_probe_features
from experiments.sparsegeo.e004a.proposals import allocate_coverage_constrained_k128, region_counts
from experiments.sparsegeo.e004a.replacement import build_post_features, replace_same_region_once


class HeadTests(unittest.TestCase):
    def test_invalid_safe_bootstrap_and_fresh_process_guard(self):
        from experiments.sparsegeo.e004a.evaluate_anchor_budget import _group_bootstrap
        from experiments.sparsegeo.e004a.aggregate_resources import run
        groups={k:"x" for k in ("displacement","rotation","scale","black_border","texture")}
        rows=[{"groups":groups,"main":{"aepe":None,"recoverable_fraction":.5}},{"groups":groups,"main":{"aepe":2.0,"recoverable_fraction":.6}}]
        result=_group_bootstrap(rows,1); self.assertEqual(result["scale"]["x"]["valid_aepe_count"],1)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"one.json"; path.write_text(json.dumps({"process":{"pid":1,"run_uuid":"x"}})); args=type("A",(),{"inputs":[path]*5,"output":Path(root)/"out.json"})()
            with self.assertRaisesRegex(RuntimeError,"distinct fresh"):
                run(args)

    def test_coarse_probe_shape_and_budget(self):
        torch.manual_seed(1); source=torch.randn(1,256,64,64); target=torch.randn(1,256,64,64)
        features,xy,scores=compute_coarse_probe_features(source,target)
        self.assertEqual(features.shape,(256,11)); self.assertEqual(xy.shape,(256,2)); self.assertEqual(scores.shape,(256,256))
        self.assertTrue(torch.isfinite(features).all())

    def test_allocator_and_replacement_invariants(self):
        scores=torch.linspace(0,1,256); primary,reserves=allocate_coverage_constrained_k128(scores)
        counts=region_counts(primary); self.assertEqual(primary.numel(),128); self.assertGreaterEqual(int(counts.min()),1); self.assertLessEqual(int(counts.max()),3)
        post=torch.zeros(256); pre=torch.linspace(0,1,256); post[reserves]=1
        final,trace=replace_same_region_once(primary,reserves,pre,post)
        self.assertEqual(final.numel(),128); self.assertEqual(torch.unique(final).numel(),128); self.assertLessEqual(len(trace),64)
        for row in trace: self.assertEqual(row["region"],int(row["primary"]//4)); self.assertEqual(row["region"],int(row["reserve"]//4))

    def test_post_features_and_gradients(self):
        scores=torch.sort(torch.randn(256,8),dim=-1,descending=True).values; xy=torch.randint(0,64,(256,8,2)).float(); coarse=torch.rand(256,2)*63
        features=build_post_features(torch.randn(256),scores,xy,coarse); self.assertEqual(features.shape,(256,15))
        for model,x in ((PreSearchRecoverabilityHead(),torch.randn(256,11)),(PostSearchVerifier(),features)):
            loss=model(x).square().mean(); loss.backward(); self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_toy_overfit(self):
        torch.manual_seed(3); x=torch.randn(128,11); y=(x[:,0]>0).float(); model=PreSearchRecoverabilityHead(); opt=torch.optim.AdamW(model.parameters(),lr=1e-2)
        initial=float(torch.nn.functional.binary_cross_entropy_with_logits(model(x),y))
        for _ in range(100):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y); opt.zero_grad(); loss.backward(); opt.step()
        final=float(torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)); self.assertLess(final,initial*.2)


if __name__=="__main__": unittest.main()
