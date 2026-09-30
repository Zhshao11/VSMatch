import unittest

import torch

from experiments.sparsegeo.e003a.candidate_cache import build_training_labels
from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
from experiments.sparsegeo.e002b.candidates import extract_candidates
from experiments.sparsegeo.e003a.losses import selector_losses
from experiments.sparsegeo.e003a.selector import (
    LearnedGeoSelector,
    affine_flow_at_pixel_points,
    apply_affine_normalized,
    dense_affine_flow,
    weighted_affine_solve,
)


def synthetic_geometry(batch=1):
    generator = torch.Generator().manual_seed(7)
    source = torch.rand(batch, 128, 2, generator=generator) * 1.8 - 0.9
    affine = torch.tensor([[1.02, 0.03, 0.08], [-0.02, 0.97, -0.05]]).expand(batch, -1, -1)
    correct = apply_affine_normalized(affine, source)
    candidates = torch.rand(batch, 128, 8, 2, generator=generator) * 2.0 - 1.0
    candidates[:, :, 0] = correct
    weights = torch.zeros(batch, 128, 8)
    weights[:, :, 0] = 1.0
    return source, candidates, weights, affine


class SelectorCoreTests(unittest.TestCase):
    def test_weighted_affine_recovers_synthetic_transform(self):
        source, targets, weights, truth = synthetic_geometry()
        prediction, condition, valid = weighted_affine_solve(source, targets, weights)
        self.assertTrue(bool(valid.all()))
        self.assertLess(float((prediction - truth).abs().max()), 2e-4)
        self.assertLess(float(condition.max()), 1e6)

    def test_degenerate_design_abstains(self):
        source = torch.zeros(1, 128, 2)
        targets = torch.zeros(1, 128, 8, 2)
        weights = torch.full((1, 128, 8), 1.0 / 8.0)
        _, condition, valid = weighted_affine_solve(source, targets, weights)
        self.assertFalse(bool(valid.item()))
        self.assertGreater(float(condition.item()), 1e6)

    def test_nonfinite_geometry_structurally_abstains(self):
        source, targets, weights, _ = synthetic_geometry()
        source[:, 0, 0] = float("nan")
        affine, _, valid = weighted_affine_solve(source, targets, weights)
        self.assertFalse(bool(valid.item()))
        self.assertTrue(bool(torch.isfinite(affine).all()))
        model = LearnedGeoSelector()
        output = model(torch.zeros(1, 128, 8, 13), source, targets, use_feedback=True)
        self.assertFalse(bool(output.solver_valid.item()))
        self.assertTrue(bool(output.abstain.item()))
        self.assertTrue(bool(torch.isfinite(output.affine_normalized).all()))

    def test_dense_flow_direction_target_minus_source(self):
        affine = torch.tensor([[[1.0, 0.0, 20.0 / 255.5], [0.0, 1.0, -8.0 / 255.5]]])
        flow = dense_affine_flow(affine)
        self.assertEqual(tuple(flow.shape), (1, 512, 512, 2))
        self.assertLess(float((flow[..., 0] - 20.0).abs().max()), 1e-4)
        self.assertLess(float((flow[..., 1] + 8.0).abs().max()), 1e-4)

    def test_raw_truth_flow_label_sampling(self):
        source, targets, _, affine = synthetic_geometry()
        source_pixel = (source[0] + 1.0) * 511.0 / 2.0
        target_pixel = (targets[0] + 1.0) * 511.0 / 2.0
        table = {
            "source_xy_pixel": source_pixel,
            "target_xy_token": target_pixel / 8.0,
        }
        dense = dense_affine_flow(affine)[0]
        labels = build_training_labels(table, dense)
        self.assertTrue(labels["uses_raw_truth_flow"])
        self.assertFalse(labels["uses_datum"])
        self.assertTrue(bool(labels["recoverable"].all()))
        self.assertTrue(torch.equal(labels["candidate_target"].argmax(dim=-1), torch.zeros(128, dtype=torch.long)))

    def test_model_shape_parameter_budget_and_gradient(self):
        source, targets, _, _ = synthetic_geometry(batch=2)
        features = torch.randn(2, 128, 8, 13)
        features[:, :, 0, 0] += 2.0
        model = LearnedGeoSelector()
        self.assertLess(model.trainable_parameter_count, 100_000)
        output = model(features, source, targets, use_feedback=True)
        self.assertEqual(output.solver_rounds, 2)
        self.assertEqual(tuple(output.candidate_weights.shape), (2, 128, 8))
        target_distribution = torch.zeros(2, 128, 9)
        target_distribution[:, :, 0] = 1.0
        axis = torch.linspace(0.0, 511.0, 16)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).expand(2, -1, -1)
        truth = affine_flow_at_pixel_points(
            torch.tensor([[1.02, 0.03, 0.08], [-0.02, 0.97, -0.05]]).expand(2, -1, -1),
            grid,
        )
        loss = selector_losses(output, target_distribution, grid, truth).total
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertTrue(any(bool((gradient != 0).any()) for gradient in gradients))

    def test_candidate_matching_never_materializes_n_by_n(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source = torch.randn(1, 256, 64, 64, device=device)
        target = torch.randn(1, 256, 64, 64, device=device)
        anchors = public_stratified_anchor_set(128, 512, 512, 8, (0, 0))
        live_shapes = []
        candidates, _ = extract_candidates(
            source,
            target,
            anchors.lattice_indices,
            score_type="cosine",
            candidate_count=8,
            target_chunk=1024,
            live_shapes=live_shapes,
        )
        self.assertEqual(tuple(candidates.scores.shape), (1, 128, 8))
        self.assertTrue(live_shapes)
        self.assertTrue(all(record["shape"][1] == 128 for record in live_shapes))
        self.assertTrue(all(record["shape"][2] <= 1024 for record in live_shapes))

    def test_toy_overfit(self):
        torch.manual_seed(9)
        source, targets, _, affine = synthetic_geometry(batch=2)
        features = torch.randn(2, 128, 8, 13) * 0.1
        features[:, :, 0, 0] = 1.0
        target_distribution = torch.zeros(2, 128, 9)
        target_distribution[:, :, 0] = 1.0
        axis = torch.linspace(0.0, 511.0, 16)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).expand(2, -1, -1)
        truth = affine_flow_at_pixel_points(affine, grid)
        model = LearnedGeoSelector()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        values = []
        for _ in range(60):
            optimizer.zero_grad()
            output = model(features, source, targets, use_feedback=True)
            loss = selector_losses(output, target_distribution, grid, truth).total
            values.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        self.assertLess(values[-1], 0.35 * values[0])


if __name__ == "__main__":
    unittest.main()
