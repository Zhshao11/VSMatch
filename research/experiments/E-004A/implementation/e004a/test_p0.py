import os
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.sparsegeo.e003a.candidate_cache import _load_image
from experiments.sparsegeo.e004a.evaluator import dense_affine_flow, pair_metrics
from experiments.sparsegeo.e004a.oracle_gate import fit_selected_oracle
from experiments.sparsegeo.e004a.proposal_cache import (
    build_pair_candidates,
    build_table_labels,
    exact_cosine_top8,
    validate_cached_pair,
)
from experiments.sparsegeo.e004a.proposals import (
    REGION_OFFSETS,
    allocate_oracle_k128,
    build_region_proposals,
    region_counts,
    uniform_region_k128,
)


class ProposalTests(unittest.TestCase):
    def test_exact_order_offsets_and_uniqueness(self):
        proposals = build_region_proposals()
        self.assertEqual(proposals.proposal_id.tolist(), list(range(256)))
        self.assertEqual(torch.unique(proposals.lattice_indices).numel(), 256)
        for proposal_id in range(256):
            region = proposal_id // 4
            offset = REGION_OFFSETS[proposal_id % 4]
            self.assertEqual(int(proposals.region_id[proposal_id]), region)
            self.assertEqual(tuple(proposals.offset_yx[proposal_id].tolist()), offset)
            ry, rx = divmod(region, 8)
            dy, dx = offset
            self.assertEqual(tuple(proposals.lattice_xy[proposal_id].tolist()), (8 * rx + dx, 8 * ry + dy))

    def test_uniform_bind_001(self):
        proposals = build_region_proposals()
        selected = uniform_region_k128(proposals)
        self.assertEqual(selected.numel(), 128)
        self.assertTrue(torch.equal(region_counts(selected, proposals), torch.full((64,), 2)))
        self.assertEqual(
            {tuple(row) for row in proposals.offset_yx[selected].tolist()}, {(2, 2), (6, 6)}
        )

    def test_allocator_budget_bounds_and_ties(self):
        proposals = build_region_proposals()
        recoverable = torch.zeros(256, dtype=torch.bool)
        error = torch.ones(256)
        selected_a = allocate_oracle_k128(recoverable, error, proposals)
        selected_b = allocate_oracle_k128(recoverable, error, proposals)
        self.assertTrue(torch.equal(selected_a, selected_b))
        self.assertEqual(selected_a.numel(), 128)
        counts = region_counts(selected_a, proposals)
        self.assertGreaterEqual(int(counts.min()), 1)
        self.assertLessEqual(int(counts.max()), 3)
        self.assertEqual(selected_a[:64].tolist(), list(range(0, 256, 4)))

    def test_allocator_prioritizes_recoverable_then_error(self):
        recoverable = torch.zeros(256, dtype=torch.bool)
        recoverable[3] = True
        error = torch.arange(256, dtype=torch.float32)
        selected = allocate_oracle_k128(recoverable, error)
        self.assertEqual(int(selected[0]), 3)


class LabelAndMatcherTests(unittest.TestCase):
    def test_cache_identity_rejects_mismatch(self):
        proposals = build_region_proposals()
        candidate = {
            "pair_id": "p", "split": "dev", "source_manifest_sha256": "source",
            "checkpoint_sha256": "checkpoint", "optical_sha256": "opt", "sar_sha256": "sar",
            "uses_ground_truth": False, "uses_datum": False,
            "proposal_m256": {
                "proposal_id": proposals.proposal_id,
                "source_lattice_indices": proposals.lattice_indices,
                "source_xy_pixel": proposals.pixel_xy,
                "candidate_scores": torch.zeros(256, 8),
                "candidate_indices": torch.zeros(256, 8, dtype=torch.long),
            },
            "fixed_public_k128": {
                "candidate_scores": torch.zeros(128, 8),
                "candidate_indices": torch.zeros(128, 8, dtype=torch.long),
            },
        }
        labels = {
            "pair_id": "p", "split": "dev", "source_manifest_sha256": "source",
            "truth_flow_sha256": "flow", "uses_raw_truth_flow": True, "uses_datum": False,
            "proposal_m256": {"recoverable": torch.zeros(256, dtype=torch.bool)},
            "fixed_public_k128": {"recoverable": torch.zeros(128, dtype=torch.bool)},
            "grid_xy_pixel": torch.zeros(256, 2), "grid_truth_flow": torch.zeros(256, 2),
        }
        record = {"pair_id": "p", "files": {
            "opt": {"sha256": "opt"}, "sar_warped": {"sha256": "sar"},
            "truth_flow": {"sha256": "flow"},
        }}
        self.assertTrue(validate_cached_pair(candidate, labels, record, "dev", "source", "checkpoint"))
        labels["truth_flow_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "truth-flow mismatch"):
            validate_cached_pair(candidate, labels, record, "dev", "source", "checkpoint")

    def test_target_minus_source_flow_and_radius(self):
        table = {
            "source_xy_pixel": torch.tensor([[80, 160], [504, 504]]),
            "target_xy_token": torch.tensor([
                [[12.0, 20.0]] + [[0.0, 0.0]] * 7,
                [[63.0, 63.0]] + [[0.0, 0.0]] * 7,
            ]),
        }
        flow = torch.zeros(512, 512, 2)
        flow[..., 0] = 16.0
        labels = build_table_labels(table, flow)
        self.assertTrue(bool(labels["recoverable"][0]))
        self.assertTrue(torch.allclose(labels["target_pixel_gt"][0], torch.tensor([96.0, 160.0])))
        self.assertFalse(bool(labels["in_bounds"][1]))
        self.assertFalse(bool(labels["recoverable"][1]))

    def test_m256_chunk_shapes_and_no_n2(self):
        generator = torch.Generator().manual_seed(7)
        source = torch.randn(1, 256, 64, 64, generator=generator)
        target = torch.randn(1, 256, 64, 64, generator=generator)
        trace = []
        scores, indices, _ = exact_cosine_top8(
            source, target, build_region_proposals().lattice_indices, live_shapes=trace
        )
        self.assertEqual(scores.shape, (256, 8))
        self.assertEqual(indices.shape, (256, 8))
        self.assertEqual([row["shape"] for row in trace], [[1, 256, 1024]] * 4)
        self.assertFalse(any(row["shape"] == [1, 4096, 4096] for row in trace))

    def test_k128_wrapper_matches_frozen_candidate_path(self):
        from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
        from experiments.sparsegeo.e002b.candidates import extract_candidates

        generator = torch.Generator().manual_seed(11)
        source = torch.randn(1, 256, 64, 64, generator=generator)
        target = torch.randn(1, 256, 64, 64, generator=generator)
        anchors = public_stratified_anchor_set(128)
        scores, indices, xy = exact_cosine_top8(source, target, anchors.lattice_indices)
        frozen, _ = extract_candidates(
            source, target, anchors.lattice_indices, "cosine", 8, target_chunk=1024
        )
        self.assertTrue(torch.equal(indices, frozen.indices[0]))
        self.assertTrue(torch.equal(scores, frozen.scores[0]))
        self.assertTrue(torch.equal(xy, frozen.xy[0]))


class OracleTests(unittest.TestCase):
    @staticmethod
    def _identity_fixture():
        proposals = build_region_proposals()
        selected = uniform_region_k128(proposals)
        source = proposals.pixel_xy.to(torch.float32)
        target = proposals.lattice_xy.to(torch.float32)[:, None, :].expand(-1, 8, -1).clone()
        candidate = {"source_xy_pixel": source, "target_xy_token": target}
        labels = {
            "recoverable": torch.ones(256, dtype=torch.bool),
            "nearest_candidate_slot": torch.zeros(256, dtype=torch.long),
        }
        return candidate, labels, selected

    def test_fp32_ordinary_affine_recovery(self):
        candidate, labels, selected = self._identity_fixture()
        candidate["target_xy_token"][:, :, 0] += 1.0
        fit = fit_selected_oracle(candidate, labels, selected)
        self.assertTrue(fit["valid"])
        expected = torch.tensor([[[1.0, 0.0, 16.0 / 511.0], [0.0, 1.0, 0.0]]])
        self.assertTrue(torch.allclose(fit["affine"], expected, atol=2e-5, rtol=0.0))

    def test_invalid_fewer_than_12(self):
        candidate, labels, selected = self._identity_fixture()
        labels["recoverable"].zero_()
        labels["recoverable"][selected[:11]] = True
        fit = fit_selected_oracle(candidate, labels, selected)
        self.assertEqual(fit["invalid_reason"], "fewer_than_12_correspondences")

    def test_invalid_fewer_than_3_quadrants(self):
        candidate, labels, selected = self._identity_fixture()
        candidate["source_xy_pixel"][selected] = torch.tensor([32.0, 32.0])
        fit = fit_selected_oracle(candidate, labels, selected)
        self.assertEqual(fit["invalid_reason"], "fewer_than_3_quadrants")

    def test_invalid_condition_and_nonfinite(self):
        candidate, labels, selected = self._identity_fixture()
        t = torch.linspace(-1.0, 1.0, 128)
        source = torch.stack((255.5 + 200.0 * t, 255.5 + 200.0 * t), dim=-1)
        source[64, 1] = 255.49
        candidate["source_xy_pixel"][selected] = source
        fit = fit_selected_oracle(candidate, labels, selected)
        self.assertEqual(fit["invalid_reason"], "condition_exceeds_20")
        candidate, labels, selected = self._identity_fixture()
        candidate["target_xy_token"][selected[0], 0, 0] = float("nan")
        fit = fit_selected_oracle(candidate, labels, selected)
        self.assertEqual(fit["invalid_reason"], "nonfinite_correspondence")

    def test_full_image_no_mask_metric_parity(self):
        affine = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        flow = torch.zeros(512, 512, 2)
        axis = torch.linspace(0.0, 511.0, 16)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
        aepe, geometry = pair_metrics(affine, flow, grid, torch.zeros(256, 2))
        self.assertLess(aepe, 1e-5)
        self.assertTrue(geometry["pair_local_ready"])
        self.assertTrue(torch.allclose(
            dense_affine_flow(affine), torch.zeros(1, 512, 512, 2), atol=1e-5, rtol=0.0
        ))


@unittest.skipUnless(os.environ.get("E004A_RUN_REAL") == "1", "set E004A_RUN_REAL=1")
class RealFrontendTests(unittest.TestCase):
    def test_checkpoint_no_grad_and_live_shapes(self):
        from experiments.sparsegeo.e002a.feature import load_slim_feature
        from experiments.sparsegeo.e003a.data_protocol import load_split_manifest

        checkpoint = Path(os.environ["E004A_CHECKPOINT"])
        records = load_split_manifest(Path(os.environ["E004A_SPLIT_MANIFEST"]), allowed_split="dev")
        device = torch.device(os.environ.get("E004A_DEVICE", "cuda:0"))
        model, _, _ = load_slim_feature(checkpoint, device)
        model.requires_grad_(False).eval()
        optical = _load_image(Path(records[0]["files"]["opt"]["path"])).to(device)
        sar = _load_image(Path(records[0]["files"]["sar_warped"]["path"])).to(device)
        trace = []
        result = build_pair_candidates(model, optical, sar, live_shapes=trace)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(result["proposal_m256"]["candidate_scores"].shape, (256, 8))
        self.assertEqual([row["shape"] for row in trace], [[1, 256, 1024]] * 4)


if __name__ == "__main__":
    unittest.main()
