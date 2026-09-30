import json
import os
import unittest
from pathlib import Path

import torch

from experiments.sparsegeo.e002a.anchors import public_stratified_anchor_set
from experiments.sparsegeo.e002a.feature import load_slim_feature
from experiments.sparsegeo.e003a.candidate_cache import _load_image, build_frozen_candidate_table


class FrozenFrontendIntegrationTests(unittest.TestCase):
    def test_real_gdros_frontend_is_frozen_and_no_grad(self):
        checkpoint = Path(os.environ["E003A_GDROS_CHECKPOINT"])
        split_manifest = Path(os.environ["E003A_SPLIT_MANIFEST"])
        record = json.loads(split_manifest.read_text().splitlines()[0])
        device = torch.device("cuda")
        model, _, _ = load_slim_feature(checkpoint, device)
        model.requires_grad_(False).eval()
        table = build_frozen_candidate_table(
            model,
            _load_image(Path(record["files"]["opt"]["path"])).to(device),
            _load_image(Path(record["files"]["sar_warped"]["path"])).to(device),
            public_stratified_anchor_set(128, 512, 512, 8, (0, 0)),
            to_cpu=False,
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        tensors = [value for value in table.values() if torch.is_tensor(value)]
        self.assertTrue(tensors)
        self.assertTrue(all(not tensor.requires_grad for tensor in tensors))
        self.assertTrue(all(tensor.grad_fn is None for tensor in tensors))


if __name__ == "__main__":
    unittest.main()
