import unittest
import sys
import json
from pathlib import Path

from experiments.sparsegeo.e004a.proposals import uniform_region_k128
from experiments.sparsegeo.e004a.spec_binding import BINDING_ID, UNIFORM_REGION_OFFSETS


class P0IsolationTests(unittest.TestCase):
    def test_binding_is_frozen(self):
        self.assertEqual(BINDING_ID, "BIND-001")
        self.assertEqual(UNIFORM_REGION_OFFSETS, ((2, 2), (6, 6)))
        self.assertEqual(uniform_region_k128().numel(), 128)

    def test_p0_source_has_no_forbidden_method_modules(self):
        root = Path(__file__).resolve().parent
        names = {path.name for path in root.glob("*.py")}
        staged = names & {"anchor_heads.py", "replacement.py", "train_anchor_heads.py"}
        if staged:
            marker = json.loads((root / "P0_PASS.json").read_text())
            self.assertEqual(marker["status"], "PASS")
            self.assertEqual(marker["pair_count"], 404)
            self.assertEqual(marker["p0_source_manifest_sha256"], "2e1b64ad31c41a455bebd68b92948c9c97e17a6c02cadab1eedbec16c3bbe801")
        source = "\n".join(path.read_text() for path in root.glob("*.py") if not path.name.startswith("test_"))
        forbidden = ("official_e000_archive", "streaming_consensus", "triple_bank", "hypothesis")
        for token in forbidden:
            self.assertNotIn(token, source.lower())

    def test_candidate_cache_schema_excludes_gt_and_datum_payloads(self):
        source = (Path(__file__).resolve().parent / "proposal_cache.py").read_text()
        self.assertIn('"uses_ground_truth": False', source)
        self.assertIn('"uses_datum": False', source)
        self.assertNotIn('candidate_record["truth_flow"', source)
        self.assertNotIn('candidate_record["datum"', source)

    def test_oracle_import_has_no_forbidden_transitive_modules(self):
        import experiments.sparsegeo.e004a.oracle_gate  # noqa: F401

        loaded = set(sys.modules)
        self.assertFalse(any(name.endswith(".consensus") for name in loaded))
        self.assertFalse(any(name.endswith(".triple_bank") for name in loaded))
        self.assertFalse(any(name.endswith(".reranker") for name in loaded))


if __name__ == "__main__":
    unittest.main()
