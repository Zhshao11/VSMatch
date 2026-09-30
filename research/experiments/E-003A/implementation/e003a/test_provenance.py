import inspect
import unittest
from pathlib import Path

from experiments.sparsegeo.e003a import candidate_cache, losses, selector, train_selector
from experiments.sparsegeo.e003a.spec_binding import EXPECTED, validate_frozen_specs


class ProvenanceTests(unittest.TestCase):
    def test_frozen_spec_hashes(self):
        self.assertEqual(validate_frozen_specs(), EXPECTED)

    def test_method_path_has_no_hypothesis_or_b2_selector(self):
        for module in (candidate_cache, selector, losses, train_selector):
            source = inspect.getsource(module).lower()
            self.assertNotIn("streaming_consensus_search", source)
            self.assertNotIn("reassign_and_refit", source)
            self.assertNotIn("131072", source)
            self.assertNotIn("4096, 4096", source)

    def test_selector_has_no_baseline_fallback(self):
        source = inspect.getsource(selector).lower()
        self.assertNotIn("baseline", source)
        self.assertNotIn("e002b", source)
        self.assertNotIn("local_refine", source)

    def test_training_has_no_testing_root(self):
        source = inspect.getsource(train_selector).lower()
        self.assertNotIn("gdros_official_e000_archive", source)
        self.assertNotIn("testing", source)


if __name__ == "__main__":
    unittest.main()
