import tempfile
import unittest
from pathlib import Path

from experiments.sparsegeo.e003a.data_protocol import enforce_split_access


class DataProtocolTests(unittest.TestCase):
    def test_split_access_accepts_bound_training_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            record = {
                "pair_id": "000000",
                "split": "train",
                "files": {"opt": {"path": str(root / "opt/000000_opt.png")}},
            }
            self.assertTrue(enforce_split_access(record, root, "train"))

    def test_split_access_rejects_wrong_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            record = {
                "pair_id": "000000",
                "split": "dev",
                "files": {"opt": {"path": str(root / "opt/000000_opt.png")}},
            }
            with self.assertRaises(PermissionError):
                enforce_split_access(record, root, "train")

    def test_split_access_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            record = {
                "pair_id": "000000",
                "split": "train",
                "files": {"opt": {"path": "/tmp/testing/000000_opt.png"}},
            }
            with self.assertRaises(PermissionError):
                enforce_split_access(record, root, "train")


if __name__ == "__main__":
    unittest.main()
