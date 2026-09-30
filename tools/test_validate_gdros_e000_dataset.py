#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).with_name("validate_gdros_e000_dataset.py")
SPEC = importlib.util.spec_from_file_location("validate_gdros_e000_dataset", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class NpyHeaderTests(unittest.TestCase):
    def test_reads_numeric_array_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric.npy"
            np.save(path, np.zeros((3, 4), dtype=np.float32))
            version, shape, fortran_order, dtype = VALIDATOR.read_npy_header(path)
            self.assertEqual(version, (1, 0))
            self.assertEqual(shape, (3, 4))
            self.assertFalse(fortran_order)
            self.assertEqual(dtype, np.dtype("float32"))

    def test_reads_object_array_header_without_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npy"
            np.save(path, np.array([{"must_not_execute": True}] * 6, dtype=object))
            version, shape, fortran_order, dtype = VALIDATOR.read_npy_header(path)
            self.assertEqual(version, (1, 0))
            self.assertEqual(shape, (6,))
            self.assertFalse(fortran_order)
            self.assertTrue(dtype.hasobject)
            with self.assertRaises(ValueError):
                np.load(path, allow_pickle=False)

    def test_rejects_a_truncated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.npy"
            path.write_bytes(b"\x93NUMPY")
            with self.assertRaisesRegex(RuntimeError, "invalid NPY header"):
                VALIDATOR.read_npy_header(path)


class ManifestGateTests(unittest.TestCase):
    def test_accepts_the_verified_official_manifest(self):
        VALIDATOR.require_expected_data_manifest(
            VALIDATOR.EXPECTED_DATA_MANIFEST_SHA256
        )

    def test_rejects_any_other_manifest(self):
        with self.assertRaisesRegex(RuntimeError, "verified official archive"):
            VALIDATOR.require_expected_data_manifest("0" * 64)


class EvaluationContentTests(unittest.TestCase):
    def test_grayscale_and_equivalent_rgb_png_have_the_same_content_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            gray_path = Path(directory) / "gray.png"
            rgb_path = Path(directory) / "rgb.png"
            pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
            Image.fromarray(pixels, mode="L").save(gray_path)
            Image.fromarray(np.repeat(pixels[..., None], 3, axis=2), mode="RGB").save(
                rgb_path
            )
            self.assertEqual(
                VALIDATOR.evaluation_content_sha256(gray_path),
                VALIDATOR.evaluation_content_sha256(rgb_path),
            )

    def test_png_pixel_change_changes_the_content_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            first = np.zeros((4, 4), dtype=np.uint8)
            second = first.copy()
            second[2, 3] = 1
            Image.fromarray(first, mode="L").save(first_path)
            Image.fromarray(second, mode="L").save(second_path)
            self.assertNotEqual(
                VALIDATOR.evaluation_content_sha256(first_path),
                VALIDATOR.evaluation_content_sha256(second_path),
            )


if __name__ == "__main__":
    unittest.main()
