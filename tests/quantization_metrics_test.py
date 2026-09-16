#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quantization_metrics", ROOT / "scripts" / "quantization_metrics.py"
)
assert SPEC and SPEC.loader
METRICS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = METRICS
SPEC.loader.exec_module(METRICS)


class QuantizationMetricsTest(unittest.TestCase):
    def test_vector_metrics_identical(self) -> None:
        result = METRICS.vector_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertAlmostEqual(result["cosine_similarity"], 1.0)
        self.assertAlmostEqual(result["relative_l2"], 0.0)
        self.assertAlmostEqual(result["norm_ratio"], 1.0)

    def test_motion_translation_and_quaternion_sign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, candidate = root / "bf16", root / "q8_0"
            reference.mkdir()
            candidate.mkdir()
            skeleton = {
                "key": "test", "frames": 2, "fps": 30,
                "names": ["root", "child"], "parents": [-1, 0],
                "offsets": [[0, 0, 0], [0, 1, 0]],
            }
            for directory in (reference, candidate):
                (directory / "skeleton.json").write_text(json.dumps(skeleton))
            ref_root = np.zeros((2, 3), dtype="<f4")
            got_root = ref_root.copy()
            got_root[:, 0] = 0.01
            ref_rotation = np.zeros((2, 2, 4), dtype="<f4")
            ref_rotation[..., 3] = 1
            got_rotation = -ref_rotation
            for directory, roots, rotations in (
                (reference, ref_root, ref_rotation),
                (candidate, got_root, got_rotation),
            ):
                roots.tofile(directory / "root_positions.f32")
                rotations.tofile(directory / "local_rotations_xyzw.f32")
                np.array([1.0, 2.0], dtype="<f4").tofile(directory / "sampling_final_state.f32")
            result = METRICS.motion_metrics(reference, candidate)
            self.assertAlmostEqual(result["world_mpjpe_cm"]["mean"], 1.0, places=5)
            self.assertAlmostEqual(result["per_frame_world_mpjpe_cm"][0], 1.0, places=5)
            self.assertAlmostEqual(result["root_relative_mpjpe_cm"]["max"], 0.0, places=5)
            self.assertAlmostEqual(result["joint_rotation_degrees"]["max"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
