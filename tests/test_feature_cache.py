import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from main import SEMStitcher, StitchingCancelled
from test_alignment_blending import texture


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        scene = texture(512)
        self.paths = []
        for row in range(3):
            for col in range(3):
                path = str(Path(self.directory.name) / f'{row}_{col}.tif')
                cv2.imwrite(path, scene[row*128:row*128+256, col*128:col*128+256])
                self.paths.append(path)
        self.stitcher = SEMStitcher()
        self.stitcher.set_progress_callback(lambda message: None)

    def test_nine_tile_cache_reduces_72_extractions_to_9_without_changing_result(self):
        cv2.setRNGSeed(42)
        baseline = self.stitcher.stitch_sequence(self.paths, use_feature_cache=False)
        baseline_report = self.stitcher.last_inspection
        baseline_metrics = self.stitcher.last_metrics.copy()
        cv2.setRNGSeed(42)
        cached = self.stitcher.stitch_sequence(self.paths)
        self.assertEqual(baseline_metrics['feature_extractions'], 72)
        self.assertEqual(self.stitcher.last_metrics['feature_extractions'], 9)
        self.assertEqual(self.stitcher.last_metrics['cache_hits'], 63)
        self.assertGreater(self.stitcher.last_metrics['descriptor_bytes'], 0)
        np.testing.assert_array_equal(cached, baseline)
        np.testing.assert_allclose(self.stitcher.last_inspection.positions, baseline_report.positions, atol=1e-8)
        self.assertIsNone(self.stitcher._feature_cache)

    def test_new_run_reextracts_features_even_with_the_same_paths(self):
        self.stitcher.stitch_sequence(self.paths[:2])
        first_signature = self.stitcher.last_inspection.signatures[0]
        cv2.imwrite(self.paths[0], texture(256, seed=99))
        with patch.object(self.stitcher, '_compute_translation', return_value=(-128, 0, 50, 50)):
            self.stitcher.stitch_sequence(self.paths[:2])
        self.assertEqual(self.stitcher.last_metrics['feature_extractions'], 2)
        self.assertNotEqual(self.stitcher.last_inspection.signatures[0], first_signature)
        self.assertIsNone(self.stitcher._feature_cache)

    def test_failure_releases_cache_and_old_inspection(self):
        self.stitcher.stitch_sequence(self.paths[:2])
        self.assertIsNotNone(self.stitcher.last_inspection)
        with patch.object(self.stitcher, '_compute_translation', side_effect=RuntimeError('match failed')):
            with self.assertRaisesRegex(RuntimeError, 'match failed'):
                self.stitcher.stitch_sequence(self.paths[:2])
        self.assertGreater(self.stitcher.last_metrics['feature_extractions'], 0)
        self.assertIsNone(self.stitcher._feature_cache)
        self.assertIsNone(self.stitcher.last_inspection)

    def test_cancellation_releases_partially_populated_cache(self):
        cancel = threading.Event()
        self.stitcher.set_cancel_callback(cancel.is_set)
        self.stitcher.set_progress_callback(lambda message: cancel.set() if 'Keypoints' in message else None)
        with self.assertRaises(StitchingCancelled):
            self.stitcher.stitch_sequence(self.paths)
        self.assertGreater(self.stitcher.last_metrics['feature_extractions'], 0)
        self.assertIsNone(self.stitcher._feature_cache)
        self.assertIsNone(self.stitcher.last_inspection)

    def test_featureless_tiles_are_cached_as_well(self):
        with patch.object(self.stitcher, '_extract_features', return_value=([], None)) as extract:
            with self.assertRaisesRegex(RuntimeError, 'reliable edges'):
                self.stitcher.stitch_sequence(self.paths)
        self.assertEqual(extract.call_count, 9)
        self.assertIsNone(self.stitcher._feature_cache)

    def test_standalone_feature_lookup_does_not_change_completed_batch_metrics(self):
        self.stitcher.stitch_sequence(self.paths[:2])
        metrics = self.stitcher.last_metrics.copy()
        self.stitcher._features_for_image(texture(256))
        self.assertEqual(self.stitcher.last_metrics, metrics)


if __name__ == '__main__':
    unittest.main()
