import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from main import SEMStitcher, StitchingCancelled
from stitching_task import StitchRequest, StitchTask, TaskState, natural_sort


class ImmediateStitcher:
    def set_cancel_callback(self, callback):
        self.cancelled = callback

    def set_progress_callback(self, callback):
        self.log = callback

    def stitch_sequence(self, paths, **options):
        self.paths, self.options = paths, options
        self.log('worker log')
        return np.full((20, 20), 120, np.uint8)


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.worker = ImmediateStitcher()
        self.task = StitchTask(lambda: self.worker)
        self.request = StitchRequest(('a.tif', 'b.tif'))

    def finish(self):
        self.task.thread.join(timeout=3)
        self.assertFalse(self.task.thread.is_alive(), 'test worker did not exit')
        return self.task.poll()

    def test_worker_result_is_only_published_when_owner_polls(self):
        self.task.start(self.request)
        self.task.thread.join(timeout=3)
        self.assertEqual(self.task.state, TaskState.RUNNING)
        self.assertIsNone(self.task.result)
        self.assertEqual(self.task.poll(), ['worker log'])
        self.assertEqual(self.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.task.result.shape, (20, 20))

    def test_new_run_clears_old_success_before_work_completes(self):
        self.task.start(self.request)
        self.finish()
        self.task.start(self.request)
        self.assertIsNone(self.task.result)
        self.assertEqual(self.task.state, TaskState.RUNNING)
        self.finish()

    def test_failure_clears_old_result_and_allows_retry(self):
        self.task.start(self.request)
        self.finish()
        with patch.object(self.worker, 'stitch_sequence', side_effect=RuntimeError('bad overlap')):
            self.task.start(self.request)
            self.finish()
        self.assertEqual(self.task.state, TaskState.FAILED)
        self.assertIsNone(self.task.result)
        self.assertIn('bad overlap', self.task.error)
        self.task.start(self.request)
        self.finish()
        self.assertEqual(self.task.state, TaskState.SUCCEEDED)
        self.assertIsNone(self.task.error)

    def test_accepted_cancel_wins_over_queued_success(self):
        self.task.start(self.request)
        self.task.thread.join(timeout=3)
        self.task.cancel()
        self.finish()
        self.assertEqual(self.task.state, TaskState.CANCELLED)
        self.assertIsNone(self.task.result)
        self.assertIsNone(self.task.error)

    def test_cancel_then_wait_before_restarting(self):
        entered, release = threading.Event(), threading.Event()

        def blocked(paths, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test timed out')
            if self.worker.cancelled():
                raise StitchingCancelled()
            return np.ones((5, 5), np.uint8)

        with patch.object(self.worker, 'stitch_sequence', side_effect=blocked):
            self.task.start(self.request)
            self.assertTrue(entered.wait(2))
            try:
                self.task.cancel()
                self.assertEqual(self.task.state, TaskState.CANCELLING)
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    self.task.start(self.request)
                with self.assertRaisesRegex(RuntimeError, 'Wait'):
                    self.task.reset()
            finally:
                release.set()
                self.finish()
        self.assertEqual(self.task.state, TaskState.CANCELLED)
        self.task.start(self.request)
        self.finish()
        self.assertEqual(self.task.state, TaskState.SUCCEEDED)

    def test_input_changes_invalidate_success(self):
        self.task.start(self.request)
        self.finish()
        self.task.reset(has_inputs=True)
        self.assertEqual(self.task.state, TaskState.READY)
        self.assertIsNone(self.task.result)
        self.assertIsNone(self.task.request)

    def test_thread_start_failure_is_recoverable(self):
        with patch('stitching_task.threading.Thread.start', side_effect=RuntimeError('no thread')):
            with self.assertRaisesRegex(RuntimeError, 'no thread'):
                self.task.start(self.request)
        self.assertEqual(self.task.state, TaskState.FAILED)
        self.assertIsNone(self.task.result)
        self.task.start(self.request)
        self.finish()
        self.assertEqual(self.task.state, TaskState.SUCCEEDED)

    def test_worker_uses_frozen_inputs_and_canonical_grid_order(self):
        paths = [f'{i}.tif' for i in range(6)]
        request = StitchRequest(paths, (2, 3), True, True)
        paths.reverse()
        self.task.start(request)
        self.finish()
        self.assertEqual(self.worker.paths, ('2.tif', '1.tif', '0.tif', '3.tif', '4.tif', '5.tif'))
        self.assertEqual(self.worker.options, {'grid_shape': (2, 3), 'serpentine': False})


class GridOrderTests(unittest.TestCase):
    def test_four_scan_patterns_map_to_physical_rows(self):
        paths = tuple(str(i) for i in range(6))
        for snake, rtl, expected in [
            (False, False, [[0, 1, 2], [3, 4, 5]]),
            (False, True, [[2, 1, 0], [5, 4, 3]]),
            (True, False, [[0, 1, 2], [5, 4, 3]]),
            (True, True, [[2, 1, 0], [3, 4, 5]]),
        ]:
            with self.subTest(snake=snake, rtl=rtl):
                request = StitchRequest(paths, (2, 3), snake, rtl)
                self.assertEqual(request.layout_indices(), expected)

    def test_natural_filename_order(self):
        self.assertEqual(natural_sort(['tile_10.tif', 'tile_2.tif', 'tile_1.tif']),
                         ['tile_1.tif', 'tile_2.tif', 'tile_10.tif'])

    def test_auto_mode_has_no_grid_options(self):
        request = StitchRequest(('a', 'b'))
        self.assertEqual(request.algorithm_inputs(), (('a', 'b'), {}))
        self.assertEqual(request.layout_indices(), [])

    def test_invalid_requests_fail_before_worker_starts(self):
        for args in [(['a'],), (['a', 'b'], (2, 2)), (['a', 'b'], (0, 2)),
                     (['a', 'b'], (1.0, 2)), (['a', 'b'], None, True)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                StitchRequest(*args)


class CoreCancellationTests(unittest.TestCase):
    def setUp(self):
        self.stitcher = SEMStitcher()
        self.stitcher.set_progress_callback(lambda message: None)
        self.cancel = threading.Event()
        self.stitcher.set_cancel_callback(self.cancel.is_set)

    def test_pre_cancelled_task_does_not_load_any_images(self):
        self.cancel.set()
        with patch('main.cv2.imread') as read:
            with self.assertRaises(StitchingCancelled):
                self.stitcher.stitch_sequence(['a', 'b'])
            read.assert_not_called()

    def test_cancel_is_checked_after_feature_extraction(self):
        detector = Mock()
        detector.detectAndCompute.side_effect = lambda *args: (self.cancel.set(), None)
        self.stitcher.sift = detector
        with self.assertRaises(StitchingCancelled):
            self.stitcher._extract_features(np.zeros((100, 100), np.uint8))

    def test_cancel_between_loading_files(self):
        self.stitcher.set_progress_callback(lambda msg: self.cancel.set() if 'Loaded:' in msg else None)
        with patch('main.cv2.imread', return_value=np.zeros((100, 100), np.uint8)) as read:
            with self.assertRaises(StitchingCancelled):
                self.stitcher.stitch_sequence(['a', 'b'])
            self.assertEqual(read.call_count, 1)

    def test_cancel_during_ransac(self):
        pairs = [[cv2.DMatch(i, i, 1), cv2.DMatch(i, (i + 1) % 10, 10)] for i in range(10)]
        matcher = Mock()
        matcher.knnMatch.return_value = pairs
        points = [cv2.KeyPoint(float(i), float(i), 1) for i in range(10)]
        descriptors = np.zeros((10, 128), np.float32)
        self.stitcher.set_cancel_callback(Mock(side_effect=[False, True]))
        with patch('main.cv2.FlannBasedMatcher', return_value=matcher):
            with self.assertRaises(StitchingCancelled):
                self.stitcher._compute_translation(points, descriptors, points, descriptors)

    def test_cancel_during_composition_discards_partial_panorama(self):
        original = self.stitcher._blend_onto_canvas

        def cancel_after_first_tile(*args):
            original(*args)
            self.cancel.set()

        image = np.ones((40, 40), np.uint8)
        with patch.object(self.stitcher, '_blend_onto_canvas', side_effect=cancel_after_first_tile) as blend:
            with self.assertRaises(StitchingCancelled):
                self.stitcher._compose_images([image, image], [(0, 0), (20, 0)])
            self.assertEqual(blend.call_count, 2)


if __name__ == '__main__':
    unittest.main()
