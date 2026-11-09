import threading
import unittest
from unittest.mock import patch

import numpy as np

from main import SEMStitcher, StitchingCancelled
from stitching_progress import StageProgress
from stitching_task import StitchRequest, StitchTask, TaskState


class ProgressTests(unittest.TestCase):
    def make_stitcher(self):
        stitcher = SEMStitcher()
        stitcher.set_progress_callback(lambda message: None)
        return stitcher

    def test_counts_cover_loaded_images_and_all_attempted_pairs_including_rejections(self):
        for grid, pairs in ((None, [(-40, 0, 20), (None, None, 0), (-40, 0, 20)]),
                            ((1, 3), [(-40, 0, 20), (-40, 0, 20)])):
            with self.subTest(grid=grid):
                stitcher = self.make_stitcher()
                events = []
                stitcher.set_stage_callback(events.append)
                with patch('main.read_image', return_value=np.full((40, 100), 12000, np.uint16)), \
                     patch('main.file_signature', return_value=(0, 0)), \
                     patch.object(stitcher, '_try_match_pair', side_effect=pairs):
                    result = stitcher.stitch_sequence(['a', 'b', 'c'], grid_shape=grid)
                self.assertEqual([(p.completed, p.total) for p in events if p.stage == 'load'],
                                 [(i, 3) for i in range(4)])
                self.assertEqual([(p.completed, p.total) for p in events if p.stage == 'match'],
                                 [(i, len(pairs)) for i in range(len(pairs)+1)])
                self.assertIn(StageProgress('solve'), events)
                self.assertEqual(events[-1], StageProgress('blend', 1, 1))
                self.assertEqual(result.dtype, np.uint16)
                self.assertTrue(np.all(result == 12000))

    def test_real_block_counts_do_not_change_pixels_and_stop_on_cancel(self):
        stitcher = self.make_stitcher()
        image = np.arange(400, dtype=np.uint16).reshape(20, 20)
        reference = stitcher._compose_images([image, image], [(0, 0), (5.5, 0)], block_size=8)
        events = []
        stitcher.set_stage_callback(events.append)
        result = stitcher._compose_images([image, image], [(0, 0), (5.5, 0)], block_size=8)
        np.testing.assert_array_equal(result, reference)
        self.assertEqual([p.completed for p in events], list(range(13)))
        self.assertTrue(all(p.total == 12 for p in events))
        cancelled = threading.Event()
        events.clear()

        def report(progress):
            events.append(progress)
            if progress.completed == 2:
                cancelled.set()

        stitcher.set_cancel_callback(cancelled.is_set)
        stitcher.set_stage_callback(report)
        with self.assertRaises(StitchingCancelled):
            stitcher._compose_images([image, image], [(0, 0), (5.5, 0)], block_size=8)
        self.assertEqual(events[-1].completed, 2)

    def test_progress_is_published_on_owner_poll_and_new_run_clears_it(self):
        class Worker:
            def set_cancel_callback(self, callback):
                pass

            def set_progress_callback(self, callback):
                self.log = callback

            def set_stage_callback(self, callback):
                self.progress = callback

            def stitch_sequence(self, paths, **options):
                self.progress(StageProgress('match', 4, 12))
                self.log('ordinary log')
                return np.ones((4, 4), np.uint8)

        task = StitchTask(Worker)
        request = StitchRequest(('a', 'b'))
        task.start(request)
        task.thread.join(3)
        self.assertIsNone(task.progress)
        self.assertEqual(task.poll(), ['ordinary log'])
        self.assertEqual(task.progress, StageProgress('match', 4, 12))
        self.assertEqual(task.state, TaskState.SUCCEEDED)
        task.start(request)
        self.assertIsNone(task.progress)
        task.thread.join(3)
        task.cancel()
        task.poll()
        self.assertEqual(task.state, TaskState.CANCELLED)
        self.assertIsNone(task.result)
        task.reset(True)
        self.assertIsNone(task.progress)


if __name__ == '__main__':
    unittest.main()
