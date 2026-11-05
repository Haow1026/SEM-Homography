import threading
import unittest
from unittest.mock import patch

from main import StitchingCancelled
from project_job import ProjectJob
from stitching_task import TaskState


class ProjectJobTests(unittest.TestCase):
    def setUp(self):
        self.job = ProjectJob()

    def finish(self):
        self.job.thread.join(timeout=3)
        self.assertFalse(self.job.thread.is_alive())
        return self.job.poll()

    def test_only_owner_poll_publishes_result(self):
        self.job.start('open', lambda cancel, log: (log('verified'), 'project')[1])
        self.job.thread.join(timeout=3)
        self.assertIsNone(self.job.result)
        self.assertEqual(self.job.state, TaskState.RUNNING)
        self.assertEqual(self.finish(), ['verified'])
        self.assertEqual(self.job.result, 'project')

    def test_open_cancel_wins_over_queued_success(self):
        self.job.start('open', lambda cancel, log: 'project')
        self.job.thread.join(timeout=3)
        self.job.cancel()
        self.finish()
        self.assertEqual(self.job.state, TaskState.CANCELLED)
        self.assertIsNone(self.job.result)

    def test_save_that_already_committed_reports_success(self):
        self.job.start('save', lambda cancel, log: 'saved.semproj')
        self.job.thread.join(timeout=3)
        self.job.cancel()
        self.finish()
        self.assertEqual(self.job.state, TaskState.SUCCEEDED)
        self.assertEqual(self.job.result, 'saved.semproj')

    def test_cancel_error_and_retry(self):
        def cancelled(cancel, log):
            raise StitchingCancelled()

        self.job.start('open', cancelled)
        self.finish()
        self.assertEqual(self.job.state, TaskState.CANCELLED)

        def failed(cancel, log):
            raise ValueError('invalid project')

        self.job.start('open', failed)
        self.finish()
        self.assertEqual(self.job.state, TaskState.FAILED)
        self.assertIn('invalid project', self.job.error)
        self.job.start('save', lambda cancel, log: 'saved.semproj')
        self.finish()
        self.assertEqual(self.job.state, TaskState.SUCCEEDED)
        self.assertIsNone(self.job.error)

    def test_no_concurrent_operation_and_closed_queue_discards_late_result(self):
        entered, release = threading.Event(), threading.Event()

        def blocked(cancel, log):
            entered.set()
            release.wait(3)
            return 'late result'

        self.job.start('open', blocked)
        self.assertTrue(entered.wait(2))
        try:
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                self.job.start('save', blocked)
            self.job.close()
        finally:
            release.set()
            self.finish()
        self.assertIsNone(self.job.result)
        self.assertTrue(self.job._events.empty())

    def test_thread_start_failure_can_be_retried(self):
        with patch('project_job.threading.Thread.start', side_effect=RuntimeError('no thread')):
            with self.assertRaisesRegex(RuntimeError, 'no thread'):
                self.job.start('open', lambda cancel, log: None)
        self.assertEqual(self.job.state, TaskState.FAILED)
        self.job.start('open', lambda cancel, log: 'project')
        self.finish()
        self.assertEqual(self.job.state, TaskState.SUCCEEDED)


if __name__ == '__main__':
    unittest.main()
