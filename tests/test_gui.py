"""Native Tk widget checks. Linux CI runs these under Xvfb."""
import os
import sys
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

try:
    from gui import LANG, SEMStitcherGUI
    from stitching_task import StitchTask, TaskState
except ImportError as error:
    raise unittest.SkipTest(f'GUI dependencies unavailable: {error}')


@unittest.skipIf(sys.platform.startswith('linux') and not os.environ.get('DISPLAY'), 'Tk needs a display; use xvfb-run')
class GUIWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.app = SEMStitcherGUI()
        self.callback_errors = []
        self.app.report_callback_exception = lambda kind, error, trace: self.callback_errors.append(str(error))
        self.app.withdraw()
        self.app.lang = 'en'
        self.app._apply_language()
        self.app.update()
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            worker.release.set()
        if not self.app._closing:
            self.app._on_close()
        for worker in self.workers:
            if worker.thread is not None:
                worker.thread.join(timeout=3)
        self.assertEqual(self.callback_errors, [], 'Unexpected Tk callback errors')

    def load(self, count=2):
        self.app._load_files([f'tile_{i}.tif' for i in range(count)])

    def install_worker(self, fail=False):
        class Worker:
            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()
                self.thread = None

            def set_cancel_callback(self, callback):
                self.cancelled = callback

            def set_progress_callback(self, callback):
                self.log = callback

            def stitch_sequence(self, paths, **options):
                self.paths, self.options = paths, options
                self.thread = threading.current_thread()
                self.entered.set()
                if not self.release.wait(3):
                    raise RuntimeError('test worker timed out')
                self.log('worker finished')
                if fail:
                    raise RuntimeError('bad overlap')
                return np.full((40, 40), 120, np.uint8)
        worker = Worker()
        self.workers.append(worker)
        self.app.task = StitchTask(lambda: worker)
        self.app._on_inputs_changed()
        return worker

    def finish_worker(self, worker):
        worker.release.set()
        deadline = time.monotonic() + 4
        while self.app.task.busy and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.assertFalse(self.app.task.busy)

    def test_initial_and_ready_controls(self):
        self.assertEqual(self.app.task.state, TaskState.IDLE)
        self.assertEqual(self.app.btn_start.cget('state'), 'disabled')
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.load()
        self.assertEqual(self.app.task.state, TaskState.READY)
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')
        self.assertEqual(self.app.rows_entry.cget('state'), 'disabled')

    def test_controls_fit_in_minimum_window(self):
        self.load()
        self.app.use_grid.set(True)
        self.app._on_inputs_changed()
        self.app.deiconify()
        self.app.geometry('1050x700')
        self.app.update()
        for widget in (self.app.btn_start, self.app.btn_cancel, self.app.btn_export,
                       self.app.rows_entry, self.app.cols_entry, self.app.layout_text,
                       self.app.file_list, self.app.lbl_validation):
            with self.subTest(widget=widget.__class__.__name__):
                self.assertGreater(widget.winfo_width(), 1)
                self.assertGreater(widget.winfo_height(), 1)
                self.assertGreaterEqual(widget.winfo_rootx(), self.app.winfo_rootx())
                self.assertGreaterEqual(widget.winfo_rooty(), self.app.winfo_rooty())
                self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                     self.app.winfo_rootx() + self.app.winfo_width())
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                     self.app.winfo_rooty() + self.app.winfo_height())

    def test_invalid_grid_blocks_start_and_disabling_grid_restores_auto(self):
        self.load()
        self.app.use_grid.set(True)
        self.app._on_inputs_changed()
        self.assertEqual(self.app.btn_start.cget('state'), 'disabled')
        self.assertIn('9 images', self.app.lbl_validation.cget('text'))
        self.app.rows.set('0')
        self.assertIn('positive', self.app.lbl_validation.cget('text'))
        self.app.use_grid.set(False)
        self.app._on_inputs_changed()
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')
        self.assertEqual(self.app.lbl_validation.cget('text'), '')

    def test_grid_preview_and_worker_receive_same_file_order(self):
        self.load(6)
        worker = self.install_worker()
        self.app.rows.set('2')
        self.app.cols.set('3')
        self.app.use_grid.set(True)
        self.app.serpentine.set(True)
        self.app.first_rtl.set(True)
        self.app._on_inputs_changed()
        self.assertEqual(self.app.layout_text.get('1.0', 'end').strip(), '03 ← 02 ← 01\n04 → 05 → 06')
        self.app._on_start()
        self.assertTrue(worker.entered.wait(2))
        self.assertEqual(worker.paths, ('tile_2.tif', 'tile_1.tif', 'tile_0.tif',
                                        'tile_3.tif', 'tile_4.tif', 'tile_5.tif'))
        self.assertEqual(worker.options, {'grid_shape': (2, 3), 'serpentine': False})
        self.finish_worker(worker)

    def test_reorder_reverse_and_natural_sort(self):
        self.app._load_files(['tile_10.tif', 'tile_1.tif', 'tile_2.tif'])
        self.assertEqual(self.app.image_paths, ['tile_1.tif', 'tile_2.tif', 'tile_10.tif'])
        self.app.file_list.selection_set(1)
        self.app._move_file(-1)
        self.assertEqual(self.app.image_paths[:2], ['tile_2.tif', 'tile_1.tif'])
        self.app._reverse_files()
        self.assertEqual(self.app.image_paths[0], 'tile_10.tif')
        self.app._sort_files()
        self.assertEqual(self.app.image_paths, ['tile_1.tif', 'tile_2.tif', 'tile_10.tif'])

    def test_running_cancelled_and_retry_states_never_expose_old_result(self):
        self.load()
        worker = self.install_worker()
        self.app.task.state = TaskState.SUCCEEDED
        self.app.task.result = np.ones((10, 10), np.uint8)
        self.app._render_state()
        self.app._on_start()
        self.assertTrue(worker.entered.wait(2))
        self.assertIsNone(self.app.stitched_result)
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.assertEqual(self.app.btn_select.cget('state'), 'disabled')
        self.assertEqual(self.app.grid_switch.cget('state'), 'disabled')
        with patch('gui.filedialog.asksaveasfilename') as dialog:
            self.app._on_export()
            dialog.assert_not_called()
        self.app._on_cancel()
        self.assertEqual(self.app.task.state, TaskState.CANCELLING)
        self.assertEqual(self.app.btn_cancel.cget('state'), 'disabled')
        self.finish_worker(worker)
        self.assertEqual(self.app.task.state, TaskState.CANCELLED)
        self.assertEqual(self.app.lbl_status.cget('text'), LANG['en']['status_cancelled'])
        self.assertIsNone(self.app.stitched_result)
        self.assertEqual(self.app.lbl_size.cget('text'), '')
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.app._on_start()
        self.finish_worker(worker)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')

    def test_failure_does_not_leave_a_preview_or_enabled_export(self):
        self.load()
        worker = self.install_worker(fail=True)
        self.app._on_start()
        self.finish_worker(worker)
        self.assertEqual(self.app.task.state, TaskState.FAILED)
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.assertIsNone(self.app.preview_label.image)
        self.assertIn('bad overlap', self.app.log_text.get('1.0', 'end'))

    def test_changes_invalidate_successful_result(self):
        self.load()
        worker = self.install_worker()
        self.app._on_start()
        self.finish_worker(worker)
        self.app._reverse_files()
        self.assertEqual(self.app.task.state, TaskState.READY)
        self.assertIsNone(self.app.stitched_result)
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.assertEqual(self.app.progress_bar.get(), 0)

    def test_success_can_select_new_images_and_complete_a_second_run(self):
        self.load()
        worker = self.install_worker()
        self.app._on_start()
        self.finish_worker(worker)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app.btn_select.cget('state'), 'normal')
        with patch('gui.filedialog.askopenfilenames', return_value=('next_2.tif', 'next_1.tif')) as dialog:
            self.app.btn_select.invoke()
            dialog.assert_called_once()
        self.assertEqual(self.app.image_paths, ['next_1.tif', 'next_2.tif'])
        self.assertEqual(self.app.file_list.get(0, 'end'), ('01  next_1.tif', '02  next_2.tif'))
        self.assertEqual(self.app.task.state, TaskState.READY)
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.assertIsNone(self.app.stitched_result)
        self.app._on_start()
        self.finish_worker(worker)
        self.assertEqual(worker.paths, ('next_1.tif', 'next_2.tif'))
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')

    def test_success_can_reload_and_change_settings_multiple_times(self):
        self.load()
        worker = self.install_worker()
        self.app._on_start()
        self.finish_worker(worker)
        for index in range(3):
            self.app._load_files([f'batch_{index}_1.tif', f'batch_{index}_2.tif'])
            self.app.rows.set('1')
            self.app.cols.set('2')
            self.app.use_grid.set(True)
            self.app._on_inputs_changed()
            self.app._set_lang('简体中文' if index % 2 == 0 else 'English')
            self.assertEqual(self.app.btn_start.cget('state'), 'normal')
            self.app._on_start()
            self.finish_worker(worker)
            self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)

    def test_dismissing_file_dialog_keeps_successful_result(self):
        self.load()
        worker = self.install_worker()
        self.app._on_start()
        self.finish_worker(worker)
        previous_result = self.app.stitched_result
        previous_paths = self.app.image_paths[:]
        with patch('gui.filedialog.askopenfilenames', return_value=()):
            self.app.btn_select.invoke()
        self.assertIs(self.app.stitched_result, previous_result)
        self.assertEqual(self.app.image_paths, previous_paths)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')

    def test_new_count_releases_previous_grid_constraint_and_allows_restart(self):
        self.load()
        worker = self.install_worker()
        self.app.rows.set('1')
        self.app.cols.set('2')
        self.app.use_grid.set(True)
        self.app._on_inputs_changed()
        self.app._on_start()
        self.finish_worker(worker)
        self.app._load_files(['next_1.tif', 'next_2.tif', 'next_3.tif'])
        self.assertFalse(self.app.use_grid.get())
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')
        self.assertEqual(self.app.lbl_validation.cget('text'), '')
        self.assertIn('Automatic matching is enabled', self.app.layout_text.get('1.0', 'end'))
        self.app._on_start()
        self.finish_worker(worker)
        self.assertEqual(worker.options, {})
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)

    def test_same_count_keeps_grid_and_scan_direction(self):
        self.load(6)
        self.app.rows.set('2')
        self.app.cols.set('3')
        self.app.use_grid.set(True)
        self.app.serpentine.set(True)
        self.app.first_rtl.set(True)
        self.app._on_inputs_changed()
        self.app._load_files([f'next_{i}.tif' for i in range(6)])
        self.assertTrue(self.app.use_grid.get())
        self.assertTrue(self.app.serpentine.get())
        self.assertTrue(self.app.first_rtl.get())
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')

    def test_new_count_keeps_grid_if_user_already_set_matching_dimensions(self):
        self.load()
        self.app.rows.set('1')
        self.app.cols.set('3')
        self.app.use_grid.set(True)
        self.app._on_inputs_changed()
        self.assertEqual(self.app.btn_start.cget('state'), 'disabled')
        self.app._load_files(['a.tif', 'b.tif', 'c.tif'])
        self.assertTrue(self.app.use_grid.get())
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')

    def test_language_switch_preserves_count_preview_and_running_controls(self):
        self.load()
        worker = self.install_worker()
        self.app._on_start()
        self.app._on_cancel()
        self.app._set_lang('简体中文')
        self.assertEqual(self.app.btn_start.cget('state'), 'disabled')
        self.assertEqual(self.app.btn_cancel.cget('text'), LANG['zh-Hans']['btn_cancelling'])
        self.assertIn('2', self.app.lbl_count.cget('text'))
        self.finish_worker(worker)
        self.app._on_start()
        self.finish_worker(worker)
        self.app._set_lang('日本語')
        self.assertEqual(self.app.preview_label.cget('text'), '')
        self.assertIsNotNone(self.app.preview_label.image)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')

    def test_close_cancels_worker_without_tk_calls_after_destroy(self):
        self.load()
        worker = self.install_worker()
        self.app._on_start()
        self.assertTrue(worker.entered.wait(2))
        self.app._on_close()
        worker.release.set()
        worker.thread.join(timeout=3)
        self.assertFalse(worker.thread.is_alive())
        self.app.task.poll()
        self.assertEqual(self.app.task.state, TaskState.CANCELLED)

    def test_real_sift_grid_stitch_reaches_exportable_gui_result(self):
        from test_alignment_blending import texture
        scene = texture(480)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for row in range(2):
                columns = reversed(range(3)) if row == 0 else range(3)
                for col in columns:
                    path = Path(directory) / f'tile_{len(paths):02d}.tif'
                    self.assertTrue(cv2.imwrite(str(path), scene[row*80:row*80+320, col*80:col*80+320]))
                    paths.append(str(path))
            self.app._load_files(paths)
            self.app.rows.set('2')
            self.app.cols.set('3')
            self.app.use_grid.set(True)
            self.app.serpentine.set(True)
            self.app.first_rtl.set(True)
            self.app._on_inputs_changed()
            self.app._on_start()
            deadline = time.monotonic() + 20
            while self.app.task.busy and time.monotonic() < deadline:
                self.app.update()
                time.sleep(0.01)
            self.assertEqual(self.app.task.state, TaskState.SUCCEEDED, self.app.task.error)
            self.assertEqual(self.app.btn_export.cget('state'), 'normal')
            self.assertIsNotNone(self.app.preview_label.image)
            height, width = self.app.stitched_result.shape
            self.assertLessEqual(abs(height - 400), 2)
            self.assertLessEqual(abs(width - 480), 2)
            # A real second batch with a different count must be usable without
            # restarting the window or inheriting the obsolete 2x3 constraint.
            self.app._load_files(paths[:2])
            self.assertFalse(self.app.use_grid.get())
            self.assertEqual(self.app.btn_start.cget('state'), 'normal')
            self.app._on_start()
            deadline = time.monotonic() + 20
            while self.app.task.busy and time.monotonic() < deadline:
                self.app.update()
                time.sleep(0.01)
            self.assertEqual(self.app.task.state, TaskState.SUCCEEDED, self.app.task.error)
            self.assertEqual(self.app.btn_export.cget('state'), 'normal')
            height, width = self.app.stitched_result.shape
            self.assertLessEqual(abs(height - 320), 2)
            self.assertLessEqual(abs(width - 400), 2)


if __name__ == '__main__':
    unittest.main()
