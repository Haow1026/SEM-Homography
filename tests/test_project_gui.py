"""Native project save/open workflows, including non-destructive failure/cancel."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

try:
    from gui import SEMStitcherGUI
    from image_io import read_image, write_image
    from inspection import PairDiagnostic, StitchInspection, file_signature
    from main import SEMStitcher
    from project import RestoredProject, save_project
    from stitching_task import StitchRequest, TaskState
except ImportError as error:
    raise unittest.SkipTest(f'GUI dependencies unavailable: {error}')


@unittest.skipIf(sys.platform.startswith('linux') and not os.environ.get('DISPLAY'), 'Tk needs a display')
class ProjectGUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name).resolve() / '项目图像'
        self.folder.mkdir()
        self.path = self.folder / 'project.semproj'
        self.app = SEMStitcherGUI()
        self.app.withdraw()
        self.app.lang = 'en'
        self.app._apply_language()
        self.errors = []
        self.app.report_callback_exception = lambda kind, error, trace: self.errors.append(str(error))
        self.releases = []
        self.paths = tuple(str(self.folder / name) for name in ('tile_10.tif', 'tile_2.tif'))
        images = [np.full((80, 90), value, np.uint16) for value in (40000, 42000)]
        for path, image in zip(self.paths, images):
            self.assertTrue(write_image(path, image))
        stitcher = SEMStitcher()
        stitcher.set_progress_callback(lambda message: None)
        self.panorama = stitcher._compose_images(images, [(0, 0), (40.375, 0)])
        self.request = StitchRequest(self.paths, (1, 2), False, True)
        self.report = StitchInspection(self.paths, ((80, 90),)*2, ((0, 0), (40.375, 0)),
                                       tuple(file_signature(p) for p in self.paths),
                                       (PairDiagnostic(0, 1, (-40.375, 0), (-40.375, 0), 50, 0, False),))
        self.restored = RestoredProject(self.request, self.panorama, self.report, {})

    def tearDown(self):
        for release in self.releases:
            release.set()
        if not self.app._closing:
            self.app._on_close()
        for thread in (self.app.task.thread, self.app.project_job.thread):
            if thread is not None:
                thread.join(timeout=3)
        self.assertEqual(self.errors, [], 'Unexpected Tk callback errors')

    def wait(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.assertTrue(predicate(), 'GUI operation timed out')
        self.app.update()

    def install_completed(self):
        self.app._apply_project(self.restored)
        self.app._render_state()

    def open_project(self):
        with patch('gui.filedialog.askopenfilename', return_value=str(self.path)):
            self.app.btn_project_open.invoke()

    def save_current(self):
        with patch('gui.filedialog.asksaveasfilename', return_value=str(self.path)):
            self.app.btn_project_save.invoke()
        self.wait(lambda: not self.app._busy)
        self.assertEqual(self.app.project_job.state, TaskState.SUCCEEDED)
        self.assertIsNone(self.app.project_job.result)

    def test_ready_save_open_restores_manual_order_and_grid_without_enabling_export(self):
        self.app._load_files(self.paths)
        self.app._reverse_files()
        self.app.rows.set('1')
        self.app.cols.set('2')
        self.app.use_grid.set(True)
        self.app.first_rtl.set(True)
        self.app._on_inputs_changed()
        expected = self.app._get_request()
        self.save_current()
        self.app._load_files(['other_1.tif', 'other_2.tif'])
        self.open_project()
        self.wait(lambda: not self.app._busy)
        self.assertEqual(self.app._get_request(), expected)
        self.assertEqual(self.app.task.state, TaskState.READY)
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')
        self.assertEqual(self.app.btn_inspect.cget('state'), 'disabled')
        self.assertIsNone(self.app.stitched_result)

    def test_completed_round_trip_restores_inspector_and_16bit_export(self):
        self.install_completed()
        self.save_current()
        self.app._load_files(['other_1.tif', 'other_2.tif'])
        with patch.object(SEMStitcher, '_extract_features', side_effect=AssertionError('must skip SIFT')):
            self.open_project()
            self.wait(lambda: not self.app._busy)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app.task.inspection, self.report)
        np.testing.assert_array_equal(self.app.stitched_result, self.panorama)
        self.assertEqual(self.app.btn_inspect.cget('state'), 'normal')
        self.app.btn_inspect.invoke()
        viewer = self.app._inspector
        self.wait(lambda: viewer._render_id is None)
        viewer.select_pair(0)
        self.wait(lambda: not viewer._loading and viewer._render_id is None)
        self.assertEqual(viewer._images[0].dtype, np.uint16)
        export = str(self.folder / 'export.tif')
        with patch('gui.filedialog.asksaveasfilename', return_value=export), patch('gui.messagebox.showinfo'):
            self.app.btn_export.invoke()
        np.testing.assert_array_equal(read_image(export), self.panorama)
        self.app._reverse_files()
        self.assertTrue(viewer.closed)
        self.assertIsNone(self.app.stitched_result)
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')

    def test_real_sift_save_open_and_restart_with_snake_first_rtl(self):
        from test_alignment_blending import texture
        scene = texture(480).astype(np.uint16) * 257
        paths = tuple(str(self.folder / f'scan_{i}.tif') for i in range(4))
        for path, (x, y) in zip(paths, ((160, 0), (0, 0), (0, 160), (160, 160))):
            self.assertTrue(write_image(path, scene[y:y+320, x:x+320]))
        self.app._load_files(paths)
        self.app.rows.set('2')
        self.app.cols.set('2')
        self.app.use_grid.set(True)
        self.app.serpentine.set(True)
        self.app.first_rtl.set(True)
        self.app._on_inputs_changed()
        self.app.btn_start.invoke()
        self.wait(lambda: not self.app._busy, timeout=15)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        original = self.app.stitched_result.copy()
        report = self.app.task.inspection
        self.save_current()
        self.app._load_files(self.paths)
        self.open_project()
        self.wait(lambda: not self.app._busy)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app.task.inspection, report)
        self.assertTrue(self.app.serpentine.get())
        self.assertTrue(self.app.first_rtl.get())
        self.assertEqual(self.app.image_paths, list(paths))
        np.testing.assert_array_equal(self.app.stitched_result, original)
        self.app._load_files(self.paths)
        self.assertEqual(self.app.task.state, TaskState.READY)
        self.assertEqual(self.app.btn_start.cget('state'), 'normal')

    def test_missing_source_open_failure_keeps_current_success_and_can_retry(self):
        self.install_completed()
        self.save_current()
        original = self.app.stitched_result
        source = Path(self.paths[0])
        before = source.read_bytes()
        source.unlink()
        with patch('gui.messagebox.showerror') as error:
            self.open_project()
            self.wait(lambda: not self.app._busy)
        error.assert_called_once()
        self.assertIn('missing', error.call_args.args[1])
        self.assertNotIn('Traceback', error.call_args.args[1])
        self.assertEqual(self.app.project_job.state, TaskState.FAILED)
        self.assertIs(self.app.stitched_result, original)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')
        self.assertEqual(self.app._get_request(), self.request)
        source.write_bytes(before)
        self.open_project()
        self.wait(lambda: not self.app._busy)
        self.assertEqual(self.app.project_job.state, TaskState.SUCCEEDED)

    def test_cancelled_open_keeps_current_session_and_blocks_other_actions(self):
        self.install_completed()
        original = self.app.stitched_result
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def blocked(*args):
            entered.set()
            release.wait(3)
            return self.restored

        with patch('gui.restore_project', side_effect=blocked):
            self.open_project()
            self.assertTrue(entered.wait(2))
            for button in (self.app.btn_select, self.app.btn_project_open, self.app.btn_project_save,
                           self.app.btn_start, self.app.btn_export, self.app.btn_inspect):
                self.assertEqual(button.cget('state'), 'disabled')
            self.app._load_files(['unwanted1.tif', 'unwanted2.tif'])
            self.app._on_start()
            self.assertEqual(self.app.image_paths, list(self.paths))
            self.app._set_lang('简体中文')
            self.assertIn('恢复项目', self.app.lbl_status.cget('text'))
            self.app.btn_cancel.invoke()
            release.set()
            self.wait(lambda: not self.app._busy)
        self.assertEqual(self.app.project_job.state, TaskState.CANCELLED)
        self.assertIs(self.app.stitched_result, original)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')
        self.assertIsNone(self.app.project_job.result)

    def test_dismissed_dialogs_keep_successful_session(self):
        self.install_completed()
        original = self.app.stitched_result
        with patch('gui.filedialog.askopenfilename', return_value=''), \
             patch('gui.filedialog.asksaveasfilename', return_value=''):
            self.app.btn_project_open.invoke()
            self.app.btn_project_save.invoke()
        self.assertIs(self.app.stitched_result, original)
        self.assertFalse(self.app._busy)

    def test_close_during_io_detaches_late_result_without_tk_callbacks(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def blocked(*args):
            entered.set()
            release.wait(3)
            return self.restored

        with patch('gui.restore_project', side_effect=blocked):
            self.open_project()
            self.assertTrue(entered.wait(2))
            self.app._on_close()
            release.set()
            self.app.project_job.thread.join(timeout=3)
        self.assertIsNone(self.app.project_job.result)
        self.assertTrue(self.app.project_job._events.empty())

    def test_project_controls_fit_in_minimum_window_in_all_languages(self):
        self.app._load_files(self.paths)
        self.app.deiconify()
        self.app.geometry('1050x700')
        for lang in ('en', 'zh-Hans', 'ja'):
            self.app.lang = lang
            self.app._apply_language()
            self.app.update()
            buttons = (self.app.btn_select, self.app.btn_project_open, self.app.btn_project_save, self.app.lbl_count)
            for widget in buttons:
                parent = widget.master
                with self.subTest(lang=lang, widget=widget):
                    self.assertGreater(widget.winfo_width(), 1)
                    self.assertLessEqual(widget.winfo_x() + widget.winfo_width(), parent.winfo_width())
            for left, right in zip(buttons, buttons[1:]):
                self.assertLessEqual(left.winfo_x() + left.winfo_width(), right.winfo_x())


if __name__ == '__main__':
    unittest.main()
