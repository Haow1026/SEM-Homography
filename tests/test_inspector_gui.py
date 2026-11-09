import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

try:
    from gui import SEMStitcherGUI
    from inspection import PairDiagnostic, StitchInspection, file_signature
    from stitching_task import TaskState
except ImportError as error:
    raise unittest.SkipTest(f'GUI dependencies unavailable: {error}')


@unittest.skipIf(sys.platform.startswith('linux') and not os.environ.get('DISPLAY'), 'Tk needs Xvfb/display')
class InspectorGUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.app = SEMStitcherGUI()
        self.addCleanup(lambda: None if self.app._closing else self.app._on_close())
        self.app.withdraw()
        self.errors = []
        self.app.report_callback_exception = lambda kind, error, trace: self.errors.append(str(error))
        y, x = np.mgrid[:512, :1040]
        self.panorama = (1000 + (x * 21 + y * 17) % 50000).astype(np.uint16)
        self.paths = tuple(str(Path(self.directory.name) / f'{i}.tif') for i in range(2))
        cv2.imwrite(self.paths[0], self.panorama[:, :640])
        cv2.imwrite(self.paths[1], self.panorama[:, 400:])
        self.report = StitchInspection(self.paths, ((512, 640), (512, 640)), ((0, 0), (400, 0)),
                                       tuple(file_signature(path) for path in self.paths),
                                       (PairDiagnostic(0, 1, (-400, 0), (-400, 0), 100, 0, False),
                                        PairDiagnostic(0, 1, (-469, 0), (-400, 0), 120, 69, True)))
        self.app.image_paths = list(self.paths)
        self.app.task.state = TaskState.SUCCEEDED
        self.app.task.result = self.panorama
        self.app.task.inspection = self.report
        self.app._refresh_layout()
        self.app._render_state()
        self.app._on_inspect()
        self.viewer = self.app._inspector
        self.app.deiconify()
        self.app.geometry('1050x700')
        self.wait(lambda: self.viewer._photo is not None and self.viewer._render_id is None)

    def tearDown(self):
        loader = self.viewer._loader
        if not self.app._closing:
            self.app._on_close()
        if loader is not None:
            loader.join(timeout=3)
        self.assertEqual(self.errors, [])

    def wait(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.assertTrue(predicate(), 'GUI operation did not complete')

    def choose(self, index):
        self.viewer.select_pair(index)
        self.wait(lambda: not self.viewer._loading and self.viewer._render_id is None)

    def test_panorama_100_percent_zoom_and_drag_do_not_change_output(self):
        before = self.panorama.copy()
        self.viewer.zoom(1.0)
        self.wait(lambda: self.viewer._render_id is None)
        self.assertEqual(self.viewer.viewport.zoom, 1.0)
        self.assertEqual(self.viewer._photo.width(), self.viewer.viewport.view_size[0])
        self.viewer.zoom(2.0)
        origin = self.viewer.viewport.origin.copy()
        self.viewer._drag_start(SimpleNamespace(x=100, y=100))
        self.viewer._drag(SimpleNamespace(x=80, y=80))
        self.assertTrue(np.all(self.viewer.viewport.origin >= origin))
        np.testing.assert_array_equal(self.app.stitched_result, before)

    def test_pair_load_overlay_and_alternation(self):
        self.choose(0)
        self.assertTrue(all(image.dtype == np.uint16 for image in self.viewer._images))
        self.assertEqual(self.viewer.viewport.zoom, 1.0)
        self.assertEqual(self.viewer.mode_button.cget('state'), 'normal')
        self.viewer.set_mode('alternate')
        self.wait(lambda: self.viewer._alternate_b)
        self.viewer.set_mode('b')
        self.assertIsNone(self.viewer._blink_id)
        self.choose(None)
        self.assertIsNone(self.viewer._images)
        self.assertIsNone(self.viewer._blink_id)

    def test_excluded_pair_is_labelled_and_language_can_change(self):
        self.choose(1)
        self.assertTrue(self.viewer._pair.excluded)
        self.app._set_lang('简体中文')
        self.assertIn('已排除', self.viewer.info_label.cget('text'))
        self.assertIn('69.00', self.viewer.info_label.cget('text'))
        self.assertIn('已排除', self.viewer.pair_menu.get())

    def test_source_change_is_reported_and_panorama_remains_available(self):
        cv2.imwrite(self.paths[0], np.ones((20, 30), np.uint8))
        self.choose(0)
        self.assertIn('changed', self.viewer._error)
        self.assertIsNone(self.viewer._images)
        self.choose(None)
        self.assertIsNotNone(self.viewer._photo)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)

    def test_new_inputs_close_inspector_and_release_source_arrays(self):
        self.choose(0)
        self.app._load_files(['next_1.tif', 'next_2.tif'])
        self.assertTrue(self.viewer.closed)
        self.assertIsNone(self.viewer._images)
        self.assertIsNone(self.viewer.panorama)
        self.assertIsNone(self.app.task.inspection)
        self.assertEqual(self.app.btn_inspect.cget('state'), 'disabled')
        self.assertEqual(self.app.btn_export.cget('state'), 'disabled')

    def test_window_can_close_during_source_loading_and_reopen(self):
        original = StitchInspection.load_pair
        import threading
        entered, release = threading.Event(), threading.Event()

        def blocked(report, index):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test loader timed out')
            return original(report, index)

        with patch.object(StitchInspection, 'load_pair', blocked):
            self.viewer.select_pair(0)
            self.assertTrue(entered.wait(2))
            self.viewer.close()
            release.set()
            self.viewer._loader.join(timeout=3)
        self.assertTrue(self.viewer.closed)
        self.app._on_inspect()
        self.assertIsNot(self.app._inspector, self.viewer)
        self.app._inspector.close()

    def test_reopening_existing_window_reuses_it(self):
        self.app._on_inspect()
        self.assertIs(self.app._inspector, self.viewer)

    def test_minimum_window_controls_fit_in_all_languages(self):
        self.choose(1)
        self.app.geometry('1050x700')
        self.app._set_log_open(True)
        for language in ('en', 'zh-Hans', 'ja'):
            self.viewer.set_language(language)
            self.app.update()
            for widget in (self.viewer.pair_menu, self.viewer.mode_button, self.viewer.fit_button,
                           self.viewer.actual_button, self.viewer.zoom_label, self.viewer.canvas,
                           self.viewer.bounds_button, self.viewer.help_label):
                with self.subTest(language=language, widget=widget.__class__.__name__):
                    self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                         self.viewer.winfo_rootx() + self.viewer.winfo_width())
                    self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                         self.viewer.winfo_rooty() + self.viewer.winfo_height())

    def test_embedded_panel_can_return_without_losing_pair_view_or_result(self):
        self.choose(0)
        self.viewer.set_mode('alternate')
        self.assertIs(self.viewer.winfo_toplevel(), self.app)
        self.assertTrue(self.app._inspection_visible)
        images = self.viewer._images
        origin = self.viewer.viewport.origin.copy()
        self.app.btn_inspect.invoke()
        self.assertFalse(self.app._inspection_visible)
        self.assertFalse(self.viewer.active)
        self.assertIsNone(self.viewer._blink_id)
        self.assertIsNone(self.viewer._render_id)
        self.assertIs(self.viewer._images, images)
        self.assertIs(self.app.stitched_result, self.panorama)
        self.app.btn_inspect.invoke()
        self.app.update()
        self.assertTrue(self.viewer.active)
        self.assertIs(self.app._inspector, self.viewer)
        np.testing.assert_array_equal(self.viewer.viewport.origin, origin)
        self.assertIsNotNone(self.viewer._blink_id)

    def test_side_by_side_shares_pixel_coordinates_and_keeps_five_pixel_gap(self):
        self.choose(0)
        before = self.panorama.copy()
        self.viewer.set_mode('side')
        self.viewer.show_bounds.set(False)
        self.viewer.viewport.center((520, 260), zoom=1.0)
        self.wait(lambda: self.viewer._render_id is None)
        from PIL import ImageTk
        from inspection import pair_frame
        shown = np.asarray(ImageTk.getimage(self.viewer._photo).convert('RGB'))
        width = self.viewer.viewport.view_size[0]
        expected = pair_frame(self.viewer._images, [(0, 0), (400, 0)], self.viewer.viewport, 'a', self.viewer._limits)
        np.testing.assert_array_equal(shown[30:, :width], expected[30:])
        expected = pair_frame(self.viewer._images, [(0, 0), (400, 0)], self.viewer.viewport, 'b', self.viewer._limits)
        np.testing.assert_array_equal(shown[30:, width+5:width*2+5], expected[30:])
        gap = np.tile(np.array([37, 42, 49], np.uint8), (shown.shape[0], 5, 1))
        np.testing.assert_array_equal(shown[:, width:width+5], gap)
        # Wheel zoom in the right pane must anchor the same source coordinate.
        world = self.viewer.viewport.origin + [60, 80]
        self.viewer._wheel(SimpleNamespace(x=width+5+60, y=80, delta=120))
        self.assertEqual(self.viewer.viewport.zoom, 1.25)
        np.testing.assert_allclose(self.viewer.viewport.origin + np.array([60, 80])/1.25, world)
        np.testing.assert_array_equal(self.panorama, before)

    def test_hidden_panel_finishes_source_loading_without_rendering_or_blinking(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        load = StitchInspection.load_pair

        def blocked(report, index):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('loader timeout')
            return load(report, index)

        with patch.object(StitchInspection, 'load_pair', blocked):
            self.viewer.select_pair(0)
            self.assertTrue(entered.wait(2))
            try:
                self.app.btn_inspect.invoke()
                release.set()
                self.wait(lambda: not self.viewer._loading)
                self.assertIsNone(self.viewer._render_id)
                self.assertIsNone(self.viewer._blink_id)
                self.assertFalse(self.app._inspection_visible)
                self.assertIs(self.app.stitched_result, self.panorama)
            finally:
                release.set()
                self.viewer._loader.join(3)
        self.app.btn_inspect.invoke()
        self.wait(lambda: self.viewer._render_id is None)
        self.assertIsNotNone(self.viewer._photo)


if __name__ == '__main__':
    unittest.main()
