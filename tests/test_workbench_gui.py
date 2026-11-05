"""Native layout/theme checks and scientific preview invariants."""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

try:
    import customtkinter as ctk
    from gui import SEMStitcherGUI
    from inspection import display_gray, display_range
    from stitching_task import TaskState
    import ui_theme as theme
except ImportError as error:
    raise unittest.SkipTest(f'GUI dependencies unavailable: {error}')


@unittest.skipIf(sys.platform.startswith('linux') and not os.environ.get('DISPLAY'), 'Tk needs a display')
class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.app = SEMStitcherGUI()
        self.app.withdraw()
        self.errors = []
        self.app.report_callback_exception = lambda kind, error, trace: self.errors.append(str(error))
        self.app._load_files([f'tile_{i}.tif' for i in range(9)])
        self.app.rows.set('3')
        self.app.cols.set('3')
        self.app.use_grid.set(True)
        self.app._on_inputs_changed()

    def tearDown(self):
        if not self.app._closing:
            self.app._on_close()
        self.assertEqual(self.errors, [])

    def show(self, size='1050x700'):
        self.app.deiconify()
        self.app.geometry(size)
        self.app.update()

    def success(self, image=None):
        image = image if image is not None else np.full((180, 320), 42000, np.uint16)
        self.app.task.result = image
        self.app.task.state = TaskState.SUCCEEDED
        self.app._update_result_metadata()
        self.app._render_state()
        return image

    def test_preview_dominates_minimum_window_in_all_languages_and_themes(self):
        self.show()
        for mode in ('Light', 'Dark'):
            self.app._set_theme_mode(mode)
            for lang in ('en', 'zh-Hans', 'ja'):
                self.app.lang = lang
                self.app._apply_language()
                self.app.update()
                with self.subTest(mode=mode, lang=lang):
                    self.assertGreater(self.app.preview_label.winfo_width(), self.app.winfo_width() * .6)
                    self.assertGreater(self.app.preview_label.winfo_height(), 300)
                    self.assertLess(self.app.sidebar.winfo_width(), self.app.winfo_width() * .33)
                    self.assertGreater(self.app.file_list.winfo_height(), 60)

    def test_expanded_log_and_layout_fit_together_without_hiding_files(self):
        self.show()
        self.app._set_log_open(True)
        # A grid error must remain visible even when both sections are expanded.
        self.app.rows.set('4')
        self.app._on_inputs_changed()
        for lang in ('en', 'zh-Hans', 'ja'):
            self.app.lang = lang
            self.app._apply_language()
            self.app.update()
            with self.subTest(lang=lang):
                self.assertGreater(self.app.file_list.winfo_height(), 30)
                self.assertGreater(self.app.preview_label.winfo_height(), 190)
                for widget in (self.app.layout_text, self.app.file_list, self.app.rows_entry, self.app.lbl_validation,
                               self.app.check_rtl, self.app.btn_export, self.app.theme_menu, self.app.lang_menu):
                    self.assertTrue(widget.winfo_ismapped())
                    self.assertGreaterEqual(widget.winfo_rootx(), self.app.winfo_rootx())
                    self.assertGreaterEqual(widget.winfo_rooty(), self.app.winfo_rooty())
                    self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),
                                         self.app.winfo_rootx()+self.app.winfo_width())
                    self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),
                                         self.app.winfo_rooty()+self.app.winfo_height())

    def test_fold_sections_preserves_result_settings_and_logs(self):
        self.show()
        source = self.success()
        before = self.app._get_request()
        dirty = self.app._project_dirty
        self.app._append_log('first line')
        self.app._append_log('second line')
        self.app.btn_layout_toggle.invoke()
        self.app.btn_log_toggle.invoke()
        self.app.update()
        self.assertFalse(self.app.layout_panel.winfo_ismapped())
        self.assertTrue(self.app.log_text.winfo_ismapped())
        self.assertIn('second line', self.app.log_text.get('1.0', 'end'))
        self.app.btn_layout_toggle.invoke()
        self.app.btn_log_toggle.invoke()
        self.assertEqual(self.app._get_request(), before)
        self.assertIs(self.app.stitched_result, source)
        self.assertEqual(self.app.task.state, TaskState.SUCCEEDED)
        self.assertEqual(self.app._project_dirty, dirty)
        self.assertEqual(self.app.btn_export.cget('state'), 'normal')

    def test_theme_switch_styles_native_file_list_without_changing_result(self):
        source = self.success()
        request = self.app._get_request()
        for mode in ('Light', 'Dark'):
            self.app._set_theme_mode(mode)
            self.app.update()
            self.assertEqual(self.app.file_list.cget('background'), theme.native_color(theme.SURFACE, mode))
            self.assertEqual(self.app.file_list.cget('foreground'), theme.native_color(theme.TEXT, mode))
            self.assertEqual(self.app.file_list.cget('selectforeground'), theme.native_color(theme.PRIMARY_TEXT, mode))
            self.assertEqual(self.app._get_request(), request)
            self.assertIs(self.app.stitched_result, source)
            self.assertEqual(self.app.btn_export.cget('state'), 'normal')

    def test_system_theme_changes_reach_native_controls_and_language_menu(self):
        with patch.object(ctk.AppearanceModeTracker, 'detect_appearance_mode', return_value=0):
            self.app._set_theme_mode('System')
        self.assertEqual(self.app.file_list.cget('background'), theme.SURFACE[0])
        with patch.object(ctk.AppearanceModeTracker, 'detect_appearance_mode', return_value=1):
            ctk.AppearanceModeTracker.init_appearance_mode()
        self.assertEqual(self.app.file_list.cget('background'), theme.SURFACE[1])
        self.app._set_lang('简体中文')
        self.assertEqual(self.app.theme_mode, 'System')
        self.assertEqual(self.app.theme_menu.get(), '跟随系统')
        self.app._set_theme('浅色')
        self.assertEqual(self.app.theme_mode, 'Light')

    def test_primary_action_changes_from_stitch_to_export_after_success(self):
        self.assertEqual(self.app.btn_start.cget('fg_color'), theme.PRIMARY)
        self.assertEqual(self.app.btn_export.cget('fg_color'), theme.SURFACE)
        self.success()
        self.assertEqual(self.app.btn_start.cget('fg_color'), theme.SURFACE)
        self.assertEqual(self.app.btn_export.cget('fg_color'), theme.PRIMARY)
        self.assertIn('16', self.app.lbl_size.cget('text'))

    def test_large_preview_uses_workspace_and_only_normalizes_a_display_sized_copy(self):
        self.show('1600x950')
        source = np.arange(2000*4000, dtype=np.uint16).reshape(2000, 4000)
        original = source.copy()
        self.success(source)
        self.app._preview_key = None
        with patch('gui.display_range', wraps=display_range) as limits, \
             patch('gui.display_gray', wraps=display_gray) as gray:
            self.app._update_preview()
            self.app._update_preview()
        width, height = self.app.preview_label.image.cget('size')
        self.assertGreater(width, 700)
        self.assertLessEqual(width, self.app.preview_label.winfo_width()-24)
        self.assertLessEqual(height, self.app.preview_label.winfo_height()-24)
        self.assertEqual(gray.call_args.args[0].shape, (height, width))
        self.assertEqual(limits.call_count, 1)
        np.testing.assert_array_equal(source, original)

    def test_nonzero_constant_16bit_preview_is_visible_and_input_changes_clear_cache(self):
        self.show()
        source = self.success()
        self.app._update_preview()
        self.assertEqual(self.app.preview_label.image.cget('light_image').getextrema(), (128, 128))
        self.app._reverse_files()
        self.assertIsNone(self.app.stitched_result)
        self.assertIsNone(self.app._preview_limits)
        self.assertIsNone(self.app._preview_key)
        np.testing.assert_array_equal(source, np.full(source.shape, 42000, np.uint16))


if __name__ == '__main__':
    unittest.main()
