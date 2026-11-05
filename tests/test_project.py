import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from inspection import PairDiagnostic, StitchInspection, file_signature
from image_io import write_image
from main import SEMStitcher, StitchingCancelled
from project import restore_project, save_project
from stitching_task import StitchRequest


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bundle = Path(self.directory.name).resolve() / '采集项目'
        self.bundle.mkdir()
        self.paths = tuple(str(self.bundle / name) for name in
                           ('tile_10.tif', 'tile_2.tif', 'tile_11.tif', 'tile_1.tif'))
        self.path = self.bundle / '拼接.semproj'
        self.request = StitchRequest(self.paths, (2, 2), True, True)
        self.stitcher = SEMStitcher()
        self.stitcher.set_progress_callback(lambda message: None)
        self.make_result(np.uint16)

    def make_result(self, dtype):
        rng = np.random.default_rng(91)
        images = [rng.integers(0, np.iinfo(dtype).max + 1, (80, 92), dtype=dtype) for _ in self.paths]
        for path, image in zip(self.paths, images):
            self.assertTrue(write_image(path, image))
        positions = ((0.375, 0.625), (45.75, 0.625), (0.375, 39.875), (45.75, 39.875))
        self.result = self.stitcher._compose_images(images, positions)
        pairs = []
        for i, j, excluded in ((1, 0, False), (0, 2, False), (2, 3, False), (3, 1, True)):
            predicted = tuple(float(v) for v in np.subtract(positions[i], positions[j]))
            measured = (predicted[0] + 2.5, predicted[1] - 1.25)
            pairs.append(PairDiagnostic(i, j, measured, predicted, 50, float(np.hypot(2.5, 1.25)), excluded))
        self.report = StitchInspection(self.paths, tuple(image.shape for image in images), positions,
                                       tuple(file_signature(path) for path in self.paths), tuple(pairs))

    def save_completed(self):
        save_project(self.path, self.request, self.report, self.result)

    def test_completed_round_trip_both_depths_skips_matching_and_preserves_diagnostics(self):
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                self.make_result(dtype)
                self.save_completed()
                with patch.object(SEMStitcher, '_extract_features', side_effect=AssertionError('must not match')), \
                     patch.object(SEMStitcher, 'stitch_sequence', side_effect=AssertionError('must not restitch')):
                    restored = restore_project(self.path)
                self.assertEqual(restored.request, self.request)
                np.testing.assert_array_equal(restored.panorama, self.result)
                self.assertEqual(restored.panorama.dtype, dtype)
                self.assertEqual(restored.inspection, self.report)
                self.assertTrue(restored.metrics['restored_project'])
                _, pair = restored.inspection.load_pair(0)
                self.assertEqual(pair[0].dtype, dtype)

    def test_ready_project_restores_file_order_and_all_four_scan_patterns(self):
        for snake in (False, True):
            for rtl in (False, True):
                request = StitchRequest(self.paths, (2, 2), snake, rtl)
                save_project(self.path, request)
                with patch('project.read_image', side_effect=AssertionError('ready projects need no decode')):
                    restored = restore_project(self.path)
                self.assertEqual(restored.request, request)
                self.assertEqual(restored.request.layout_indices(), request.layout_indices())
                self.assertIsNone(restored.panorama)
                self.assertIsNone(restored.inspection)
        save_project(self.path, StitchRequest(self.paths))
        self.assertIsNone(restore_project(self.path).request.grid_shape)

    def test_relative_bundle_can_move_and_timestamps_can_change_without_changing_pixels(self):
        self.save_completed()
        document = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertTrue(all(not Path(source['path']).is_absolute() for source in document['sources']))
        moved = Path(self.directory.name).resolve() / 'moved'
        shutil.copytree(self.bundle, moved)
        shutil.rmtree(self.bundle)
        for path in moved.glob('*.tif'):
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
        restored = restore_project(moved / self.path.name)
        self.assertEqual(restored.request.paths, tuple(str(moved / Path(p).name) for p in self.paths))
        np.testing.assert_array_equal(restored.panorama, self.result)
        restored.inspection.load_pair(0)  # New signatures, not the old timestamps.
        save_project(moved / 'again.semproj', restored.request, restored.inspection, restored.panorama)

    def test_missing_source_is_reported(self):
        self.save_completed()
        Path(self.paths[0]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'missing'):
            restore_project(self.path)

    def test_same_size_same_mtime_content_change_is_detected(self):
        self.save_completed()
        source = Path(self.paths[0])
        info = source.stat()
        data = bytearray(source.read_bytes())
        data[-1] ^= 1
        source.write_bytes(data)
        os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'changed'):
            restore_project(self.path)

    def test_changed_source_cannot_be_saved_as_the_old_completed_result(self):
        write_image(self.paths[0], np.ones((80, 92), np.uint16))
        with self.assertRaisesRegex(ValueError, 'changed since stitching'):
            self.save_completed()
        self.assertFalse(self.path.exists())

    def test_cannot_overwrite_a_source_image(self):
        before = Path(self.paths[0]).read_bytes()
        with self.assertRaisesRegex(ValueError, 'overwrite a source'):
            save_project(self.paths[0], self.request)
        self.assertEqual(Path(self.paths[0]).read_bytes(), before)

    def test_failed_atomic_replace_preserves_previous_project_and_removes_temporary_file(self):
        self.save_completed()
        before = self.path.read_bytes()
        with patch('project.os.replace', side_effect=OSError('disk unavailable')):
            with self.assertRaisesRegex(OSError, 'disk unavailable'):
                save_project(self.path, self.request)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.bundle.glob('.*.tmp')), [])

    def test_cancel_after_writing_temporary_file_preserves_previous_project(self):
        self.save_completed()
        before = self.path.read_bytes()
        cancel = threading.Event()
        fsync = os.fsync

        def cancel_at_commit(fd):
            fsync(fd)
            cancel.set()

        with patch('project.os.fsync', side_effect=cancel_at_commit):
            with self.assertRaises(StitchingCancelled):
                save_project(self.path, self.request, cancelled=cancel.is_set)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.bundle.glob('.*.tmp')), [])

    def test_cancel_during_restore_composition_does_not_return_a_partial_result(self):
        self.save_completed()
        cancel = threading.Event()
        compose = SEMStitcher._compose_images

        def cancel_after_compose(stitcher, *args):
            result = compose(stitcher, *args)
            cancel.set()
            return result

        with patch.object(SEMStitcher, '_compose_images', new=cancel_after_compose):
            with self.assertRaises(StitchingCancelled):
                restore_project(self.path, cancelled=cancel.is_set)

    def test_source_change_during_later_source_read_is_detected(self):
        from project import _digest

        def change_earlier_source(path, cancelled):
            digest = _digest(path, cancelled)
            if path == self.paths[-1]:
                os.utime(self.paths[0], ns=(1, 1))
            return digest

        for completed in (False, True):
            self.make_result(np.uint16)
            if completed:
                self.save_completed()
            else:
                save_project(self.path, self.request)
            with self.subTest(completed=completed), patch('project._digest', side_effect=change_earlier_source):
                with self.assertRaisesRegex(ValueError, 'changed while restoring'):
                    restore_project(self.path)

    def test_malformed_metadata_is_rejected_before_source_decode(self):
        self.save_completed()
        original = json.loads(self.path.read_text(encoding='utf-8'))
        mutations = [
            lambda d: d.update(version=2),
            lambda d: d.update(version=True),
            lambda d: d['scan'].update(grid_shape=[3, 3]),
            lambda d: d['scan'].update(serpentine='yes'),
            lambda d: d['alignment'].update(blend='unknown'),
            lambda d: d['alignment'].update(dtype='float64'),
            lambda d: d['alignment']['positions'].__setitem__(0, [float('nan'), 0]),
            lambda d: d['alignment']['positions'].__setitem__(0, [1e100, 0]),
            lambda d: d['alignment']['shapes'].__setitem__(0, [-1, 92]),
            lambda d: d['alignment']['pairs'][0].update(i=99),
            lambda d: d['alignment']['pairs'][0].update(predicted=[0, 0]),
            lambda d: d['alignment']['output'].update(shape=[1, 1]),
            lambda d: d['sources'][0].update(sha256='bad'),
        ]
        for mutation in mutations:
            document = copy.deepcopy(original)
            mutation(document)
            self.path.write_text(json.dumps(document), encoding='utf-8')
            with self.subTest(document=document), patch('project.read_image') as read:
                with self.assertRaises(ValueError):
                    restore_project(self.path)
                read.assert_not_called()

    def test_output_pixel_mismatch_is_rejected(self):
        self.save_completed()
        document = json.loads(self.path.read_text(encoding='utf-8'))
        document['alignment']['output']['sha256'] = '0'*64
        self.path.write_text(json.dumps(document), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pixels differ'):
            restore_project(self.path)

    def test_invalid_json_and_oversized_files_are_rejected(self):
        self.path.write_text('{ broken', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Cannot read SEM project'):
            restore_project(self.path)
        self.path.write_text('too much metadata', encoding='utf-8')
        with patch('project.MAX_PROJECT_BYTES', 8):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                restore_project(self.path)


if __name__ == '__main__':
    unittest.main()
