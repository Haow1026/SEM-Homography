"""Entry point for the standalone GUI and its packaged smoke check."""
import json
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np

from gui import SEMStitcherGUI
from image_io import read_image, write_image
from stitching_task import TaskState


def smoke_test():
    """Exercise the frozen runtime with real image I/O, SIFT and Tk widgets."""
    errors = []
    app = SEMStitcherGUI()
    app.report_callback_exception = lambda kind, error, trace: errors.append(str(error))
    app.geometry('1050x700')

    def wait_for(predicate, description):
        deadline = time.monotonic() + 25
        while True:
            app.update()
            if errors:
                raise RuntimeError('; '.join(errors))
            if predicate():
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Timed out: {description}')
            time.sleep(.02)

    try:
        app._set_theme_mode('Dark')
        app._set_lang('简体中文')
        wait_for(lambda: app.preview_label.winfo_ismapped() and app.preview_label.winfo_width() > 1,
                 'visible workbench')
        with tempfile.TemporaryDirectory(prefix='sem-smoke-') as temp:
            folder = Path(temp) / '显微图像'
            folder.mkdir()
            cv2.setNumThreads(1)
            rng = np.random.default_rng(812)
            texture = rng.integers(1000, 63000, (384, 576), dtype=np.uint16)
            scene = cv2.GaussianBlur(texture, (5, 5), 1.1)
            paths = [folder / '源图_01.tif', folder / '源图_02.tif']
            for path, image in zip(paths, (scene[:, :384], scene[:, 192:])):
                if not write_image(path, image):
                    raise OSError(f'Cannot write {path.name}')
            app._load_files(paths)
            app.btn_start.invoke()
            wait_for(lambda: not app._busy, 'real SIFT stitching')
            if app.task.state != TaskState.SUCCEEDED:
                raise RuntimeError(app.task.error or 'Stitching did not succeed')
            result = app.stitched_result
            if result.dtype != np.uint16 or result.shape[1] < 570:
                raise RuntimeError(f'Unexpected panorama: {result.shape}, {result.dtype}')

            app.btn_inspect.invoke()
            inspector = app._inspector
            inspector.select_pair(0)
            wait_for(lambda: not inspector._loading, 'source pair load')
            if inspector._images is None:
                raise RuntimeError(inspector._error or 'No images in source pair')
            inspector.set_mode('side')
            wait_for(lambda: inspector._photo is not None and inspector._render_id is None,
                     'side-by-side render')

            output = folder / '导出全景.tif'
            if not write_image(output, result) or not np.array_equal(read_image(output), result):
                raise RuntimeError('16-bit TIFF export changed pixels')
            return dict(ok=True, frozen=getattr(sys, 'frozen', False), machine=platform.machine(),
                        bits=struct.calcsize('P') * 8, os=platform.platform(),
                        shape=list(result.shape), dtype=str(result.dtype),
                        checks=['workbench', 'real SIFT stitch', 'seam comparison', '16-bit TIFF round trip'])
    finally:
        app._on_close()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--smoke-test':
        try:
            report = smoke_test()
        except Exception:
            report = dict(ok=False, error=traceback.format_exc())
        Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        raise SystemExit(0 if report['ok'] else 1)
    SEMStitcherGUI().mainloop()
