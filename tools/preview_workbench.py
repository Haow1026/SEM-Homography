"""Open a synthetic workbench demo, or capture native UI screenshots under Xvfb."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from gui import SEMStitcherGUI
from image_io import write_image
from inspection import StitchInspection, file_signature
from main import SEMStitcher
from project import RestoredProject
from stitching_task import StitchRequest


def demo(directory):
    rng = np.random.default_rng(58)
    scene = cv2.GaussianBlur(rng.uniform(0, 65535, (768, 1024)).astype(np.float32), (0, 0), 1.6)
    scene = cv2.normalize(scene, None, 3000, 58000, cv2.NORM_MINMAX).astype(np.uint16)
    paths, images, positions = [], [], []
    for row in range(3):
        for col in (range(3) if row % 2 == 0 else reversed(range(3))):
            x, y = col * 256, row * 192
            image = scene[y:y+384, x:x+512].copy()
            path = str(directory / f'demo_{len(paths)+1:03d}.tif')
            if not write_image(path, image):
                raise OSError(path)
            paths.append(path)
            images.append(image)
            positions.append((x, y))
    stitcher = SEMStitcher()
    stitcher.set_progress_callback(lambda message: None)
    panorama = stitcher._compose_images(images, positions)
    request = StitchRequest(tuple(paths), (3, 3), True)
    report = StitchInspection(tuple(paths), tuple(image.shape for image in images), tuple(positions),
                              tuple(file_signature(path) for path in paths), ())
    return RestoredProject(request, panorama, report, {})


def prepare(restored, language, mode, compact=False, ready=False):
    app = SEMStitcherGUI()
    app.lang = language
    app._apply_language()
    app._set_theme_mode(mode)
    app.geometry(('1050x700' if compact else '1240x820') + '+20+20')
    app._apply_project(restored if not ready else RestoredProject(restored.request, None, None, {}))
    app._session_path = 'Synthetic demo.semproj'
    app._project_dirty = False
    app._refresh_session()
    app._append_log('Synthetic UI demo — no microscope images or measured matches.')
    app._append_log('9 grayscale source tiles · 3×3 serpentine layout · uint16 output.')
    app._append_log('Preview and original-resolution seam inspection are display-only.')
    app._render_state()
    if compact:
        app._set_log_open(True)
    app.update()
    time.sleep(.16)
    app.update()
    if app.stitched_result is not None:
        app._update_preview()
    app.update()
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='capture six theme/language views, a minimum-window view and a ready view')
    parser.add_argument('--language', default='zh-Hans', choices=('en', 'zh-Hans', 'ja'))
    parser.add_argument('--theme', default='Dark', choices=('System', 'Light', 'Dark'))
    parser.add_argument('--check', action='store_true', help='build and close all views without taking screenshots')
    args = parser.parse_args()
    if args.output and not args.check and (not sys.platform.startswith('linux') or not os.environ.get('DISPLAY')):
        parser.error('screenshots require Linux with DISPLAY; use xvfb-run or --check')
    with tempfile.TemporaryDirectory() as directory:
        restored = demo(Path(directory))
        if not args.output and not args.check:
            app = prepare(restored, args.language, args.theme)
            app.mainloop()
            return
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
        cases = [(lang, mode, False, False) for lang in ('en', 'zh-Hans', 'ja') for mode in ('Light', 'Dark')]
        cases += [('en', 'Dark', True, False), ('zh-Hans', 'Light', False, True)]
        for language, mode, compact, ready in cases:
            app = prepare(restored, language, mode, compact, ready)
            try:
                name = f'{language}-{mode.lower()}' + ('-minimum-log' if compact else '-ready' if ready else '')
                if args.output and not args.check:
                    from PIL import ImageGrab
                    bounds = (app.winfo_rootx(), app.winfo_rooty(),
                              app.winfo_rootx()+app.winfo_width(), app.winfo_rooty()+app.winfo_height())
                    ImageGrab.grab(bbox=bounds).save(args.output / (name + '.png'))
                print(f'{name}: {app.winfo_width()}x{app.winfo_height()}, '
                      f'preview {app.preview_label.winfo_width()}x{app.preview_label.winfo_height()}')
            finally:
                app._on_close()


if __name__ == '__main__':
    main()
