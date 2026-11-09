"""Read-only full-resolution inspection panel embedded in the workbench."""
import os
import queue
import threading
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

from inspection import Viewport, display_gray, display_range, pair_frame, sample_view
import ui_theme as theme
from workbench import button_style


TEXT = {
    'en': dict(title='Seam inspection', panorama='Panorama', overlay='Red/cyan overlay', alternate='Alternate A/B',
               fit='Fit', boundaries='Tile boundaries / excluded links', excluded='Excluded', retained='Retained',
               loading='Loading source pair…', error='Cannot load source pair',
               help='Drag to pan · wheel to zoom · 100% = one image pixel per canvas pixel. Display only; export is unchanged.',
               info='{status} · inliers {inliers} · residual {residual:.2f} px\nMeasured Δ=({mx:.2f}, {my:.2f}); final Δ=({px:.2f}, {py:.2f})'),
    'zh-Hans': dict(title='接缝检查', panorama='全景图', overlay='红青叠加', alternate='交替显示 A/B',
                    fit='适应窗口', boundaries='图像边界 / 已排除连线', excluded='已排除', retained='已保留',
                    loading='正在加载原始图像对…', error='无法加载原始图像对',
                    help='拖动平移 · 滚轮缩放 · 100% 为图像与画布像素一一对应。仅改变显示，不影响导出。',
                    info='{status} · 内点 {inliers} · 残差 {residual:.2f} 像素\n测量位移 Δ=({mx:.2f}, {my:.2f})；最终位移 Δ=({px:.2f}, {py:.2f})'),
    'ja': dict(title='継ぎ目を確認', panorama='全景画像', overlay='赤・シアン重ね合わせ', alternate='A/B を交互表示',
               fit='全体表示', boundaries='画像境界 / 除外した接続', excluded='除外', retained='採用',
               loading='元画像を読み込み中…', error='元画像を読み込めません',
               help='ドラッグで移動 · ホイールで拡大 · 100% は画像とキャンバスの画素が1対1です。表示のみ変更します。',
               info='{status} · インライア {inliers} · 残差 {residual:.2f} px\n測定位移 Δ=({mx:.2f}, {my:.2f}); 最終位移 Δ=({px:.2f}, {py:.2f})'),
}
for language, label in (('en', 'A | B side by side'), ('zh-Hans', 'A/B 并排对照'), ('ja', 'A/B 並列表示')):
    TEXT[language]['side'] = label


def _load_pair(report, index, generation, events):
    # This worker never references Tk objects, including after window closure.
    try:
        pair, images = report.load_pair(index)
        events.put((generation, pair, images, None))
    except Exception as error:
        events.put((generation, None, None, str(error)))


class SeamInspector(ctk.CTkFrame):
    def __init__(self, parent, panorama, report, lang='en'):
        super().__init__(parent)
        self.configure(fg_color='transparent', width=1, height=1)
        self.report = report
        self.panorama = panorama
        self.lang = lang
        self.closed = False
        self.active = True
        self.comparison_gap = 5
        self._render_id = self._blink_id = self._load_id = None
        self._photo = None
        self._pair_index = None
        self._pair = None
        self._images = None
        self._limits = display_range([panorama])
        self._mode = 'overlay'
        self._alternate_b = False
        self._generation = 0
        self._events = queue.Queue()
        self._loading = False
        self._loader = None
        self._error = ''
        self._fit_mode = True
        self._drag_at = None
        self.viewport = Viewport((panorama.shape[1], panorama.shape[0]))
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.pair_menu = ctk.CTkOptionMenu(self, values=[''], command=self._choose_label, width=340,
                                          fg_color=theme.SUBTLE, button_color=theme.SUBTLE,
                                          button_hover_color=theme.BORDER, text_color=theme.TEXT,
                                          dropdown_fg_color=theme.SURFACE, dropdown_text_color=theme.TEXT,
                                          dropdown_hover_color=theme.SUBTLE)
        self.pair_menu.grid(row=0, column=0, sticky='ew', padx=4, pady=(0, 6))
        tools = ctk.CTkFrame(self, fg_color='transparent')
        tools.grid(row=2, column=0, sticky='ew', padx=4, pady=6)
        self.mode_button = ctk.CTkSegmentedButton(self, values=['A', 'B'], command=self._choose_mode,
                                                font=ctk.CTkFont(size=12), fg_color=theme.BORDER,
                                                selected_color=theme.SELECTION, selected_hover_color=theme.SELECTION,
                                                unselected_color=theme.SUBTLE, unselected_hover_color=theme.BORDER,
                                                text_color=theme.TEXT)
        self.mode_button.grid(row=1, column=0, sticky='w', padx=4)
        self.fit_button = ctk.CTkButton(tools, text='', width=85, command=self.fit, **button_style())
        self.fit_button.pack(side='left', padx=(0, 5))
        self.actual_button = ctk.CTkButton(tools, text='100%', width=65, command=lambda: self.zoom(1.0), **button_style())
        self.actual_button.pack(side='left', padx=(0, 5))
        for label, factor in (('−', 0.8), ('+', 1.25)):
            ctk.CTkButton(tools, text=label, width=35, command=lambda f=factor: self.zoom(self.viewport.zoom * f),
                           **button_style()).pack(side='left', padx=(0, 4))
        self.zoom_label = ctk.CTkLabel(tools, text='', text_color=theme.TEXT)
        self.zoom_label.pack(side='left', padx=5)
        self.show_bounds = tk.BooleanVar(self, value=True)
        self.bounds_button = ctk.CTkCheckBox(tools, text='', variable=self.show_bounds, command=self._request_render,
                                            font=ctk.CTkFont(size=11), checkbox_width=16, checkbox_height=16,
                                            text_color=theme.TEXT, fg_color=theme.PRIMARY,
                                            checkmark_color=theme.PRIMARY_TEXT, border_color=theme.BORDER)
        self.bounds_button.pack(side='left', padx=8)
        self.info_label = ctk.CTkLabel(self, text='', width=1, anchor='w', justify='left', wraplength=560,
                                      font=ctk.CTkFont(size=12))
        self.info_label.grid(row=3, column=0, sticky='ew', padx=4, pady=(0, 6))
        self.canvas = tk.Canvas(self, bg=theme.CANVAS, highlightthickness=0)
        self.canvas.grid(row=4, column=0, sticky='nsew', padx=4)
        footer = ctk.CTkFrame(self, fg_color='transparent')
        footer.grid(row=5, column=0, sticky='ew', padx=4, pady=(4, 0))
        self.help_label = ctk.CTkLabel(footer, text='', width=1, anchor='w', justify='left', wraplength=560,
                                      font=ctk.CTkFont(size=11), text_color=theme.MUTED)
        self.help_label.pack(fill='x')
        self.canvas.bind('<Configure>', self._resize)
        self.canvas.bind('<ButtonPress-1>', self._drag_start)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>', self._wheel)
        self.canvas.bind('<Button-5>', self._wheel)
        self.set_language(lang)

    def t(self, key, **values):
        return TEXT[self.lang][key].format(**values)

    def set_language(self, lang):
        self.lang = lang
        self._pair_labels = [self.t('panorama')] + [
            f'{p.i+1:02d} ↔ {p.j+1:02d}' + (' [' + self.t('excluded') + ']' if p.excluded else '')
            for p in self.report.pairs]
        self.pair_menu.configure(values=self._pair_labels)
        self.pair_menu.set(self._pair_labels[0 if self._pair_index is None else self._pair_index + 1])
        self._mode_labels = {'a': 'A', 'b': 'B', 'overlay': self.t('overlay'), 'alternate': self.t('alternate'),
                             'side': self.t('side')}
        self.mode_button.configure(values=list(self._mode_labels.values()))
        self.mode_button.set(self._mode_labels[self._mode])
        self.fit_button.configure(text=self.t('fit'))
        self.bounds_button.configure(text=self.t('boundaries'))
        self.help_label.configure(text=self.t('help'))
        self._refresh_info()
        self._request_render()

    def _refresh_info(self):
        self.pair_menu.configure(state='disabled' if self._loading else 'normal')
        self.mode_button.configure(state='normal' if self._images is not None else 'disabled')
        if self._loading:
            text = self.t('loading')
        elif self._error:
            text = self.t('error') + ': ' + self._error
        elif self._pair is None:
            text = self.t('panorama')
        else:
            p = self._pair
            text = f'A: {os.path.basename(self.report.paths[p.i])}    B: {os.path.basename(self.report.paths[p.j])}\n'
            text += self.t('info', status=self.t('excluded' if p.excluded else 'retained'),
                           inliers=p.inliers, residual=p.residual, mx=p.measured[0], my=p.measured[1],
                           px=p.predicted[0], py=p.predicted[1])
        color = theme.ERROR if self._error or (self._pair is not None and self._pair.excluded) else theme.TEXT
        self.info_label.configure(text=text, text_color=color)

    def _choose_label(self, label):
        index = self._pair_labels.index(label)
        self.select_pair(None if index == 0 else index - 1)

    def select_pair(self, index):
        self._generation += 1
        self._stop_blink()
        self._pair_index = index
        self._pair = self._images = None
        self._error = ''
        self._alternate_b = False
        self._set_view_size()
        self.pair_menu.set(self._pair_labels[0 if index is None else index + 1])
        if index is None:
            self._loading = False
            self._limits = display_range([self.panorama])
            self._refresh_info()
            self.fit()
            return
        self._loading = True
        self._refresh_info()
        self._request_render()
        self._loader = threading.Thread(target=_load_pair,
                                       args=(self.report, index, self._generation, self._events), daemon=True)
        self._loader.start()
        if self._load_id is None:
            self._load_id = self.after(30, self._poll_load)

    def _poll_load(self):
        self._load_id = None
        if self.closed:
            return
        while True:
            try:
                generation, pair, images, error = self._events.get_nowait()
            except queue.Empty:
                break
            if generation != self._generation:
                continue
            self._loading = False
            self._pair, self._images, self._error = pair, images, error or ''
            if images is not None:
                self._limits = display_range(images)
                self._mode = 'overlay'
                self._set_view_size()
                self.mode_button.set(self._mode_labels[self._mode])
                a, b = (np.array(self.report.positions[i]) for i in (pair.i, pair.j))
                ah, aw = images[0].shape
                bh, bw = images[1].shape
                lower = np.maximum(a, b)
                upper = np.minimum(a + [aw, ah], b + [bw, bh])
                self.viewport.center((lower + upper) / 2, zoom=1.0)
                self._fit_mode = False
            self._refresh_info()
            self._request_render()
        if self._loading:
            self._load_id = self.after(30, self._poll_load)

    def _choose_mode(self, label):
        self.set_mode(next(key for key, value in self._mode_labels.items() if value == label))

    def set_mode(self, mode):
        self._stop_blink()
        self._mode = mode
        self._set_view_size()
        self._alternate_b = False
        self.mode_button.set(self._mode_labels[mode])
        if mode == 'alternate' and self._images is not None and self.active:
            self._blink_id = self.after(450, self._blink)
        self._request_render()

    def _stop_blink(self):
        if self._blink_id is not None:
            self.after_cancel(self._blink_id)
            self._blink_id = None

    def _blink(self):
        self._blink_id = None
        if self.closed or not self.active or self._images is None or self._mode != 'alternate':
            return
        self._alternate_b = not self._alternate_b
        self._request_render()
        self._blink_id = self.after(450, self._blink)

    def _resize(self, event):
        self._set_view_size(event.width, event.height)
        logical_width = max(1, int(event.width / ctk.ScalingTracker.get_widget_scaling(self)))
        self.info_label.configure(wraplength=logical_width)
        self.help_label.configure(wraplength=logical_width)
        if self._fit_mode:
            self.fit()
        else:
            self.viewport.clamp()
            self._request_render()

    def _set_view_size(self, width=None, height=None):
        center = self.viewport.origin + np.array(self.viewport.view_size) / (2 * self.viewport.zoom)
        width = self.canvas.winfo_width() if width is None else width
        height = self.canvas.winfo_height() if height is None else height
        if self._mode == 'side' and self._images is not None:
            width = (width - self.comparison_gap) // 2
        self.viewport.view_size = (max(1, width), max(1, height))
        self.viewport.center(center, self.viewport.zoom)

    def set_active(self, active):
        """Pause image rendering/blinking while the ordinary preview is shown."""
        self.active = active
        self._stop_blink()
        if self._render_id is not None:
            self.after_cancel(self._render_id)
            self._render_id = None
        if active:
            if self._mode == 'alternate' and self._images is not None:
                self._blink_id = self.after(450, self._blink)
            self._request_render()

    def fit(self):
        self._fit_mode = True
        if self._pair is None or self._images is None:
            self.viewport.fit()
        else:
            positions = np.array([self.report.positions[self._pair.i], self.report.positions[self._pair.j]])
            sizes = np.array([(image.shape[1], image.shape[0]) for image in self._images])
            low, high = positions.min(axis=0), (positions + sizes).max(axis=0)
            zoom = min(np.array(self.viewport.view_size) / (high - low))
            self.viewport.center((low + high) / 2, zoom)
        self._request_render()

    def zoom(self, value, point=None):
        self._fit_mode = False
        self.viewport.zoom_at(value, point)
        self._request_render()

    def _wheel(self, event):
        direction = 1 if getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0 else -1
        x = event.x
        if self._mode == 'side' and self._images is not None and x >= self.viewport.view_size[0]:
            x = max(0, x - self.viewport.view_size[0] - self.comparison_gap)
        self.zoom(self.viewport.zoom * (1.25 if direction > 0 else 0.8), (x, event.y))
        return 'break'

    def _drag_start(self, event):
        self._drag_at = (event.x, event.y)

    def _drag(self, event):
        if self._drag_at is not None:
            self._fit_mode = False
            self.viewport.pan(event.x - self._drag_at[0], event.y - self._drag_at[1])
            self._drag_at = (event.x, event.y)
            self._request_render()

    def _request_render(self):
        if not self.closed and self.active and self._render_id is None:
            self._render_id = self.after(25, self._render)

    def _render(self):
        self._render_id = None
        if self.closed or not self.active:
            return
        self.canvas.delete('all')  # Remove Tk image references before releasing photos.
        if self._pair_index is not None and self._images is None:
            self._photo = None
            return
        if self._pair_index is None:
            sampled, valid = sample_view(self.panorama, (0, 0), self.viewport)
            shown = display_gray(sampled, valid, self._limits)
            frame = np.repeat(shown[:, :, None], 3, axis=2)
        else:
            positions = [self.report.positions[self._pair.i], self.report.positions[self._pair.j]]
            if self._mode == 'side':
                width, height = self.viewport.view_size
                color = tuple(bytes.fromhex(theme.CANVAS[1:]))
                frame = np.full((height, max(self.canvas.winfo_width(), width * 2 + self.comparison_gap), 3), color, np.uint8)
                for index, mode in enumerate(('a', 'b')):
                    offset = index * (width + self.comparison_gap)
                    frame[:, offset:offset+width] = pair_frame(self._images, positions, self.viewport, mode, self._limits)
            else:
                frame = pair_frame(self._images, positions, self.viewport, self._mode, self._limits, self._alternate_b)
        self._photo = ImageTk.PhotoImage(Image.fromarray(frame), master=self.canvas)
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        self.zoom_label.configure(text=f'{self.viewport.zoom * 100:.1f}%')
        if self._mode == 'side' and self._images is not None:
            width = self.viewport.view_size[0]
            for index, source_index in enumerate((self._pair.i, self._pair.j)):
                offset = index * (width + self.comparison_gap)
                self.canvas.create_rectangle(offset+4, 4, offset+24, 24, fill=theme.CANVAS, outline='')
                self.canvas.create_text(offset+14, 14, text=('A', 'B')[index], fill='white')
                if self.show_bounds.get():
                    self._draw_clipped_bounds(source_index, offset, width)
            return
        if self.show_bounds.get():
            for i, ((x, y), (h, w)) in enumerate(zip(self.report.positions, self.report.shapes)):
                if self._pair is not None and i not in (self._pair.i, self._pair.j):
                    continue
                x, y = (np.array([x, y]) - self.viewport.origin) * self.viewport.zoom
                self.canvas.create_rectangle(x, y, x + w * self.viewport.zoom, y + h * self.viewport.zoom,
                                             outline='#d9b75f', width=1)
            if self._pair_index is None:
                for p in self.report.pairs:
                    if p.excluded:
                        points = []
                        for i in (p.i, p.j):
                            h, w = self.report.shapes[i]
                            center = np.array(self.report.positions[i]) + [w / 2, h / 2]
                            points.extend((center - self.viewport.origin) * self.viewport.zoom)
                        self.canvas.create_line(*points, fill='#ff6655', width=2)

    def _draw_clipped_bounds(self, index, offset, width):
        """Clip boundary segments to their own pane so none cross the gap."""
        height = self.viewport.view_size[1]
        x, y = (np.array(self.report.positions[index]) - self.viewport.origin) * self.viewport.zoom
        h, w = np.array(self.report.shapes[index]) * self.viewport.zoom
        for edge in (x, x+w):
            if 0 <= edge < width and y <= height and y+h >= 0:
                self.canvas.create_line(offset+edge, max(0, y), offset+edge, min(height, y+h), fill='#d9b75f')
        for edge in (y, y+h):
            if 0 <= edge < height and x <= width and x+w >= 0:
                self.canvas.create_line(offset+max(0, x), edge, offset+min(width, x+w), edge, fill='#d9b75f')

    def close(self):
        if self.closed:
            return
        self.closed = True
        for timer in (self._render_id, self._blink_id, self._load_id):
            if timer is not None:
                self.after_cancel(timer)
        self.canvas.delete('all')
        self._photo = self._images = self.panorama = None
        self.report = None
        # A late I/O result belongs to the worker's old queue and is released
        # when that worker exits; a closed window must not retain source arrays.
        self._events = queue.Queue()
        self.destroy()
