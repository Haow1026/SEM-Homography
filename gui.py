"""
SEM Image Stitching Tool — Mac Desktop GUI
===========================================
CustomTkinter dark-mode interface: file selection, real-time log,
async stitching, preview & export.  Supports en / 繁體中文 / 日本語.
"""

import os
import re
import locale
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np

from main import SEMStitcher

# ============================================================
# 多语言翻译表  i18n
# ============================================================
LANG = {
    "en": {
        "title": "SEM Image Stitching Tool",
        "heading": "SEM Image Stitching",
        "subtitle": "Select TIFF images to stitch (multi-select, sorted by filename)",
        "btn_select": "Select Images",
        "lbl_none": "None selected",
        "lbl_count": "{n} image(s) selected",
        "log_title": "Log",
        "btn_start": "Start Stitching",
        "btn_start_disabled": "Stitching…",
        "preview_title": "Preview",
        "preview_placeholder": "Result will appear here\nafter stitching",
        "btn_export": "Export Panorama",
        "lbl_size": "Size: {w} × {h}",
        "msg_few": "Not enough images",
        "msg_few_detail": "{n} image(s) selected, need at least 2.",
        "log_loaded": "Loaded {n} images:",
        "log_start": "Starting stitch…",
        "log_done": "Stitching complete!",
        "log_fail": "Stitching failed. Check log.",
        "log_error": "Error: {e}",
        "log_exported": "Exported: {path}",
        "msg_export_ok": "Export successful",
        "msg_export_detail": "Saved to:\n{path}",
        "file_filter_tiff": "TIFF images",
        "file_filter_all": "All files",
        "dialog_select": "Select TIFF images to stitch",
        "dialog_export": "Export panorama",
        "lang_label": "Language",
    },
    "zh-Hans": {
        "title": "SEM 图像拼接工具",
        "heading": "SEM 图像拼接",
        "subtitle": "选择要拼接的 TIFF 图像（可多选，按文件名排序）",
        "btn_select": "选择图片文件",
        "lbl_none": "尚未选择",
        "lbl_count": "已选择 {n} 张图像",
        "log_title": "运行日志",
        "btn_start": "开始拼接 (Start Stitching)",
        "btn_start_disabled": "拼接中…",
        "preview_title": "预览",
        "preview_placeholder": "拼接完成后\n在此预览",
        "btn_export": "导出全景图 (Export)",
        "lbl_size": "尺寸：{w} × {h}",
        "msg_few": "图像不足",
        "msg_few_detail": "已选择 {n} 张图像，至少需要 2 张。",
        "log_loaded": "已加载 {n} 张图像：",
        "log_start": "开始拼接…",
        "log_done": "拼接完成！",
        "log_fail": "拼接失败，请检查日志。",
        "log_error": "错误：{e}",
        "log_exported": "已导出：{path}",
        "msg_export_ok": "导出成功",
        "msg_export_detail": "已保存至：\n{path}",
        "file_filter_tiff": "TIFF 图像",
        "file_filter_all": "所有文件",
        "dialog_select": "选择要拼接的 TIFF 图像",
        "dialog_export": "导出全景图",
        "lang_label": "语言",
    },
    "ja": {
        "title": "SEM 画像 stitching ツール",
        "heading": "SEM 画像 stitching",
        "subtitle": "TIFF 画像を選択（複数選択可、ファイル名順）",
        "btn_select": "画像ファイルを選択",
        "lbl_none": "未選択",
        "lbl_count": "{n} 枚選択済み",
        "log_title": "ログ",
        "btn_start": "Stitching 開始 (Start)",
        "btn_start_disabled": "処理中…",
        "preview_title": "プレビュー",
        "preview_placeholder": "Stitching 完了後\nここに表示されます",
        "btn_export": "全景画像を書き出し (Export)",
        "lbl_size": "サイズ：{w} × {h}",
        "msg_few": "画像が不足しています",
        "msg_few_detail": "{n} 枚選択されました。2 枚以上必要です。",
        "log_loaded": "{n} 枚の画像を読み込みました：",
        "log_start": "Stitching を開始します…",
        "log_done": "Stitching が完了しました！",
        "log_fail": "Stitching に失敗しました。ログを確認してください。",
        "log_error": "エラー：{e}",
        "log_exported": "書き出し完了：{path}",
        "msg_export_ok": "書き出し成功",
        "msg_export_detail": "保存先：\n{path}",
        "file_filter_tiff": "TIFF 画像",
        "file_filter_all": "すべてのファイル",
        "dialog_select": "Stitching する TIFF 画像を選択",
        "dialog_export": "全景画像を書き出し",
        "lang_label": "言語",
    },
}

# ============================================================
# 全局配置
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

WIN_WIDTH, WIN_HEIGHT = 1050, 750
PREVIEW_MAX = 700


def _detect_lang():
    """Auto-detect system language, fallback to English."""
    try:
        loc = locale.getdefaultlocale()[0] or ""
    except Exception:
        loc = ""
    if loc.startswith("zh"):
        return "zh-Hans"
    if loc.startswith("ja"):
        return "ja"
    return "en"


# ============================================================
# 主窗口
# ============================================================
class SEMStitcherGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = _detect_lang()
        self._apply_title()
        self.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.minsize(950, 600)

        # ---- 状态 ----
        self.image_paths = []
        self.stitched_result = None
        self.is_running = False
        self.stitcher_thread = None

        # ---- 保存所有需要翻译的 widget 引用 ----
        self._i18n_widgets = {}

        # ---- 构建界面 ----
        self._build_ui()
        self._apply_language()

    # ================================================================
    # 翻译辅助
    # ================================================================
    def t(self, key: str, **fmt) -> str:
        """Get translated string for key, with optional formatting."""
        s = LANG.get(self.lang, LANG["en"]).get(key, key)
        if fmt:
            s = s.format(**fmt)
        return s

    def _apply_title(self):
        self.title(LANG.get(self.lang, LANG["en"])["title"])

    def _apply_language(self):
        """Refresh all UI text for current language."""
        self._apply_title()
        for widget, (key, fmt) in self._i18n_widgets.items():
            text = self.t(key, **fmt)
            try:
                widget.configure(text=text)
            except Exception:
                pass  # widget may have been destroyed
        if not self.is_running and self.stitched_result is None:
            self.lbl_count.configure(text=self.t("lbl_none"))
        elif self.stitched_result is not None:
            h, w = self.stitched_result.shape
            self.lbl_size.configure(text=self.t("lbl_size", w=w, h=h))

    def _reg(self, widget, key: str, **fmt):
        """Register a widget for i18n updates."""
        self._i18n_widgets[widget] = (key, fmt)

    def _set_lang(self, choice: str):
        lang_map = {"English": "en", "简体中文": "zh-Hans", "日本語": "ja"}
        self.lang = lang_map.get(choice, "en")
        self._apply_language()

    # ================================================================
    # 构建 UI
    # ================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ===== 语言选择器（右上角） =====
        lang_frame = ctk.CTkFrame(self, fg_color="transparent")
        lang_frame.grid(row=0, column=1, padx=(7, 22), pady=(15, 0), sticky="ne")
        lbl_lang = ctk.CTkLabel(
            lang_frame, text="", font=ctk.CTkFont(size=11), text_color="gray60",
        )
        lbl_lang.pack(side="left", padx=(0, 5))
        self._reg(lbl_lang, "lang_label")

        lang_options = ["English", "简体中文", "日本語"]
        lang_default = {"en": "English", "zh-Hans": "简体中文", "ja": "日本語"}[self.lang]
        self.lang_menu = ctk.CTkOptionMenu(
            lang_frame, values=lang_options,
            command=self._set_lang,
            font=ctk.CTkFont(size=12), width=110,
        )
        self.lang_menu.set(lang_default)
        self.lang_menu.pack(side="left")

        # ===== 左侧面板 =====
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=(15, 7), pady=15, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(0, weight=0)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_rowconfigure(2, weight=0)

        # --- 顶部：文件选择 ---
        file_frame = ctk.CTkFrame(left_frame)
        file_frame.grid(row=0, column=0, padx=12, pady=(12, 0), sticky="ew")
        file_frame.grid_columnconfigure(0, weight=1)

        lbl_heading = ctk.CTkLabel(
            file_frame, text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        lbl_heading.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        self._reg(lbl_heading, "heading")

        lbl_sub = ctk.CTkLabel(
            file_frame, text="",
            font=ctk.CTkFont(size=12), text_color="gray70",
        )
        lbl_sub.grid(row=1, column=0, padx=15, pady=(0, 5), sticky="w")
        self._reg(lbl_sub, "subtitle")

        btn_bar = ctk.CTkFrame(file_frame, fg_color="transparent")
        btn_bar.grid(row=2, column=0, padx=15, pady=(5, 15), sticky="ew")

        self.btn_select = ctk.CTkButton(
            btn_bar, text="",
            command=self._on_select_files,
            height=36, font=ctk.CTkFont(size=13),
        )
        self.btn_select.pack(side="left", padx=(0, 8))
        self._reg(self.btn_select, "btn_select")

        self.lbl_count = ctk.CTkLabel(
            btn_bar, text="",
            font=ctk.CTkFont(size=12), text_color="gray60",
        )
        self.lbl_count.pack(side="left", padx=8)

        # --- 中间：日志区 ---
        log_frame = ctk.CTkFrame(left_frame)
        log_frame.grid(row=1, column=0, padx=12, pady=(8, 0), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=0)
        log_frame.grid_rowconfigure(1, weight=1)

        lbl_log = ctk.CTkLabel(
            log_frame, text="",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        lbl_log.grid(row=0, column=0, padx=12, pady=(10, 5), sticky="w")
        self._reg(lbl_log, "log_title")

        self.log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Menlo", size=11),
            wrap="word", activate_scrollbars=True,
        )
        self.log_text.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        # --- 底部：按钮 ---
        bottom_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, padx=12, pady=(8, 12), sticky="ew")

        self.btn_start = ctk.CTkButton(
            bottom_frame, text="",
            command=self._on_start,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            state="disabled",
        )
        self.btn_start.pack(fill="x")
        self._reg(self.btn_start, "btn_start")

        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.pack(fill="x", pady=(8, 0))
        self.progress_bar.set(0)
        self.progress_bar.configure(mode="indeterminate")

        # ===== 右侧面板：预览 =====
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=(7, 15), pady=(50, 15), sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=0)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_rowconfigure(2, weight=0)
        right_frame.grid_rowconfigure(3, weight=0)

        lbl_preview = ctk.CTkLabel(
            right_frame, text="",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        lbl_preview.grid(row=0, column=0, padx=12, pady=(12, 5), sticky="w")
        self._reg(lbl_preview, "preview_title")

        self.preview_label = ctk.CTkLabel(
            right_frame, text="",
            font=ctk.CTkFont(size=12), text_color="gray50",
            width=300, height=300,
            fg_color="gray15", corner_radius=8,
        )
        self.preview_label.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="nsew")
        self._reg(self.preview_label, "preview_placeholder")

        self.btn_export = ctk.CTkButton(
            right_frame, text="",
            command=self._on_export,
            height=36, font=ctk.CTkFont(size=13),
            state="disabled",
        )
        self.btn_export.grid(row=2, column=0, padx=12, pady=(0, 15), sticky="ew")
        self._reg(self.btn_export, "btn_export")

        self.lbl_size = ctk.CTkLabel(
            right_frame, text="",
            font=ctk.CTkFont(size=11), text_color="gray60",
        )
        self.lbl_size.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="w")

    # ================================================================
    # 文件选择
    # ================================================================
    def _on_select_files(self):
        paths = filedialog.askopenfilenames(
            title=self.t("dialog_select"),
            filetypes=[
                (self.t("file_filter_tiff"), "*.tif *.tiff"),
                (self.t("file_filter_all"), "*.*"),
            ],
        )
        if not paths:
            return
        self._load_files(list(paths))

    def _load_files(self, paths):
        def _sort_key(p):
            name = os.path.basename(p)
            nums = re.findall(r"\d+", name)
            if nums:
                return (0, int(nums[-1]), name)
            return (1, 0, name)

        paths = sorted(paths, key=_sort_key)
        if len(paths) < 2:
            messagebox.showwarning(
                self.t("msg_few"),
                self.t("msg_few_detail", n=len(paths)),
            )
            return
        self.image_paths = paths
        self.lbl_count.configure(text=self.t("lbl_count", n=len(paths)))
        self.btn_start.configure(state="normal")
        self._append_log(self.t("log_loaded", n=len(paths)))
        for p in paths:
            self._append_log(f"  - {os.path.basename(p)}")

    # ================================================================
    # 日志
    # ================================================================
    def _append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.update_idletasks()

    # ================================================================
    # 开始拼接
    # ================================================================
    def _on_start(self):
        if self.is_running:
            return
        self.is_running = True
        self.btn_start.configure(state="disabled", text=self.t("btn_start_disabled"))
        self.btn_select.configure(state="disabled")
        self.progress_bar.start()
        self.log_text.delete("1.0", "end")
        self._append_log("=" * 50)
        self._append_log(self.t("log_start"))

        self.stitcher_thread = threading.Thread(
            target=self._run_stitching, daemon=True
        )
        self.stitcher_thread.start()
        self._poll_thread()

    def _run_stitching(self):
        try:
            stitcher = SEMStitcher()
            stitcher.set_progress_callback(self._log_from_thread)
            result = stitcher.stitch_sequence(self.image_paths)
            self.stitched_result = result
        except Exception as e:
            import traceback
            self._log_from_thread(self.t("log_error", e=e))
            self._log_from_thread(traceback.format_exc())
            self.stitched_result = None

    def _log_from_thread(self, msg: str):
        self.after(0, self._append_log, msg)

    def _poll_thread(self):
        if self.stitcher_thread and self.stitcher_thread.is_alive():
            self.after(200, self._poll_thread)
        else:
            self._on_stitching_done()

    def _on_stitching_done(self):
        self.is_running = False
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.btn_select.configure(state="normal")
        self.btn_start.configure(state="normal")
        self._apply_language()  # restore button text

        if self.stitched_result is not None:
            h, w = self.stitched_result.shape
            self._append_log(self.t("log_done"))
            self._append_log(f"  {self.t('lbl_size', w=w, h=h)}")
            self.lbl_size.configure(text=self.t("lbl_size", w=w, h=h))
            self.btn_export.configure(state="normal")
            self._update_preview()
        else:
            self._append_log(self.t("log_fail"))
            self.progress_bar.set(0)

    # ================================================================
    # 预览
    # ================================================================
    def _update_preview(self):
        if self.stitched_result is None:
            return
        img_arr = self.stitched_result
        h, w = img_arr.shape
        scale = min(PREVIEW_MAX / max(w, h), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        from cv2 import resize, INTER_AREA
        resized = resize(img_arr, (new_w, new_h), interpolation=INTER_AREA)
        pil_img = Image.fromarray(resized)
        ctk_img = ctk.CTkImage(
            light_image=pil_img, dark_image=pil_img, size=(new_w, new_h)
        )
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

    # ================================================================
    # 导出
    # ================================================================
    def _on_export(self):
        if self.stitched_result is None:
            return
        path = filedialog.asksaveasfilename(
            title=self.t("dialog_export"),
            defaultextension=".tif",
            filetypes=[
                (self.t("file_filter_tiff"), "*.tif"),
                ("PNG", "*.png"),
                (self.t("file_filter_all"), "*.*"),
            ],
            initialfile="stitched_panorama.tif",
        )
        if not path:
            return
        from cv2 import imwrite
        imwrite(path, self.stitched_result)
        messagebox.showinfo(
            self.t("msg_export_ok"),
            self.t("msg_export_detail", path=path),
        )
        self._append_log(self.t("log_exported", path=path))


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    app = SEMStitcherGUI()
    app.mainloop()
