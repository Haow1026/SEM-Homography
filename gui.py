"""
SEM Image Stitching Tool — Mac Desktop GUI
===========================================
CustomTkinter dark-mode interface: file selection, real-time log,
async stitching, preview & export.  Supports en / 简体中文 / 日本語.
"""

import os
import locale
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np

from main import SEMStitcher
from stitching_task import StitchRequest, StitchTask, TaskState, natural_sort
from inspector import SeamInspector
from project import restore_project, save_project
from project_job import ProjectJob

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

LANG['en'].update({
    'subtitle': 'Check acquisition order and scan layout before stitching',
    'tab_files': 'Files & layout', 'tab_log': 'Log',
    'grid_enabled': 'Use a known grid', 'rows': 'Rows', 'cols': 'Columns',
    'serpentine': 'Serpentine scan', 'first_rtl': 'First row: right to left',
    'btn_up': 'Move up', 'btn_down': 'Move down', 'btn_reverse': 'Reverse', 'btn_sort': 'Sort by name',
    'order_hint': 'List = acquisition order; select a file to move it.',
    'layout_hint': 'Top row first; numbers refer to the file list.',
    'layout_auto': 'Automatic matching (no grid constraint)',
    'layout_empty': 'Select at least two images to preview the layout.',
    'layout_grid_reset': 'The new image count does not fit the previous grid. Automatic matching is enabled; set a new grid if needed.',
    'grid_positive': 'Rows and columns must be positive whole numbers.',
    'grid_count': 'Grid needs {expected} images; {actual} selected.',
    'msg_settings': 'Check scan settings', 'btn_cancel': 'Cancel', 'btn_cancelling': 'Cancelling…',
    'status_idle': 'Select images to begin', 'status_ready': 'Ready',
    'status_running': 'Stitching…', 'status_cancelling': 'Cancelling; waiting for the current step…',
    'status_succeeded': 'Complete — ready to export', 'status_failed': 'Failed — check the log',
    'status_cancelled': 'Cancelled — no result to export', 'log_cancelled': 'Stitching cancelled.',
    'log_cancelling': 'Cancellation requested. Waiting for the current operation to finish.',
    'btn_inspect': 'Inspect seams / 100%',
})
LANG['zh-Hans'].update({
    'subtitle': '开始前请核对文件采集顺序和扫描布局',
    'tab_files': '文件与布局', 'tab_log': '运行日志',
    'grid_enabled': '使用已知网格', 'rows': '行数', 'cols': '列数',
    'serpentine': '蛇形扫描', 'first_rtl': '首行从右向左',
    'btn_up': '上移', 'btn_down': '下移', 'btn_reverse': '倒序', 'btn_sort': '按文件名排序',
    'order_hint': '列表为采集顺序；选中文件后可调整位置。',
    'layout_hint': '上方为首行；数字对应文件列表编号。',
    'layout_auto': '自动匹配（不限制网格邻接关系）',
    'layout_empty': '选择至少两张图像后预览布局。',
    'layout_grid_reset': '新批次图像数量与原网格不符，已切换为自动匹配；如需网格，请重新设置行列数。',
    'grid_positive': '行数和列数必须为正整数。',
    'grid_count': '网格需要 {expected} 张图像，当前选择 {actual} 张。',
    'msg_settings': '请检查扫描设置', 'btn_cancel': '取消', 'btn_cancelling': '正在取消…',
    'status_idle': '请选择图像', 'status_ready': '准备就绪',
    'status_running': '正在拼接…', 'status_cancelling': '正在取消，等待当前步骤结束…',
    'status_succeeded': '拼接完成，可以导出', 'status_failed': '拼接失败，请查看日志',
    'status_cancelled': '已取消，无可导出的结果', 'log_cancelled': '拼接已取消。',
    'log_cancelling': '已请求取消，等待当前操作完成。',
    'btn_inspect': '接缝检查 / 原分辨率',
})
LANG['ja'].update({
    'subtitle': '開始前に取得順序とスキャン配置を確認してください',
    'tab_files': 'ファイルと配置', 'tab_log': 'ログ',
    'grid_enabled': '既知のグリッドを使用', 'rows': '行数', 'cols': '列数',
    'serpentine': '蛇行スキャン', 'first_rtl': '最初の行は右から左',
    'btn_up': '上へ', 'btn_down': '下へ', 'btn_reverse': '逆順', 'btn_sort': '名前順',
    'order_hint': '一覧は取得順です。選択して順序を変更できます。',
    'layout_hint': '上から順に表示。番号はファイル一覧に対応します。',
    'layout_auto': '自動マッチング（グリッド制約なし）',
    'layout_empty': '画像を2枚以上選択すると配置を表示します。',
    'layout_grid_reset': '新しい画像数が以前のグリッドと異なるため自動マッチングに切り替えました。必要に応じてグリッドを再設定してください。',
    'grid_positive': '行数と列数は正の整数で指定してください。',
    'grid_count': 'グリッドには {expected} 枚必要です。選択済み: {actual} 枚。',
    'msg_settings': 'スキャン設定を確認', 'btn_cancel': 'キャンセル', 'btn_cancelling': 'キャンセル中…',
    'status_idle': '画像を選択してください', 'status_ready': '準備完了',
    'status_running': '処理中…', 'status_cancelling': '現在の処理の終了を待っています…',
    'status_succeeded': '完了 — 書き出し可能', 'status_failed': '失敗 — ログを確認',
    'status_cancelled': 'キャンセル済み — 結果なし', 'log_cancelled': '処理をキャンセルしました。',
    'log_cancelling': 'キャンセルを要求しました。現在の処理の終了を待っています。',
    'btn_inspect': '継ぎ目 / 原寸を確認',
})

LANG['en'].update({
    'btn_project_open': 'Open project', 'btn_project_save': 'Save project',
    'project_filter': 'SEM project', 'project_open': 'Open SEM project', 'project_save': 'Save SEM project',
    'status_project_open': 'Verifying sources and restoring project…',
    'status_project_save': 'Verifying sources and saving project…',
    'project_saved': 'Project saved: {path}', 'project_restored': 'Project restored: {path}',
    'project_inputs': 'Saved inputs restored. Start stitching to create a panorama.',
    'project_cancelled': 'Project operation cancelled; current session kept.',
})
LANG['zh-Hans'].update({
    'btn_project_open': '打开项目', 'btn_project_save': '保存项目',
    'project_filter': 'SEM 项目', 'project_open': '打开 SEM 项目', 'project_save': '保存 SEM 项目',
    'status_project_open': '正在校验源图并恢复项目…', 'status_project_save': '正在校验源图并保存项目…',
    'project_saved': '项目已保存：{path}', 'project_restored': '项目已恢复：{path}',
    'project_inputs': '已恢复文件与布局，点击开始拼接生成全景图。',
    'project_cancelled': '项目操作已取消，保留当前会话。',
})
LANG['ja'].update({
    'btn_project_open': 'プロジェクトを開く', 'btn_project_save': 'プロジェクト保存',
    'project_filter': 'SEM プロジェクト', 'project_open': 'SEM プロジェクトを開く',
    'project_save': 'SEM プロジェクトを保存',
    'status_project_open': '元画像を検証してプロジェクトを復元中…',
    'status_project_save': '元画像を検証してプロジェクトを保存中…',
    'project_saved': 'プロジェクトを保存しました：{path}', 'project_restored': 'プロジェクトを復元しました：{path}',
    'project_inputs': 'ファイルと配置を復元しました。開始ボタンで全景画像を作成してください。',
    'project_cancelled': 'プロジェクト操作をキャンセルしました。現在の状態を保持します。',
})

# ============================================================
# 全局配置
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

WIN_WIDTH, WIN_HEIGHT = 1180, 760
PREVIEW_MAX = 700


def _detect_lang():
    """Auto-detect system language, fallback to English."""
    try:
        loc = locale.getlocale()[0] or ""
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
        self.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")
        self.minsize(1050, 700)
        self.image_paths = []
        self.task = StitchTask()
        self.project_job = ProjectJob()
        self._applying_project = False
        self._closing = False
        self._poll_id = None
        self._preview_id = None
        self._inspector = None
        self._layout_valid = False
        self._layout_notice = False
        self._i18n_widgets = {}
        self.use_grid = tk.BooleanVar(self, value=False)
        self.rows = tk.StringVar(self, value='3')
        self.cols = tk.StringVar(self, value='3')
        self.serpentine = tk.BooleanVar(self, value=False)
        self.first_rtl = tk.BooleanVar(self, value=False)
        self._build_ui()
        self.rows.trace_add('write', self._on_inputs_changed)
        self.cols.trace_add('write', self._on_inputs_changed)
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._apply_language()

    @property
    def stitched_result(self):
        return self.task.result

    @property
    def _busy(self):
        return self.task.busy or self.project_job.busy

    def t(self, key, **fmt):
        return LANG[self.lang].get(key, key).format(**fmt)

    def _reg(self, widget, key, **fmt):
        self._i18n_widgets[widget] = (key, fmt)

    def _apply_language(self):
        self.title(self.t('title'))
        for widget, (key, fmt) in self._i18n_widgets.items():
            widget.configure(text=self.t(key, **fmt))
        for key, old_name in list(self._tab_names.items()):
            new_name = self.t(key)
            if old_name != new_name:
                self.tabs.rename(old_name, new_name)
                self._tab_names[key] = new_name
        self._refresh_layout()
        self._render_state()
        if self._inspector is not None and not self._inspector.closed:
            self._inspector.set_language(self.lang)
        if self.stitched_result is None:
            self.preview_label.configure(text=self.t('preview_placeholder'))
        else:
            h, w = self.stitched_result.shape
            self.lbl_size.configure(text=self.t('lbl_size', w=w, h=h))

    def _set_lang(self, choice):
        self.lang = {'English': 'en', '简体中文': 'zh-Hans', '日本語': 'ja'}[choice]
        self._apply_language()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, padx=(15, 7), pady=15, sticky='nsew')
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)
        heading = ctk.CTkLabel(left, text='', font=ctk.CTkFont(size=20, weight='bold'))
        heading.grid(row=0, column=0, padx=18, pady=(15, 0), sticky='w')
        self._reg(heading, 'heading')
        header = ctk.CTkFrame(left, fg_color='transparent')
        header.grid(row=1, column=0, padx=15, pady=(5, 0), sticky='ew')
        subtitle = ctk.CTkLabel(header, text='', text_color='gray70')
        subtitle.pack(anchor='w', pady=(0, 8))
        self._reg(subtitle, 'subtitle')
        self.btn_select = ctk.CTkButton(header, text='', width=130, command=self._on_select_files)
        self.btn_select.pack(side='left')
        self._reg(self.btn_select, 'btn_select')
        self.btn_project_open = ctk.CTkButton(header, text='', width=105, command=self._on_open_project)
        self.btn_project_open.pack(side='left', padx=(6, 0))
        self._reg(self.btn_project_open, 'btn_project_open')
        self.btn_project_save = ctk.CTkButton(header, text='', width=105, command=self._on_save_project)
        self.btn_project_save.pack(side='left', padx=(6, 0))
        self._reg(self.btn_project_save, 'btn_project_save')
        self.lbl_count = ctk.CTkLabel(header, text='')
        self.lbl_count.pack(side='left', padx=12)

        self.tabs = ctk.CTkTabview(left)
        self.tabs.grid(row=2, column=0, padx=12, pady=8, sticky='nsew')
        self._tab_names = {key: self.t(key) for key in ('tab_files', 'tab_log')}
        files = self.tabs.add(self._tab_names['tab_files'])
        logs = self.tabs.add(self._tab_names['tab_log'])
        files.grid_columnconfigure(0, weight=1)
        files.grid_rowconfigure(2, weight=1)
        settings = ctk.CTkFrame(files)
        settings.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        self.grid_switch = ctk.CTkSwitch(settings, text='', variable=self.use_grid, command=self._on_inputs_changed)
        self.grid_switch.grid(row=0, column=0, columnspan=4, padx=12, pady=(10, 8), sticky='w')
        self._reg(self.grid_switch, 'grid_enabled')
        for col, key, variable in ((0, 'rows', self.rows), (2, 'cols', self.cols)):
            label = ctk.CTkLabel(settings, text='')
            label.grid(row=1, column=col, padx=(12, 6), sticky='w')
            self._reg(label, key)
            entry = ctk.CTkEntry(settings, width=65, textvariable=variable)
            entry.grid(row=1, column=col + 1, padx=(0, 12), sticky='w')
            setattr(self, key + '_entry', entry)
        self.check_serpentine = ctk.CTkCheckBox(settings, text='', variable=self.serpentine, command=self._on_inputs_changed)
        self.check_serpentine.grid(row=2, column=0, columnspan=2, padx=12, pady=10, sticky='w')
        self._reg(self.check_serpentine, 'serpentine')
        self.check_rtl = ctk.CTkCheckBox(settings, text='', variable=self.first_rtl, command=self._on_inputs_changed)
        self.check_rtl.grid(row=2, column=2, columnspan=2, padx=12, pady=10, sticky='w')
        self._reg(self.check_rtl, 'first_rtl')
        hint = ctk.CTkLabel(files, text='', text_color='gray70')
        hint.grid(row=1, column=0, sticky='w')
        self._reg(hint, 'order_hint')
        list_frame = ctk.CTkFrame(files)
        list_frame.grid(row=2, column=0, sticky='nsew', pady=(4, 6))
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        self.file_list = tk.Listbox(list_frame, bg='#242424', fg='#eeeeee',
                                   selectbackground='#1f6aa5', selectforeground='white',
                                   font=('TkDefaultFont', 12), height=5, exportselection=False,
                                   relief='flat', highlightthickness=0)
        self.file_list.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        scrollbar = ctk.CTkScrollbar(list_frame, command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind('<<ListboxSelect>>', lambda event: self._render_state())
        reorder = ctk.CTkFrame(files, fg_color='transparent')
        reorder.grid(row=3, column=0, sticky='ew', pady=(0, 6))
        self._order_buttons = []
        for key, command in (
            ('btn_up', lambda: self._move_file(-1)), ('btn_down', lambda: self._move_file(1)),
            ('btn_reverse', self._reverse_files), ('btn_sort', self._sort_files),
        ):
            button = ctk.CTkButton(reorder, text='', width=75, command=command)
            button.pack(side='left', padx=(0, 6))
            self._reg(button, key)
            self._order_buttons.append(button)
        self.lbl_layout_hint = ctk.CTkLabel(files, text='', text_color='gray70')
        self.lbl_layout_hint.grid(row=4, column=0, sticky='w')
        self._reg(self.lbl_layout_hint, 'layout_hint')
        self.layout_text = ctk.CTkTextbox(files, height=85, font=ctk.CTkFont(size=13), wrap='none')
        self.layout_text.grid(row=5, column=0, sticky='ew', pady=4)
        self.lbl_validation = ctk.CTkLabel(files, text='', text_color='#ff8b80', wraplength=520)
        self.lbl_validation.grid(row=6, column=0, sticky='w')
        logs.grid_rowconfigure(0, weight=1)
        logs.grid_columnconfigure(0, weight=1)
        self.log_text = ctk.CTkTextbox(logs, font=ctk.CTkFont(size=12), wrap='word')
        self.log_text.grid(row=0, column=0, sticky='nsew')

        bottom = ctk.CTkFrame(left, fg_color='transparent')
        bottom.grid(row=3, column=0, sticky='ew', padx=15, pady=(0, 15))
        bottom.grid_columnconfigure(0, weight=1)
        self.lbl_status = ctk.CTkLabel(bottom, text='', anchor='w')
        self.lbl_status.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        self.btn_start = ctk.CTkButton(bottom, text='', height=40, command=self._on_start)
        self.btn_start.grid(row=1, column=0, sticky='ew', padx=(0, 8))
        self.btn_cancel = ctk.CTkButton(bottom, text='', height=40, width=110, command=self._on_cancel,
                                       fg_color='#555555', hover_color='#666666')
        self.btn_cancel.grid(row=1, column=1)
        self.progress_bar = ctk.CTkProgressBar(bottom, mode='indeterminate')
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        self.progress_bar.set(0)

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, padx=(7, 15), pady=15, sticky='nsew')
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        language = ctk.CTkFrame(right, fg_color='transparent')
        language.grid(row=0, column=0, sticky='e', padx=12, pady=12)
        label = ctk.CTkLabel(language, text='')
        label.pack(side='left', padx=(0, 8))
        self._reg(label, 'lang_label')
        self.lang_menu = ctk.CTkOptionMenu(language, values=['English', '简体中文', '日本語'],
                                          width=110, command=self._set_lang)
        self.lang_menu.set({'en': 'English', 'zh-Hans': '简体中文', 'ja': '日本語'}[self.lang])
        self.lang_menu.pack(side='left')
        label = ctk.CTkLabel(right, text='', font=ctk.CTkFont(size=14, weight='bold'))
        label.grid(row=1, column=0, sticky='w', padx=15, pady=(0, 8))
        self._reg(label, 'preview_title')
        # CTkLabel.configure(image=None) does not clear the underlying Tk image.
        # Keep a real, transparent placeholder so releasing a completed preview
        # cannot leave Tk referencing a deleted PhotoImage on the next run.
        self._empty_preview = ctk.CTkImage(Image.new('RGBA', (1, 1)), size=(1, 1))
        self.preview_label = ctk.CTkLabel(right, text='', image=self._empty_preview, width=300, height=250,
                                          fg_color='gray15', corner_radius=8, text_color='gray60')
        self.preview_label.grid(row=2, column=0, sticky='nsew', padx=12)
        self.preview_label.bind('<Configure>', self._on_preview_resize)
        self.lbl_size = ctk.CTkLabel(right, text='')
        self.lbl_size.grid(row=3, column=0, sticky='w', padx=15, pady=8)
        self.btn_inspect = ctk.CTkButton(right, text='', height=34, command=self._on_inspect)
        self.btn_inspect.grid(row=4, column=0, sticky='ew', padx=12, pady=(0, 8))
        self._reg(self.btn_inspect, 'btn_inspect')
        self.btn_export = ctk.CTkButton(right, text='', height=38, command=self._on_export)
        self.btn_export.grid(row=5, column=0, sticky='ew', padx=12, pady=(0, 15))
        self._reg(self.btn_export, 'btn_export')

    def _get_request(self):
        if not self.use_grid.get():
            return StitchRequest(tuple(self.image_paths))
        try:
            rows, cols = int(self.rows.get()), int(self.cols.get())
            if rows < 1 or cols < 1:
                raise ValueError
        except ValueError as error:
            raise ValueError(self.t('grid_positive')) from error
        if rows * cols != len(self.image_paths):
            raise ValueError(self.t('grid_count', expected=rows * cols, actual=len(self.image_paths)))
        return StitchRequest(tuple(self.image_paths), (rows, cols), self.serpentine.get(), self.first_rtl.get())

    def _refresh_layout(self):
        self.lbl_count.configure(text=self.t('lbl_count', n=len(self.image_paths)) if self.image_paths else self.t('lbl_none'))
        error = ''
        self._layout_valid = False
        if len(self.image_paths) < 2:
            text = self.t('layout_empty')
        else:
            try:
                request = self._get_request()
                self._layout_valid = True
                if request.grid_shape is None:
                    text = self.t('layout_auto')
                    if self._layout_notice:
                        text += '\n\n' + self.t('layout_grid_reset')
                else:
                    lines = []
                    for indices in request.layout_indices():
                        arrow = ' ← ' if len(indices) > 1 and indices[0] > indices[-1] else ' → '
                        lines.append(arrow.join(f'{index + 1:02d}' for index in indices))
                    text = '\n'.join(lines)
            except ValueError as exc:
                text, error = '—', str(exc)
        self.layout_text.configure(state='normal')
        self.layout_text.delete('1.0', 'end')
        self.layout_text.insert('end', text)
        self.layout_text.configure(state='disabled')
        self.lbl_validation.configure(text=error)

    def _render_state(self):
        busy = self._busy
        self.btn_select.configure(state='disabled' if busy else 'normal')
        self.btn_project_open.configure(state='disabled' if busy else 'normal')
        self.btn_project_save.configure(state='normal' if self._layout_valid and not busy else 'disabled')
        self.grid_switch.configure(state='disabled' if busy else 'normal')
        grid_state = 'normal' if self.use_grid.get() and not busy else 'disabled'
        for widget in (self.rows_entry, self.cols_entry, self.check_serpentine, self.check_rtl):
            widget.configure(state=grid_state)
        self.btn_start.configure(state='normal' if self._layout_valid and not busy else 'disabled',
                                 text=self.t('btn_start_disabled' if self.task.busy else 'btn_start'))
        state = self.project_job.state if self.project_job.busy else self.task.state
        self.btn_cancel.configure(state='normal' if state == TaskState.RUNNING else 'disabled',
                                  text=self.t('btn_cancelling' if state == TaskState.CANCELLING else 'btn_cancel'))
        self.btn_export.configure(state='normal' if self.task.state == TaskState.SUCCEEDED and not busy else 'disabled')
        self.btn_inspect.configure(state='normal' if self.task.state == TaskState.SUCCEEDED and not busy
                                  and self.task.inspection is not None else 'disabled')
        self.file_list.configure(state='disabled' if busy else 'normal')
        selection = self.file_list.curselection()
        for index, button in enumerate(self._order_buttons):
            enabled = not busy and len(self.image_paths) > 1
            if index < 2:
                enabled = enabled and bool(selection) and 0 <= selection[0] + (-1 if index == 0 else 1) < len(self.image_paths)
            button.configure(state='normal' if enabled else 'disabled')
        color = {TaskState.FAILED: '#ff8b80', TaskState.SUCCEEDED: '#8bd5a0',
                 TaskState.CANCELLED: '#f4c57a', TaskState.CANCELLING: '#f4c57a'}.get(state, 'gray75')
        status_key = ('status_project_' + self.project_job.kind if self.project_job.state == TaskState.RUNNING
                      else 'status_' + state.value)
        self.lbl_status.configure(text=self.t(status_key), text_color=color)

    def _clear_preview(self):
        if self._inspector is not None:
            self._inspector.close()
            self._inspector = None
        if self._preview_id is not None:
            self.after_cancel(self._preview_id)
            self._preview_id = None
        self.preview_label.configure(image=self._empty_preview, text=self.t('preview_placeholder'))
        self.preview_label.image = None
        self.lbl_size.configure(text='')

    def _on_inputs_changed(self, *args):
        if self._busy or self._applying_project:
            return
        self.task.reset(has_inputs=len(self.image_paths) >= 2)
        self._layout_notice = False
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self._clear_preview()
        self._refresh_layout()
        self._render_state()

    def _on_select_files(self):
        if self._busy:
            return
        paths = filedialog.askopenfilenames(title=self.t('dialog_select'), filetypes=[
            (self.t('file_filter_tiff'), '*.tif *.tiff'), (self.t('file_filter_all'), '*.*')])
        if paths:
            self._load_files(list(paths))

    def _load_files(self, paths):
        if self._busy:
            return
        if len(paths) < 2:
            messagebox.showwarning(self.t('msg_few'), self.t('msg_few_detail', n=len(paths)))
            return
        previous_count = len(self.image_paths)
        self.image_paths = natural_sort(paths)
        reset_grid = False
        if previous_count and previous_count != len(paths) and self.use_grid.get():
            try:
                self._get_request()
            except ValueError:
                self.use_grid.set(False)
                reset_grid = True
        self._refresh_file_list()
        self._on_inputs_changed()
        if reset_grid:
            self._layout_notice = True
            self._refresh_layout()
            self._append_log(self.t('layout_grid_reset'))
        self.tabs.set(self._tab_names['tab_files'])

    def _refresh_file_list(self, selected=None):
        self.file_list.delete(0, 'end')
        for index, path in enumerate(self.image_paths):
            self.file_list.insert('end', f'{index + 1:02d}  {os.path.basename(path)}')
        if selected is not None:
            self.file_list.selection_set(selected)
            self.file_list.see(selected)

    def _move_file(self, delta):
        if self._busy:
            return
        selection = self.file_list.curselection()
        if not selection:
            return
        index, target = selection[0], selection[0] + delta
        if 0 <= target < len(self.image_paths):
            self.image_paths[index], self.image_paths[target] = self.image_paths[target], self.image_paths[index]
            self._refresh_file_list(target)
            self._on_inputs_changed()

    def _reverse_files(self):
        if not self._busy:
            self.image_paths.reverse()
            self._refresh_file_list()
            self._on_inputs_changed()

    def _sort_files(self):
        if not self._busy:
            self.image_paths = natural_sort(self.image_paths)
            self._refresh_file_list()
            self._on_inputs_changed()

    def _append_log(self, message):
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')

    def _on_start(self):
        if self._busy:
            return
        try:
            request = self._get_request()
        except ValueError as error:
            messagebox.showwarning(self.t('msg_settings'), str(error))
            return
        self._clear_preview()
        self.log_text.delete('1.0', 'end')
        self._append_log(self.t('log_start'))
        self._append_log(self.t('log_loaded', n=len(request.paths)))
        for index, path in enumerate(request.paths):
            self._append_log(f'  {index + 1:02d}  {os.path.basename(path)}')
        self.tabs.set(self._tab_names['tab_log'])
        try:
            self.task.start(request)
        except Exception as error:
            self._append_log(self.t('log_error', e=error))
            self._render_state()
            return
        self.progress_bar.set(0)
        self.progress_bar.start()
        self._render_state()
        self._poll_task()

    def _on_cancel(self):
        active = self.project_job if self.project_job.busy else self.task
        if active.state == TaskState.RUNNING:
            active.cancel()
            self._append_log(self.t('log_cancelling'))
            self._render_state()

    def _on_save_project(self):
        if self._busy:
            return
        try:
            request = self._get_request()
        except ValueError as error:
            messagebox.showwarning(self.t('msg_settings'), str(error))
            return
        path = filedialog.asksaveasfilename(title=self.t('project_save'), defaultextension='.semproj',
                                           filetypes=[(self.t('project_filter'), '*.semproj')],
                                           initialfile='stitch.semproj')
        if not path:
            return
        inspection = self.task.inspection if self.task.state == TaskState.SUCCEEDED else None
        panorama = self.stitched_result if inspection is not None else None
        self._start_project_job('save', path, lambda cancel, log: save_project(
            path, request, inspection, panorama, cancelled=cancel, log=log))

    def _on_open_project(self):
        if self._busy:
            return
        path = filedialog.askopenfilename(title=self.t('project_open'), filetypes=[
            (self.t('project_filter'), '*.semproj'), (self.t('file_filter_all'), '*.*')])
        if path:
            self._start_project_job('open', path, lambda cancel, log: restore_project(path, cancel, log))

    def _start_project_job(self, kind, path, operation):
        self._project_path = path
        self.tabs.set(self._tab_names['tab_log'])
        self._append_log(self.t('status_project_' + kind))
        try:
            self.project_job.start(kind, operation)
        except Exception as error:
            self._append_log(self.t('log_error', e=error))
            self._render_state()
            return
        self.progress_bar.start()
        self._render_state()
        self._poll_project_job()

    def _poll_project_job(self):
        self._poll_id = None
        if self._closing:
            return
        for message in self.project_job.poll():
            self._append_log(message)
        if self.project_job.busy:
            self._poll_id = self.after(75, self._poll_project_job)
            return
        self.progress_bar.stop()
        if self.project_job.state == TaskState.SUCCEEDED:
            if self.project_job.kind == 'open':
                self._apply_project(self.project_job.result)
                self._append_log(self.t('project_restored', path=self._project_path))
                if self.stitched_result is None:
                    self._append_log(self.t('project_inputs'))
            else:
                self._append_log(self.t('project_saved', path=self.project_job.result))
        elif self.project_job.state == TaskState.CANCELLED:
            self._append_log(self.t('project_cancelled'))
        else:
            error = self.project_job.error or self.t('log_fail')
            self._append_log(error)
            messagebox.showerror(self.t('project_' + self.project_job.kind), error.rstrip().splitlines()[-1])
        self.project_job.result = None
        self.progress_bar.set(1 if self.task.state == TaskState.SUCCEEDED else 0)
        self._render_state()

    def _apply_project(self, restored):
        # Apply once on the Tk owner, preserving acquisition order exactly.
        self._clear_preview()
        request = restored.request
        self._applying_project = True
        try:
            self.image_paths = list(request.paths)
            self.use_grid.set(request.grid_shape is not None)
            if request.grid_shape is not None:
                self.rows.set(str(request.grid_shape[0]))
                self.cols.set(str(request.grid_shape[1]))
            self.serpentine.set(request.serpentine)
            self.first_rtl.set(request.first_row_rtl)
        finally:
            self._applying_project = False
        self.task.reset(has_inputs=True)
        if restored.panorama is not None:
            self.task.request = request
            self.task.result = restored.panorama
            self.task.inspection = restored.inspection
            self.task.metrics = restored.metrics
            self.task.state = TaskState.SUCCEEDED
            h, w = restored.panorama.shape
            self.lbl_size.configure(text=self.t('lbl_size', w=w, h=h))
            self._update_preview()
        self._layout_notice = False
        self._refresh_file_list()
        self._refresh_layout()

    def _poll_task(self):
        self._poll_id = None
        if self._closing:
            return
        for message in self.task.poll():
            self._append_log(message)
        if self.task.busy:
            self._poll_id = self.after(75, self._poll_task)
            return
        self.progress_bar.stop()
        self.progress_bar.set(1 if self.task.state == TaskState.SUCCEEDED else 0)
        self._render_state()
        if self.task.state == TaskState.SUCCEEDED:
            self._append_log(self.t('log_done'))
            h, w = self.stitched_result.shape
            self.lbl_size.configure(text=self.t('lbl_size', w=w, h=h))
            self._update_preview()
        elif self.task.state == TaskState.CANCELLED:
            self._append_log(self.t('log_cancelled'))
            self._clear_preview()
        elif self.task.state == TaskState.FAILED:
            self._append_log(self.task.error or self.t('log_fail'))
            self._append_log(self.t('log_fail'))
            self._clear_preview()

    def _on_close(self):
        self._closing = True
        if self._inspector is not None:
            self._inspector.close()
        self.task.cancel()
        self.project_job.close()
        for timer in (self._poll_id, self._preview_id):
            if timer is not None:
                self.after_cancel(timer)
        self.destroy()

    def _on_preview_resize(self, event):
        if self.stitched_result is not None and not self._closing:
            if self._preview_id is not None:
                self.after_cancel(self._preview_id)
            self._preview_id = self.after(120, self._update_preview)

    # ================================================================
    # 预览
    # ================================================================
    def _on_inspect(self):
        if self._busy or self.task.state != TaskState.SUCCEEDED or self.task.inspection is None:
            return
        if self._inspector is not None and not self._inspector.closed:
            self._inspector.lift()
            return
        self._inspector = SeamInspector(self, self.stitched_result, self.task.inspection, self.lang)

    def _update_preview(self):
        self._preview_id = None
        if self._closing or self.stitched_result is None:
            return
        # Normalize only the display copy; keep the full 16-bit result for export.
        img_arr = SEMStitcher._matching_image(self.stitched_result)
        h, w = img_arr.shape
        available_w = max(1, self.preview_label.winfo_width() - 24)
        available_h = max(1, self.preview_label.winfo_height() - 24)
        scale = min(available_w / w, available_h / h, PREVIEW_MAX / max(w, h), 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
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
        if self._busy or self.task.state != TaskState.SUCCEEDED or self.stitched_result is None:
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
        try:
            if (self.stitched_result.dtype == np.uint16
                    and os.path.splitext(path)[1].lower() not in (".tif", ".tiff", ".png")):
                raise ValueError("16-bit images must be exported as TIFF or PNG")
            if not imwrite(path, self.stitched_result):
                raise OSError(f"Cannot write image: {path}")
        except Exception as error:
            message = self.t("log_error", e=error)
            self._append_log(message)
            messagebox.showerror(self.t("dialog_export"), message)
            return
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
