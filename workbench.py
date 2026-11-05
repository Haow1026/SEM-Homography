"""Workbench widget layout; task lifecycle remains in the GUI controller."""
import tkinter as tk
import customtkinter as ctk
from PIL import Image

import ui_theme as theme


def button_style(primary=False):
    return dict(fg_color=theme.PRIMARY if primary else theme.SURFACE,
                hover_color=theme.PRIMARY_HOVER if primary else theme.SUBTLE,
                text_color=theme.PRIMARY_TEXT if primary else theme.TEXT,
                border_color=theme.PRIMARY if primary else theme.BORDER, border_width=1)


def build_workbench(app):
    app.configure(fg_color=theme.BACKGROUND)
    app.grid_columnconfigure(0, weight=1)
    app.grid_rowconfigure(2, weight=1)
    app._font_body = ctk.CTkFont(size=13)
    app._font_small = ctk.CTkFont(size=12)

    def label(parent, key, small=False, **kwargs):
        widget = ctk.CTkLabel(parent, text='', font=app._font_small if small else app._font_body,
                             text_color=theme.MUTED if small else theme.TEXT, **kwargs)
        app._reg(widget, key)
        return widget

    def button(parent, key, command, width=110, primary=False):
        widget = ctk.CTkButton(parent, text='', command=command, height=34, width=width,
                              corner_radius=6, font=app._font_body, **button_style(primary))
        app._reg(widget, key)
        return widget

    app.header = ctk.CTkFrame(app, fg_color='transparent')
    app.header.grid(row=0, column=0, sticky='ew', padx=16, pady=(12, 8))
    app.header.grid_columnconfigure(1, weight=1)
    heading = label(app.header, 'heading')
    heading.configure(font=ctk.CTkFont(size=18, weight='bold'))
    heading.grid(row=0, column=0, sticky='w')
    app.lbl_session = ctk.CTkLabel(app.header, text='', text_color=theme.MUTED,
                                  font=app._font_small, anchor='w')
    app.lbl_session.grid(row=0, column=1, sticky='w', padx=16)
    preferences = ctk.CTkFrame(app.header, fg_color='transparent')
    preferences.grid(row=0, column=2, sticky='e')
    label(preferences, 'theme_label', small=True).pack(side='left', padx=(0, 6))
    app.theme_menu = ctk.CTkOptionMenu(preferences, values=['System', 'Light', 'Dark'],
                                     command=app._set_theme, width=112, height=30,
                                     font=app._font_small, fg_color=theme.SUBTLE,
                                     button_color=theme.SUBTLE, button_hover_color=theme.BORDER,
                                     text_color=theme.TEXT)
    app.theme_menu.pack(side='left', padx=(0, 14))
    label(preferences, 'lang_label', small=True).pack(side='left', padx=(0, 6))
    app.lang_menu = ctk.CTkOptionMenu(preferences, values=['English', '简体中文', '日本語'],
                                    command=app._set_lang, width=104, height=30,
                                    font=app._font_small, fg_color=theme.SUBTLE,
                                    button_color=theme.SUBTLE, button_hover_color=theme.BORDER,
                                    text_color=theme.TEXT)
    app.lang_menu.set({'en': 'English', 'zh-Hans': '简体中文', 'ja': '日本語'}[app.lang])
    app.lang_menu.pack(side='left')

    app.toolbar = ctk.CTkFrame(app, fg_color='transparent')
    app.toolbar.grid(row=1, column=0, sticky='ew', padx=16, pady=(0, 10))
    app.toolbar.grid_columnconfigure(0, weight=1)
    project_actions = ctk.CTkFrame(app.toolbar, fg_color='transparent')
    project_actions.grid(row=0, column=0, sticky='w')
    app.btn_project_open = button(project_actions, 'btn_project_open', app._on_open_project)
    app.btn_project_open.pack(side='left', padx=(0, 8))
    app.btn_project_save = button(project_actions, 'btn_project_save', app._on_save_project)
    app.btn_project_save.pack(side='left', padx=(0, 8))
    app.btn_select = button(project_actions, 'btn_select', app._on_select_files, width=130)
    app.btn_select.pack(side='left')
    task_actions = ctk.CTkFrame(app.toolbar, fg_color='transparent')
    task_actions.grid(row=0, column=1, sticky='e')
    app.btn_start = button(task_actions, 'btn_start', app._on_start, width=128, primary=True)
    app.btn_start.pack(side='left', padx=(0, 8))
    app.btn_cancel = button(task_actions, 'btn_cancel', app._on_cancel, width=100)
    app.btn_cancel.pack(side='left', padx=(0, 8))
    app.btn_export = button(task_actions, 'btn_export', app._on_export, width=120)
    app.btn_export.pack(side='left')

    body = ctk.CTkFrame(app, fg_color='transparent')
    body.grid(row=2, column=0, sticky='nsew', padx=16, pady=(0, 8))
    body.grid_columnconfigure(1, weight=1)
    body.grid_rowconfigure(0, weight=1)
    app.sidebar = ctk.CTkFrame(body, width=310, fg_color=theme.SURFACE, corner_radius=8,
                              border_width=1, border_color=theme.BORDER)
    app.sidebar.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
    app.sidebar.grid_propagate(False)
    app.sidebar.grid_columnconfigure(0, weight=1)
    app.sidebar.grid_rowconfigure(1, weight=1)
    source_heading = ctk.CTkFrame(app.sidebar, fg_color='transparent')
    source_heading.grid(row=0, column=0, sticky='ew', padx=12, pady=(8, 2))
    source_heading.grid_columnconfigure(0, weight=1)
    label(source_heading, 'files_title').grid(row=0, column=0, sticky='w')
    app.lbl_count = ctk.CTkLabel(source_heading, text='', text_color=theme.MUTED, font=app._font_small)
    app.lbl_count.grid(row=0, column=1, sticky='e')
    list_frame = ctk.CTkFrame(app.sidebar, fg_color='transparent')
    list_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=4)
    list_frame.grid_rowconfigure(0, weight=1)
    list_frame.grid_columnconfigure(0, weight=1)
    app.file_list = tk.Listbox(list_frame, height=4, exportselection=False,
                               relief='flat', highlightthickness=1, activestyle='none',
                               selectborderwidth=0)
    app.file_list.grid(row=0, column=0, sticky='nsew')
    scrollbar = ctk.CTkScrollbar(list_frame, command=app.file_list.yview, width=12)
    scrollbar.grid(row=0, column=1, sticky='ns')
    horizontal = ctk.CTkScrollbar(list_frame, command=app.file_list.xview,
                                 orientation='horizontal', height=10)
    horizontal.grid(row=1, column=0, sticky='ew')
    app.file_list.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
    app.file_list.bind('<<ListboxSelect>>', lambda event: app._render_state())
    reorder = ctk.CTkFrame(app.sidebar, fg_color='transparent')
    reorder.grid(row=2, column=0, sticky='ew', padx=10, pady=(2, 6))
    reorder.grid_columnconfigure((0, 1), weight=1)
    app._order_buttons = []
    for index, (key, command) in enumerate((
        ('btn_up', lambda: app._move_file(-1)), ('btn_down', lambda: app._move_file(1)),
        ('btn_reverse', app._reverse_files), ('btn_sort', app._sort_files),
    )):
        widget = button(reorder, key, command, width=100)
        widget.configure(height=28, font=app._font_small)
        widget.grid(row=index//2, column=index % 2, sticky='ew', padx=2, pady=2)
        app._order_buttons.append(widget)
    app.btn_layout_toggle = ctk.CTkButton(app.sidebar, text='', command=app._toggle_layout,
                                         height=28, anchor='w', fg_color='transparent',
                                         hover_color=theme.SUBTLE, text_color=theme.TEXT,
                                         font=app._font_body, corner_radius=4)
    app.btn_layout_toggle.grid(row=3, column=0, sticky='ew', padx=10)
    app.layout_panel = ctk.CTkFrame(app.sidebar, fg_color='transparent')
    app.layout_panel.grid(row=4, column=0, sticky='ew', padx=12, pady=(2, 8))
    app.layout_panel.grid_columnconfigure(0, weight=1)
    app.grid_switch = ctk.CTkSwitch(app.layout_panel, text='', variable=app.use_grid,
                                   command=app._on_inputs_changed, font=app._font_small,
                                   text_color=theme.TEXT, progress_color=theme.PRIMARY,
                                   switch_width=32, switch_height=18)
    app.grid_switch.grid(row=0, column=0, sticky='w', pady=4)
    app._reg(app.grid_switch, 'grid_enabled')
    dimensions = ctk.CTkFrame(app.layout_panel, fg_color='transparent')
    dimensions.grid(row=1, column=0, sticky='ew', pady=4)
    for col, key, variable in ((0, 'rows', app.rows), (2, 'cols', app.cols)):
        label(dimensions, key, small=True).grid(row=0, column=col, padx=(0, 6))
        entry = ctk.CTkEntry(dimensions, width=52, height=28, textvariable=variable,
                            font=app._font_small, fg_color=theme.BACKGROUND, text_color=theme.TEXT,
                            border_color=theme.BORDER)
        entry.grid(row=0, column=col+1, padx=(0, 12))
        setattr(app, key + '_entry', entry)
    for row, key, variable, attribute in (
        (2, 'serpentine', app.serpentine, 'check_serpentine'),
        (3, 'first_rtl', app.first_rtl, 'check_rtl'),
    ):
        check = ctk.CTkCheckBox(app.layout_panel, text='', variable=variable,
                               command=app._on_inputs_changed, font=app._font_small,
                               text_color=theme.TEXT, checkbox_width=18, checkbox_height=18,
                               fg_color=theme.PRIMARY, checkmark_color=theme.PRIMARY_TEXT)
        check.grid(row=row, column=0, sticky='w', pady=3)
        app._reg(check, key)
        setattr(app, attribute, check)
    app.layout_text = ctk.CTkTextbox(app.layout_panel, height=64, font=app._font_small,
                                    fg_color=theme.BACKGROUND, text_color=theme.TEXT, wrap='word',
                                    corner_radius=6)
    app.layout_text.grid(row=4, column=0, sticky='ew', pady=(6, 0))
    app.lbl_validation = ctk.CTkLabel(app.sidebar, text='', text_color=theme.ERROR,
                                     font=app._font_small, wraplength=276, justify='left', anchor='w')
    app.lbl_validation.grid(row=5, column=0, sticky='ew', padx=12, pady=(0, 8))

    app.workspace = ctk.CTkFrame(body, fg_color=theme.SURFACE, corner_radius=8,
                                border_width=1, border_color=theme.BORDER)
    app.workspace.grid(row=0, column=1, sticky='nsew')
    app.workspace.grid_columnconfigure(0, weight=1)
    app.workspace.grid_rowconfigure(1, weight=1)
    preview_tools = ctk.CTkFrame(app.workspace, fg_color='transparent')
    preview_tools.grid(row=0, column=0, sticky='ew', padx=12, pady=8)
    preview_tools.grid_columnconfigure(0, weight=1)
    label(preview_tools, 'preview_title').grid(row=0, column=0, sticky='w')
    app.btn_inspect = button(preview_tools, 'btn_inspect', app._on_inspect, width=166)
    app.btn_inspect.grid(row=0, column=1, sticky='e')
    # A real placeholder keeps Tk from referencing a released PhotoImage.
    app._empty_preview = ctk.CTkImage(Image.new('RGBA', (1, 1)), size=(1, 1))
    app.preview_label = ctk.CTkLabel(app.workspace, text='', image=app._empty_preview,
                                    width=1, height=1, fg_color=theme.CANVAS, text_color=theme.CANVAS_TEXT,
                                    font=app._font_body, corner_radius=6)
    app.preview_label.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0, 8))
    app.preview_label.bind('<Configure>', app._on_preview_resize)
    label(app.workspace, 'fit_hint', small=True).grid(row=2, column=0, sticky='w', padx=12, pady=(0, 6))

    app.log_panel = ctk.CTkFrame(app, fg_color=theme.SURFACE, corner_radius=8,
                                border_width=1, border_color=theme.BORDER)
    app.log_panel.grid(row=3, column=0, sticky='ew', padx=16)
    app.log_panel.grid_columnconfigure(0, weight=1)
    app.btn_log_toggle = ctk.CTkButton(app.log_panel, text='', command=app._toggle_log,
                                      height=28, anchor='w', fg_color='transparent', hover_color=theme.SUBTLE,
                                      text_color=theme.MUTED, font=app._font_small, corner_radius=6)
    app.btn_log_toggle.grid(row=0, column=0, sticky='ew', padx=6, pady=3)
    app.log_text = ctk.CTkTextbox(app.log_panel, height=120, font=app._font_small,
                                 wrap='word', fg_color=theme.BACKGROUND, text_color=theme.TEXT)
    app.log_text.grid(row=1, column=0, sticky='ew', padx=10, pady=(0, 10))
    app.log_text.grid_remove()
    footer = ctk.CTkFrame(app, fg_color='transparent')
    footer.grid(row=4, column=0, sticky='ew', padx=16, pady=(6, 10))
    footer.grid_columnconfigure(0, weight=1)
    app.lbl_status = ctk.CTkLabel(footer, text='', anchor='w', text_color=theme.MUTED,
                                 font=app._font_small)
    app.lbl_status.grid(row=0, column=0, sticky='ew')
    app.lbl_size = ctk.CTkLabel(footer, text='', text_color=theme.MUTED, font=app._font_small)
    app.lbl_size.grid(row=0, column=1, sticky='e', padx=(12, 0))
    app.progress_bar = ctk.CTkProgressBar(footer, mode='indeterminate', height=4,
                                        fg_color=theme.BORDER, progress_color=theme.PRIMARY)
    app.progress_bar.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(3, 0))
    app.progress_bar.set(0)
    app._sync_native_palette(ctk.get_appearance_mode())
