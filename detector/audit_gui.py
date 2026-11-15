import os
import queue
import sqlite3
import threading
import tkinter as tk
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from urllib.error import URLError

from PIL import Image, ImageOps, ImageTk

from .audit_review import COLOR, REASONS, Review, Vision, crop_evidence, export_train, history_rows, make_box, scan_train
from .plate_reader import required_env


BG = '#0d1117'
PAPER = '#161b22'
RAISED = '#21262d'
HOVER = '#30363d'
BORDER = '#30363d'
INK = '#e6edf3'
MUTED = '#8b949e'
DISABLED = '#545d68'
CANVAS = '#010409'
ACCENT = '#2f81f7'
ACCENT_HOVER = '#58a6ff'
OK = '#3fb950'
WARN = '#d29922'
BAD = '#f85149'
STATE_COLORS = {
    'correcta': OK,
    'mal_delimitada': WARN,
    'no_es_patente': BAD,
    'dudosa': WARN,
    'error': BAD,
    'pending': MUTED,
}
STATES = {
    'correcta': 'Aprobada por IA',
    'mal_delimitada': 'Mal delimitada',
    'no_es_patente': 'No es patente',
    'dudosa': 'Dudosa',
    'error': 'Error de auditoría',
    'pending': 'Por auditar',
}
DECISIONS = {
    'keep': 'Confirmada',
    'remove': 'Quitada',
    'adjust': 'Redibujada',
    'add': 'Agregada',
}


class AuditWindow:
    def __init__(self, root, train, work):
        self.root, self.train, self.work = root, train, work
        self.cases = scan_train(train)
        self.variants = Counter(case.photo for case in self.cases)
        self.model = os.environ.get('AUDIT_MODEL', '').strip()
        self.review = Review(work / 'revision.sqlite3')
        self.log_path = work / 'auditoria.log'
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.running = False
        self.selected = None
        self.photo = None
        self.drag = None
        self.draw_mode = None
        self.rows = {}
        self.status = tk.StringVar(value='Selecciona una alerta o inicia la auditoría visual.')
        self.stats = {name: tk.StringVar(value='—') for name in ('boxes', 'correcta', 'pending', 'view')}
        self.progress_value = tk.DoubleVar()
        self.heading = tk.StringVar(value='Sin selección')
        self.detail = tk.StringVar(value='Selecciona una alerta de la cola.')
        self.filename = tk.StringVar(value='Imagen completa')
        self.history_count = tk.StringVar()
        self.history_cases = {}
        self.show_all = tk.BooleanVar()
        self.follow = tk.BooleanVar(value=True)
        self.build()
        if self.log_path.is_file():
            with self.log_path.open(encoding='utf-8') as saved:
                self.show_log(''.join(deque(saved, maxlen=120)))
        self.record(f'Sesión abierta · {sum(len(case.images) for case in self.cases)} imágenes · modelo {self.model or "sin configurar"}')
        self.refresh()
        self.root.after(150, self.poll)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def build(self):
        root = self.root
        root.title('EV03 · Revisión de patentes')
        root.geometry('1280x830+30+10')
        root.minsize(1000, 790)
        root.configure(bg=BG)
        self.apply_style()
        header = ttk.Frame(root, padding=(24, 18, 24, 6))
        header.pack(fill='x')
        titles = ttk.Frame(header)
        titles.pack(side='left')
        ttk.Label(titles, text='Revisión de patentes', style='Title.TLabel').pack(anchor='w')
        ttk.Label(titles, text='Auditoría visual de las cajas de entrenamiento', style='Muted.TLabel').pack(anchor='w')
        cards = ttk.Frame(header)
        cards.pack(side='right')
        for key, label, color in (('boxes', 'Cajas', INK), ('correcta', 'Aprobadas por IA', OK), ('pending', 'Por auditar', WARN), ('view', 'Imágenes en la cola', INK)):
            card = tk.Frame(cards, bg=PAPER, highlightbackground=BORDER, highlightthickness=1)
            card.pack(side='left', padx=(8, 0))
            tk.Label(card, textvariable=self.stats[key], bg=PAPER, fg=color, font=('Segoe UI', 17, 'bold')).pack(anchor='w', padx=14, pady=(8, 0))
            tk.Label(card, text=label, bg=PAPER, fg=MUTED, font=('Segoe UI', 9)).pack(anchor='w', padx=14, pady=(0, 8))
        tools = ttk.Frame(root, padding=(24, 10, 24, 10))
        tools.pack(fill='x')
        self.batch = ttk.Button(tools, text='Auditar todas', command=self.audit, style='Primary.TButton')
        self.batch.pack(side='left', padx=(0, 8))
        self.sample = ttk.Button(tools, text='Auditar 50 cajas', command=lambda: self.audit(50))
        self.sample.pack(side='left', padx=(0, 8))
        self.pause = ttk.Button(tools, text='Pausar', command=self.stop.set, state='disabled', style='Warn.TButton')
        self.pause.pack(side='left', padx=(0, 14))
        ttk.Checkbutton(tools, text='Seguir auditoría', variable=self.follow).pack(side='left')
        ttk.Button(tools, text='Exportar copia de train', command=self.export).pack(side='right')
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill='both', expand=True, padx=24)
        main = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.queue_tab = main
        self.tabs.add(main, text='Cola de revisión')
        self.build_history()
        self.tabs.bind('<<NotebookTabChanged>>', lambda event: self.refresh_history())
        main.rowconfigure(1, weight=1)
        main.columnconfigure(1, weight=1)
        ttk.Label(main, text='Cola de revisión', style='Section.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(main, textvariable=self.filename, style='Section.TLabel').grid(row=0, column=1, sticky='w', padx=14)
        ttk.Label(main, text='Detalle de la caja', style='Section.TLabel').grid(row=0, column=2, sticky='w')
        sidebar = ttk.Frame(main, width=270)
        sidebar.grid(row=1, column=0, sticky='nsew', pady=(8, 0))
        sidebar.pack_propagate(False)
        ttk.Checkbutton(sidebar, text='Mostrar todas las cajas', variable=self.show_all, command=self.refresh).pack(side='bottom', anchor='w', pady=(10, 0))
        tree_frame = self.card(sidebar)
        tree_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(tree_frame, show='tree', selectmode='browse', height=12)
        self.tree.column('#0', width=242, minwidth=210, stretch=True)
        scroll = ttk.Scrollbar(tree_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.tree.pack(side='left', fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.choose)
        self.tag_states(self.tree)
        canvas_frame = self.card(main, bg=CANVAS)
        canvas_frame.grid(row=1, column=1, sticky='nsew', padx=14, pady=(8, 0))
        self.canvas = tk.Canvas(canvas_frame, bg=CANVAS, highlightthickness=0, width=450, height=360)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.canvas.bind('<ButtonPress-1>', self.drag_start)
        self.canvas.bind('<B1-Motion>', self.drag_move)
        self.canvas.bind('<ButtonRelease-1>', self.drag_end)
        side = ttk.Frame(main, width=260)
        side.grid(row=1, column=2, sticky='ns', pady=(8, 0))
        side.pack_propagate(False)
        crop_frame = self.card(side, bg=CANVAS)
        crop_frame.pack(fill='x')
        self.crop = tk.Canvas(crop_frame, bg=CANVAS, width=258, height=104, highlightthickness=0)
        self.crop.pack(fill='x')
        self.badge = tk.Label(side, textvariable=self.heading, bg=PAPER, fg=MUTED, font=('Segoe UI', 10, 'bold'), anchor='w', padx=10, pady=5)
        self.badge.pack(fill='x', pady=(12, 0))
        ttk.Label(side, textvariable=self.detail, wraplength=250, justify='left').pack(fill='x', pady=8)
        self.actions = ttk.Frame(side)
        self.actions.pack(fill='x', anchor='n')
        log_area = ttk.Frame(root, padding=(24, 10, 24, 0))
        log_area.pack(fill='x')
        ttk.Label(log_area, text='Registro de auditoría', style='Section.TLabel').pack(anchor='w')
        log_frame = self.card(log_area, bg=CANVAS)
        log_frame.pack(fill='x', pady=(6, 0))
        self.log = tk.Text(log_frame, height=3, wrap='none', bg=CANVAS, fg='#9fb1c5', insertbackground=INK, relief='flat', padx=10, pady=6, font=('Consolas', 10), state='disabled')
        log_scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side='right', fill='y')
        self.log.pack(side='left', fill='x', expand=True)
        footer = ttk.Frame(root, padding=(24, 10, 24, 12))
        footer.pack(fill='x')
        ttk.Label(footer, textvariable=self.status, style='Muted.TLabel').pack(side='left')
        ttk.Progressbar(footer, variable=self.progress_value, maximum=1, length=220, style='Audit.Horizontal.TProgressbar').pack(side='right')
        self.draw()
        self.darken_titlebar()

    def card(self, parent, bg=PAPER):
        return tk.Frame(parent, bg=bg, highlightbackground=BORDER, highlightthickness=1)

    def tag_states(self, tree):
        for state, color in STATE_COLORS.items():
            if state not in {'correcta', 'pending'}:
                tree.tag_configure(state, foreground=color)

    def apply_style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background=BG, foreground=INK, bordercolor=BORDER, lightcolor=BG, darkcolor=BG, troughcolor=PAPER, focuscolor=BG)
        style.configure('Title.TLabel', font=('Segoe UI', 21, 'bold'))
        style.configure('Section.TLabel', font=('Segoe UI', 10, 'bold'), foreground=MUTED)
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('TButton', padding=(14, 6), background=RAISED, foreground=INK, bordercolor=BORDER, borderwidth=1, relief='flat')
        style.map('TButton', background=[('disabled', PAPER), ('pressed', HOVER), ('active', HOVER)], foreground=[('disabled', DISABLED)], bordercolor=[('active', MUTED)])
        style.configure('Primary.TButton', background=ACCENT, foreground='#ffffff', bordercolor=ACCENT)
        style.map('Primary.TButton', background=[('disabled', PAPER), ('pressed', ACCENT_HOVER), ('active', ACCENT_HOVER)], foreground=[('disabled', DISABLED)], bordercolor=[('disabled', BORDER), ('active', ACCENT_HOVER)])
        style.configure('Ok.TButton', foreground=OK)
        style.configure('Warn.TButton', foreground=WARN)
        style.configure('Danger.TButton', foreground=BAD)
        style.map('Ok.TButton', foreground=[('disabled', DISABLED)])
        style.map('Warn.TButton', foreground=[('disabled', DISABLED)])
        style.map('Danger.TButton', foreground=[('disabled', DISABLED)])
        style.configure('TCheckbutton', background=BG, foreground=INK, indicatorcolor=RAISED, indicatorbackground=RAISED, indicatorforeground=INK)
        style.map('TCheckbutton', background=[('active', BG)], indicatorcolor=[('selected', ACCENT), ('!selected', RAISED)])
        style.configure('Treeview', background=PAPER, fieldbackground=PAPER, foreground=INK, rowheight=32, borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f3b63')], foreground=[('selected', '#ffffff')])
        style.configure('Treeview.Heading', background=RAISED, foreground=MUTED, font=('Segoe UI', 9, 'bold'), relief='flat', padding=(10, 7))
        style.map('Treeview.Heading', background=[('active', HOVER)])
        style.configure('TNotebook', background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure('TNotebook.Tab', background=BG, foreground=MUTED, padding=(20, 9), borderwidth=1, bordercolor=BORDER)
        style.map('TNotebook.Tab', background=[('selected', RAISED), ('active', PAPER)], foreground=[('selected', INK), ('active', INK)], expand=[('selected', (0, 0, 0, 0))], padding=[('selected', (20, 9))])
        style.layout('TNotebook.Tab', [('Notebook.tab', {'sticky': 'nswe', 'children': [('Notebook.padding', {'side': 'top', 'sticky': 'nswe', 'children': [('Notebook.label', {'side': 'top', 'sticky': ''})]})]})])
        style.layout('Vertical.TScrollbar', [('Vertical.Scrollbar.trough', {'sticky': 'ns', 'children': [('Vertical.Scrollbar.thumb', {'expand': '1', 'sticky': 'nswe'})]})])
        style.configure('Vertical.TScrollbar', background=RAISED, troughcolor=PAPER, bordercolor=PAPER, gripcount=0, width=10)
        style.map('Vertical.TScrollbar', background=[('active', HOVER)])
        style.configure('Audit.Horizontal.TProgressbar', background=ACCENT, troughcolor=RAISED, bordercolor=RAISED, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)

    def darken_titlebar(self):
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
        except (AttributeError, OSError):
            pass

    def build_history(self):
        frame = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.history_tab = frame
        self.tabs.add(frame, text='Historial')
        ttk.Label(frame, textvariable=self.history_count, style='Muted.TLabel').pack(anchor='w', pady=(0, 8))
        table = self.card(frame)
        table.pack(fill='both', expand=True)
        columns = (('image', 'Imagen', 420), ('box', 'Caja', 60), ('state', 'Veredicto', 150), ('reason', 'Motivo', 300), ('decision', 'Decisión humana', 150))
        self.history = ttk.Treeview(table, columns=[column[0] for column in columns], show='headings', selectmode='browse')
        for key, title, width in columns:
            self.history.heading(key, text=title, anchor='w')
            self.history.column(key, width=width, minwidth=50, anchor='w', stretch=key in {'image', 'reason'})
        scroll = ttk.Scrollbar(table, command=self.history.yview)
        self.history.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.history.pack(side='left', fill='both', expand=True)
        self.history.bind('<<TreeviewSelect>>', self.open_history)
        self.tag_states(self.history)

    def refresh_history(self):
        if self.tabs.select() != str(self.history_tab):
            return
        rows = history_rows(self.cases, self.review, self.model)
        self.history_cases.clear()
        self.history.delete(*self.history.get_children())
        for case, target, state, reason, action in rows:
            item = self.history.insert('', 'end', values=(
                case.images[0].name,
                target.split('-')[-1],
                STATES.get(state, state),
                REASONS.get(reason, ('', '—'))[1],
                DECISIONS.get(action, action or '—'),
            ), tags=(state,))
            self.history_cases[item] = case, target
        self.history_count.set(f'Cajas auditadas: {len(rows):,}'.replace(',', '.') + '. Selecciona una para abrirla en la cola de revisión.')

    def open_history(self, event=None):
        selection = self.history.selection()
        if not selection or selection[0] not in self.history_cases:
            return
        if self.running:
            self.follow.set(False)
        self.tabs.select(self.queue_tab)
        self.preview(*self.history_cases[selection[0]])

    def show_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.configure(state='disabled')

    def record(self, message):
        line = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n'
        self.show_log(line)
        try:
            with self.log_path.open('a', encoding='utf-8') as saved:
                saved.write(line)
        except OSError:
            self.show_log('No se pudo guardar el registro en disco.\n')

    def refresh(self, keep_preview=False):
        self.refresh_history()
        old = self.selected
        groups = defaultdict(list)
        counts = defaultdict(int)
        for case in self.cases:
            decisions = self.review.decisions(case)
            inspections = self.review.inspections(case, self.model)
            targets = []
            if decisions.get('case', ('', None))[0] != 'exclude':
                if self.variants[case.photo] > 1 and ('conflict' not in decisions or self.show_all.get()):
                    targets.append(('conflict', 'Variante de etiquetas', 'dudosa'))
                for box in case.boxes:
                    state, reason = inspections.get(box.key, ('pending', None))
                    counts['boxes'] += 1
                    counts[state] += 1
                    resolved = box.key in decisions
                    if self.show_all.get() or (not resolved and state not in {'correcta', 'pending'}):
                        targets.append((box.key, f'Caja {box.key.split("-")[-1]} · {STATES[state]}', state))
                for key, reason in case.errors.items():
                    negative = not case.boxes and decisions.get('image', ('', None))[0] == 'negative'
                    if self.show_all.get() or (key not in decisions and not negative):
                        targets.append((key, reason, 'error'))
            if self.show_all.get() and not targets:
                targets.append(('case', 'Decisión de la imagen', 'pending'))
            if targets:
                groups[case.photo].append((case, targets))
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        selected_row = None
        for photo, variants in groups.items():
            name = variants[0][0].images[0].name
            parent = self.tree.insert('', 'end', text=name, open=len(groups) < 15)
            for case, targets in variants:
                for target, title, tag in targets:
                    row = self.tree.insert(parent, 'end', text=title, tags=(tag,))
                    self.rows[row] = case, target
                    if old and (case.key, target) == (old[0].key, old[1]):
                        selected_row = row
        for key, value in (('boxes', counts['boxes']), ('correcta', counts['correcta']), ('pending', counts['pending']), ('view', len(groups))):
            self.stats[key].set(f'{value:,}'.replace(',', '.'))
        if keep_preview:
            return
        selected_row = selected_row or next(iter(self.rows), None)
        if selected_row:
            self.tree.selection_set(selected_row)
            self.tree.see(selected_row)
            self.choose()
        else:
            self.selected = None
            self.photo = None
            self.filename.set('Imagen completa')
            self.heading.set('Sin selección')
            self.badge.configure(fg=MUTED)
            self.detail.set('No hay alertas en la cola. Puedes auditar más cajas o mostrar todas.')
            self.update_actions()
            self.draw()

    def choose(self, event=None):
        selection = self.tree.selection()
        if not selection:
            return
        row = selection[0]
        if row not in self.rows:
            children = self.tree.get_children(row)
            if children:
                self.tree.selection_set(children[0])
            return
        if self.running:
            self.follow.set(False)
        self.preview(*self.rows[row])

    def preview(self, case, target):
        selected = case, target
        if self.selected and (self.selected[0].key, self.selected[1]) == (selected[0].key, selected[1]):
            self.show_details()
            return
        self.selected = selected
        self.draw_mode = None
        try:
            with Image.open(selected[0].images[0]) as image:
                self.photo = image.convert('RGB')
        except (OSError, Image.DecompressionBombError):
            self.photo = None
        self.show_details()

    def active_box(self):
        if self.selected:
            case, target = self.selected
            return next((box for box in case.boxes if box.key == target), None)
        return None

    def show_details(self):
        case, target = self.selected
        variants = self.variants[case.photo] > 1
        image_names = f'\n\n{", ".join(image.name for image in case.images)}' if variants else ''
        suffix = '\nEsta foto tiene variantes de etiquetas.' if variants else ''
        self.filename.set(case.images[0].name)
        state, reason = self.review.inspections(case, self.model).get(target, ('pending', None))
        description = case.errors.get(target) or REASONS.get(reason, ('', STATES[state]))[1]
        if target == 'conflict':
            description = 'Confirma cómo tratar esta variante antes de cerrar el conflicto.'
        if state == 'error':
            description = 'El modelo no devolvió un veredicto válido. Puedes volver a auditar o decidir aquí.'
        decision = self.review.decisions(case).get(target)
        saved = '\nDecisión humana guardada' if decision else ''
        box = self.active_box()
        self.heading.set(f'Caja {target.split("-")[-1]} · {STATES[state]}' if box else 'Imagen completa')
        self.badge.configure(fg=STATE_COLORS[state] if box else MUTED)
        self.detail.set(f'{description}{saved}{suffix}{image_names}')
        self.update_actions()
        self.draw()

    def update_actions(self):
        for child in self.actions.winfo_children():
            child.destroy()
        if not self.selected:
            return
        case, target = self.selected
        buttons = [('Excluir imagen', lambda: self.decide('case', 'exclude', confirm=True))]
        if self.photo is not None:
            buttons.insert(0, ('Dibujar caja nueva', lambda: self.start_drawing('add')))
        if target == 'conflict':
            buttons.insert(0, ('Confirmar variante', lambda: self.decide('conflict', 'keep')))
        if self.active_box():
            buttons[0:0] = [
                ('Confirmar caja', lambda: self.decide(target, 'keep')),
                ('Redibujar caja', lambda: self.start_drawing('adjust')),
                ('Quitar caja', lambda: self.decide(target, 'remove', confirm=True)),
            ]
            if not self.running:
                buttons.insert(0, ('Auditar esta caja', lambda: self.audit(selection=(case, self.active_box()))))
        elif target in case.errors and target not in {'ambiguous', 'unreadable'}:
            if not case.boxes:
                buttons.insert(0, ('Confirmar imagen negativa', lambda: self.decide('image', 'negative', confirm=True)))
            if target.startswith('line-'):
                buttons.insert(0, ('Descartar línea inválida', lambda: self.decide(target, 'remove', confirm=True)))
        if self.review.decisions(case).get('case', ('', None))[0] == 'exclude':
            buttons = [('Reincorporar imagen', lambda: self.decide('case', 'include'))]
        for label, command in buttons:
            ttk.Button(self.actions, text=label, command=command, style=self.action_style(label)).pack(fill='x', pady=(0, 5))

    def action_style(self, label):
        if label.startswith(('Confirmar', 'Reincorporar')):
            return 'Ok.TButton'
        if label.startswith(('Quitar', 'Excluir', 'Descartar')):
            return 'Danger.TButton'
        return 'TButton'

    def draw(self):
        self.canvas.delete('all')
        self.crop.delete('all')
        if self.photo is None:
            for canvas, label in [(self.canvas, 'Sin imagen seleccionada'), (self.crop, 'Sin recorte')]:
                canvas.create_text(max(canvas.winfo_width(), 245) / 2, max(canvas.winfo_height(), 120) / 2, text=label, fill=MUTED)
            return
        canvas = self.canvas
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        image = self.photo.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        start_x, start_y = (width - image.width) / 2, (height - image.height) / 2
        self.area = start_x, start_y, image.width, image.height
        self.full_tk = ImageTk.PhotoImage(image)
        canvas.create_image(start_x, start_y, anchor='nw', image=self.full_tk)
        case, target = self.selected
        decisions = self.review.decisions(case)
        for box in case.boxes:
            action, coords = decisions.get(box.key, ('', None))
            if action != 'remove':
                self.draw_box(make_box(box.key, coords) if action == 'adjust' else box, target)
        for key, (action, coords) in decisions.items():
            if action == 'add':
                self.draw_box(make_box(key, coords), target)
        box = self.active_box()
        if box:
            action, coords = decisions.get(box.key, ('', None))
            box = make_box(box.key, coords) if action == 'adjust' else box
            crop = ImageOps.contain(crop_evidence(case.images[0], box), (250, 98), Image.Resampling.LANCZOS)
            self.crop_tk = ImageTk.PhotoImage(crop)
            self.crop.create_image(129, 52, image=self.crop_tk)
        else:
            self.crop.create_text(129, 52, text='Selecciona una caja', fill=MUTED)

    def draw_box(self, box, target):
        x, y, width, height = self.area
        left, top, right, bottom = box.pixels((width, height))
        focused = box.key == target
        self.canvas.create_rectangle(x + left, y + top, x + right, y + bottom, outline=COLOR if focused else '#9ddae0', width=3 if focused else 1)

    def decide(self, target, action, coords=None, confirm=False):
        if confirm and not messagebox.askyesno('Confirmar decisión', '¿Aplicar esta decisión a la copia de entrenamiento?', parent=self.root):
            return
        case, _ = self.selected
        try:
            self.review.save_decision(case, target, action, coords)
        except sqlite3.Error:
            self.status.set('No se pudo guardar. Conservamos la selección para que puedas reintentar.')
            return
        self.status.set('Decisión guardada.')
        self.record(f'{case.images[0].name} · {target} · decisión humana guardada')
        self.refresh()

    def start_drawing(self, action):
        self.draw_mode = action
        self.status.set('Arrastra el cursor sobre la imagen para dibujar la caja.')

    def drag_start(self, event):
        if self.draw_mode:
            self.drag = event.x, event.y

    def drag_move(self, event):
        if self.drag:
            self.canvas.delete('drag')
            self.canvas.create_rectangle(*self.drag, event.x, event.y, outline=COLOR, width=2, tags='drag')

    def drag_end(self, event):
        if not self.drag:
            return
        x, y, width, height = self.area
        start_x, start_y = self.drag
        self.drag = None
        self.canvas.delete('drag')
        left, right = sorted(max(0, min(1, (point - x) / width)) for point in [start_x, event.x])
        top, bottom = sorted(max(0, min(1, (point - y) / height)) for point in [start_y, event.y])
        if left == right or top == bottom:
            return
        coords = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        action = self.draw_mode
        self.draw_mode = None
        case, target = self.selected
        if action == 'add':
            target = f'added-{len(self.review.decisions(case)) + 1}'
        self.decide(target, action, coords, confirm=True)

    def audit(self, limit=None, selection=None):
        try:
            vision = Vision()
        except ValueError as error:
            self.status.set(str(error))
            return
        tasks = []
        for case in self.cases:
            if {'unreadable', 'ambiguous'} & case.errors.keys():
                continue
            inspections = self.review.inspections(case, self.model)
            decisions = self.review.decisions(case)
            if decisions.get('case', ('', None))[0] == 'exclude':
                continue
            tasks.extend(
                (case, box) for box in case.boxes
                if box.key not in decisions and inspections.get(box.key, ('pending', None))[0] in {'pending', 'error'}
            )
        tasks = [selection] if selection else tasks[:limit]
        if not tasks:
            self.status.set('No quedan cajas por auditar o reintentar.')
            return
        self.stop.clear()
        self.progress_value.set(0)
        self.set_running(True)
        self.update_actions()
        if selection:
            self.follow.set(True)
        self.status.set(f'Auditando {len(tasks)} cajas con {vision.model}.')
        self.record(f'Inicio del lote · {len(tasks)} cajas · {vision.model}')
        threading.Thread(target=self.run_audit, args=(vision, tasks), daemon=True).start()

    def run_audit(self, vision, tasks):
        message = 'Terminó la auditoría de este lote.'
        for index, (case, box) in enumerate(tasks, 1):
            if self.stop.is_set():
                message = 'Auditoría pausada. Puedes continuar las cajas pendientes.'
                break
            self.events.put(('start', case, box, index, len(tasks)))
            try:
                result, reason = vision.inspect(case, box)
            except (OSError, URLError, ValueError):
                self.events.put(('result', case, box, 'error', '', index, len(tasks)))
                message = 'No se pudo obtener un veredicto. Comprueba que Ollama esté abierto y que el modelo esté descargado.'
                break
            self.events.put(('result', case, box, result, reason, index, len(tasks)))
        self.events.put(('done', message))

    def set_running(self, running):
        self.running = running
        for button in (self.sample, self.batch):
            button.configure(state='disabled' if running else 'normal')
        self.pause.configure(state='normal' if running else 'disabled')

    def poll(self):
        refresh = False
        while not self.events.empty():
            event, *payload = self.events.get_nowait()
            if event == 'done':
                self.set_running(False)
                self.status.set(payload[0])
                self.record(payload[0])
                self.update_actions()
            elif event == 'start':
                case, box, index, total = payload
                self.status.set(f'Evaluando caja {index} de {total} con {self.model}.')
                self.record(f'{index}/{total} · {case.images[0].name} · caja {box.key.split("-")[-1]} · enviando recorte al modelo local')
                if self.follow.get():
                    self.preview(case, box.key)
            else:
                case, box, result, reason, index, total = payload
                try:
                    self.review.save_inspection(case, box, self.model, result, reason)
                except sqlite3.Error:
                    self.stop.set()
                    self.status.set('No se pudo guardar el resultado. Revisa el acceso a la carpeta de trabajo.')
                    continue
                self.status.set(f'Auditoría visual · {index} de {total} cajas')
                self.progress_value.set(index / total)
                note = REASONS.get(reason, ('', 'Comprueba Ollama y vuelve a auditar'))[1]
                self.record(f'{index}/{total} · {STATES[result]} · {note}')
                if self.follow.get():
                    self.preview(case, box.key)
                refresh = True
        if refresh:
            self.refresh(keep_preview=self.follow.get())
        self.root.after(150, self.poll)

    def export(self):
        if self.running:
            self.status.set('Pausa la auditoría antes de exportar.')
            return
        if not messagebox.askyesno('Exportar', '¿Crear una copia de train? Las etiquetas válidas pendientes conservan su versión de origen.', parent=self.root):
            return
        try:
            fresh = scan_train(self.train)
            destination = self.work / f'train-revisado-{len(list(self.work.glob("train-revisado-*"))) + 1:03d}'
            export_train(fresh, self.review, destination, self.model)
        except (OSError, ValueError, sqlite3.Error) as error:
            self.status.set(str(error))
            return
        self.status.set(f'Copia exportada en {destination}')
        self.record(f'Copia exportada · {destination.name}')

    def close(self):
        self.stop.set()
        self.review.db.close()
        self.root.destroy()


def main():
    try:
        dataset = Path(required_env('DATASET_DIR')).resolve(strict=True)
        work = Path(required_env('AUDIT_WORK_DIR')).resolve()
        checkout = Path(__file__).resolve().parent
        if any(work == root or work.is_relative_to(root) for root in (dataset, checkout)):
            raise ValueError('AUDIT_WORK_DIR debe estar fuera del corpus y del repositorio')
        train = (dataset / 'train').resolve(strict=True)
        if train.parent != dataset:
            raise ValueError('train debe estar dentro de DATASET_DIR')
        work.mkdir(parents=True, exist_ok=True)
        root = tk.Tk()
        AuditWindow(root, train, work)
        root.mainloop()
    except (OSError, ValueError, sqlite3.Error, tk.TclError) as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    main()
