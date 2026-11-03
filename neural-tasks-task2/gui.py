import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import data_loader as dl
import models as mdl
import metrics as mtr

BG = '#1a1a2e'
PANEL_BG = '#16213e'
ACCENT = '#0f3460'
BTN_COLOR = '#e94560'
BTN_FG = '#ffffff'
FG = '#e0e0e0'
FG_DIM = '#a0a0b0'
ENTRY_BG = '#0f3460'
SUCCESS = '#4caf50'
ERROR = '#f44336'
WARN = '#ff9800'

CLASS_COLORS = ['#e94560', '#00d4ff']
BOUNDARY_COLOR = '#f5d042'

FONT_HEADER = ('Segoe UI', 13, 'bold')
FONT_LABEL = ('Segoe UI', 9)
FONT_ENTRY = ('Consolas', 10)
FONT_MONO = ('Consolas', 9)
FONT_RESULT = ('Consolas', 10, 'bold')


def _styled_label(parent, text, font=None, fg=FG, **kw):
    f = font or FONT_LABEL
    return tk.Label(parent, text=text, bg=parent['bg'], fg=fg, font=f, **kw)


def _styled_entry(parent, textvariable, width=10):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=ENTRY_BG, fg=FG, insertbackground=FG,
                    font=FONT_ENTRY, relief='flat', bd=4)


def _styled_button(parent, text, command, width=20, bg=BTN_COLOR):
    btn = tk.Button(parent, text=text, command=command, width=width,
                    bg=bg, fg=BTN_FG, font=('Segoe UI', 10, 'bold'),
                    relief='flat', bd=0, activebackground=ACCENT,
                    activeforeground=BTN_FG, cursor='hand2')
    btn.bind('<Enter>', lambda e: btn.config(bg=ACCENT))
    btn.bind('<Leave>', lambda e: btn.config(bg=bg))
    return btn


def _styled_combo(parent, values, textvariable, width=22):
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Dark.TCombobox',
                    fieldbackground=ENTRY_BG,
                    background=ENTRY_BG,
                    foreground=FG,
                    arrowcolor=FG,
                    selectbackground=ACCENT,
                    selectforeground=FG)
    style.map('Dark.TCombobox',
              fieldbackground=[('readonly', ENTRY_BG), ('!disabled', ENTRY_BG)],
              foreground=[('readonly', FG), ('!disabled', FG)],
              selectbackground=[('readonly', ENTRY_BG), ('!disabled', ENTRY_BG)],
              selectforeground=[('readonly', FG), ('!disabled', FG)],
              background=[('readonly', ENTRY_BG), ('!disabled', ENTRY_BG)])
    cb = ttk.Combobox(parent, values=values, textvariable=textvariable,
                      width=width, style='Dark.TCombobox', state='readonly',
                      font=FONT_ENTRY, exportselection=False)
    cb.bind('<<ComboboxSelected>>', lambda e: cb.selection_clear())
    cb.bind('<FocusOut>', lambda e: cb.selection_clear())
    return cb


class PenguinApp(tk.Tk):

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self.title('Penguin Classifier - Perceptron & Adaline Lab')
        self.configure(bg=BG)
        self.option_add('*TCombobox*Listbox.background', ENTRY_BG)
        self.option_add('*TCombobox*Listbox.foreground', FG)
        self.option_add('*TCombobox*Listbox.selectBackground', ACCENT)
        self.option_add('*TCombobox*Listbox.selectForeground', FG)
        self.resizable(True, True)

        self._model = None
        self._scaler = None
        self._feat1 = None
        self._feat2 = None
        self._class1 = None
        self._class2 = None
        self._X_test = None
        self._y_test = None
        self._X_train = None
        self._y_train = None
        self._test_point = None

        self._build_ui()
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.mainloop()

    def _on_close(self):
        try:
            self.unbind_all('<MouseWheel>')
        except Exception:
            pass
        self.destroy()

    def _build_ui(self):
        title_bar = tk.Frame(self, bg=ACCENT, height=50)
        title_bar.pack(fill='x', side='top')
        tk.Label(title_bar, text='Penguin Neural Classifier - Perceptron & Adaline',
                 bg=ACCENT, fg=FG, font=('Segoe UI', 14, 'bold'),
                 pady=12).pack(side='left', padx=20)

        main_frame = tk.Frame(self, bg=BG)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        left = tk.Frame(main_frame, bg=PANEL_BG, width=340, bd=0)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)

        right = tk.Frame(main_frame, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._build_controls(left)
        self._build_canvas(right)

    def _build_controls(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL_BG, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL_BG)

        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scroll.set)

        canvas.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_wheel)

        pad = {'padx': 14, 'pady': 4}

        self._section(inner, 'Feature Selection')
        self._feat1_var = tk.StringVar(value=dl.FEATURE_NAMES[0])
        self._feat2_var = tk.StringVar(value=dl.FEATURE_NAMES[1])
        self._row(inner, 'Feature 1',
                  _styled_combo(inner, dl.FEATURE_NAMES, self._feat1_var, 22))
        self._row(inner, 'Feature 2',
                  _styled_combo(inner, dl.FEATURE_NAMES, self._feat2_var, 22))

        self._section(inner, 'Class Pair')
        self._class_var = tk.StringVar(value=dl.CLASS_PAIRS[0])
        for cp in dl.CLASS_PAIRS:
            tk.Radiobutton(inner, text=cp, variable=self._class_var, value=cp,
                           bg=PANEL_BG, fg=FG, selectcolor=ACCENT,
                           activebackground=PANEL_BG, activeforeground=FG,
                           font=FONT_LABEL).pack(anchor='w', **pad)

        self._section(inner, 'Hyperparameters')
        self._eta_var = tk.StringVar(value='0.01')
        self._ep_var = tk.StringVar(value='100')
        self._mse_var = tk.StringVar(value='0.01')
        self._seed_var = tk.StringVar(value='42')

        self._row(inner, 'Learning Rate (eta)', _styled_entry(inner, self._eta_var))
        self._row(inner, 'Epochs (m)', _styled_entry(inner, self._ep_var))
        self._row(inner, 'MSE Threshold', _styled_entry(inner, self._mse_var))
        self._row(inner, 'Random Seed', _styled_entry(inner, self._seed_var))

        self._section(inner, 'Options')
        self._bias_var = tk.BooleanVar(value=True)
        self._algo_var = tk.StringVar(value='Perceptron')

        tk.Checkbutton(inner, text='Add Bias Term', variable=self._bias_var,
                       bg=PANEL_BG, fg=FG, selectcolor=ACCENT,
                       activebackground=PANEL_BG, activeforeground=FG,
                       font=FONT_LABEL).pack(anchor='w', **pad)

        algo_frame = tk.Frame(inner, bg=PANEL_BG)
        algo_frame.pack(anchor='w', **pad)
        _styled_label(algo_frame, 'Algorithm: ').pack(side='left')
        for algo in ['Perceptron', 'Adaline']:
            tk.Radiobutton(algo_frame, text=algo, variable=self._algo_var,
                           value=algo, bg=PANEL_BG, fg=FG, selectcolor=ACCENT,
                           activebackground=PANEL_BG, activeforeground=FG,
                           font=FONT_LABEL).pack(side='left', padx=4)

        tk.Frame(inner, bg=PANEL_BG, height=10).pack()
        _styled_button(inner, 'Train & Test', self._run_pipeline, width=26,
                       bg=BTN_COLOR).pack(padx=14, pady=6, fill='x')

        self._section(inner, 'Results')
        self._result_text = tk.Text(inner, height=12, width=36,
                                    bg='#0a0a1a', fg=FG, font=FONT_MONO,
                                    relief='flat', bd=4, wrap='none')
        self._result_text.pack(padx=14, pady=4)
        self._result_text.config(state='disabled')

        self._section(inner, 'Classify Single Sample')
        self._s1_var = tk.StringVar()
        self._s2_var = tk.StringVar()

        self._s1_lbl = _styled_label(inner, 'Feature 1 value:')
        self._s1_lbl.pack(anchor='w', padx=14, pady=(4, 0))
        _styled_entry(inner, self._s1_var, width=28).pack(padx=14, pady=2)

        self._s2_lbl = _styled_label(inner, 'Feature 2 value:')
        self._s2_lbl.pack(anchor='w', padx=14, pady=(4, 0))
        _styled_entry(inner, self._s2_var, width=28).pack(padx=14, pady=2)

        self._origin_hint = _styled_label(
            inner,
            "OriginLocation hint:\n  0=Biscoe  1=Dream  2=Torgersen",
            fg=FG_DIM)
        self._origin_hint.pack(anchor='w', padx=14)

        _styled_button(inner, 'Classify Sample', self._classify_sample,
                       width=26, bg='#2a7a4e').pack(padx=14, pady=6, fill='x')

        self._pred_label = _styled_label(inner, '', font=('Segoe UI', 11, 'bold'),
                                         fg=SUCCESS)
        self._pred_label.pack(padx=14, pady=2)

    def _section(self, parent, title: str):
        tk.Frame(parent, bg=ACCENT, height=2).pack(fill='x', padx=8, pady=(14, 0))
        _styled_label(parent, title, font=FONT_HEADER, fg=FG).pack(
            anchor='w', padx=14, pady=(4, 2))

    def _row(self, parent, label: str, widget):
        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill='x', padx=14, pady=3)
        _styled_label(frame, label + ':', fg=FG_DIM).pack(anchor='w')
        widget.pack(anchor='w', pady=1)

    def _build_canvas(self, parent):
        self._fig = Figure(figsize=(7, 6), facecolor='#0a0a1a')
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor('#0d0d1f')
        self._fig.tight_layout(pad=2)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill='both', expand=True)

        self._ax.text(0.5, 0.5, 'Train the model to see the decision boundary',
                      transform=self._ax.transAxes,
                      ha='center', va='center', fontsize=12, color=FG_DIM)
        self._canvas_widget.draw()

    def _parse_hyperparams(self):
        try:
            eta = float(self._eta_var.get())
            if eta <= 0:
                raise ValueError('Learning rate must be positive.')
        except ValueError as exc:
            raise ValueError(f'Invalid learning rate: {exc}')

        try:
            epochs = int(self._ep_var.get())
            if epochs < 1:
                raise ValueError('Epochs must be >= 1.')
        except ValueError as exc:
            raise ValueError(f'Invalid epochs: {exc}')

        try:
            mse_thr = float(self._mse_var.get())
            if mse_thr < 0:
                raise ValueError('MSE threshold must be >= 0.')
        except ValueError as exc:
            raise ValueError(f'Invalid MSE threshold: {exc}')

        try:
            seed = int(self._seed_var.get())
        except ValueError:
            raise ValueError('Random seed must be an integer.')

        return {'eta': eta, 'epochs': epochs, 'mse_threshold': mse_thr,
                'seed': seed}

    def _get_class_pair(self):
        pair_map = {
            dl.CLASS_PAIRS[0]: ('Adelie', 'Chinstrap'),
            dl.CLASS_PAIRS[1]: ('Adelie', 'Gentoo'),
            dl.CLASS_PAIRS[2]: ('Chinstrap', 'Gentoo'),
        }
        return pair_map[self._class_var.get()]

    def _run_pipeline(self):
        f1 = self._feat1_var.get()
        f2 = self._feat2_var.get()
        if f1 == f2:
            messagebox.showerror('Input Error', 'Feature 1 and Feature 2 must differ.')
            return

        try:
            hp = self._parse_hyperparams()
        except ValueError as exc:
            messagebox.showerror('Input Error', str(exc))
            return

        class1, class2 = self._get_class_pair()
        algo = self._algo_var.get()
        use_bias = self._bias_var.get()

        self._feat1 = f1
        self._feat2 = f2
        self._class1 = class1
        self._class2 = class2

        try:
            X_train, X_test, y_train, y_test, scaler = dl.load_and_prepare(
                self.csv_path, class1, class2, f1, f2,
                random_seed=hp['seed'])
        except Exception as exc:
            messagebox.showerror('Data Error', str(exc))
            return

        self._scaler = scaler
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._test_point = None

        if algo == 'Perceptron':
            model = mdl.Perceptron(eta=hp['eta'], epochs=hp['epochs'],
                                   use_bias=use_bias, random_seed=hp['seed'])
        else:
            model = mdl.Adaline(eta=hp['eta'], epochs=hp['epochs'],
                                mse_threshold=hp['mse_threshold'],
                                use_bias=use_bias, random_seed=hp['seed'])

        model.fit(X_train, y_train)
        self._model = model

        y_pred = model.predict(X_test)
        cm = mtr.compute_confusion_matrix(y_pred=y_pred, y_true=y_test,
                                          class1_name=class1,
                                          class2_name=class2)
        acc = mtr.compute_accuracy(y_true=y_test, y_pred=y_pred)

        self._update_results(cm, acc, model, algo)
        self._update_plot(X_train, y_train, X_test, y_test, model, f1, f2,
                          class1, class2)

    def _update_results(self, cm: dict, acc: float, model, algo: str):
        self._result_text.config(state='normal')
        self._result_text.delete('1.0', 'end')

        lines = []
        lines.append(f'Algorithm  : {algo}')
        lines.append(f'Epochs run : {model.epochs_run}')
        if algo == 'Adaline':
            mse = model.get_mse()
            lines.append(f'Final MSE  : {mse:.6f}')
        lines.append('')
        lines.append('--- Confusion Matrix ---')
        lines.append(f'Classes: +1={cm["class1_name"]}, -1={cm["class2_name"]}')
        lines.append(f'  TP = {cm["TP"]:4d}   FN = {cm["FN"]:4d}')
        lines.append(f'  FP = {cm["FP"]:4d}   TN = {cm["TN"]:4d}')
        lines.append('')
        lines.append(f'Accuracy   : {acc:.2f}%')

        self._result_text.insert('1.0', '\n'.join(lines))
        self._result_text.config(state='disabled')

    def _update_plot(self, X_train, y_train, X_test, y_test,
                     model, f1: str, f2: str, class1: str, class2: str):
        self._ax.cla()
        self._ax.set_facecolor('#0d0d1f')

        # Scatter points: training samples (circles) and test samples (squares)
        for label, name, color, train_marker, test_marker in [
            (1, class1, CLASS_COLORS[0], 'o', 's'),
            (-1, class2, CLASS_COLORS[1], 'o', 's'),
        ]:
            tr_mask = y_train == label
            te_mask = y_test == label
            if tr_mask.any():
                self._ax.scatter(X_train[tr_mask, 0], X_train[tr_mask, 1],
                                 c=color, marker=train_marker, s=60, alpha=0.75,
                                 label=f'{name} (train)', edgecolors='white',
                                 linewidths=0.4)
            if te_mask.any():
                self._ax.scatter(X_test[te_mask, 0], X_test[te_mask, 1],
                                 c=color, marker=test_marker, s=80, alpha=0.9,
                                 label=f'{name} (test)', edgecolors='white',
                                 linewidths=0.6)

        # Decision boundary: w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b)/w2
        w1, w2, b = model.get_decision_boundary()
        x1_min, x1_max = self._ax.get_xlim()
        y_min, y_max = self._ax.get_ylim()
        margin = (x1_max - x1_min) * 0.15
        x1_vals = np.linspace(x1_min - margin, x1_max + margin, 300)

        if abs(w2) > 1e-10:
            x2_vals = -(w1 * x1_vals + b) / w2
            self._ax.plot(x1_vals, x2_vals, color=BOUNDARY_COLOR,
                          linewidth=2.0, linestyle='--',
                          label='Decision boundary', zorder=5)
            # Restore plot limits to prevent boundary line from rescaling data view
            self._ax.set_xlim(x1_min, x1_max)
            self._ax.set_ylim(y_min, y_max)
        else:
            # Vertical boundary when w2 is 0: x1 = -b / w1
            if abs(w1) > 1e-10:
                x1_vert = -b / w1
                self._ax.axvline(x=x1_vert, color=BOUNDARY_COLOR,
                                 linewidth=2.0, linestyle='--',
                                 label='Decision boundary (vertical)', zorder=5)
                self._ax.set_xlim(x1_min, x1_max)
                self._ax.set_ylim(y_min, y_max)

        if self._test_point is not None:
            self._ax.scatter([self._test_point[0]], [self._test_point[1]],
                             marker='*', s=300, c='#ffff00', zorder=10,
                             edgecolors='black', linewidths=0.8,
                             label='Single sample')

        self._ax.set_xlabel(f1, color=FG, fontsize=10)
        self._ax.set_ylabel(f2, color=FG, fontsize=10)
        self._ax.set_title(f'{class1} vs {class2}  |  {f1} & {f2}',
                           color=FG, fontsize=11, fontweight='bold')
        self._ax.tick_params(colors=FG)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(ACCENT)
        self._ax.legend(fontsize=8, facecolor='#111122',
                        labelcolor=FG, framealpha=0.8,
                        loc='best')
        self._fig.tight_layout(pad=2)
        self._canvas_widget.draw()

    def _classify_sample(self):
        if self._model is None:
            messagebox.showwarning('No Model', 'Please train the model first.')
            return

        raw1_str = self._s1_var.get().strip()
        raw2_str = self._s2_var.get().strip()

        if not raw1_str or not raw2_str:
            messagebox.showerror('Input Error', 'Please enter values for both features.')
            return

        def _parse_val(val_str: str, feat_name: str) -> float:
            val_str = val_str.strip()
            if feat_name == 'OriginLocation':
                loc_map = dl.get_location_map()
                for name, code in loc_map.items():
                    if val_str.lower() == name.lower():
                        return float(code)
                try:
                    num = float(val_str)
                    if num in loc_map.values():
                        return num
                except ValueError:
                    pass
                names = ', '.join(loc_map.keys())
                codes = ', '.join(str(c) for c in loc_map.values())
                raise ValueError(
                    f"OriginLocation must be one of: {names} or codes: {codes}.")
            try:
                return float(val_str)
            except ValueError:
                raise ValueError(f"Feature '{feat_name}' must be numeric, got '{val_str}'.")

        try:
            raw1 = _parse_val(raw1_str, self._feat1)
            raw2 = _parse_val(raw2_str, self._feat2)
        except ValueError as exc:
            messagebox.showerror('Input Error', str(exc))
            return

        try:
            x_norm = dl.encode_single_sample(
                [raw1, raw2], [self._feat1, self._feat2], self._scaler)
        except Exception as exc:
            messagebox.showerror('Encoding Error', str(exc))
            return

        pred = self._model.predict_single(x_norm)
        species = self._class1 if pred == 1 else self._class2

        self._pred_label.config(
            text=f'Prediction: {species}  (label={pred:+d})',
            fg=CLASS_COLORS[0] if pred == 1 else CLASS_COLORS[1])

        self._test_point = x_norm
        self._update_plot(self._X_train, self._y_train,
                          self._X_test, self._y_test,
                          self._model,
                          self._feat1, self._feat2,
                          self._class1, self._class2)


def launch(csv_path: str):
    PenguinApp(csv_path)
