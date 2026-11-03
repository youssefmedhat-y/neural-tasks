import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import task2_data as dl
import backprop as mdl
import metrics as mtr

BG = '#1a1a2e'
PANEL_BG = '#16213e'
ACCENT = '#0f3460'
BTN_COLOR = '#e94560'
FG = '#e0e0e0'
FG_DIM = '#a0a0b0'
ENTRY_BG = '#0f3460'
SUCCESS = '#4caf50'

FONT_HEADER = ('Segoe UI', 13, 'bold')
FONT_LABEL = ('Segoe UI', 9)
FONT_ENTRY = ('Consolas', 10)


def _styled_label(parent, text, font=None, fg=FG, **kw):
    return tk.Label(parent, text=text, bg=parent['bg'], fg=fg,
                    font=font or FONT_LABEL, **kw)


def _styled_entry(parent, textvariable, width=10):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=ENTRY_BG, fg=FG, insertbackground=FG,
                    font=FONT_ENTRY, relief='flat', bd=4)


def _styled_button(parent, text, command, bg=BTN_COLOR):
    return tk.Button(parent, text=text, command=command, bg=bg, fg=FG,
                     font=('Segoe UI', 10, 'bold'), relief='flat',
                     activebackground=ACCENT, activeforeground=FG, cursor='hand2')


def _styled_combo(parent, values, textvariable):
    return ttk.Combobox(parent, values=values, textvariable=textvariable,
                        state='readonly', style='Dark.TCombobox',
                        font=FONT_ENTRY, exportselection=False)


class PenguinApp(tk.Tk):

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.title('Penguin Classifier - Backpropagation Lab')
        self.geometry('1240x850')
        self.minsize(1100, 780)
        self.configure(bg=BG)

        self._model = None
        self._scaler = None
        self._result = None
        self._training = False
        self._events = queue.Queue()
        self._poll_id = None

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Dark.TCombobox', fieldbackground=ENTRY_BG,
                        background=ENTRY_BG, foreground=FG, arrowcolor=FG)
        style.map('Dark.TCombobox', fieldbackground=[('readonly', ENTRY_BG)],
                  foreground=[('readonly', FG)], selectbackground=[('readonly', ENTRY_BG)])
        self.option_add('*TCombobox*Listbox.background', ENTRY_BG)
        self.option_add('*TCombobox*Listbox.foreground', FG)

        self._build_ui()
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._poll_id = self.after(100, self._poll_training)

    def _on_close(self):
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.destroy()

    def _build_ui(self):
        title_bar = tk.Frame(self, bg=ACCENT)
        title_bar.pack(fill='x')
        _styled_label(title_bar, 'Penguin Neural Classifier - Backpropagation',
                      font=('Segoe UI', 14, 'bold')).pack(side='left', padx=20, pady=12)

        main_frame = tk.Frame(self, bg=BG)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        left = tk.Frame(main_frame, bg=PANEL_BG, width=300)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        right = tk.Frame(main_frame, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self._build_controls(left)
        self._build_canvas(right)
        self._build_sample_inputs(right)

    def _section(self, parent, title):
        tk.Frame(parent, bg=ACCENT, height=2).pack(fill='x', padx=8, pady=(12, 0))
        _styled_label(parent, title, font=FONT_HEADER).pack(anchor='w', padx=14, pady=(4, 2))

    def _row(self, parent, label, variable):
        _styled_label(parent, label + ':', fg=FG_DIM).pack(anchor='w', padx=14, pady=(6, 0))
        _styled_entry(parent, variable).pack(fill='x', padx=14, pady=2)

    def _build_controls(self, parent):
        self._section(parent, 'Network Settings')
        self._layers_var = tk.StringVar(value='1')
        self._neurons_var = tk.StringVar(value='8')
        self._eta_var = tk.StringVar(value='0.05')
        self._ep_var = tk.StringVar(value='1000')
        self._seed_var = tk.StringVar(value='42')
        self._bias_var = tk.BooleanVar(value=True)
        self._activation_var = tk.StringVar(value='sigmoid')

        self._row(parent, 'Number of hidden layers', self._layers_var)
        self._row(parent, 'Neurons per layer (e.g. 8, 6)', self._neurons_var)
        self._row(parent, 'Learning Rate (eta)', self._eta_var)
        self._row(parent, 'Epochs (m)', self._ep_var)
        self._row(parent, 'Random Seed', self._seed_var)

        self._section(parent, 'Options')
        _styled_label(parent, 'Activation function:', fg=FG_DIM).pack(anchor='w', padx=14, pady=4)
        _styled_combo(parent, ['sigmoid', 'tanh'], self._activation_var).pack(fill='x', padx=14)
        tk.Checkbutton(parent, text='Add Bias Term', variable=self._bias_var,
                       bg=PANEL_BG, fg=FG, selectcolor=ACCENT,
                       activebackground=PANEL_BG, activeforeground=FG,
                       font=FONT_LABEL).pack(anchor='w', padx=14, pady=8)

        self._train_button = _styled_button(parent, 'Train & Test', self._run_pipeline)
        self._train_button.pack(fill='x', padx=14, pady=6, ipady=7)
        self._export_button = _styled_button(parent, 'Export Results', self._export_result, bg=ACCENT)
        self._export_button.pack(fill='x', padx=14, pady=4, ipady=4)
        self._export_button.config(state='disabled')

        self._section(parent, 'Training Status')
        self._status_var = tk.StringVar(value='Ready to train.')
        _styled_label(parent, '', textvariable=self._status_var, fg=FG_DIM,
                      wraplength=270, justify='left').pack(anchor='w', padx=14, pady=6)
        _styled_label(parent, '5 features / 3 penguin species\n90 training / 60 test samples',
                      fg=FG_DIM, justify='left').pack(anchor='w', padx=14, pady=8)

    def _build_canvas(self, parent):
        self._summary_var = tk.StringVar(value='Train the network to see the results.')
        self._config_var = tk.StringVar(value='First 30 samples per species train; remaining 20 test.')
        _styled_label(parent, '', textvariable=self._summary_var,
                      font=FONT_HEADER, fg=SUCCESS).pack(anchor='w', padx=8, pady=6)
        _styled_label(parent, '', textvariable=self._config_var, fg=FG_DIM,
                      wraplength=850, justify='left').pack(anchor='w', padx=8)
        self._fig = Figure(figsize=(8, 4), facecolor=BG)
        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill='both', expand=True)
        self._update_plot()

    def _build_sample_inputs(self, parent):
        sample = tk.Frame(parent, bg=PANEL_BG, padx=14, pady=10)
        sample.pack(fill='x', pady=(8, 0))
        _styled_label(sample, 'Classify a Single Penguin', font=FONT_HEADER).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        defaults = ['39.1', '18.7', '181', 'Torgersen', '3750']
        self._sample_vars = []
        for i, name in enumerate(dl.FEATURE_NAMES):
            row = 1 + (i // 3) * 2
            column = i % 3
            variable = tk.StringVar(value=defaults[i])
            self._sample_vars.append(variable)
            _styled_label(sample, name, fg=FG_DIM).grid(row=row, column=column, sticky='w')
            if name == 'OriginLocation':
                widget = _styled_combo(sample, ['Biscoe', 'Dream', 'Torgersen'], variable)
            else:
                widget = _styled_entry(sample, variable, width=18)
            widget.grid(row=row + 1, column=column, sticky='ew', padx=(0, 12), pady=(2, 6))
        for column in range(3):
            sample.columnconfigure(column, weight=1)

        self._classify_button = _styled_button(sample, 'Classify Penguin', self._classify_sample, bg='#2a7a4e')
        self._classify_button.grid(row=4, column=2, sticky='ew', padx=(0, 12), pady=(2, 6))
        self._classify_button.config(state='disabled')
        self._prediction_var = tk.StringVar(value='Enter the five features to classify a penguin.')
        _styled_label(sample, '', textvariable=self._prediction_var,
                      wraplength=800, justify='left').grid(row=5, column=0, columnspan=3, sticky='w', pady=4)

    def _parse_hyperparams(self):
        try:
            count = int(self._layers_var.get())
            hidden = []
            for value in self._neurons_var.get().split(','):
                hidden.append(int(value.strip()))
            eta = float(self._eta_var.get())
            epochs = int(self._ep_var.get())
            seed = int(self._seed_var.get())
        except ValueError:
            raise ValueError('Enter whole numbers for layers, neurons, epochs and seed, and a number for learning rate.')
        if count < 1 or count != len(hidden):
            raise ValueError('Enter one neuron count for each hidden layer.')
        return {'hidden_layers': hidden, 'eta': eta, 'epochs': epochs,
                'random_seed': seed, 'activation': self._activation_var.get(),
                'use_bias': self._bias_var.get()}

    def _set_config(self, config):
        self._layers_var.set(str(len(config['hidden_layers'])))
        self._neurons_var.set(', '.join(map(str, config['hidden_layers'])))
        self._eta_var.set(str(config['eta']))
        self._ep_var.set(str(config['epochs']))
        self._seed_var.set(str(config['random_seed']))
        self._activation_var.set(config['activation'])
        self._bias_var.set(config['use_bias'])

    def _run_pipeline(self):
        if self._training:
            return
        try:
            config = self._parse_hyperparams()
            model = mdl.BackpropNetwork(**config)
        except ValueError as exc:
            messagebox.showerror('Input Error', str(exc), parent=self)
            return

        self._training = True
        self._train_button.config(state='disabled')
        self._classify_button.config(state='disabled')
        self._export_button.config(state='disabled')
        self._status_var.set('Training...')
        # Train in the background so the window stays responsive.
        thread = threading.Thread(target=self._train_model, args=(model, config), daemon=True)
        thread.start()

    def _report_progress(self, epoch, loss):
        self._events.put(('progress', (epoch, loss)))

    def _train_model(self, model, config):
        try:
            # Load data, train the network, then evaluate both sets.
            X_train, X_test, y_train, y_test, scaler = dl.load_and_prepare(self.csv_path)
            model.fit(X_train, y_train, progress=self._report_progress)

            train_result = mtr.evaluate(model, X_train, y_train)
            test_result = mtr.evaluate(model, X_test, y_test)
            result = {
                'config': config,
                'classes': dl.CLASS_NAMES,
                'features': dl.FEATURE_NAMES,
                'train': train_result,
                'test': test_result,
                'loss_history': model.loss_history,
            }
            self._events.put(('done', (model, scaler, result)))
        except Exception as exc:
            self._events.put(('error', str(exc)))

    def _poll_training(self):
        # Read the worker's messages here, on Tkinter's main thread.
        try:
            while True:
                kind, value = self._events.get_nowait()
                if kind == 'progress':
                    epoch, loss = value
                    self._status_var.set(f'Epoch: {epoch}\nLoss: {loss:.5f}')
                elif kind == 'done':
                    self._model, self._scaler, self._result = value
                    self._training = False
                    self._train_button.config(state='normal')
                    self._classify_button.config(state='normal')
                    self._export_button.config(state='normal')
                    self._status_var.set(f'Training complete.\nEpochs run: {self._model.epochs_run}')
                    self._update_results()
                elif kind == 'error':
                    self._training = False
                    self._model = None
                    self._result = None
                    self._scaler = None
                    self._train_button.config(state='normal')
                    self._status_var.set('Training failed.')
                    self._summary_var.set('No trained model available.')
                    self._config_var.set('')
                    self._prediction_var.set('')
                    self._update_plot()
                    messagebox.showerror('Training Error', value, parent=self)
        except queue.Empty:
            pass
        self._poll_id = self.after(100, self._poll_training)

    def _update_results(self):
        result = self._result
        config = result['config']
        train_accuracy = result['train']['accuracy']
        test_accuracy = result['test']['accuracy']
        summary = f'Train accuracy: {train_accuracy:.2f}%    Test accuracy: {test_accuracy:.2f}%'
        self._summary_var.set(summary)

        layer_sizes = [5] + list(config['hidden_layers']) + [3]
        layers = ' -> '.join(map(str, layer_sizes))
        bias = 'on' if config['use_bias'] else 'off'
        self._config_var.set(f"{config['activation'].capitalize()} | {layers} | LR {config['eta']} | "
                             f"{config['epochs']} epochs | Bias {bias} | Seed {config['random_seed']}")
        self._prediction_var.set('Enter the five features to classify a penguin.')
        self._update_plot()

    def _update_plot(self):
        self._fig.clear()
        loss_ax, matrix_ax = self._fig.subplots(1, 2)
        for ax in (loss_ax, matrix_ax):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=FG_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(ACCENT)
        loss_ax.set_title('Training loss', color=FG, fontsize=11)
        loss_ax.set_xlabel('Epoch', color=FG_DIM)
        loss_ax.set_ylabel('Mean half-squared error', color=FG_DIM)
        matrix_ax.set_title('Test confusion matrix (60 penguins)', color=FG, fontsize=11)
        matrix_ax.set_xticks(range(3), dl.CLASS_NAMES, rotation=15)
        matrix_ax.set_yticks(range(3), dl.CLASS_NAMES)
        matrix_ax.set_xlabel('Predicted species', color=FG_DIM)
        matrix_ax.set_ylabel('Actual species', color=FG_DIM)

        if self._result is not None:
            loss = self._result['loss_history']
            loss_ax.plot(range(1, len(loss) + 1), loss, color=BTN_COLOR, linewidth=1.7)
            matrix = np.array(self._result['test']['confusion_matrix'])
            matrix_ax.imshow(matrix, cmap='Blues', vmin=0, vmax=20)
            for row in range(3):
                for column in range(3):
                    color = 'white' if matrix[row, column] > 10 else '#172334'
                    matrix_ax.text(column, row, str(matrix[row, column]),
                                   ha='center', va='center', color=color, fontsize=15)
        self._fig.tight_layout(pad=2.0)
        self._canvas_widget.draw()

    def _classify_sample(self):
        if self._model is None:
            messagebox.showwarning('No Model', 'Please train the network first.', parent=self)
            return
        try:
            raw_values = []
            for variable in self._sample_vars:
                raw_values.append(variable.get().strip())
            if '' in raw_values:
                raise ValueError('Please enter all five feature values.')

            # Use the same preprocessing values learned from training data.
            x = dl.encode_single_sample(raw_values, self._scaler)
            scores = self._model.predict_scores(x[None, :])[0]
            predicted = int(np.argmax(scores))
            one_hot = np.eye(3, dtype=int)[predicted].tolist()

            # Display the predicted class and the three output activations.
            lines = [f'Predicted penguin: {dl.CLASS_NAMES[predicted]} (class {predicted + 1}) | Output: {one_hot}']
            score_labels = []
            for name, score in zip(dl.CLASS_NAMES, scores):
                score_labels.append(f'{name}: {score:.4f}')
            lines.append('Activation scores: ' + ' / '.join(score_labels))
            self._prediction_var.set('\n'.join(lines))
        except (ValueError, TypeError) as exc:
            messagebox.showerror('Input Error', str(exc), parent=self)

    def _export_result(self):
        if self._result is None:
            return
        filename = filedialog.asksaveasfilename(parent=self, defaultextension='.json',
                                               initialfile='penguin_result.json',
                                               filetypes=[('JSON result', '*.json')])
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as stream:
                    json.dump(self._result, stream, indent=2)
            except OSError as exc:
                messagebox.showerror('Export Error', str(exc), parent=self)


def launch(csv_path, preset=None):
    app = PenguinApp(csv_path)
    if preset:
        with open(preset, encoding='utf-8') as stream:
            result = json.load(stream)
        app._set_config(result['config'])
        app.after(300, app._run_pipeline)
    app.mainloop()
