"""Tkinter interface for the Task 2 backpropagation experiment."""
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from backprop import BackpropNetwork, evaluate
from task2_data import load_task2
from data_loader import CLASS_NAMES, FEATURE_NAMES

BG, PANEL, FG, MUTED, ACCENT = '#101827', '#1b293d', '#edf3fc', '#a7bad2', '#53d4c0'


class Task2App(tk.Tk):
    def __init__(self, csv_path):
        super().__init__()
        self.title('Task 2 | Penguin Backpropagation')
        self.geometry('1240x850')
        self.minsize(1100, 780)
        self.configure(bg=BG)
        self.csv_path = str(csv_path)
        self.model = self.data = self.result = None
        self.events = queue.Queue()
        self.busy = False
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground='white', foreground='#172334')
        self._build()
        self.after(100, self._poll)

    def label(self, parent, text, size=10, color=FG, **kw):
        return tk.Label(parent, text=text, bg=parent['bg'], fg=color,
                        font=('Segoe UI', size), **kw)

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill='x', padx=24, pady=(16, 10))
        self.label(top, 'PENGUIN LAB / TASK 2', 10, ACCENT).pack(anchor='w')
        self.label(top, 'Multilayer backpropagation', 24).pack(anchor='w')
        self.label(top, '5 inputs  /  3 species  /  90 training samples  /  60 test samples', 11, MUTED).pack(anchor='w')
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True, padx=24, pady=(0, 18))
        left = tk.Frame(body, bg=PANEL, width=290, padx=16, pady=12)
        left.pack(side='left', fill='y', padx=(0, 16))
        left.pack_propagate(False)
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)
        self.label(left, 'Network settings', 15).pack(anchor='w', pady=(0, 8))
        self.variables = {}
        settings = [('layers', 'Number of hidden layers', '1'),
                    ('neurons', 'Neurons per layer (comma separated)', '8'),
                    ('eta', 'Learning rate', '0.05'),
                    ('epochs', 'Training epochs', '1000'),
                    ('seed', 'Random seed', '42')]
        for key, caption, default in settings:
            self.label(left, caption, 9, MUTED).pack(anchor='w', pady=(6, 2))
            var = tk.StringVar(value=default)
            self.variables[key] = var
            tk.Entry(left, textvariable=var, font=('Consolas', 11), relief='flat').pack(fill='x', ipady=3)
        self.label(left, 'Activation (all hidden + output layers)', 9, MUTED).pack(anchor='w', pady=(10, 2))
        self.activation = tk.StringVar(value='sigmoid')
        ttk.Combobox(left, textvariable=self.activation, values=['sigmoid', 'tanh'], state='readonly').pack(fill='x')
        self.bias = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text='Include bias', variable=self.bias, bg=PANEL, fg=FG,
                       selectcolor=BG, activebackground=PANEL, activeforeground=FG).pack(anchor='w', pady=10)
        self.train_button = tk.Button(left, text='Train and evaluate', command=self.start_training,
                                      bg=ACCENT, fg=BG, relief='flat', font=('Segoe UI', 11, 'bold'))
        self.train_button.pack(fill='x', ipady=8)
        self.save_button = tk.Button(left, text='Export result JSON', command=self.export, state='disabled')
        self.save_button.pack(fill='x', pady=(10, 0))
        self.status = tk.StringVar(value='Ready to train.')
        self.label(left, '', 10, MUTED, textvariable=self.status, wraplength=250, justify='left').pack(anchor='w', pady=14)
        self.label(left, 'Split: first 30 rows per species train; remaining 20 test.\n\nMissing values and scaling are fitted on training rows only.',
                   9, MUTED, wraplength=250, justify='left').pack(anchor='w')

        self.summary = tk.StringVar(value='Train a network to see accuracy and the confusion matrix.')
        self.label(right, '', 13, ACCENT, textvariable=self.summary, anchor='w').pack(fill='x', pady=(0, 6))
        self.config_text = tk.StringVar(value='Classification uses the output neuron with the highest activation.')
        self.label(right, '', 9, MUTED, textvariable=self.config_text, anchor='w').pack(fill='x')
        self.figure = Figure(figsize=(8.5, 3.5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=8)
        self._draw()
        sample = tk.Frame(right, bg=PANEL, padx=14, pady=10)
        sample.pack(fill='x')
        self.label(sample, 'Classify a single bird', 14).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 6))
        self.sample_vars = []
        for i, (name, default) in enumerate(zip(FEATURE_NAMES, ['39.1', '18.7', '181', 'Torgersen', '3750'])):
            row, column = 1 + (i // 3) * 2, i % 3
            self.label(sample, name, 9, MUTED).grid(row=row, column=column, sticky='w', padx=(0, 12))
            var = tk.StringVar(value=default)
            self.sample_vars.append(var)
            if name == 'OriginLocation':
                widget = ttk.Combobox(sample, textvariable=var, values=['Biscoe', 'Dream', 'Torgersen'], state='readonly', width=18)
            else:
                widget = tk.Entry(sample, textvariable=var, width=20)
            widget.grid(row=row + 1, column=column, sticky='ew', padx=(0, 12), pady=(2, 6))
        for column in range(3):
            sample.columnconfigure(column, weight=1)
        self.classify_button = tk.Button(sample, text='Classify bird', command=self.classify, state='disabled', bg=ACCENT, fg=BG)
        self.classify_button.grid(row=4, column=2, sticky='ew', padx=(0, 12), pady=(2, 6))
        self.prediction = tk.StringVar(value='Prediction will appear here. Scores are activations, not probabilities.')
        self.label(sample, '', 10, textvariable=self.prediction, wraplength=800, justify='left').grid(row=5, column=0, columnspan=3, sticky='w', pady=(6, 0))

    def get_config(self):
        try:
            count = int(self.variables['layers'].get())
            hidden = tuple(int(value.strip()) for value in self.variables['neurons'].get().split(','))
            config = dict(hidden_layers=hidden, eta=float(self.variables['eta'].get()),
                          epochs=int(self.variables['epochs'].get()), random_seed=int(self.variables['seed'].get()),
                          activation=self.activation.get(), use_bias=self.bias.get())
        except ValueError:
            raise ValueError('Enter integer layer/neuron/epoch/seed values and a numeric learning rate.') from None
        if count != len(hidden) or count < 1:
            raise ValueError('The number of neuron counts must match the number of hidden layers.')
        BackpropNetwork(**config)  # Validate before starting a worker.
        return config

    def set_config(self, config):
        self.variables['layers'].set(str(len(config['hidden_layers'])))
        self.variables['neurons'].set(','.join(map(str, config['hidden_layers'])))
        for key in ('eta', 'epochs'):
            self.variables[key].set(str(config[key]))
        self.variables['seed'].set(str(config['random_seed']))
        self.activation.set(config['activation'])
        self.bias.set(config['use_bias'])

    def start_training(self):
        if self.busy:
            return
        try:
            config = self.get_config()
        except ValueError as exc:
            messagebox.showerror('Invalid settings', str(exc), parent=self)
            return
        self.busy = True
        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.classify_button.config(state='disabled')
        self.status.set('Loading data and training...')
        self.summary.set('Training a new network...')
        self.config_text.set('Previous plots remain visible until this run finishes.')
        def worker():
            try:
                data = load_task2(self.csv_path)
                model = BackpropNetwork(**config)
                model.fit(data.X_train, data.y_train,
                          progress=lambda epoch, loss: self.events.put(('progress', (epoch, loss, config['epochs']))))
                result = {'config': config, 'classes': CLASS_NAMES, 'features': FEATURE_NAMES,
                          'train': evaluate(model, data.X_train, data.y_train),
                          'test': evaluate(model, data.X_test, data.y_test),
                          'loss_history': model.loss_history}
                self.events.put(('done', (model, data, result)))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    epoch, loss, total = value
                    self.status.set(f'Epoch {epoch} / {total}\nLoss: {loss:.5f}')
                elif kind == 'done':
                    self.model, self.data, self.result = value
                    self.busy = False
                    self.train_button.config(state='normal')
                    self.save_button.config(state='normal')
                    self.classify_button.config(state='normal')
                    self.status.set(f'Training complete.\n{self.model.epochs_run} epochs performed.')
                    self.show_result()
                elif kind == 'error':
                    self.busy = False
                    self.model = self.data = self.result = None
                    self.train_button.config(state='normal')
                    self.status.set('Training failed. Check the settings or dataset.')
                    self.summary.set('No trained model available.')
                    self.config_text.set('')
                    self._draw()
                    messagebox.showerror('Training failed', value, parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def show_result(self):
        result = self.result
        config = result['config']
        self.summary.set(f"Train accuracy  {result['train']['accuracy']:.2f}%       Test accuracy  {result['test']['accuracy']:.2f}%")
        self.config_text.set(f"{config['activation'].upper()}  |  5 -> {' -> '.join(map(str, config['hidden_layers']))} -> 3  |  LR {config['eta']}  |  {config['epochs']} epochs  |  Bias {'on' if config['use_bias'] else 'off'}  |  Seed {config['random_seed']}")
        self.prediction.set('Prediction will appear here. Scores are activations, not probabilities.')
        self._draw()

    def _draw(self):
        self.figure.clear()
        axes = self.figure.subplots(1, 2)
        for ax in axes:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
        axes[0].set_title('Training loss', color=FG, fontsize=11)
        axes[0].set_xlabel('Epoch', color=MUTED)
        axes[0].set_ylabel('Mean half-squared error', color=MUTED)
        axes[1].set_title('Test confusion matrix (60 birds)', color=FG, fontsize=11)
        if self.result:
            loss = self.result['loss_history']
            axes[0].plot(np.arange(1, len(loss) + 1), loss, color=ACCENT, linewidth=1.7)
            matrix = np.asarray(self.result['test']['confusion_matrix'])
            axes[1].imshow(matrix, cmap='Blues', vmin=0, vmax=20)
            for row in range(3):
                for column in range(3):
                    axes[1].text(column, row, str(matrix[row, column]), ha='center', va='center',
                                 color='white' if matrix[row, column] > 10 else '#172334', fontsize=15)
        axes[1].set_xticks(range(3), CLASS_NAMES, rotation=15)
        axes[1].set_yticks(range(3), CLASS_NAMES)
        axes[1].set_xlabel('Predicted species', color=MUTED)
        axes[1].set_ylabel('Actual species', color=MUTED)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def classify(self):
        if self.model is None:
            return
        try:
            values = [var.get().strip() for var in self.sample_vars]
            if any(value == '' for value in values):
                raise ValueError('Enter all five feature values.')
            x = self.data.preprocessor.single(values)
            scores = self.model.predict_scores(x[None, :])[0]
            predicted = int(np.argmax(scores))
            one_hot = np.eye(3, dtype=int)[predicted].tolist()
            self.prediction.set(f'{CLASS_NAMES[predicted]} (class {predicted + 1})  |  One-hot: {one_hot}\n'
                                + '  /  '.join(f'{name}: {score:.4f}' for name, score in zip(CLASS_NAMES, scores)))
        except (ValueError, TypeError) as exc:
            messagebox.showerror('Invalid sample', str(exc), parent=self)

    def export(self):
        if self.result is None:
            return
        filename = filedialog.asksaveasfilename(parent=self, defaultextension='.json', initialfile='task2_result.json',
                                               filetypes=[('JSON result', '*.json')])
        if filename:
            try:
                Path(filename).write_text(json.dumps(self.result, indent=2), encoding='utf-8')
            except OSError as exc:
                messagebox.showerror('Export failed', str(exc), parent=self)


def launch(csv_path, preset=None):
    app = Task2App(csv_path)
    if preset:
        result = json.loads(Path(preset).read_text(encoding='utf-8'))
        app.set_config(result['config'])
        app.after(300, app.start_training)
    app.mainloop()
