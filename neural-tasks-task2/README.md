# Task 2: Penguin Classifier

## Installation

```powershell
python -m pip install -r requirements.txt
```

Use Python 3.10 or later with Tkinter installed (included with the standard Windows Python installer).

## Task 2: multilayer backpropagation

```powershell
python task2.py
```

You can also double-click `run_task2.bat`.

The interface supports:

- Number of hidden layers and neurons in each layer.
- Learning rate, epochs, random seed, and optional bias.
- Sigmoid or tanh activation.
- Training and test accuracy, training loss, and a confusion matrix.
- Classification of a single penguin using its five features.

Task 2 uses all three species. The first 30 rows per species form the training set; the remaining 20 form the test set. Missing-value handling and normalization are fitted only on training data. Backpropagation is implemented with NumPy.

The final report is `Task2_Report.pdf`. An editable Markdown version is available as `Task2_Report.md`, with its screenshots in `report_images/`.

## Files

- `task2.py`: Task 2 entry point.
- `task2_gui.py`: Task 2 interface.
- `backprop.py`: Multilayer neural network.
- `metrics.py`: Three-class confusion matrix, accuracy, and evaluation.
- `task2_data.py`: Task 2 data preparation.
- `penguins.csv`: Dataset.
- `test_task2.py`: Model and preprocessing checks.

Run the Task 2 checks with:

```powershell
python -m unittest test_task2 -v
```
