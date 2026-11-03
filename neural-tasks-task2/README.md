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

## Reading the code for the discussion

1. `task2.py` finds the dataset and opens the window.
2. `task2_data.py` splits the data, fills missing values, and normalizes the features.
3. `backprop.py` contains the network. Start with `fit`: each sample goes through `_forward`, then `_deltas`, then the weight update. Loss is recorded after each epoch.
4. `metrics.py` counts correct predictions and builds the confusion matrix.
5. `task2_gui.py` connects the inputs, training button, plots, and single-penguin prediction. Training runs in a background thread so the window stays responsive.

In the network, `@` multiplies inputs by weights, `np.outer(inputs, delta)` calculates one update for every connection, and `np.argmax` chooses the output neuron with the highest activation. The one-hot targets are `[1, 0, 0]`, `[0, 1, 0]`, and `[0, 0, 1]`.
