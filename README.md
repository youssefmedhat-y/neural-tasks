# Penguin Classifier: Perceptron & Adaline

A modular Python application demonstrating from-scratch implementations of the **Single-Layer Perceptron** and **Adaptive Linear Neuron (Adaline)** on the Penguins dataset.

## Features
- **No high-level ML libraries**: Algorithms, data imputation, categorical encoding, Z-score normalization, and metrics (confusion matrix, accuracy) are implemented from scratch using only `numpy` and `pandas` (for CSV I/O).
- **Interactive Tkinter GUI**:
  - Dynamic 2D scatter plot with the decision boundary line.
  - Pairwise class and feature selection.
  - Hyperparameter tuning (learning rate, epochs, bias toggle, MSE threshold for Adaline).
  - Confusion matrix display and overall accuracy calculation.
  - Interactive single-sample classification.

## Project Structure
- `data_loader.py`: CSV loading, per-class mean/mode imputation without dropping rows, stratified train/test split, and Z-score normalization.
- `models.py`: From-scratch `Perceptron` and `Adaline` classifier classes.
- `metrics.py`: From-scratch confusion matrix and accuracy calculations.
- `gui.py`: Tkinter-based dark theme interface with embedded Matplotlib visualizations.
- `main.py`: Application entry point.
- `penguins.csv`: Dataset with morphological measurements for Adelie, Chinstrap, and Gentoo penguins.

## Installation & Usage

1. Install dependencies:
```bash
pip install numpy pandas matplotlib
```

2. Run the application:
```bash
python main.py
```
