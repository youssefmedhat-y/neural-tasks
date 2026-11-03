"""Task 2: fixed per-class split and preprocessing fitted on training rows only."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from data_loader import FEATURE_NAMES, CLASS_NAMES, get_location_map


class Preprocessor:
    def _numeric(self, frame):
        frame = frame[FEATURE_NAMES].copy()
        locations = frame['OriginLocation']
        invalid = locations.notna() & ~locations.isin(get_location_map())
        if invalid.any():
            raise ValueError('OriginLocation must be Biscoe, Dream, or Torgersen.')
        frame['OriginLocation'] = locations.map(get_location_map())
        for column in FEATURE_NAMES:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        values = frame.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('Features must be finite numbers or missing values.')
        return values

    def fit(self, frame):
        values = self._numeric(frame)
        if np.isnan(values).all(axis=0).any():
            raise ValueError('A training feature is entirely missing.')
        self.fill_values = np.nanmean(values, axis=0)
        location_index = FEATURE_NAMES.index('OriginLocation')
        locations = values[:, location_index]
        choices, counts = np.unique(locations[~np.isnan(locations)], return_counts=True)
        self.fill_values[location_index] = choices[counts.argmax()]
        filled = np.where(np.isnan(values), self.fill_values, values)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, frame):
        if not hasattr(self, 'mean'):
            raise RuntimeError('Fit the preprocessor before transforming samples.')
        values = self._numeric(frame)
        filled = np.where(np.isnan(values), self.fill_values, values)
        return (filled - self.mean) / self.std

    def single(self, values):
        if len(values) != 5:
            raise ValueError('Enter all five features in the displayed order.')
        return self.transform(pd.DataFrame([values], columns=FEATURE_NAMES))[0]


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: Preprocessor
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def read_split(csv_path):
    frame = pd.read_csv(csv_path, na_values=['NA', 'na', 'N/A', ''])
    missing = set(['Species'] + FEATURE_NAMES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if set(frame['Species'].unique()) != set(CLASS_NAMES):
        raise ValueError('Expected exactly Adelie, Chinstrap, and Gentoo classes.')
    if frame.duplicated(subset=['Species'] + FEATURE_NAMES).any():
        raise ValueError('Repeated samples found; the task requires 50 unique samples per class.')
    train, test = [], []
    for name in CLASS_NAMES:
        rows = frame.loc[frame['Species'] == name]
        if len(rows) != 50:
            raise ValueError(f'{name} must contain exactly 50 samples; found {len(rows)}.')
        train.append(rows.iloc[:30])
        test.append(rows.iloc[30:])
    return pd.concat(train), pd.concat(test)


def labels(frame):
    return frame['Species'].map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy(dtype=int)


def load_task2(csv_path):
    train, test = read_split(csv_path)
    preprocessor = Preprocessor().fit(train)
    return Dataset(preprocessor.transform(train), preprocessor.transform(test),
                   labels(train), labels(test), preprocessor, train, test)
