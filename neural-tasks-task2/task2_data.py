import numpy as np
import pandas as pd

FEATURE_NAMES = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
CLASS_NAMES = ['Adelie', 'Chinstrap', 'Gentoo']
_LOCATION_MAP = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}


def read_split(csv_path):
    df = pd.read_csv(csv_path, na_values=['NA', 'na', 'N/A', ''])
    required = ['Species'] + FEATURE_NAMES
    for column in required:
        if column not in df.columns:
            raise ValueError(f'Missing CSV column: {column}')
    if set(df['Species'].unique()) != set(CLASS_NAMES):
        raise ValueError('Expected Adelie, Chinstrap, and Gentoo penguins.')
    if df.duplicated(subset=required).any():
        raise ValueError('The dataset contains repeated samples.')

    train_parts = []
    test_parts = []
    for species in CLASS_NAMES:
        rows = df[df['Species'] == species]
        if len(rows) != 50:
            raise ValueError(f'{species} must have 50 samples.')
        # Keep file order: first 30 for training, last 20 for testing.
        train_parts.append(rows.iloc[:30])
        test_parts.append(rows.iloc[30:])

    return pd.concat(train_parts), pd.concat(test_parts)


def _encode_features(df):
    df = df[FEATURE_NAMES].copy()
    locations = df['OriginLocation']
    invalid = locations.notna() & ~locations.isin(_LOCATION_MAP)
    if invalid.any():
        raise ValueError('OriginLocation must be Biscoe, Dream, or Torgersen.')
    df['OriginLocation'] = locations.map(_LOCATION_MAP)
    values = df.values.astype(float)
    if np.isinf(values).any():
        raise ValueError('Feature values cannot be infinite.')
    return values


def _fit_scaler(X_train):
    if np.isnan(X_train).all(axis=0).any():
        raise ValueError('A training feature is entirely missing.')

    # Use training means for numerical features and the mode for location.
    fill_values = np.nanmean(X_train, axis=0)
    location_index = FEATURE_NAMES.index('OriginLocation')
    locations = X_train[:, location_index]
    locations = locations[~np.isnan(locations)]
    values, counts = np.unique(locations, return_counts=True)
    fill_values[location_index] = values[counts.argmax()]

    filled = np.where(np.isnan(X_train), fill_values, X_train)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std[std == 0] = 1.0
    return {'mean': mean, 'std': std, 'fill_values': fill_values}


def _normalize(X, scaler):
    filled = np.where(np.isnan(X), scaler['fill_values'], X)
    return (filled - scaler['mean']) / scaler['std']


def load_and_prepare(csv_path):
    # 1. Split the three classes before preprocessing.
    train_df, test_df = read_split(csv_path)
    X_train = _encode_features(train_df)
    X_test = _encode_features(test_df)

    # 2. Class labels: Adelie -> 0, Chinstrap -> 1, Gentoo -> 2.
    label_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    y_train = train_df['Species'].map(label_map).values.astype(int)
    y_test = test_df['Species'].map(label_map).values.astype(int)

    # 3. Fit on training data, then apply the same values to both sets.
    scaler = _fit_scaler(X_train)
    X_train = _normalize(X_train, scaler)
    X_test = _normalize(X_test, scaler)
    return X_train, X_test, y_train, y_test, scaler


def encode_single_sample(raw_values, scaler):
    if len(raw_values) != 5:
        raise ValueError('Please enter all five feature values.')
    df = pd.DataFrame([raw_values], columns=FEATURE_NAMES)
    values = _encode_features(df)
    return _normalize(values, scaler)[0]
