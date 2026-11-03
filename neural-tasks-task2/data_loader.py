import numpy as np
import pandas as pd

FEATURE_NAMES = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
CLASS_NAMES = ['Adelie', 'Chinstrap', 'Gentoo']
CLASS_PAIRS = [
    'C1 & C2 (Adelie & Chinstrap)',
    'C1 & C3 (Adelie & Gentoo)',
    'C2 & C3 (Chinstrap & Gentoo)',
]

_LOCATION_MAP = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}


def _impute_per_class(df):
    df = df.copy()
    num_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']

    for species in ['Adelie', 'Chinstrap', 'Gentoo']:
        mask = df['Species'] == species

        # Fill missing numerical values with the mean for this species
        for col in num_cols:
            mean_val = df.loc[mask, col].mean()
            df.loc[mask & df[col].isna(), col] = mean_val

        # Fill missing OriginLocation with the mode (most common island) for this species
        mode_val = df.loc[mask, 'OriginLocation'].mode()[0]
        df.loc[mask & df['OriginLocation'].isna(), 'OriginLocation'] = mode_val

    return df


def _stratified_split(df_class, n_train, n_test, rng):
    # Randomly shuffle indices for this class
    idx = np.arange(len(df_class))
    rng.shuffle(idx)
    # Take first n_train for training and next n_test for testing
    train_df = df_class.iloc[idx[:n_train]]
    test_df = df_class.iloc[idx[n_train:n_train + n_test]]
    return train_df, test_df


def load_and_prepare(csv_path, class1, class2, feat1, feat2, random_seed=42):
    rng = np.random.RandomState(random_seed)

    # 1. Load CSV
    df = pd.read_csv(csv_path, na_values=['NA', 'na', 'N/A', ''])

    # 2. Encode categorical OriginLocation into numbers (0, 1, 2)
    df['OriginLocation'] = df['OriginLocation'].map(_LOCATION_MAP)

    # 3. Impute missing values per species (no rows dropped)
    df = _impute_per_class(df)

    # 4. Filter to the two selected classes
    df_c1 = df[df['Species'] == class1].reset_index(drop=True)
    df_c2 = df[df['Species'] == class2].reset_index(drop=True)

    # 5. Stratified 30 train / 20 test split per class
    c1_train, c1_test = _stratified_split(df_c1, 30, 20, rng)
    c2_train, c2_test = _stratified_split(df_c2, 30, 20, rng)

    # 6. Extract the two selected features as float numpy arrays
    features = [feat1, feat2]
    X_c1_train = c1_train[features].values.astype(float)
    X_c2_train = c2_train[features].values.astype(float)
    X_c1_test = c1_test[features].values.astype(float)
    X_c2_test = c2_test[features].values.astype(float)

    # 7. Labels: Class 1 -> +1, Class 2 -> -1
    y_c1_train = np.ones(len(X_c1_train))
    y_c2_train = -np.ones(len(X_c2_train))
    y_c1_test = np.ones(len(X_c1_test))
    y_c2_test = -np.ones(len(X_c2_test))

    # 8. Combine and shuffle the training set
    X_train = np.vstack([X_c1_train, X_c2_train])
    y_train = np.concatenate([y_c1_train, y_c2_train])

    shuffle_idx = np.arange(len(y_train))
    rng.shuffle(shuffle_idx)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # Combine test set
    X_test = np.vstack([X_c1_test, X_c2_test])
    y_test = np.concatenate([y_c1_test, y_c2_test])

    # 9. Z-score normalization: z = (x - mean) / std (fit on train, apply to test)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    scaler = {'mean': mean, 'std': std}

    return X_train, X_test, y_train, y_test, scaler


def encode_single_sample(raw_values, feat_names, scaler):
    values = []
    for val, name in zip(raw_values, feat_names):
        if name == 'OriginLocation':
            if isinstance(val, str):
                values.append(float(_LOCATION_MAP[val]))
            else:
                values.append(float(val))
        else:
            values.append(float(val))

    x = np.array(values, dtype=float)
    # Z-score normalize using the fitted training mean and std
    return (x - scaler['mean']) / scaler['std']


def get_location_map():
    return dict(_LOCATION_MAP)
