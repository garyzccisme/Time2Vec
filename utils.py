from typing import List, Union

import numpy as np
import pandas as pd


def sliding_window(df: pd.DataFrame, value_feature_cols: List, time_feature_cols: List,
                   target_col: str, window_size: int = 24, step_size: int = 1):
    """
    x1: lookback value features, i.e. ['water_level'].
    x2: lookback time features, i.e. ['timestamp'] or ['year', 'monthofyear', ...].
    x3: forecast point time features, has same index with target.
    y: forecast point target value.
    Note: some windows might not useful for some cases, comment to skip to save time.

    Output Shape:
        x1: [window_num, window_size, value_feature_num]
        x2: [window_num, window_size, time_feature_num]
        x3: [window_num, time_feature_num]
        y: [window_num]
    """
    x1, x2, x3, y = [], [], [], []
    for i in range(window_size, len(df), step_size):
        x1.append(df.loc[i - window_size: i - 1, value_feature_cols].values)
        x2.append(df.loc[i - window_size: i - 1, time_feature_cols].astype(float).values)
        x3.append(df.loc[i, time_feature_cols].astype(float).values)
        y.append(df.loc[i, target_col])
    return np.stack(x1), np.stack(x2), np.stack(x3), np.array(y)


def train_test_split(*arrays, test_size: Union[int, float] = 0.3, verbose: bool = False):
    """
    Split train set & test set by time order.

    Args:
        *arrays: Tuple of arrays with same length, i.e. (X, y) or (X1, X2, y).
        test_size: Test set size, if is float (< 1) then represents ratio; if is integer then represents length.
        verbose: If True, print split output shape.

    Returns: (train_set: List, test_set: List).

    """
    sample_size = arrays[0].shape[0]
    assert all(sample_size == array.shape[0] for array in arrays), "Size mismatch between arrays"

    if isinstance(test_size, int):
        split = sample_size - test_size
    else:
        split = int(sample_size * (1 - test_size))
    train_set, test_set = [], []
    for array in arrays:
        train_set.append(array[:split])
        test_set.append(array[split:])
    if verbose:
        print(f"Train Set Shape: {[array.shape for array in train_set]}")
        print(f"Test Set Shape: {[array.shape for array in test_set]}")
    return train_set, test_set
