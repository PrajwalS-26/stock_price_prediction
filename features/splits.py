"""Chronological and walk-forward split utilities for time-series data."""

import numpy as np


def chronological_split(n, train_frac=0.7, val_frac=0.15):
    """Split first `n` rows into train / val / test boundaries.

    Returns (train_end, val_end): rows [0, train_end) are train,
    [train_end, val_end) are val, [val_end, n) are test.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")
    if n < 20:
        raise ValueError(f"Dataset too small to split (n={n})")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end < 5 or val_end <= train_end or n - val_end < 2:
        raise ValueError(f"Dataset too small to split (n={n})")
    return train_end, val_end


def walk_forward_folds(n, initial_train, test_size, step=None):
    """Generate walk-forward (rolling) folds over `n` rows.

    Returns a list of (train_end, test_start, test_end) tuples where the model
    is trained on rows [0, train_end) and tested on rows [test_start, test_end).
    The training window always expands; the test window rolls forward by
    `step` rows (defaults to `test_size`).
    """
    if initial_train <= 0 or initial_train >= n:
        raise ValueError("initial_train must be in (0, n)")
    step = step or test_size

    folds = []
    start = initial_train
    while start < n:
        end = min(start + test_size, n)
        folds.append((start, start, end))
        start += step
    if not folds:
        raise ValueError("No walk-forward folds could be created")
    return folds


def train_val_test_dates(dates, train_frac=0.7, val_frac=0.15):
    """Return (train_date_end, val_date_end) timestamps for date-based slicing."""
    dates = np.asarray(dates)
    train_end, val_end = chronological_split(len(dates), train_frac, val_frac)
    return dates[train_end], dates[val_end]
