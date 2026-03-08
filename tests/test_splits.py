import pytest

from features.splits import chronological_split, walk_forward_folds


def test_chronological_split_boundaries():
    train_end, val_end = chronological_split(1000, 0.7, 0.15)
    assert train_end == 700
    assert val_end == 850


def test_chronological_split_validation():
    with pytest.raises(ValueError):
        chronological_split(100, 1.2, 0.1)
    with pytest.raises(ValueError):
        chronological_split(100, 0.7, 0.4)
    with pytest.raises(ValueError):
        chronological_split(5, 0.7, 0.15)  # too small


def test_walk_forward_folds_expand_and_roll():
    folds = walk_forward_folds(n=300, initial_train=150, test_size=50, step=50)
    assert folds[0] == (150, 150, 200)
    # training window expands from the start on every fold
    prev_train_end = 0
    for train_end, test_start, test_end in folds:
        assert train_end == test_start
        assert train_end > prev_train_end
        prev_train_end = train_end
        assert test_end - test_start <= 50
    # test windows are contiguous and cover the tail exactly
    starts = [f[1] for f in folds]
    assert starts == sorted(starts)
    assert folds[-1][2] == 300


def test_walk_forward_folds_last_window_partial():
    folds = walk_forward_folds(n=290, initial_train=150, test_size=50, step=50)
    assert folds[-1][2] == 290  # final window truncated to n


def test_walk_forward_validation():
    with pytest.raises(ValueError):
        walk_forward_folds(n=100, initial_train=100, test_size=10)
    with pytest.raises(ValueError):
        walk_forward_folds(n=100, initial_train=0, test_size=10)
    # test_size larger than remaining rows truncates every fold's window
    folds = walk_forward_folds(n=50, initial_train=10, test_size=100, step=10)
    assert folds == [(10, 10, 50), (20, 20, 50), (30, 30, 50), (40, 40, 50)]
