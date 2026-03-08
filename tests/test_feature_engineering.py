import numpy as np
import pandas as pd

from features.feature_engineering import (BASE_FEATURES, MARKET_FEATURES,
                                          available_features, create_features)
from features.pipeline import usable_features


def test_create_features_columns_and_targets(synthetic_stock):
    df = create_features(synthetic_stock)
    for col in BASE_FEATURES:
        assert col in df.columns, f"missing feature {col}"
    for col in ["Target", "Target_5d", "Target_21d", "Direction" ]:
        if col == "Direction":
            continue
        assert col in df.columns

    # core feature columns must be fully populated
    for col in BASE_FEATURES:
        assert col in df.columns, f"missing feature {col}"
        assert df[col].notna().all(), f"NaNs in feature {col}"

    # targets: last row is NaN (future unknown) but earlier rows are valid
    assert df["Target"].iloc[-1] != df["Target"].iloc[-1]  # NaN
    assert df["Target"].iloc[100] == df["Target"].iloc[100]  # finite


def test_target_is_forward_return(synthetic_stock):
    df = create_features(synthetic_stock)
    i = 100
    expected = df["Price"].iloc[i + 1] / df["Price"].iloc[i] - 1
    assert df["Target"].iloc[i] == np.float64(expected) or abs(df["Target"].iloc[i] - expected) < 1e-12

    expected_5d = df["Price"].iloc[i + 5] / df["Price"].iloc[i] - 1
    assert abs(df["Target_5d"].iloc[i] - expected_5d) < 1e-9


def test_market_features_merged(synthetic_stock, synthetic_market, synthetic_vix):
    df = create_features(synthetic_stock, market_df=synthetic_market, vix_df=synthetic_vix)
    for col in MARKET_FEATURES:
        assert col in df.columns
        assert df[col].notna().sum() > 400

    # excess return = stock return - market return
    i = 150
    assert abs(df["Excess_Return"].iloc[i]
               - (df["Return"].iloc[i] - df["Mkt_Return"].iloc[i])) < 1e-9
    assert (df["VIX"] > 0).all()


def test_without_market_no_market_feature(synthetic_stock):
    df = create_features(synthetic_stock)
    feats = usable_features(df)
    assert "Excess_Return" not in feats
    assert "VIX" not in feats
    assert all(f in BASE_FEATURES for f in feats)


def test_empty_frame_raises():
    import pytest
    with pytest.raises(ValueError):
        create_features(pd.DataFrame())
