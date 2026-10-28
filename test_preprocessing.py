"""
Unit tests for preprocessing and train/test splitting.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import PreprocessingPipeline, split_dataset


@pytest.fixture
def sample_weather_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
        "temperature_2m": [22.0, 24.0, 26.0, 28.0, 31.0, 33.0, 35.0, 32.0, 27.0, 23.0],
        "relative_humidity_2m": [70.0, 65.0, 60.0, 50.0, 45.0, 40.0, 38.0, 48.0, 62.0, 75.0],
        "wind_speed_10m": [8.0, 9.0, 11.0, 14.0, 16.0, 15.0, 12.0, 10.0, 7.0, 6.0],
        "surface_solar_radiation": [0.0, 120.0, 350.0, 600.0, 800.0, 750.0, 500.0, 200.0, 0.0, 0.0],
        "heat_risk_category": ["Low", "Low", "Low", "Moderate", "Moderate", "High", "High", "Moderate", "Low", "Low"],
    })


def test_chronological_split(sample_weather_df):
    train_df, test_df, meta = split_dataset(sample_weather_df, strategy="chronological", test_size=0.3)
    assert len(train_df) == 7
    assert len(test_df) == 3
    # Ensure chronological order (no future data in train)
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_random_split(sample_weather_df):
    train_df, test_df, meta = split_dataset(sample_weather_df, strategy="random", test_size=0.2, random_seed=42)
    assert len(train_df) == 8
    assert len(test_df) == 2


def test_split_empty_raises():
    with pytest.raises(ValueError):
        split_dataset(pd.DataFrame())


def test_preprocessing_pipeline_fit_transform(sample_weather_df, tmp_path):
    features = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_solar_radiation"]
    target = "heat_risk_category"

    pipeline = PreprocessingPipeline(feature_columns=features, target_column=target, quantum_qubits=4)
    X_train_c, y_train, X_train_q = pipeline.fit_transform_train(sample_weather_df)

    assert X_train_c.shape == (10, 4)
    assert len(y_train) == 10
    assert X_train_q.shape == (10, 4)
    # Quantum values must be strictly bounded in [0, 2*pi]
    assert np.all(X_train_q >= -1e-5)
    assert np.all(X_train_q <= 2.0 * np.pi + 1e-5)

    # Test single input transformation
    single_input = {
        "temperature_2m": 34.0,
        "relative_humidity_2m": 42.0,
        "wind_speed_10m": 12.0,
        "surface_solar_radiation": 700.0,
    }
    x_c_single, x_q_single = pipeline.transform_single_input(single_input)
    assert x_c_single.shape == (1, 4)
    assert x_q_single.shape == (1, 4)

    # Test save and reload
    save_path = tmp_path / "prep.joblib"
    pipeline.save(str(save_path))
    loaded = PreprocessingPipeline.load(str(save_path))
    assert loaded.is_fitted is True
    assert loaded.classes_ == pipeline.classes_
