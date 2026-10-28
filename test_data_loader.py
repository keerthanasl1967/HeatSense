"""
Unit tests for data_loader module in HeatSense Bengaluru.
"""

import io
import pandas as pd
import pytest
from src.data_loader import load_dataset, PHYSICAL_BOUNDS


def test_load_valid_csv(tmp_path):
    csv_file = tmp_path / "valid_weather.csv"
    csv_file.write_text(
        "timestamp,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_solar_radiation\n"
        "2023-04-01 12:00:00,32.5,45.0,12.0,650.0\n"
        "2023-04-01 13:00:00,33.0,42.0,14.0,700.0\n"
    )
    df, report = load_dataset(str(csv_file))
    assert report["success"] is True
    assert len(df) == 2
    assert report["timestamp_parsed"] is True
    assert report["duplicates"] == 0


def test_load_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("timestamp,temperature_2m,relative_humidity_2m\n")
    df, report = load_dataset(str(csv_file))
    assert df.empty
    assert len(report["errors"]) > 0


def test_missing_values_detection():
    data = {
        "timestamp": ["2023-04-01 10:00", "2023-04-01 11:00"],
        "temperature_2m": [30.0, None],
        "relative_humidity_2m": [50.0, 55.0],
    }
    df_raw = pd.DataFrame(data)
    df, report = load_dataset(df_raw)
    assert report["success"] is True
    assert report["missing_counts"]["temperature_2m"] == 1


def test_duplicate_records_detection():
    data = {
        "timestamp": ["2023-04-01 10:00", "2023-04-01 10:00"],
        "temperature_2m": [30.0, 30.0],
        "relative_humidity_2m": [50.0, 50.0],
    }
    df_raw = pd.DataFrame(data)
    df, report = load_dataset(df_raw)
    assert report["duplicates"] == 1


def test_out_of_bounds_detection():
    data = {
        "timestamp": ["2023-04-01 10:00"],
        "temperature_2m": [85.0],  # Physically impossible ambient temp in Bengaluru
        "relative_humidity_2m": [120.0],  # Impossible >100%
    }
    df_raw = pd.DataFrame(data)
    df, report = load_dataset(df_raw)
    assert "temperature_2m" in report["out_of_bounds"]
    assert "relative_humidity_2m" in report["out_of_bounds"]
    assert len(report["warnings"]) >= 2


def test_nonexistent_file():
    df, report = load_dataset("non_existent_file.csv")
    assert report["success"] is False
    assert any("not found" in err.lower() for err in report["errors"])
