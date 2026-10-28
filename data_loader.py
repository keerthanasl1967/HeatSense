"""
Data loading and validation module for HeatSense Bengaluru.
Supports reading local CSVs, uploaded file buffers, Open-Meteo API downloads,
and performing thorough meteorological data integrity checks.
"""

import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Sensible meteorological boundaries for Bengaluru (tropical urban highland, ~920m elev.)
PHYSICAL_BOUNDS = {
    "temperature_2m": (-5.0, 55.0),       # °C
    "temperature": (-5.0, 55.0),
    "relative_humidity_2m": (0.0, 100.0), # %
    "relative_humidity": (0.0, 100.0),
    "wind_speed_10m": (0.0, 150.0),       # km/h
    "wind_speed": (0.0, 150.0),
    "surface_solar_radiation": (0.0, 1500.0), # W/m2
    "solar_radiation": (0.0, 1500.0),
    "apparent_temperature": (-5.0, 65.0), # °C
}


def load_dataset(
    source: Union[str, io.BytesIO, io.StringIO, pd.DataFrame],
    timestamp_col: Optional[str] = "timestamp",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads and validates a meteorological dataset.
    Returns the loaded DataFrame and a validation report dictionary.
    """
    report: Dict[str, Any] = {
        "success": False,
        "total_rows": 0,
        "valid_rows": 0,
        "columns": [],
        "missing_counts": {},
        "duplicates": 0,
        "out_of_bounds": {},
        "timestamp_parsed": False,
        "date_range": None,
        "errors": [],
        "warnings": [],
    }

    try:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, str):
            if not os.path.exists(source):
                report["errors"].append(f"File not found: {source}")
                return pd.DataFrame(), report
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source)

        report["total_rows"] = len(df)
        report["columns"] = list(df.columns)

        if df.empty:
            report["errors"].append("Dataset is completely empty.")
            return df, report

        # Check duplicates
        report["duplicates"] = int(df.duplicated().sum())

        # Check missing values
        report["missing_counts"] = df.isnull().sum().to_dict()

        # Parse timestamp if present
        if timestamp_col and timestamp_col in df.columns:
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
                valid_dates = df[timestamp_col].dropna()
                if not valid_dates.empty:
                    report["timestamp_parsed"] = True
                    report["date_range"] = (
                        str(valid_dates.min()),
                        str(valid_dates.max()),
                    )
                    # Sort chronologically by default
                    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
            except Exception as e:
                report["warnings"].append(f"Timestamp parsing failed: {e}")

        # Validate physical meteorological boundaries
        for col, (low, high) in PHYSICAL_BOUNDS.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                outliers = ((df[col] < low) | (df[col] > high)).sum()
                if outliers > 0:
                    report["out_of_bounds"][col] = int(outliers)
                    report["warnings"].append(
                        f"Column '{col}' has {outliers} values outside realistic physical bounds [{low}, {high}]."
                    )

        report["valid_rows"] = len(df.dropna())
        report["success"] = True
        return df, report

    except Exception as e:
        report["errors"].append(f"Failed to read CSV: {str(e)}")
        return pd.DataFrame(), report


def fetch_open_meteo_historical(
    start_date: str = "2023-01-01",
    end_date: str = "2023-06-30",
    latitude: float = 12.9716,
    longitude: float = 77.5946,
    timeout_seconds: int = 20,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Retrieves real historical meteorological records from the Open-Meteo Historical Weather API.
    Does NOT require an API key. Documented at https://open-meteo.com/en/docs/historical-weather-api
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,direct_normal_irradiance,apparent_temperature",
        "timezone": "Asia/Kolkata",
    }
    url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "HeatSense-Bengaluru-Research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if "hourly" not in payload:
                return None, "Invalid response from Open-Meteo API: missing 'hourly' field."

            hourly = payload["hourly"]
            df = pd.DataFrame(
                {
                    "timestamp": hourly.get("time", []),
                    "temperature_2m": hourly.get("temperature_2m", []),
                    "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
                    "wind_speed_10m": hourly.get("wind_speed_10m", []),
                    "surface_solar_radiation": hourly.get("direct_normal_irradiance", []),
                    "apparent_temperature": hourly.get("apparent_temperature", []),
                }
            )
            df = df.dropna().reset_index(drop=True)
            return df, None

    except urllib.error.HTTPError as e:
        return None, f"Open-Meteo HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"Network connection error contacting Open-Meteo: {e.reason}"
    except Exception as e:
        return None, f"Unexpected error querying historical weather: {str(e)}"
