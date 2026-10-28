"""
Heat-risk labeling module for HeatSense Bengaluru.

Provides scientifically documented, configurable methodologies for deriving
urban heat-risk categories from raw meteorological observations while preventing
target leakage.

Authoritative Standards Supported:
1. NOAA / National Weather Service (NWS) Steadman Heat Index (Rothfusz equation)
2. Indian Meteorological Department (IMD) Tropical Urban Heat Stress Criteria
3. Australian Bureau of Meteorology (BOM) Apparent Temperature / WBGT Proxy
4. Pre-existing dataset risk labels
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_noaa_heat_index(temp_c: float, rel_humidity: float) -> float:
    """
    Computes the NOAA / NWS Heat Index (apparent temperature in °C) using the
    standard Steadman formulation and Rothfusz multiple regression equations.

    Reference:
    NOAA Technical Attachment SR 90-23 (1990) / Steadman (1979).
    """
    # Convert Celsius to Fahrenheit
    T = (temp_c * 9.0 / 5.0) + 32.0
    R = max(0.0, min(100.0, rel_humidity))

    # Simple Steadman equation for cooler conditions
    hi_simple = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))
    if hi_simple < 80.0:
        heat_index_f = hi_simple
    else:
        # Full Rothfusz regression equation
        hi = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * R
            - 0.22475541 * T * R
            - 0.00683783 * (T**2)
            - 0.05481717 * (R**2)
            + 0.00122874 * (T**2) * R
            + 0.00085282 * T * (R**2)
            - 0.00000199 * (T**2) * (R**2)
        )

        # Adjustments
        if R < 13.0 and 80.0 <= T <= 112.0:
            adjustment = ((13.0 - R) / 4.0) * math.sqrt((17.0 - abs(T - 95.0)) / 17.0)
            hi -= adjustment
        elif R > 85.0 and 80.0 <= T <= 87.0:
            adjustment = ((R - 85.0) / 10.0) * ((87.0 - T) / 5.0)
            hi += adjustment

        heat_index_f = hi

    # Convert back to Celsius
    heat_index_c = (heat_index_f - 32.0) * 5.0 / 9.0
    return heat_index_c


def categorize_noaa_heat_index(hi_c: float, moderate_thresh: float = 30.0, high_thresh: float = 34.0) -> str:
    """
    Categorizes NOAA Heat Index into 3 risk levels for research classification:
    Configurable thresholds allow adaptation to tropical urban highland contexts like Bengaluru
    where heat stress manifests at lower absolute values than desert environments.
    Default thresholds:
    - Low: < moderate_thresh (default 30.0°C)
    - Moderate: moderate_thresh to high_thresh (default 30.0°C to 34.0°C)
    - High: >= high_thresh (default 34.0°C)
    """
    if hi_c < moderate_thresh:
        return "Low"
    elif hi_c < high_thresh:
        return "Moderate"
    else:
        return "High"


def categorize_imd_criteria(
    temp_c: float,
    rel_humidity: float,
    moderate_temp: float = 32.0,
    high_temp: float = 35.0,
) -> str:
    """
    Categorizes heat risk based on Indian Meteorological Department (IMD)
    heat wave and tropical urban heat stress advisory guidelines:
    - High: Air temperature >= high_temp OR (temp >= moderate_temp and RH >= 65%)
    - Moderate: Air temperature >= moderate_temp OR (temp >= 29.0 and RH >= 60%)
    - Low: Below moderate criteria
    """
    if temp_c >= high_temp or (temp_c >= moderate_temp and rel_humidity >= 65.0):
        return "High"
    elif temp_c >= moderate_temp or (temp_c >= 29.0 and rel_humidity >= 60.0):
        return "Moderate"
    else:
        return "Low"


def categorize_bom_apparent_temperature(temp_c: float, rel_humidity: float, wind_speed_kmh: float) -> str:
    """
    Australian Bureau of Meteorology (BOM) Apparent Temperature formulation:
    AT = Ta + 0.33 * e - 0.70 * ws - 4.00
    where:
    Ta = Dry bulb temp (°C)
    e = Water vapor pressure (hPa) = (RH / 100) * 6.105 * exp((17.27 * Ta) / (237.7 + Ta))
    ws = Wind speed (m/s)
    """
    ws_ms = wind_speed_kmh / 3.6
    e = (rel_humidity / 100.0) * 6.105 * math.exp((17.27 * temp_c) / (237.7 + temp_c))
    at = temp_c + (0.33 * e) - (0.70 * ws_ms) - 4.00

    if at < 30.0:
        return "Low"
    elif at < 36.0:
        return "Moderate"
    else:
        return "High"


def apply_heat_risk_labeling(
    df: pd.DataFrame,
    method: str = "heat_index_noaa",
    existing_target_col: Optional[str] = None,
    temp_col: str = "temperature_2m",
    humidity_col: str = "relative_humidity_2m",
    wind_col: str = "wind_speed_10m",
    output_target_col: str = "heat_risk_category",
    moderate_threshold: float = 30.0,
    high_threshold: float = 34.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies the chosen labeling methodology to the dataset.
    Returns the updated DataFrame and a labeling metadata dictionary.
    
    CRITICAL (Target Leakage Prevention):
    Any auxiliary calculation columns (e.g. intermediate heat_index_calc)
    are retained ONLY for reference and MUST NOT be passed into model training.
    """
    df_out = df.copy()
    meta: Dict[str, Any] = {
        "method": method,
        "target_col": output_target_col,
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "leakage_warning_columns": [],
        "class_distribution": {},
        "notes": "",
    }

    if method == "existing_column":
        if not existing_target_col or existing_target_col not in df_out.columns:
            raise ValueError(f"Existing target column '{existing_target_col}' not found in dataset.")
        df_out[output_target_col] = df_out[existing_target_col].astype(str).str.strip()
        meta["notes"] = f"Using pre-existing risk column '{existing_target_col}'."
        meta["class_distribution"] = df_out[output_target_col].value_counts().to_dict()
        return df_out, meta

    # Check for required temperature and humidity columns
    if temp_col not in df_out.columns:
        raise ValueError(f"Required temperature column '{temp_col}' is missing.")
    if humidity_col not in df_out.columns:
        raise ValueError(f"Required humidity column '{humidity_col}' is missing.")

    if method == "heat_index_noaa":
        calc_hi = [
            compute_noaa_heat_index(float(t), float(h))
            for t, h in zip(df_out[temp_col], df_out[humidity_col])
        ]
        df_out["_derived_heat_index_c"] = calc_hi
        df_out[output_target_col] = [
            categorize_noaa_heat_index(hi, moderate_thresh=moderate_threshold, high_thresh=high_threshold)
            for hi in calc_hi
        ]
        meta["leakage_warning_columns"].append("_derived_heat_index_c")
        if "apparent_temperature" in df_out.columns:
            meta["leakage_warning_columns"].append("apparent_temperature")
        meta["notes"] = f"Derived via NOAA Steadman Rothfusz Heat Index [Mod: {moderate_threshold}°C, High: {high_threshold}°C]. Intermediate index column excluded from training."

    elif method == "imd_criteria":
        df_out[output_target_col] = [
            categorize_imd_criteria(float(t), float(h), moderate_temp=moderate_threshold, high_temp=high_threshold)
            for t, h in zip(df_out[temp_col], df_out[humidity_col])
        ]
        meta["notes"] = f"Derived via IMD tropical urban heat stress advisory criteria [Mod: {moderate_threshold}°C, High: {high_threshold}°C]."

    elif method == "apparent_temp_wbgt":
        wind = df_out[wind_col] if wind_col in df_out.columns else np.zeros(len(df_out))
        df_out[output_target_col] = [
            categorize_bom_apparent_temperature(float(t), float(h), float(w))
            for t, h, w in zip(df_out[temp_col], df_out[humidity_col], wind)
        ]
        meta["notes"] = "Derived via Australian BOM Apparent Temperature thresholding."

    else:
        raise ValueError(f"Unknown labeling method: {method}")

    meta["class_distribution"] = df_out[output_target_col].value_counts().to_dict()
    return df_out, meta
