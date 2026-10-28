"""
Unit tests for prediction pipeline and input validation.
"""

import numpy as np
import pandas as pd
import pytest
from src.classical_models import ClassicalModelTrainer
from src.prediction import PredictionService, validate_prediction_inputs
from src.preprocessing import PreprocessingPipeline


def test_input_validation():
    # Valid Bengaluru weather input
    valid_inputs = {
        "temperature_2m": 33.5,
        "relative_humidity_2m": 55.0,
        "wind_speed_10m": 14.0,
        "surface_solar_radiation": 750.0,
    }
    is_valid, errors = validate_prediction_inputs(valid_inputs)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid impossible inputs
    invalid_inputs = {
        "temperature_2m": 85.0,  # Unrealistic
        "relative_humidity_2m": -5.0,  # Negative humidity
        "wind_speed_10m": 12.0,
        "surface_solar_radiation": 750.0,
    }
    is_valid, errors = validate_prediction_inputs(invalid_inputs)
    assert is_valid is False
    assert len(errors) == 2


def test_prediction_service_execution():
    df = pd.DataFrame({
        "temperature_2m": [22.0, 35.0],
        "relative_humidity_2m": [70.0, 40.0],
        "wind_speed_10m": [8.0, 15.0],
        "surface_solar_radiation": [100.0, 800.0],
        "heat_risk_category": ["Low", "High"],
    })
    features = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_solar_radiation"]
    prep = PreprocessingPipeline(feature_columns=features, target_column="heat_risk_category", quantum_qubits=2)
    X_train_c, y_train, _ = prep.fit_transform_train(df)

    rf = ClassicalModelTrainer(model_type="random_forest", params={"n_estimators": 5})
    rf.train(X_train_c, y_train)

    service = PredictionService(preprocessor=prep, classical_models={"random_forest": rf})

    res = service.predict({
        "temperature_2m": 36.0,
        "relative_humidity_2m": 38.0,
        "wind_speed_10m": 12.0,
        "surface_solar_radiation": 850.0,
    })

    assert res["success"] is True
    assert "random_forest" in res["predictions"]
    assert res["predictions"]["random_forest"]["status"] == "Success"
    assert "disclaimer" in res
    assert "NOT as emergency medical advisories" in res["disclaimer"]
