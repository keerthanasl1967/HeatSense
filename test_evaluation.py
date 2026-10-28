"""
Unit tests for evaluation metrics and high-risk FNR.
"""

import numpy as np
import pytest
from src.evaluation import compute_metrics, load_results, save_results


def test_compute_metrics_standard():
    class_names = ["Low", "Moderate", "High"]
    # 0: Low, 1: Moderate, 2: High
    y_true = np.array([0, 1, 2, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1, 0, 1, 2, 0])  # One high misclassified as moderate (FN)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        high_risk_classes=["High"],
        training_time_sec=1.5,
        inference_time_sec=0.02,
    )

    assert metrics["accuracy"] == 7.0 / 8.0
    assert metrics["high_risk_counts"]["total_actual_high_risk"] == 3
    assert metrics["high_risk_counts"]["TP"] == 2
    assert metrics["high_risk_counts"]["FN"] == 1
    # Recall = 2 / 3 = ~0.6667
    assert pytest.approx(metrics["high_risk_recall"], 0.01) == (2.0 / 3.0)
    # FNR = FN / (FN + TP) = 1 / 3 = ~0.3333
    assert pytest.approx(metrics["high_risk_fnr"], 0.01) == (1.0 / 3.0)
    assert metrics["training_time_seconds"] == 1.5


def test_missing_high_risk_class_zero_division():
    class_names = ["Low", "Moderate", "High"]
    # Only Low and Moderate in test set
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0])

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        high_risk_classes=["High"],
    )

    # When no actual high risk cases exist, must report None (N/A) without crashing
    assert metrics["high_risk_recall"] is None
    assert metrics["high_risk_fnr"] is None
    assert metrics["high_risk_counts"]["total_actual_high_risk"] == 0


def test_empty_evaluation_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]), class_names=["Low", "High"])


def test_results_serialization(tmp_path):
    res = {
        "accuracy": 0.95,
        "high_risk_fnr": 0.05,
        "model": "Random Forest",
    }
    path = tmp_path / "res.json"
    save_results(res, str(path))
    loaded = load_results(str(path))
    assert loaded["accuracy"] == 0.95
    assert loaded["high_risk_fnr"] == 0.05
