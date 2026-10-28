"""
Evaluation metrics module for HeatSense Bengaluru.

Calculates standardized classification metrics across both classical and quantum
models under identical experimental definitions.

Special focus on:
- High-Risk Recall: Capability to flag critical heat-stress events.
- High-Risk False-Negative Rate (FNR = FN / (FN + TP)): Frequency of dangerously
  failing to detect high-risk thermal conditions.
- Strict zero_division safety and explicit handling of missing classes.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    high_risk_classes: Optional[List[str]] = None,
    training_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics on test predictions.
    Supports both multiclass and high-risk binary evaluation.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    if high_risk_classes is None:
        high_risk_classes = ["High", "High Risk", "Extreme Risk"]

    # Overall metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Detailed per-class report
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # High-Risk Specific Metrics (Binary evaluation of high-risk vs non-high-risk)
    # Identify indices corresponding to high-risk classes
    high_risk_indices = [
        i for i, name in enumerate(class_names)
        if name in high_risk_classes or any(hr.lower() in name.lower() for hr in high_risk_classes)
    ]

    is_high_risk_true = np.isin(y_true, high_risk_indices)
    is_high_risk_pred = np.isin(y_pred, high_risk_indices)

    tp_high = int(np.sum(is_high_risk_true & is_high_risk_pred))
    fn_high = int(np.sum(is_high_risk_true & (~is_high_risk_pred)))
    fp_high = int(np.sum((~is_high_risk_true) & is_high_risk_pred))
    tn_high = int(np.sum((~is_high_risk_true) & (~is_high_risk_pred)))

    total_actual_high = tp_high + fn_high

    if total_actual_high > 0:
        high_risk_recall = float(tp_high / total_actual_high)
        high_risk_fnr = float(fn_high / total_actual_high)
    else:
        # If test set contains zero actual high-risk samples, mark as None (N/A)
        high_risk_recall = None
        high_risk_fnr = None

    sample_count = len(y_true)
    latency_ms_per_sample = (
        (inference_time_sec * 1000.0) / sample_count if sample_count > 0 else 0.0
    )

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "high_risk_recall": high_risk_recall,
        "high_risk_fnr": high_risk_fnr,
        "high_risk_counts": {
            "TP": tp_high,
            "FN": fn_high,
            "FP": fp_high,
            "TN": tn_high,
            "total_actual_high_risk": total_actual_high,
        },
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "training_time_seconds": round(training_time_sec, 4),
        "inference_time_seconds": round(inference_time_sec, 4),
        "latency_ms_per_sample": round(latency_ms_per_sample, 4),
        "test_sample_count": sample_count,
    }


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Saves structured experiment results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads saved experiment results."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
