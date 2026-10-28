"""
Classical machine learning models for HeatSense Bengaluru.
Implements Logistic Regression and Random Forest classifiers.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ClassicalModelTrainer:
    """
    Manages training, timing, prediction, and serialization of classical ML models.
    """

    def __init__(self, model_type: str = "random_forest", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.params = params or {}
        self.model: Optional[LogisticRegression | RandomForestClassifier] = None
        self.training_time_sec: float = 0.0
        self.is_trained: bool = False
        self._init_model()

    def _init_model(self) -> None:
        if self.model_type == "logistic_regression":
            default_params = {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": 42,
            }
            default_params.update(self.params)
            self.model = LogisticRegression(**default_params)
        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 4,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.params)
            self.model = RandomForestClassifier(**default_params)
        else:
            raise ValueError(f"Unsupported classical model type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Trains model and records wall-clock training duration in seconds.
        """
        if self.model is None:
            self._init_model()

        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        end = time.perf_counter()

        self.training_time_sec = end - start
        self.is_trained = True
        return self.training_time_sec

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predicts classes and records total inference duration.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"{self.model_type} has not been trained yet.")

        start = time.perf_counter()
        preds = self.model.predict(X)
        end = time.perf_counter()

        return preds, (end - start)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns class probabilities if supported.
        """
        if not self.is_trained or self.model is None:
            return None
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, filepath: str) -> None:
        """Saves trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model_type": self.model_type, "model": self.model, "training_time": self.training_time_sec}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClassicalModelTrainer":
        """Loads trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        data = joblib.load(filepath)
        trainer = cls(model_type=data["model_type"])
        trainer.model = data["model"]
        trainer.training_time_sec = data.get("training_time", 0.0)
        trainer.is_trained = True
        return trainer
