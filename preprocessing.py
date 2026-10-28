"""
Preprocessing pipeline for HeatSense Bengaluru.

Handles:
- Feature/label separation and target leakage prevention.
- Imputation of missing meteorological values.
- Label encoding with standardized class order.
- Feature scaling (StandardScaler / MinMaxScaler).
- Chronological (time-series) and random train/test splits.
- Quantum-tailored dimensionality reduction (PCA mapped to [0, 2pi]).
- Serialization and loading of fitted preprocessing components.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class PreprocessingPipeline:
    """
    Encapsulates all data transformation states to ensure zero test leakage
    and reproducible inference.
    """

    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        scaler_type: str = "standard",  # "standard" or "minmax"
        quantum_qubits: int = 4,
    ):
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.scaler_type = scaler_type
        self.quantum_qubits = quantum_qubits

        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.quantum_pca: Optional[PCA] = None
        self.quantum_scaler: Optional[MinMaxScaler] = None
        self.classes_: List[str] = []
        self.is_fitted = False

    def fit_transform_train(
        self, df_train: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits transformers on training split ONLY and transforms training data.
        Returns:
            X_train_scaled: Classical scaled features
            y_train_encoded: Label-encoded targets
            X_train_quantum: PCA-reduced, [0, 2pi]-scaled quantum features
        """
        # Validate features
        missing = [c for c in self.feature_columns if c not in df_train.columns]
        if missing:
            raise ValueError(f"Missing required feature columns in training set: {missing}")
        if self.target_column not in df_train.columns:
            raise ValueError(f"Target column '{self.target_column}' missing from training set.")

        X_raw = df_train[self.feature_columns].copy()
        # Impute missing values with column medians from training split
        self.train_medians = X_raw.median().to_dict()
        X_imputed = X_raw.fillna(self.train_medians).values.astype(np.float64)

        # 1. Label Encoder
        self.label_encoder = LabelEncoder()
        # Ensure consistent class ordering: Low, Moderate, High
        y_raw = df_train[self.target_column].astype(str).values
        self.label_encoder.fit(y_raw)
        self.classes_ = list(self.label_encoder.classes_)
        y_encoded = self.label_encoder.transform(y_raw)

        # 2. Classical Feature Scaler
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        # 3. Quantum Dimensionality Reduction (PCA to match qubit count)
        n_features = X_scaled.shape[1]
        n_components = min(self.quantum_qubits, n_features)
        self.quantum_pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.quantum_pca.fit_transform(X_scaled)

        # Scale quantum inputs strictly to [0, 2 * pi] for phase-rotation feature maps
        self.quantum_scaler = MinMaxScaler(feature_range=(0.0, 2.0 * np.pi))
        X_quantum = self.quantum_scaler.fit_transform(X_pca)

        self.is_fitted = True
        return X_scaled, y_encoded, X_quantum

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Applies fitted transformations to unseen evaluation or inference data.
        """
        if not self.is_fitted or self.scaler is None or self.label_encoder is None:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X_raw = df[self.feature_columns].copy()
        X_imputed = X_raw.fillna(self.train_medians).values.astype(np.float64)

        X_scaled = self.scaler.transform(X_imputed)

        y_encoded: Optional[np.ndarray] = None
        if self.target_column in df.columns:
            y_raw = df[self.target_column].astype(str).values
            y_encoded = self.label_encoder.transform(y_raw)

        X_pca = self.quantum_pca.transform(X_scaled) if self.quantum_pca else X_scaled
        X_quantum = self.quantum_scaler.transform(X_pca) if self.quantum_scaler else X_pca

        return X_scaled, y_encoded, X_quantum

    def transform_single_input(self, input_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a single row dictionary of inputs for live prediction.
        Returns:
            x_classical: 2D array of shape (1, n_features)
            x_quantum: 2D array of shape (1, n_qubits)
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Pipeline must be fitted before transforming inputs.")

        row = [input_dict.get(c, self.train_medians.get(c, 0.0)) for c in self.feature_columns]
        x_raw = np.array([row], dtype=np.float64)
        x_scaled = self.scaler.transform(x_raw)
        x_pca = self.quantum_pca.transform(x_scaled) if self.quantum_pca else x_scaled
        x_quantum = self.quantum_scaler.transform(x_pca) if self.quantum_scaler else x_pca

        return x_scaled, x_quantum

    def save(self, file_path: str = "models/preprocessor.joblib") -> None:
        """Saves fitted preprocessor components."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str = "models/preprocessor.joblib") -> "PreprocessingPipeline":
        """Loads fitted preprocessor components."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor not found at {file_path}")
        return joblib.load(file_path)


def split_dataset(
    df: pd.DataFrame,
    strategy: str = "chronological",
    test_size: float = 0.25,
    random_seed: int = 42,
    timestamp_col: Optional[str] = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Splits the dataset into training and test partitions.
    
    Why chronological is preferred for urban heat-risk:
    Meteorological records have strong temporal autocorrelation and diurnal/seasonal patterns.
    Random shuffling leaks future temperature trends into past predictions, causing
    overly optimistic performance metrics. Chronological splitting mirrors real-world
    deployment where models trained on past seasons predict future periods.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    split_meta: Dict[str, Any] = {
        "strategy": strategy,
        "test_size": test_size,
        "random_seed": random_seed if strategy == "random" else None,
        "train_rows": 0,
        "test_rows": 0,
        "train_date_range": None,
        "test_date_range": None,
    }

    if strategy == "chronological":
        # Ensure chronological ordering
        if timestamp_col and timestamp_col in df.columns:
            df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)

        split_idx = int(len(df_sorted) * (1.0 - test_size))
        train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

        if timestamp_col and timestamp_col in df.columns:
            split_meta["train_date_range"] = (
                str(train_df[timestamp_col].min()),
                str(train_df[timestamp_col].max()),
            )
            split_meta["test_date_range"] = (
                str(test_df[timestamp_col].min()),
                str(test_df[timestamp_col].max()),
            )
    elif strategy == "random":
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_seed
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    split_meta["train_rows"] = len(train_df)
    split_meta["test_rows"] = len(test_df)

    return train_df, test_df, split_meta
