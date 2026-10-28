"""
Experiment Manager for HeatSense Bengaluru.

Coordinates:
- Dataset ingestion and validation.
- Scientifically grounded labeling (NOAA, IMD, or BOM).
- Strict chronological train/test splitting and leakage-free preprocessing.
- Training of classical baselines (Logistic Regression, Random Forest).
- Training of quantum classifiers (QSVC with genuine Qiskit circuit simulation).
- Standardized evaluation on identical held-out test sets.
- Persistence of trained artifacts in models/ and results in results/.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from src.classical_models import ClassicalModelTrainer
from src.data_loader import load_dataset
from src.evaluation import compute_metrics, save_results
from src.labeling import apply_heat_risk_labeling
from src.preprocessing import PreprocessingPipeline, split_dataset
from src.quantum_models import QuantumClassifierTrainer


class ExperimentManager:
    """
    Executes and coordinates reproducible comparative experiments.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()

        self.df_raw: Optional[pd.DataFrame] = None
        self.df_labeled: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.preprocessor: Optional[PreprocessingPipeline] = None
        self.classical_models: Dict[str, ClassicalModelTrainer] = {}
        self.quantum_model: Optional[QuantumClassifierTrainer] = None
        self.results: Dict[str, Any] = {}
        self.is_completed = False

    def load_config(self) -> None:
        """Loads experiment configuration from YAML file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    def run_pipeline(
        self,
        dataset_path: Optional[str] = None,
        uploaded_df: Optional[pd.DataFrame] = None,
        labeling_method: Optional[str] = None,
        moderate_threshold: float = 30.0,
        high_threshold: float = 34.0,
        train_quantum: bool = True,
        quantum_qubits: int = 4,
        quantum_train_samples: int = 100,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline:
        Load -> Label -> Preprocess -> Train Classical -> Train Quantum -> Evaluate
        """
        def report_progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # 1. Load Data
        report_progress("Loading meteorological dataset...")
        path = dataset_path or self.config.get("data", {}).get(
            "default_dataset_path", "data/bengaluru_heat_records.csv"
        )
        if uploaded_df is not None:
            self.df_raw, report = load_dataset(uploaded_df)
        else:
            self.df_raw, report = load_dataset(path)

        if not report["success"]:
            raise ValueError(f"Dataset loading failed: {report['errors']}")

        # 2. Labeling
        report_progress("Applying heat-risk labeling...")
        method = labeling_method or self.config.get("data", {}).get(
            "labeling", {}
        ).get("method", "heat_index_noaa")

        target_col = self.config.get("data", {}).get("target_column", "heat_risk_category")
        self.df_labeled, label_meta = apply_heat_risk_labeling(
            self.df_raw,
            method=method,
            output_target_col=target_col,
            moderate_threshold=moderate_threshold,
            high_threshold=high_threshold,
        )

        # 3. Chronological Train/Test Split
        report_progress("Partitioning data (chronological split)...")
        split_cfg = self.config.get("split", {})
        strategy = split_cfg.get("strategy", "chronological")
        test_size = float(split_cfg.get("test_size", 0.25))

        self.train_df, self.test_df, split_meta = split_dataset(
            self.df_labeled,
            strategy=strategy,
            test_size=test_size,
        )

        # 4. Leakage-Free Preprocessing
        report_progress("Fitting preprocessing and PCA quantum encoders...")
        features = self.config.get("data", {}).get(
            "feature_columns",
            ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_solar_radiation"],
        )
        # Ensure target and leakage variables are strictly removed from feature list
        clean_features = [
            f for f in features
            if f in self.train_df.columns
            and f not in label_meta.get("leakage_warning_columns", [])
            and f != target_col
        ]

        self.preprocessor = PreprocessingPipeline(
            feature_columns=clean_features,
            target_column=target_col,
            quantum_qubits=quantum_qubits,
        )

        X_train_c, y_train, X_train_q = self.preprocessor.fit_transform_train(self.train_df)
        X_test_c, y_test, X_test_q = self.preprocessor.transform(self.test_df)
        class_names = self.preprocessor.classes_

        # Save preprocessor
        self.preprocessor.save("models/preprocessor.joblib")

        self.results = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_records": len(self.df_labeled),
                "train_records": len(self.train_df),
                "test_records": len(self.test_df),
                "features": clean_features,
                "target_column": target_col,
                "labeling_method": method,
                "class_names": class_names,
                "split_strategy": strategy,
            },
            "models": {},
        }

        # 5. Train Classical Baselines
        # Model 1: Logistic Regression
        report_progress("Training Classical Baseline 1: Logistic Regression...")
        lr_trainer = ClassicalModelTrainer(model_type="logistic_regression")
        lr_time = lr_trainer.train(X_train_c, y_train)
        lr_preds, lr_inf_time = lr_trainer.predict(X_test_c)
        lr_trainer.save("models/logistic_regression.joblib")
        self.classical_models["logistic_regression"] = lr_trainer

        self.results["models"]["Logistic Regression"] = compute_metrics(
            y_true=y_test,
            y_pred=lr_preds,
            class_names=class_names,
            training_time_sec=lr_time,
            inference_time_sec=lr_inf_time,
        )

        # Model 2: Random Forest
        report_progress("Training Classical Baseline 2: Random Forest...")
        rf_trainer = ClassicalModelTrainer(model_type="random_forest")
        rf_time = rf_trainer.train(X_train_c, y_train)
        rf_preds, rf_inf_time = rf_trainer.predict(X_test_c)
        rf_trainer.save("models/random_forest.joblib")
        self.classical_models["random_forest"] = rf_trainer

        self.results["models"]["Random Forest"] = compute_metrics(
            y_true=y_test,
            y_pred=rf_preds,
            class_names=class_names,
            training_time_sec=rf_time,
            inference_time_sec=rf_inf_time,
        )

        # 6. Train Quantum Classifier (QSVC)
        if train_quantum:
            report_progress("Simulating Qiskit Quantum Feature Map & Training QSVC...")
            try:
                self.quantum_model = QuantumClassifierTrainer(
                    algorithm="QSVC",
                    feature_map_type="ZZFeatureMap",
                    num_qubits=quantum_qubits,
                    reps=2,
                    entanglement="linear",
                    max_train_samples=quantum_train_samples,
                    max_test_samples=min(60, len(X_test_q)),
                )
                q_train_meta = self.quantum_model.train(X_train_q, y_train, subsample=True)
                q_preds, q_inf_time, eval_indices = self.quantum_model.predict(X_test_q, subsample=True)

                self.quantum_model.save("models/quantum_model.joblib")

                y_test_eval = y_test[eval_indices] if eval_indices is not None else y_test
                q_metrics = compute_metrics(
                    y_true=y_test_eval,
                    y_pred=q_preds,
                    class_names=class_names,
                    training_time_sec=q_train_meta["training_time_sec"],
                    inference_time_sec=q_inf_time,
                )
                q_metrics["circuit_info"] = self.quantum_model.circuit_info
                q_metrics["trained_samples"] = q_train_meta["trained_samples"]
                q_metrics["simulation_note"] = (
                    f"Trained on {q_train_meta['trained_samples']} samples and evaluated on "
                    f"{len(y_test_eval)} test samples using statevector simulation to ensure CPU tractability."
                )

                self.results["models"]["QSVC (Quantum Kernel)"] = q_metrics

            except Exception as e:
                self.results["models"]["QSVC (Quantum Kernel)"] = {
                    "error": str(e),
                    "status": "Quantum simulation failed or was skipped.",
                }
        else:
            self.results["models"]["QSVC (Quantum Kernel)"] = {
                "status": "Quantum model training was deselected.",
            }

        report_progress("Finalizing experiment results and saving artifacts...")
        save_results(self.results, "results/latest_experiment.json")
        self.is_completed = True
        return self.results

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Extracts a clean comparative DataFrame from results.
        """
        rows = []
        for model_name, m in self.results.get("models", {}).items():
            if "accuracy" not in m:
                continue

            hr_fnr = m.get("high_risk_fnr")
            hr_rec = m.get("high_risk_recall")

            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": f"{m.get('accuracy', 0.0):.3f}",
                    "Precision (Macro)": f"{m.get('precision_macro', 0.0):.3f}",
                    "Recall (Macro)": f"{m.get('recall_macro', 0.0):.3f}",
                    "Macro F1": f"{m.get('f1_macro', 0.0):.3f}",
                    "Weighted F1": f"{m.get('f1_weighted', 0.0):.3f}",
                    "High-Risk Recall": f"{hr_rec:.3f}" if hr_rec is not None else "N/A",
                    "High-Risk FNR": f"{hr_fnr:.3f}" if hr_fnr is not None else "N/A",
                    "Training Time (s)": f"{m.get('training_time_seconds', 0.0):.2f}",
                    "Latency (ms/sample)": f"{m.get('latency_ms_per_sample', 0.0):.2f}",
                    "Test Samples": m.get("test_sample_count", 0),
                    # Raw numeric values for plotting
                    "accuracy": m.get("accuracy", 0.0),
                    "f1_macro": m.get("f1_macro", 0.0),
                    "high_risk_recall": hr_rec if hr_rec is not None else np.nan,
                    "high_risk_fnr": hr_fnr if hr_fnr is not None else np.nan,
                    "training_time_seconds": m.get("training_time_seconds", 0.0),
                    "latency_ms_per_sample": m.get("latency_ms_per_sample", 0.0),
                }
            )
        return pd.DataFrame(rows)
