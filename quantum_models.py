"""
Quantum Machine Learning module for HeatSense Bengaluru.

Implements genuine quantum classifiers using Qiskit and Qiskit Machine Learning:
1. Primary: Quantum Support Vector Classifier (QSVC) via FidelityQuantumKernel
   with a Second-Order Pauli-Z (ZZFeatureMap) entangling circuit.
2. Optional: Variational Quantum Classifier (VQC) with RealAmplitudes ansatz.

Features:
- Genuine quantum circuit simulation (Statevector simulator).
- Configurable qubit count (2 to 4) and circuit depth (reps).
- Phase encoding of PCA-projected weather features in [0, 2pi].
- Subsampling controls for CPU simulation feasibility.
- Circuit architecture inspection (depth, gates, parameters, ASCII diagram).
- Strict separation of quantum simulation time vs classical execution.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Qiskit imports with fallback safety
try:
    import qiskit
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import QSVC, VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class QuantumClassifierTrainer:
    """
    Manages genuine Qiskit quantum classifier training, circuit inspection,
    simulation metrics, and inference.
    """

    def __init__(
        self,
        algorithm: str = "QSVC",
        feature_map_type: str = "ZZFeatureMap",
        num_qubits: int = 4,
        reps: int = 2,
        entanglement: str = "linear",
        max_train_samples: int = 120,
        max_test_samples: int = 60,
    ):
        self.algorithm = algorithm.upper()
        self.feature_map_type = feature_map_type
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples

        self.model: Optional[Any] = None
        self.feature_map: Optional[Any] = None
        self.kernel: Optional[Any] = None
        self.training_time_sec: float = 0.0
        self.is_trained: bool = False
        self.circuit_info: Dict[str, Any] = {}

        if QISKIT_AVAILABLE:
            self._build_circuit()

    def _build_circuit(self) -> None:
        """Constructs the quantum feature map circuit and records architecture stats."""
        if self.feature_map_type == "ZFeatureMap":
            self.feature_map = ZFeatureMap(
                feature_dimension=self.num_qubits, reps=self.reps
            )
        else:
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.reps,
                entanglement=self.entanglement,
            )

        decomp = self.feature_map.decompose()
        self.circuit_info = {
            "num_qubits": self.num_qubits,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "feature_map": self.feature_map_type,
            "circuit_depth": int(decomp.depth()),
            "gate_count": dict(decomp.count_ops()),
            "num_parameters": len(self.feature_map.parameters),
            "ascii_diagram": str(decomp.draw("text")),
        }

    def train(
        self,
        X_train_quantum: np.ndarray,
        y_train: np.ndarray,
        subsample: bool = True,
    ) -> Dict[str, Any]:
        """
        Trains the quantum classifier using quantum statevector simulation.
        If subsample is True, trains on up to max_train_samples to ensure
        laptop/CPU simulation finishes within reasonable seconds.
        """
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "Qiskit or Qiskit Machine Learning is not installed. "
                "Run '.venv/bin/pip install qiskit qiskit-machine-learning' to enable QML."
            )

        # Slice features to match num_qubits
        X_sub = X_train_quantum[:, : self.num_qubits]
        y_sub = y_train

        # Subsample if requested for computational tractability
        actual_train_count = len(X_sub)
        if subsample and len(X_sub) > self.max_train_samples:
            # Preserve class representation if possible
            indices = np.linspace(0, len(X_sub) - 1, self.max_train_samples, dtype=int)
            X_sub = X_sub[indices]
            y_sub = y_sub[indices]
            actual_train_count = len(X_sub)

        start = time.perf_counter()

        if self.algorithm == "QSVC":
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
            self.model = QSVC(quantum_kernel=self.kernel)
            self.model.fit(X_sub, y_sub)
        elif self.algorithm == "VQC":
            ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=1)
            optimizer = COBYLA(maxiter=20)
            self.model = VQC(
                feature_map=self.feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
            )
            self.model.fit(X_sub, y_sub)
        else:
            raise ValueError(f"Unknown quantum algorithm: {self.algorithm}")

        end = time.perf_counter()
        self.training_time_sec = end - start
        self.is_trained = True

        return {
            "training_time_sec": self.training_time_sec,
            "trained_samples": actual_train_count,
            "num_qubits": self.num_qubits,
            "algorithm": self.algorithm,
        }

    def predict(
        self,
        X_test_quantum: np.ndarray,
        subsample: bool = True,
    ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """
        Runs quantum inference.
        Returns (predictions, inference_time_sec, evaluated_indices).
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Quantum model has not been trained yet.")

        X_sub = X_test_quantum[:, : self.num_qubits]
        indices = None

        if subsample and len(X_sub) > self.max_test_samples:
            indices = np.linspace(0, len(X_sub) - 1, self.max_test_samples, dtype=int)
            X_eval = X_sub[indices]
        else:
            X_eval = X_sub

        start = time.perf_counter()
        preds = self.model.predict(X_eval)
        end = time.perf_counter()

        return preds, (end - start), indices

    def predict_single(self, x_quantum: np.ndarray) -> int:
        """Predicts risk class for a single transformed input vector."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Quantum model has not been trained yet.")
        x_in = x_quantum[:, : self.num_qubits]
        pred = self.model.predict(x_in)
        return int(pred[0])

    def save(self, filepath: str = "models/quantum_model.joblib") -> None:
        """Saves trained quantum model and circuit architecture."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "algorithm": self.algorithm,
                "feature_map_type": self.feature_map_type,
                "num_qubits": self.num_qubits,
                "reps": self.reps,
                "entanglement": self.entanglement,
                "model": self.model,
                "circuit_info": self.circuit_info,
                "training_time_sec": self.training_time_sec,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str = "models/quantum_model.joblib") -> "QuantumClassifierTrainer":
        """Loads trained quantum model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Quantum model file not found at {filepath}")
        data = joblib.load(filepath)
        trainer = cls(
            algorithm=data.get("algorithm", "QSVC"),
            feature_map_type=data.get("feature_map_type", "ZZFeatureMap"),
            num_qubits=data.get("num_qubits", 4),
            reps=data.get("reps", 2),
            entanglement=data.get("entanglement", "linear"),
        )
        trainer.model = data.get("model")
        trainer.circuit_info = data.get("circuit_info", {})
        trainer.training_time_sec = data.get("training_time_sec", 0.0)
        trainer.is_trained = True
        return trainer
