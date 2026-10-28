"""
Unit tests for quantum machine learning models (Qiskit & Qiskit ML).
"""

import numpy as np
import pytest
from src.quantum_models import QISKIT_AVAILABLE, QuantumClassifierTrainer


def test_quantum_circuit_initialization():
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not installed.")

    trainer = QuantumClassifierTrainer(
        algorithm="QSVC",
        feature_map_type="ZZFeatureMap",
        num_qubits=2,
        reps=1,
    )
    assert trainer.circuit_info["num_qubits"] == 2
    assert trainer.circuit_info["reps"] == 1
    assert "gate_count" in trainer.circuit_info
    assert "ascii_diagram" in trainer.circuit_info


def test_quantum_small_train_and_predict(tmp_path):
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not installed.")

    # Small 8-sample dataset with 2 features in [0, 2pi]
    np.random.seed(42)
    X = np.random.uniform(0.0, 2.0 * np.pi, size=(8, 2))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    trainer = QuantumClassifierTrainer(
        algorithm="QSVC",
        feature_map_type="ZZFeatureMap",
        num_qubits=2,
        reps=1,
        max_train_samples=8,
        max_test_samples=4,
    )

    train_meta = trainer.train(X, y, subsample=False)
    assert trainer.is_trained is True
    assert train_meta["training_time_sec"] > 0

    preds, inf_time, _ = trainer.predict(X[:4], subsample=False)
    assert len(preds) == 4
    assert inf_time > 0

    single_pred = trainer.predict_single(X[:1])
    assert single_pred in [0, 1]

    # Test serialization
    m_path = tmp_path / "quantum.joblib"
    trainer.save(str(m_path))
    loaded = QuantumClassifierTrainer.load(str(m_path))
    assert loaded.is_trained is True
    loaded_preds, _, _ = loaded.predict(X[:4], subsample=False)
    np.testing.assert_array_equal(preds, loaded_preds)
