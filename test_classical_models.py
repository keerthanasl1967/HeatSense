"""
Unit tests for classical models (Logistic Regression & Random Forest).
"""

import numpy as np
import pytest
from src.classical_models import ClassicalModelTrainer


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    # 3 classes: 0, 1, 2
    y = np.random.choice([0, 1, 2], size=50)
    return X, y


def test_logistic_regression_train_and_predict(dummy_train_data, tmp_path):
    X, y = dummy_train_data
    trainer = ClassicalModelTrainer(model_type="logistic_regression")
    assert trainer.is_trained is False

    t_sec = trainer.train(X, y)
    assert t_sec > 0
    assert trainer.is_trained is True

    preds, inf_time = trainer.predict(X[:10])
    assert len(preds) == 10
    assert inf_time > 0

    probas = trainer.predict_proba(X[:5])
    assert probas is not None
    assert probas.shape == (5, 3)

    # Test serialization
    m_path = tmp_path / "lr.joblib"
    trainer.save(str(m_path))
    loaded = ClassicalModelTrainer.load(str(m_path))
    assert loaded.is_trained is True
    l_preds, _ = loaded.predict(X[:5])
    np.testing.assert_array_equal(preds[:5], l_preds)


def test_random_forest_train_and_predict(dummy_train_data):
    X, y = dummy_train_data
    trainer = ClassicalModelTrainer(model_type="random_forest", params={"n_estimators": 10})
    trainer.train(X, y)
    assert trainer.is_trained is True

    preds, _ = trainer.predict(X[:5])
    assert len(preds) == 5


def test_predict_before_train_raises():
    trainer = ClassicalModelTrainer(model_type="random_forest")
    with pytest.raises(RuntimeError):
        trainer.predict(np.zeros((2, 4)))
