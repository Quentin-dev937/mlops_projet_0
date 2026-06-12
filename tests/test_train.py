import os
import skops.io as sio
import pytest
import numpy as np
from sklearn.base import BaseEstimator
from src.models.train import train

@pytest.fixture(scope="module")
def trained_model():
    if os.path.exists("models/model.skops"):
        os.remove("models/model.skops")
    train()
    model = sio.load("models/model.skops", trusted=sio.get_untrusted_types(file="models/model.skops"))
    return model

def test_train_creates_model_file(trained_model):
    """Test 1 : Le fichier est-il créé ?"""

    assert os.path.exists("models/model.skops"), "Le fichier modèle n'a pas été créé."

def test_train_model_is_loadable(trained_model):
    """Test 2 : Le fichier est-il un vrai modèle ?"""

    model = trained_model
    assert isinstance(model, BaseEstimator), "Objet invalide."
    assert hasattr(model, "classes_"), "Modèle non entraîné."

def test_train_model_predicts_correctly(trained_model):
    """Test 3 : Le modèle fonctionne-t-il ?"""

    model = trained_model
    dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(dummy_input)
    assert len(prediction) == 1, "Erreur de format de prédiction."