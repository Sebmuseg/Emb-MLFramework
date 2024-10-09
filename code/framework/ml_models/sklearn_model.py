# ml_models/sklearn_model.py
from sklearn.externals import joblib

class SklearnModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, input_data):
        """Führt eine Vorhersage basierend auf einem scikit-learn-Modell durch."""
        return self.model.predict(input_data)