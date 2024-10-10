# ml_models/catboost_model.py
from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, model_path):
        """
        Load the CatBoost model from the given path.
        """
        self.model = CatBoostClassifier()  # Initialize the CatBoost model
        self.model.load_model(model_path)  # Load the model from file

    def predict(self, input_data):
        """
        Predict using the CatBoost model.

        Parameters:
        - input_data: The data on which to make predictions (numpy array, list, etc.)

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)