# ml_models/catboost_model.py
from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the CatBoost model. Load the model from a file if a model_path is provided,
        otherwise use the provided model instance.

        Parameters:
        - model_path: Path to a pre-trained CatBoost model file (optional).
        - model: An existing CatBoostClassifier model instance (optional).
        """
        if model_path:
            self.model = CatBoostClassifier()  # Initialize the CatBoost model
            self.model.load_model(model_path)  # Load the model from the provided path
        elif model:
            self.model = model  # Use the provided CatBoost model instance
        else:
            raise ValueError("Either model_path or model must be provided.")

    def predict(self, input_data):
        """
        Predict using the CatBoost model.

        Parameters:
        - input_data: The input data for making predictions (numpy array, pandas DataFrame, list, etc.)

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_path):
        """
        Save the current model to disk.

        Parameters:
        - file_path: The file path to save the model.
        """
        self.model.save_model(file_path)
        print(f"Model saved to {file_path}")