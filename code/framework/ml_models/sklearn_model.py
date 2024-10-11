# ml_models/sklearn_model.py
import joblib

class SklearnModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the Scikit-learn model. Load the model from a file if a model_path is provided,
        otherwise use the provided scikit-learn model instance.

        Parameters:
        - model_path: Path to the Scikit-learn model file (optional).
        - model: An existing Scikit-learn model instance (optional).
        """
        if model_path:
            self.model = joblib.load(model_path)  # Load the model from a file
        elif model:
            self.model = model  # Use the provided Scikit-learn model instance
        else:
            raise ValueError("Either model_path or model must be provided.")

    def predict(self, input_data):
        """
        Predict using the Scikit-learn model.

        Parameters:
        - input_data: Input data for the model (numpy array, pandas DataFrame, etc.).

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_path):
        """
        Save the Scikit-learn model to disk.

        Parameters:
        - file_path: Path to save the model file.
        """
        joblib.dump(self.model, file_path)
        print(f"Scikit-learn model saved to {file_path}")