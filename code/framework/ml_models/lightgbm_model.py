# ml_models/lightgbm_model.py
import lightgbm as lgb

class LightGBMModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the LightGBM model. Load the model from a file if a model_path is provided,
        otherwise use the provided LightGBM model instance.

        Parameters:
        - model_path: Path to the LightGBM model file (optional).
        - model: An existing LightGBM model instance (optional).
        """
        if model_path:
            self.model = lgb.Booster(model_file=model_path)  # Load model from file
        elif model:
            self.model = model  # Use the provided LightGBM model instance
        else:
            raise ValueError("Either model_path or model must be provided.")
    
    def predict(self, input_data):
        """
        Predict using the LightGBM model.

        Parameters:
        - input_data: The input data for the model (can be numpy array, pandas DataFrame, or similar).

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_path):
        """
        Save the LightGBM model to disk.

        Parameters:
        - file_path: The file path to save the model.
        """
        self.model.save_model(file_path)
        print(f"LightGBM model saved to {file_path}")