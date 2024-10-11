# ml_models/xgboost_model.py
import xgboost as xgb

class XGBoostModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the XGBoost model. Load the model from a file if a model_path is provided,
        otherwise use the provided XGBoost model instance.

        Parameters:
        - model_path: Path to the XGBoost model file (optional).
        - model: An existing XGBoost model instance (optional).
        """
        if model_path:
            self.model = xgb.Booster()  # Initialize an XGBoost booster
            self.model.load_model(model_path)  # Load the model from a file
        elif model:
            self.model = model  # Use the provided XGBoost model instance
        else:
            raise ValueError("Either model_path or model must be provided.")
    
    def predict(self, input_data):
        """
        Predict using the XGBoost model.

        Parameters:
        - input_data: The input data for the model in the form of a DMatrix (xgb.DMatrix).

        Returns:
        - Predictions from the model.
        """
        if not isinstance(input_data, xgb.DMatrix):
            input_data = xgb.DMatrix(input_data)  # Convert input to DMatrix if not already
        return self.model.predict(input_data)

    def save(self, file_path):
        """
        Save the XGBoost model to disk.

        Parameters:
        - file_path: The file path to save the model.
        """
        self.model.save_model(file_path)
        print(f"XGBoost model saved to {file_path}")