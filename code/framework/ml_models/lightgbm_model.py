# ml_models/lightgbm_model.py
import lightgbm as lgb
from pathlib import Path
from utils.logging_utils import log_deployment_event

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
        
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def predict(self, input_data):
        """
        Predict using the LightGBM model.

        Parameters:
        - input_data: The input data for the model (can be numpy array, pandas DataFrame, or similar).

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_name):
        """
        Save the LightGBM model to disk.

        Parameters:
        - file_name: The file name to save the model.
        
        Returns:
        - A dictionary with the status and the path of the saved model.
        """
        file_path = self.data_dir / file_name.with_suffix('.txt')
        try:
            self.model.save_model(file_path)
            log_deployment_event(f"LightGBM model saved to {file_path}")
            return {"status": "success", "model_path": str(file_path)}
        except Exception as e:
            log_deployment_event(f"Error saving LightGBM model: {str(e)}", log_level="error")
            return {"status": "error", "message": f"Error saving model: {str(e)}"}
        
        
    def train(self, train_data_path, params, num_rounds=100):
        """
        Train the LightGBM model with the provided training data and parameters.

        Parameters:
        - train_data_path: Path to the training data (in CSV format).
        - params: Dictionary of LightGBM training parameters.
        - num_rounds: Number of boosting iterations (default: 100).

        Returns:
        - Trained LightGBM model.
        """
        try:
            # Load the training data from the provided path
            train_data = lgb.Dataset(train_data_path)
            
            # Train the model
            self.model = lgb.train(params, train_data, num_boost_round=num_rounds)

            # Save the trained model to the `data` directory
            model_file_path = self.data_dir / "trained_lightgbm_model.txt"
            self.model.save_model(str(model_file_path))

            # Log success
            log_deployment_event(f"Model trained and saved to {model_file_path}")

            return self.model

        except Exception as e:
            # Log the error
            log_deployment_event(f"Error during training: {str(e)}", log_level='error')
            return {"status": "error", "message": str(e)}