# ml_models/xgboost_model.py
import xgboost as xgb
from pathlib import Path
from utils.logging_utils import log_deployment_event

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

        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
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

    def save(self, file_name):
        """
        Save the XGBoost model to disk.

        Parameters:
        - file_name: The file name to save the model.
        
        Returns:
        - A dictionary with the status and the path of the saved model.
        """
        file_path = self.data_dir / file_name.with_suffix('.json')
        try:
            self.model.save_model(str(file_path))  # Save the XGBoost model to a JSON file
            log_deployment_event(f"XGBoost model saved to {file_path}")
            return {"status": "success", "model_path": str(file_path)}
        except Exception as e:
            log_deployment_event(f"Error saving XGBoost model: {str(e)}", log_level='error')
            return {"status": "error", "message": str(e)}
        
    def train(self, train_data, train_labels, params=None):
        """
        Train the XGBoost model.

        Parameters:
        - train_data: The training data (e.g., a DMatrix or NumPy array).
        - train_labels: The training labels (e.g., NumPy array).
        - params: A dictionary of parameters for training the model (optional).
        
        Returns:
        - A dictionary containing the status and training result.
        """
        try:
            if not params:
                # Default training parameters if none provided
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 100
                }

            # Convert training data to DMatrix (required by XGBoost)
            dtrain = xgb.DMatrix(train_data, label=train_labels)

            # Train the model
            self.model = xgb.train(params, dtrain)

            log_deployment_event(f"XGBoost model training completed.")
            return {"status": "success", "message": "Training completed."}
        except Exception as e:
            log_deployment_event(f"Error during XGBoost model training: {str(e)}", log_level='error')
            return {"status": "error", "message": str(e)}