# ml_models/catboost_model.py
from catboost import CatBoostClassifier, Pool
from pathlib import Path
from utils.logging_utils import log_deployment_event

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
        
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, input_data):
        """
        Predict using the CatBoost model.

        Parameters:
        - input_data: The input data for making predictions (numpy array, pandas DataFrame, list, etc.)

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)
        
    def save(self, file_name):
        """
        Save the model to the 'data' directory.
        Parameters:
        - file_name: The file name to save the model.
        """
        file_path = self.data_dir / file_name
        self.model.save_model(str(file_path))
        log_deployment_event(f"Model saved to {file_path}")
        
    def train(self, data_path, params):
        """
        Train the CatBoost model using the provided data and parameters.

        Parameters:
        - data_path: Path to the training dataset (in a format like CSV or already preprocessed).
        - params: A dictionary of CatBoost training parameters (e.g., iterations, depth, learning rate).

        Returns:
        - A string confirming the model has been trained.
        """
        try:
            # Step 1: Load the training data
            train_data = self.load_data(data_path)

            # Step 2: Set default parameters if not provided
            default_params = {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.1,
                'loss_function': 'Logloss',  # Example for classification
                'verbose': True
            }
            # Update default parameters with any overrides from `params`
            training_params = {**default_params, **params}

            # Step 3: Train the model
            self.model = CatBoostClassifier(**training_params)
            self.model.fit(train_data)

            # Step 4: Save the trained model 
            file_path = self.data_dir / "trained_catboost_model.cbm"
            self.model.save_model(file_path)
            
            # Log success
            log_deployment_event(f"Model trained and saved to {file_path}")

            return "Training completed successfully."
        except Exception as e:
            # Log the error
            log_deployment_event(f"Error during training: {str(e)}", log_level='error')
            raise e