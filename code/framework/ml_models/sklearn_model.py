# ml_models/sklearn_model.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
from utils.logging_utils import log_deployment_event


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
        
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, input_data):
        """
        Predict using the Scikit-learn model.

        Parameters:
        - input_data: Input data for the model (numpy array, pandas DataFrame, etc.).

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_name):
        """
        Save the Scikit-learn model to disk.

        Parameters:
        - file_name: The file name to save the model file.
        """
        file_path = self.data_dir / file_name.with_suffix('.pkl')
        try:
            joblib.dump(self.model, file_path)
            log_deployment_event(f"Scikit-learn model saved to {file_path}")
            return {"status": "success", "model_path": str(file_path)}
        except Exception as e:
            log_deployment_event(f"Error saving Scikit-learn model: {str(e)}", log_level="error")
            return {"status": "error", "message": f"Error saving model: {str(e)}"}
                
    def train(self, X, y, test_size=0.2, random_state=42, model_params=None):
        """
        Train the Scikit-learn model using the provided data and parameters.

        Parameters:
        - X: Features (input data).
        - y: Labels (target data).
        - test_size: Fraction of the data to be used as the test set (default: 0.2).
        - random_state: Random seed for reproducibility (default: 42).
        - model_params: Dictionary of parameters for model training (default: None).

        Returns:
        - Dictionary containing training accuracy, testing accuracy, and the path where the model is saved.
        """
        try:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # If model_params are provided, update the model parameters
            if model_params:
                self.model.set_params(**model_params)

            # Log training start
            log_deployment_event("Starting training for the Scikit-learn model.")

            # Train the model
            self.model.fit(X_train, y_train)

            # Log the completion of training
            log_deployment_event("Training completed for the Scikit-learn model.")

            # Evaluate the model
            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            test_acc = accuracy_score(y_test, self.model.predict(X_test))

            # Log the accuracies
            log_deployment_event(f"Training accuracy: {train_acc:.4f}")
            log_deployment_event(f"Testing accuracy: {test_acc:.4f}")

            # Save the trained model
            model_file_path = self.data_dir / "trained_sklearn_model.pkl"
            joblib.dump(self.model, model_file_path)
            log_deployment_event(f"Scikit-learn model saved to {model_file_path}")

            # Return the training and testing accuracies, and saved model path
            return {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "model_path": str(model_file_path)
            }

        except Exception as e:
            # Log the error and raise it
            log_deployment_event(f"Error during Scikit-learn model training: {str(e)}", log_level='error')
            return {"status": "error", "message": str(e)}
        