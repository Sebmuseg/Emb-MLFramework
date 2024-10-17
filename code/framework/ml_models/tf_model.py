# ml_models/tf_model.py
import tensorflow as tf
from pathlib import Path
from utils.logging_utils import log_deployment_event

class TensorFlowModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the TensorFlow model. Load the model from a file if a model_path is provided,
        otherwise use the provided TensorFlow model instance.

        Parameters:
        - model_path: Path to the saved TensorFlow model file (optional).
        - model: An existing TensorFlow model instance (optional).
        """
        if model_path:
            self.model = tf.keras.models.load_model(model_path)  # Load the model from the provided path
        elif model:
            self.model = model  # Use the provided TensorFlow model instance
        else:
            raise ValueError("Either model_path or model must be provided.")
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            
    def predict(self, input_data):
        """
        Predict using the TensorFlow model.

        Parameters:
        - input_data: Input data for the model (numpy array, pandas DataFrame, etc.).

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_name):
        """
        Save the TensorFlow model to disk.

        Parameters:
        - file_name:  The file name to save the model (in TensorFlow's SavedModel or HDF5 format).
        
        Returns:
        - A dictionary with the status and the path of the saved model.
        """
        file_path = self.data_dir / file_name.with_suffix('.h5')
        try:
            self.model.save(file_path)
            log_deployment_event(f"TensorFlow model saved to {file_path}")
            return {"status": "success", "model_path": str(file_path)}
        except Exception as e:
            log_deployment_event(f"Error saving TensorFlow model: {str(e)}", log_level="error")
            return {"status": "error", "message": f"Error saving model: {str(e)}"}
    
    def train(self, train_data, train_labels, epochs=10, batch_size=32, validation_data=None, model_save_path="trained_tensorflow_model.h5"):
        """
        Train the TensorFlow model using the provided training data.

        Parameters:
        - train_data: The data used to train the model (e.g., numpy arrays or a TensorFlow dataset).
        - train_labels: The labels corresponding to the training data.
        - epochs: Number of epochs for training.
        - batch_size: Size of each batch of data.
        - validation_data: Tuple (validation_data, validation_labels) for model validation (optional).
        - model_save_path: Path to save the trained model (default: 'trained_tensorflow_model.h5').

        Returns:
        - A dictionary containing the training history and status.
        """
        # Train the model
        try:
            history = self.model.fit(
                train_data,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data
            )

            # Save the trained model to the specified path
            model_save_path = self.data_dir / model_save_path
            self.model.save(model_save_path)

            # Log the event
            log_deployment_event(f"TensorFlow model successfully trained and saved to {model_save_path}")

            # Return the training history and model save path
            return {
                "status": "success",
                "model_path": str(model_save_path),
                "history": history.history
            }
        
        except Exception as e:
            log_deployment_event(f"Error during TensorFlow model training: {str(e)}", log_level="error")
            return {
                "status": "error",
                "message": f"Error during training: {str(e)}"
            }