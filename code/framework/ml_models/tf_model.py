# ml_models/tf_model.py
import tensorflow as tf

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
    
    def predict(self, input_data):
        """
        Predict using the TensorFlow model.

        Parameters:
        - input_data: Input data for the model (numpy array, pandas DataFrame, etc.).

        Returns:
        - Predictions from the model.
        """
        return self.model.predict(input_data)

    def save(self, file_path):
        """
        Save the TensorFlow model to disk.

        Parameters:
        - file_path: Path to save the model (in TensorFlow's SavedModel or HDF5 format).
        """
        self.model.save(file_path)
        print(f"TensorFlow model saved to {file_path}")