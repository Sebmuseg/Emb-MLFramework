# ml_models/tflite_model.py
import tensorflow as tf

class TFLiteModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the TensorFlow Lite model. Load the model from a file if a model_path is provided,
        otherwise use the provided TFLite interpreter instance.

        Parameters:
        - model_path: Path to the TFLite model file (optional).
        - model: An existing TensorFlow Lite Interpreter instance (optional).
        """
        if model_path:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)  # Load the TFLite model
            self.interpreter.allocate_tensors()  # Allocate memory for model tensors
        elif model:
            self.interpreter = model  # Use the provided TensorFlow Lite interpreter instance
        else:
            raise ValueError("Either model_path or model must be provided.")
    
    def predict(self, input_data):
        """
        Predict using the TensorFlow Lite model.

        Parameters:
        - input_data: Input data to be passed to the TFLite model.

        Returns:
        - The model's prediction.
        """
        # Get input and output details for the interpreter
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Set the input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        # Invoke the interpreter (run the model)
        self.interpreter.invoke()

        # Get the output tensor
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def save(self, file_path):
        """
        Save the TensorFlow Lite model to disk.

        Parameters:
        - file_path: Path to save the TFLite model (usually after converting from TensorFlow model).
        """
        # For TensorFlow Lite models, they are saved as FlatBuffers, typically after conversion
        with open(file_path, 'wb') as f:
            f.write(self.interpreter._get_model())  # Save the flatbuffer model
        print(f"TFLite model saved to {file_path}")