# ml_models/tflite_model.py
import tensorflow as tf

class TFLiteModel:
    def __init__(self, model_path):
        """
        Load the TensorFlow Lite model from the given path.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)  # Load the TFLite model
        self.interpreter.allocate_tensors()  # Allocate memory for the model tensors

    def predict(self, input_data):
        """
        Predict using the TensorFlow Lite model.

        Parameters:
        - input_data: The input data formatted for TFLite.

        Returns:
        - Predictions from the model.
        """
        input_details = self.interpreter.get_input_details()  # Get input tensor details
        output_details = self.interpreter.get_output_details()  # Get output tensor details

        # Set the input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run the interpreter
        self.interpreter.invoke()

        # Get the output tensor
        return self.interpreter.get_tensor(output_details[0]['index'])