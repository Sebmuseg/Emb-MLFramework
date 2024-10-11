# ml_models/onnx_model.py
import onnx
import onnxruntime as rt

class ONNXModel:
    def __init__(self, model_path=None, model=None):
        """
        Initialize the ONNX model. Load the model from a file if a model_path is provided,
        otherwise use the provided ONNX model instance.

        Parameters:
        - model_path: Path to the ONNX model file (optional).
        - model: An existing ONNX model session instance (optional).
        """
        if model_path:
            self.model = rt.InferenceSession(model_path)  # Load the ONNX model into an inference session
        elif model:
            self.model = model  # Use the provided ONNX model session
        else:
            raise ValueError("Either model_path or model must be provided.")
    
    def predict(self, input_data):
        """
        Predict using the ONNX model.

        Parameters:
        - input_data: The input data for making predictions (numpy array or compatible input).

        Returns:
        - Predictions from the ONNX model.
        """
        input_name = self.model.get_inputs()[0].name  # Get input tensor name
        return self.model.run(None, {input_name: input_data})[0]  # Run the model and return predictions

    def save(self, file_path):
        """
        Save the ONNX model to disk.

        Parameters:
        - file_path: The file path to save the ONNX model.
        """
        # ONNX models typically use the `onnx.save_model()` function
        onnx.save(self.model, file_path)
        print(f"ONNX model saved to {file_path}")