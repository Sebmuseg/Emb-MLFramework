# ml_models/onnx_model.py
import onnxruntime as rt

class ONNXModel:
    def __init__(self, model_path):
        """
        Load the ONNX model from the given path using the ONNX runtime.
        """
        self.session = rt.InferenceSession(model_path)  # Load the ONNX model into an inference session

    def predict(self, input_data):
        """
        Predict using the ONNX model.

        Parameters:
        - input_data: The input data as a numpy array or another compatible format.

        Returns:
        - Predictions from the model.
        """
        input_name = self.session.get_inputs()[0].name  # Get input tensor name
        return self.session.run(None, {input_name: input_data})[0]  # Run the model and return the predictions