# ml_models/onnx_model.py
import onnx
import onnxruntime as rt
from pathlib import Path
from utils.logging_utils import log_deployment_event

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
        
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
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

    def save(self, file_name):
        """
        Save the ONNX model to disk.

        Parameters:
        - file_name: The file name to save the ONNX model.
        
        Returns:
        - A dictionary with the status and the path of the saved model.
        """
        file_path = self.data_dir / file_name.with_suffix('.onnx')
        try:
            onnx.save(self.model, file_path)
            log_deployment_event(f"ONNX model saved to {file_path}")
            return {"status": "success", "model_path": str(file_path)}
        except Exception as e:
            log_deployment_event(f"Error saving ONNX model: {str(e)}", log_level="error")
            return {"status": "error", "message": f"Error saving model: {str(e)}"}
        