# conversion/xgboost_to_onnx.py
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools

def convert_xgboost_model_to_onnx(model, input_shape, model_path):
    """Converts an XGBoost model to ONNX format."""
    
    # Define the input type, using FloatTensorType from skl2onnx
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    
    # Convert the XGBoost model to ONNX format
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    
    # Save the ONNX model to the specified path
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"XGBoost Model converted to ONNX and saved at {model_path}")