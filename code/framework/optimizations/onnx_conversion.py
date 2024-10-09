# conversion/onnx_conversion.py
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def convert_sklearn_model_to_onnx(model, model_path, input_shape):
    """Konvertiert ein scikit-learn Modell in ONNX-Format."""
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Speichert das konvertierte Modell im ONNX-Format
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"scikit-learn Model converted to ONNX and saved to {model_path}")