# conversion/onnx_conversion.py
import torch
import tf2onnx
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import lightgbm as lgb
import tensorflow as tf
import catboost as cb

def convert_model_to_onnx(model, onnx_file_path):
    """
    Convert a model to ONNX format based on its framework type and save it to the specified path.
    
    Parameters:
    - model: The model instance to convert.
    - onnx_file_path: Path where the ONNX model will be saved.
    """
    if isinstance(model, torch.nn.Module):
        # PyTorch to ONNX
        dummy_input = torch.randn(1, *model.input_shape)  # Example input shape
        torch.onnx.export(model, dummy_input, onnx_file_path)
    
    elif 'tensorflow' in str(type(model)):
        # TensorFlow to ONNX
        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
        with open(onnx_file_path, "wb") as f:
            f.write(model_proto.SerializeToString())
    
    elif 'sklearn' in str(type(model)):
        # Scikit-learn to ONNX
        initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    elif isinstance(model, lgb.Booster):
        # LightGBM to ONNX
        initial_type = [('float_input', FloatTensorType([None, model.num_feature()]))]
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    elif 'xgboost' in str(type(model)):
        # XGBoost to ONNX
        initial_type = [('float_input', FloatTensorType([None, model.feature_names.shape[1]]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    
    elif isinstance(model, cb.CatBoost):
        # CatBoost to ONNX
        initial_type = [('float_input', FloatTensorType([None, model.feature_names_.shape[1]]))]
        onnx_model = onnxmltools.convert_catboost(model, initial_types=initial_type)
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"CatBoost model converted to ONNX and saved at {onnx_file_path}")
    
    else:
        raise ValueError("Unsupported model type for ONNX conversion.")