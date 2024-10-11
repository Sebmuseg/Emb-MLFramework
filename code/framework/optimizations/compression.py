# optimizations/compression.py
import torch
import tensorflow as tf
import gzip
import shutil
import onnx
from onnx import optimizer as onnx_optimizer
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def compress_model(model, framework_type, compressed_file_path):
    """
    Compress a model based on its framework type and save it to the specified path.

    Parameters:
    - model: The model to compress.
    - framework_type: The type of model framework ('pytorch', 'tensorflow', 'onnx', 'sklearn', 'xgboost', 'lightgbm', 'catboost').
    - compressed_file_path: Path to save the compressed model.
    """
    if framework_type == 'pytorch':
        # PyTorch: Use TorchScript to optimize and compress
        scripted_model = torch.jit.script(model)  # Script the model for optimization
        scripted_model.save(compressed_file_path)  # Save the optimized model
        
    elif framework_type == 'tensorflow':
        # TensorFlow: Save model with reduced precision (e.g., float16 or int8)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        compressed_model = converter.convert()
        with open(compressed_file_path, 'wb') as f:
            f.write(compressed_model)

    elif framework_type == 'onnx':
        # ONNX: Apply graph optimizations and save the compressed model
        passes = ["eliminate_deadend", "eliminate_identity", "eliminate_nop_transpose"]
        optimized_model = onnx_optimizer.optimize(model, passes)
        onnx.save(optimized_model, compressed_file_path)
    
    elif framework_type == 'sklearn':
        # Scikit-learn: Compress the serialized model using gzip
        with gzip.open(compressed_file_path, 'wb') as f:
            f.write(model.dumps())  # Serialize and compress

    elif framework_type == 'xgboost':
        # XGBoost: Compress the serialized model file
        model.save_model(compressed_file_path)
        with open(compressed_file_path, 'rb') as f_in, gzip.open(f"{compressed_file_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    elif framework_type == 'lightgbm':
        # LightGBM: Compress the serialized model file
        model.save_model(compressed_file_path)
        with open(compressed_file_path, 'rb') as f_in, gzip.open(f"{compressed_file_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    elif framework_type == 'catboost':
        # CatBoost: Compress the serialized model file
        model.save_model(compressed_file_path)
        with open(compressed_file_path, 'rb') as f_in, gzip.open(f"{compressed_file_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    else:
        raise ValueError(f"Unsupported framework for compression: {framework_type}")