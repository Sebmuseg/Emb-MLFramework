# utils/model_utils.py
def get_model_extension(model_format):
    """
    Returns the file extension for the given model format.
    
    Parameters:
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow', 'tflite', 'catboost', 'lightgbm').
    
    Returns:
    - A string representing the file extension (e.g., '.onnx', '.pb', '.cbm').
    """
    extensions = {
        'onnx': '.onnx',                # ONNX models
        'tensorflow': '.pb',            # TensorFlow models
        'tflite': '.tflite',            # TensorFlow Lite models
        'pytorch': '.pth',              # PyTorch models
        'openvino': '.xml',             # OpenVINO uses .xml for structure and .bin for weights
        'catboost': '.cbm',             # CatBoost models use the .cbm extension
        'lightgbm': '.txt',             # LightGBM models are often stored as .txt files
        'xgboost': '.model',            # XGBoost models typically use the .model extension
        'sklearn': '.pkl'               # scikit-learn models are often saved as pickle files (.pkl)
    }
    return extensions.get(model_format, '')


def get_inference_service_name(model_format):
    """
    Returns the name of the inference service based on the model format.
    
    Parameters:
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow', 'tflite', 'catboost', 'lightgbm').
    
    Returns: The name of the inference service (e.g., 'onnx-inference-service').
    """
    services = {
        'onnx': 'onnx-inference-service',               # ONNX Runtime service
        'tensorflow': 'tensorflow-inference-service',   # TensorFlow Serving service
        'tflite': 'tflite-inference-service',           # TensorFlow Lite service
        'pytorch': 'pytorch-inference-service',         # PyTorch service
        'openvino': 'openvino-inference-service',       # OpenVINO Runtime service
        'catboost': 'catboost-inference-service',       # Custom service for CatBoost
        'lightgbm': 'lightgbm-inference-service',       # Custom service for LightGBM
        'xgboost': 'xgboost-inference-service',         # Custom service for XGBoost
        'sklearn': 'sklearn-inference-service'          # Custom service for scikit-learn
    }
    return services.get(model_format, 'unknown-inference-service')

