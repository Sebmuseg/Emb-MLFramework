# utils/model_utils.py

from fastapi import HTTPException
import os
from deployment.service_management import restart_inference_service
from deployment.transfer_model import transfer_model_to_device
from deployment.backup_replace import backup_existing_model, replace_model_on_device
from utils.logging_utils import log_deployment_event

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


def evaluate_model(model, eval_data, eval_labels):
    """
    Evaluate a model using the provided evaluation data.

    Parameters:
    - model: The model object to be evaluated. This can be any ML model class (e.g., TensorFlowModel, SklearnModel, XGBoostModel).
    - eval_data: The evaluation data (e.g., features as NumPy array or pandas DataFrame).
    - eval_labels: The true labels for the evaluation data (e.g., NumPy array or pandas DataFrame).

    Returns:
    - A dictionary containing evaluation metrics (accuracy, loss, etc.).
    """
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Call the model's evaluate method, which will handle framework-specific evaluation logic
        evaluation_result = model.evaluate(eval_data, eval_labels)
        return {"status": "success", "evaluation_result": evaluation_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model evaluation: {str(e)}")
    
def update_model_on_device(model_name: str, device_ip: str, deployment_path: str, model_format: str) -> bool:
    """
    Updates an already deployed model on the target device by transferring the new model,
    stopping the inference service, replacing the model, and restarting the service.

    Parameters:
    - model_name: The name of the model to update.
    - device_ip: The IP address of the target device.
    - deployment_path: The path where the model is deployed on the device.
    - model_format: The format of the model (e.g., ONNX, TensorFlow, etc.).

    Returns:
    - True if the update was successful, False otherwise.
    """
    try:
        # Step 1: Transfer the new model to the device
        model_file = f"{model_name}.{model_format.lower()}"
        transfer_success = transfer_model_to_device(device_ip, model_file, deployment_path)

        if not transfer_success:
            log_deployment_event(f"Failed to transfer model {model_name} to device {device_ip}.", log_level="error")
            return False

        # Step 2: Backup the existing model
        backup_success = backup_existing_model(device_ip, deployment_path, model_name, model_format)

        if not backup_success:
            log_deployment_event(f"Failed to backup the existing model {model_name} on device {device_ip}.", log_level="error")
            return False

        # Step 3: Replace the existing model with the new one
        replace_success = replace_model_on_device(device_ip, deployment_path, model_name, model_format)

        if not replace_success:
            log_deployment_event(f"Failed to replace the model {model_name} on device {device_ip}.", log_level="error")
            return False

        # Step 4: Restart the inference service to activate the new model
        service_name = f"{model_format.lower()}-inference-service"
        restart_success = restart_inference_service(device_ip, "username", service_name)  # Replace 'username' as appropriate

        if not restart_success:
            log_deployment_event(f"Failed to restart inference service {service_name} for model {model_name}.", log_level="error")
            return False

        # Log the success and return True
        log_deployment_event(f"Successfully updated model {model_name} on device {device_ip}.", log_level="info")
        return True

    except Exception as e:
        log_deployment_event(f"Error during model update: {str(e)}", log_level="error")
        return False
    
def rollback_model_on_device(device_ip: str, backup_path: str, deployment_path: str, model_format: str, username: str) -> bool:
    """
    Rolls back the current model on the device to a previous version.

    Parameters:
    - device_ip: The IP address of the target device.
    - backup_path: The path to the backup model file.
    - deployment_path: The current deployment path of the model.
    - model_format: The format of the model (e.g., ONNX, TensorFlow, etc.).
    - username: The SSH login username for the device.

    Returns:
    - True if the rollback was successful, False otherwise.
    """
    try:
        # Step 1: Check if backup model exists
        if not os.path.exists(backup_path):
            log_deployment_event(f"Backup model not found at {backup_path} for rollback on device {device_ip}.", log_level="error")
            return False
        
        # Step 2: Replace the current model with the backup model
        replace_success = replace_model_on_device(device_ip, deployment_path, os.path.basename(backup_path), model_format, is_backup=True)

        if not replace_success:
            log_deployment_event(f"Failed to replace the current model with backup on device {device_ip}.", log_level="error")
            return False

        # Step 3: Restart the inference service to load the rolled-back model
        service_name = f"{model_format.lower()}-inference-service"
        restart_success = restart_inference_service(device_ip, username, service_name)  

        if not restart_success:
            log_deployment_event(f"Failed to restart inference service {service_name} after rollback on device {device_ip}.", log_level="error")
            return False

        # Log the success and return True
        log_deployment_event(f"Model rollback successful on device {device_ip}.", log_level="info")
        return True

    except Exception as e:
        log_deployment_event(f"Error during model rollback on device {device_ip}: {str(e)}", log_level="error")
        return False
