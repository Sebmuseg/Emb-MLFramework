# api/api.py
from framework.core import MLFramework
from ml_models.tf_model import TensorFlowModel
from ml_models.sklearn_model import SklearnModel
from ml_models.xgboost_model import XGBoostModel
from ml_models.lightgbm_model import LightGBMModel
from ml_models.pytorch_model import PyTorchModel
from ml_models.catboost_model import CatBoostModel
from ml_models.onnx_model import ONNXModel
from ml_models.tflite_model import TFLiteModel
from optimizations.compression import compress_model
from optimizations.prune import prune_model
from optimizations.quantize import quantize_model
from conversion.onnx_conversion import convert_model_to_onnx
from conversion.openvino_conversion import convert_to_openvino
from deployment.deploy_model import deploy_model, package_model_in_docker, update_model
from deployment.backup_replace import check_backup_exists, transfer_backup_model
from deployment.service_management import restart_inference_service
from utils.logging_utils import log_deployment_event





class FrameworkAPI:
    # -------------------------------
    #  Model Loading and Management
    # -------------------------------
    def __init__(self):
        self.framework = MLFramework()

    def _get_model_class(self, framework_type, model_path=None):
        """
        Helper method to get the correct model class based on the framework type.
        """
        if framework_type == 'tensorflow':
            return TensorFlowModel(model_path)
        elif framework_type == 'sklearn':
            return SklearnModel(model_path)
        elif framework_type == 'xgboost':
            return XGBoostModel(model_path)
        elif framework_type == 'lightgbm':
            return LightGBMModel(model_path)
        elif framework_type == 'pytorch':
            return PyTorchModel(model_path)
        elif framework_type == 'catboost':
            return CatBoostModel(model_path)
        elif framework_type == 'onnx':
            return ONNXModel(model_path)
        elif framework_type == 'tflite':
            return TFLiteModel(model_path)
        else:
            raise ValueError(f"Unsupported framework: {framework_type}")

    def load_model(self, model_name, model_path, framework_type):
        """
        Loads a model from disk based on the type of framework.
        """
        model = self._get_model_class(framework_type, model_path)
        self.framework.models[model_name] = model
        log_deployment_event(f"Model {model_name} successfully loaded.")

    def remove_model(self, model_name):
        """
        API call to remove a model from the framework.
        """
        self.framework.remove_model(model_name)
        log_deployment_event(f"Model {model_name} removed from framework.")

    def predict(self, model_name, input_data):
        """
        API call to predict using the loaded model in the framework.
        """
        return self.framework.predict(model_name, input_data)

    def list_models(self):
        """
        API call to list all loaded models in the framework.
        """
        return self.framework.list_models()

    def save_model(self, model_name, file_path):
        """
        Saves the model to disk.
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")
        
        model.save(file_path)  # Directly call the save method of the model
        log_deployment_event(f"Model {model_name} saved to {file_path}.")
        
    #-------------------------------
    #  Model Optimization Functions
    # -------------------------------
        
    def quantize_model(self, model_name):
        """
        Apply quantization to a loaded model to reduce size and make it edge-compatible.
        
        Parameters:
        - model_name: The name of the model to be quantized.
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")
        
        quantized_model = quantize_model(model)  # Call the quantize utility function
        self.framework.models[model_name] = quantized_model
        log_deployment_event(f"Model {model_name} quantized.")

    def prune_model(self, model_name, amount=0.3):
        """
        Apply pruning to reduce the size of a model by removing unimportant weights.
        
        Parameters:
        - model_name: The name of the model to prune.
        - amount: The percentage of weights to prune (default: 30%).
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")
        
        pruned_model = prune_model(model, amount)  # Call the prune utility function
        self.framework.models[model_name] = pruned_model
        log_deployment_event(f"Model {model_name} pruned by {amount*100}%.")
        
    def compress_model(self, model_name, compressed_file_path):
        """
        Apply model compression to reduce the size of the model.

        Parameters:
        - model_name: The name of the model to compress.
        - compressed_file_path: Path where the compressed model will be saved.
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")
        
        compress_model(model, compressed_file_path)
        log_deployment_event(f"Model {model_name} compressed and saved to {compressed_file_path}")
    
    
    #-------------------------------
    #  Model Conversion Functions
    # -------------------------------
    
    def convert_to_onnx(self, model_name, onnx_file_path):
        """
        Convert the loaded model to ONNX format.

        Parameters:
        - model_name: The name of the model to be converted.
        - onnx_file_path: The path where the ONNX model will be saved.
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")
        
        # Call a utility function to convert the model to ONNX
        convert_model_to_onnx(model, onnx_file_path)
        log_deployment_event(f"Model {model_name} converted to ONNX and saved to {onnx_file_path}")

    # This is the function to convert models to OpenVINO format. For now, it's commented out.
    '''
    def convert_to_openvino(self, model_name, framework_type, input_shape, output_dir):
        """
        Convert a loaded model to OpenVINO format using the OpenVINO Model Optimizer.

        Parameters:
        - model_name: The name of the model to be converted (as loaded in the framework).
        - framework_type: The type of model framework (e.g., 'tensorflow', 'onnx').
        - input_shape: The input shape for the model (required by some frameworks like TensorFlow).
        - output_dir: The directory where the OpenVINO IR files (.xml, .bin) will be saved.
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")

        # Retrieve the model path
        model_path = model.get_model_path()  # Assuming models store their paths

        # Call the OpenVINO conversion function
        convert_to_openvino(model_path, output_dir, framework_type, input_shape)
        print(f"Model {model_name} successfully converted to OpenVINO format.")
    '''
    
    # -------------------------------
    #  Model Deployment Functions
    # -------------------------------
    
    def deploy_model(self, model_name, device_ip, username, deployment_path, is_docker=False):
        """
        API method to deploy a model to a device.
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")

        model_path = model.get_model_path()  
        
        success = deploy_model(model_name, model_path, device_ip, username, deployment_path, is_docker)
        return success
    
    
    def deploy_model_in_docker(self, model_name, model_path, dockerfile_template, output_dir):
        """
        API method to package a model into a Docker container for deployment.

        Parameters:
        - model_name: The name of the model to package.
        - model_path: The path to the model file.
        - dockerfile_template: The path to the Dockerfile template.
        - output_dir: The directory to store the Docker image.

        Returns: True if packaging was successful, False otherwise.
        """
        # Input validation
        if not model_name or not model_path or not dockerfile_template or not output_dir:
            log_deployment_event(f"Invalid input parameters for Docker packaging of model {model_name}", log_level="error")
            return False

        try:
            # Call the function to package the model into Docker
            package_model_in_docker(model_name, model_path, dockerfile_template, output_dir)
            
            # Log successful packaging
            log_deployment_event(f"Model {model_name} successfully packaged into Docker container.", log_level="info")
            return True

        except Exception as e:
            # Log and handle any errors that occurred during packaging
            log_deployment_event(f"Error packaging model {model_name} into Docker container: {e}", log_level="error")
            return False
    
    def update_model_on_device(self, model_name, device_ip, username, deployment_path, model_format, backup_existing=True, restart_service=True):
        """
        API method to update an already deployed model on the target device with a new version.

        Parameters:
        - model_name: The name of the model to update.
        - device_ip: The IP address of the target device.
        - deployment_path: The file path where the model is stored on the device.
        - model_format: The format of the model (e.g., ONNX, TensorFlow Lite).
        - backup_existing: Boolean indicating whether to backup the existing model (default is True).
        - restart_service: Boolean indicating whether to restart the inference service after deployment (default is True).

        Returns:
        - True if the update was successful, False otherwise.
        """
        # Step 1: Check if the model exists in the framework
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")

        # Step 2: Get the model path from the model object (assuming this is implemented)
        model_path = model.get_model_path()  # e.g., "/path/to/model.onnx"

        # Step 3: Call the deployment logic for updating the model
        success = update_model(
            model_name=model_name,
            model_path=model_path,
            model_format=model_format,
            device_ip=device_ip,
            device_username=username,  
            deployment_path=deployment_path,
            backup_existing=backup_existing,
            restart_service=restart_service
        )

        return success
    
    def rollback_model_on_device(self, model_name, device_ip, user, backup_path, deployment_path, model_format):
        """
        Rolls back the current model on the device to a previous version.

        Parameters:
        - model_name: The name of the model to rollback.
        - device_ip: The IP address of the target device.
        - backup_path: The file path where the previous model version is stored.
        - deployment_path: The current deployment path of the model.
        - model_format: The format of the model (e.g., ONNX, OpenVINO, TensorFlow Lite).

        Returns:
        - True if rollback was successful, False otherwise.
        """
        # Step 1: Validate if backup exists
        backup_exists = check_backup_exists(device_ip, user, backup_path)
        if not backup_exists:
            log_deployment_event(f"Backup not found for {model_name} on device {device_ip}.", log_level="error")
            return False

        # Step 2: Transfer the backup model to the deployment path
        transfer_success = transfer_backup_model(device_ip, user, backup_path, deployment_path, model_format)
        if not transfer_success:
            log_deployment_event(f"Failed to transfer backup model {model_name} to device {device_ip}.", log_level="error")
            return False

        # Step 3: Restart the inference service if necessary
        restart_success = restart_inference_service(device_ip, "username", service_name)
        if not restart_success:
            log_deployment_event(f"Failed to restart service for {model_name} on device {device_ip}.", log_level="error")
            return False

        log_deployment_event(f"Successfully rolled back model {model_name} on device {device_ip}.", log_level="info")
        return True
    
    # -------------------------------
    #  Model Monitoring Functions
    # -------------------------------
    
    def monitor_device_resources(device_ip, resource_thresholds):
        """
        Monitors the resource usage (CPU, memory, GPU) of the device running the model.

        Parameters:
        - device_ip: The IP address of the target device.
        - resource_thresholds: A dictionary of resource usage limits (e.g., {'cpu': 80%, 'memory': 70%}).

        Steps:
        - Continuously monitor CPU, memory, and GPU usage.
        - Log the resource usage at regular intervals.
        - Trigger alerts if usage exceeds the defined thresholds.
        """
        pass
    
    def log_predictions(device_ip, log_path, model_name):
        """
        Logs all predictions made by the deployed model for auditing and monitoring.

        Parameters:
        - device_ip: The IP address of the target device.
        - log_path: The file path where predictions will be logged.
        - model_name: The name of the deployed model.

        Steps:  
        - Intercept model predictions on the device.
        - Log prediction inputs, outputs, and timestamps.
        """
        pass
        

        