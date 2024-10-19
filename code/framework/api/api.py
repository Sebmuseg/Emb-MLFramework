
from optimizations.compression import compress_model
from optimizations.prune import prune_model
from optimizations.quantize import quantize_model
from conversion.onnx_conversion import convert_model_to_onnx
from conversion.openvino_conversion import convert_to_openvino

from utils.logging_utils import log_deployment_event

class DeploymentAPI:
        
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
        

        