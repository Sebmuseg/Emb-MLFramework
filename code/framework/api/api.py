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
from sklearn.metrics import accuracy_score, mean_squared_error



class FrameworkAPI:
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
        print(f"Model {model_name} successfully loaded.")

    def remove_model(self, model_name):
        """
        API call to remove a model from the framework.
        """
        self.framework.remove_model(model_name)
        print(f"Model {model_name} removed from framework.")

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
        print(f"Model {model_name} saved to {file_path}.")
        
        
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
        print(f"Model {model_name} quantized.")

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
        print(f"Model {model_name} pruned by {amount*100}%.")
        
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
        print(f"Model {model_name} compressed and saved to {compressed_file_path}")
    
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
        print(f"Model {model_name} converted to ONNX and saved to {onnx_file_path}")

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

        
    def evaluate_model(self, model_name, X_test, y_test):
        # Evaluates the model's performance on a test set
        model = self.models.get(model_name)
        if model:
            predictions = model.predict(X_test)
            if hasattr(model, "predict_proba"):  # If it's a classifier
                accuracy = accuracy_score(y_test, predictions)
                print(f"Accuracy for model {model_name}: {accuracy:.2f}")
                return accuracy
            else:  # Assume it's a regressor if no probability prediction available
                mse = mean_squared_error(y_test, predictions)
                print(f"Mean Squared Error for model {model_name}: {mse:.2f}")
                return mse
        else:
            raise ValueError(f"Model {model_name} not found!")
        