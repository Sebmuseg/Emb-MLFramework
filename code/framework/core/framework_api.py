from framework import MLFramework
from ml_models.catboost_model import CatBoostModel
from ml_models.lightgbm_model import LightGBMModel
from ml_models.onnx_model import ONNXModel
from ml_models.pytorch_model import PyTorchModel
from ml_models.sklearn_model import SklearnModel
from ml_models.tf_model import TensorFlowModel
from ml_models.tflite_model import TFLiteModel
from ml_models.xgboost_model import XGBoostModel
from utils.logging_utils import log_deployment_event

class FrameworkAPI:
    """
    FrameworkAPI serves as an interface to manage machine learning models, 
    load them into the framework, perform predictions, and handle deployment.
    """

    def __init__(self):
        """
        Initializes the FrameworkAPI by creating an instance of the MLFramework 
        which will store and manage models.
        """
        self.framework = MLFramework()

    def _get_model_class(self, framework_type, model_path=None):
        """
        Helper method to return the appropriate model class based on the framework type.

        Parameters:
        - framework_type: The type of the machine learning framework (e.g., 'tensorflow', 'sklearn', etc.).
        - model_path: The path to the model file that needs to be loaded.

        Returns: An instance of the corresponding model class (e.g., TensorFlowModel).

        Raises:
        - ValueError: If the framework type is not supported.
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
        Loads a machine learning model into the framework.

        Parameters:
        - model_name: The name to assign to the model in the framework.
        - model_path: The path to the model file that needs to be loaded.
        - framework_type: The type of the machine learning framework (e.g., 'tensorflow', 'sklearn', etc.).

        Logs a deployment event upon successful loading of the model.
        """
        model = self._get_model_class(framework_type, model_path)
        self.framework.models[model_name] = model
        log_deployment_event(f"Model {model_name} successfully loaded.")

    def remove_model(self, model_name):
        """
        Removes a model from the framework.

        Parameters:
        - model_name: The name of the model to be removed from the framework.

        Logs a deployment event upon successful removal of the model.

        Raises:
        - ValueError: If the model is not found in the framework.
        """
        self.framework.remove_model(model_name)
        log_deployment_event(f"Model {model_name} removed from framework.")

    def predict(self, model_name, input_data):
        """
        Executes a prediction using the specified model in the framework.

        Parameters:
        - model_name: The name of the model to use for prediction.
        - input_data: The data to make predictions on.

        Returns: The model's prediction output.

        Raises:
        - ValueError: If the model is not found in the framework.
        """
        return self.framework.predict(model_name, input_data)

    def list_models(self):
        """
        Lists all models currently loaded in the framework.

        Returns: A list of model names that are currently loaded.
        """
        return self.framework.list_models()

    def save_model(self, model_name, file_path):
        """
        Saves the specified model to disk.

        Parameters:
        - model_name: The name of the model to save.
        - file_path: The path where the model should be saved.

        Logs a deployment event upon successful saving of the model.

        Raises:
        - ValueError: If the model is not found in the framework.
        """
        model = self.framework.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found!")
        
        model.save(file_path)  # Directly call the save method of the model
        log_deployment_event(f"Model {model_name} saved to {file_path}.")
    
    