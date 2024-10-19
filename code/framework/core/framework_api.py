# core/framework.py
from core.framework import MLFramework
from utils.logging_utils import log_deployment_event
import subprocess
import importlib

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
        
    def _install_if_missing(self, package_name):
        try:
            importlib.import_module(package_name)
        except ImportError:
            subprocess.check_call(["pip", "install", package_name])

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
            self._install_if_missing('tensorflow')
            TensorFlowModel = importlib.import_module('ml_models.tf_model').TensorFlowMode
            return TensorFlowModel(model_path)
        elif framework_type == 'sklearn':
            self._install_if_missing('scikit-learn')
            SklearnModel = importlib.import_module('ml_models.sklearn_model').SklearnModel
            return SklearnModel(model_path)
        elif framework_type == 'xgboost':
            self._install_if_missing('xgboost')
            XGBoostModel = importlib.import_module('ml_models.xgboost_model').XGBoostModel
            return XGBoostModel(model_path)
        elif framework_type == 'lightgbm':
            self._install_if_missing('lightgbm')
            LightGBMModel = importlib.import_module('ml_models.lightgbm_model').LightGBMModel
            return LightGBMModel(model_path)
        elif framework_type == 'pytorch':
            self._install_if_missing('torch')
            PyTorchModel = importlib.import_module('ml_models.pytorch_model').PyTorchModel
            return PyTorchModel(model_path)
        elif framework_type == 'catboost':
            self._install_if_missing('catboost')
            CatBoostModel = importlib.import_module('ml_models.catboost_model').CatBoostModel
            return CatBoostModel(model_path)
        elif framework_type == 'onnx':
            self._install_if_missing('onnx')
            ONNXModel = importlib.import_module('ml_models.onnx_model').ONNXModel
            return ONNXModel(model_path)
        elif framework_type == 'tflite':
            self._install_if_missing('tflite')
            TFLiteModel = importlib.import_module('ml_models.tflite_model').TFLiteModel
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

        Returns: True if the model is loaded successfully, False otherwise.

        Raises:
        - ValueError: If the framework type is not supported.
        """
        try:
            model = self._get_model_class(framework_type, model_path)
            self.framework.models[model_name] = model
            log_deployment_event(f"Model {model_name} successfully loaded.")
            return True
        except ValueError as ve:
            log_deployment_event(f"Failed to load model {model_name}: {str(ve)}")
            raise
        except Exception as e:
            log_deployment_event(f"Unexpected error while loading model {model_name}: {str(e)}")
            return False

    def remove_model(self, model_name):
        """
        Removes a model from the framework.

        Parameters:
        - model_name: The name of the model to be removed from the framework.

        Logs a deployment event upon successful removal of the model.

        Raises:
        - ValueError: If the model is not found in the framework.
        """
        try:
            if model_name not in self.framework.models:
                raise ValueError(f"Model {model_name} not found!")
            
            self.framework.remove_model(model_name)
            log_deployment_event(f"Model {model_name} removed from framework.")
            return True
        except Exception as e:
            log_deployment_event(f"Failed to remove model {model_name}: {str(e)}")
            return False

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
        try:
            if model_name not in self.framework.models:
                raise ValueError(f"Model {model_name} not found!")
            
            prediction = self.framework.predict(model_name, input_data)
            return prediction
        except Exception as e:
            log_deployment_event(f"Failed to make prediction with model {model_name}: {str(e)}")
            raise

    def list_models(self):
        """
        Lists all models currently loaded in the framework.

        Returns: A list of model names that are currently loaded.
        """
        try:
            model_list = self.framework.list_models()
            log_deployment_event("Successfully listed all models.")
            return model_list
        except Exception as e:
            log_deployment_event(f"Failed to list models: {str(e)}")
            return []

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
        try:
            model = self.framework.models.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found!")
            
            model.save(file_path)  # Directly call the save method of the model
            log_deployment_event(f"Model {model_name} saved to {file_path}.")
            return True
        except ValueError as ve:
            log_deployment_event(f"Failed to save model {model_name}: {str(ve)}")
            raise
        except Exception as e:
            log_deployment_event(f"Unexpected error while saving model {model_name}: {str(e)}")
            return False
    
    