# core/framework.py
from core.framework import MLFramework
from utils.logging_utils import log_deployment_event
import subprocess
import importlib
import logging
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FrameworkAPI:
    """
    FrameworkAPI serves as an interface to manage machine learning models, 
    load them into the framework, perform predictions, and handle deployment.
    """

    def __init__(self, metadata_dir: str = './model_metadata'):
        """
        Initializes the FrameworkAPI by creating an instance of the MLFramework 
        which will store and manage models.
        
        Attributes:
        - framework: An instance of MLFramework to manage machine learning models.
        - installed_packages: A set to keep track of installed packages.
        - install_lock: A threading lock to ensure thread-safe package installation.
        """
        self.framework = MLFramework(metadata_dir=metadata_dir)
        self.installed_packages = set()
        self.install_lock = threading.Lock()
        
    def _install_if_missing(self, package_name: str) -> None:
        """
        Installs the specified package if it is not already installed.

        Parameters:
        - package_name: The name of the package to check and install if missing.

        Raises:
        - RuntimeError: If the package could not be installed or imported.
        """
        with self.install_lock:
            if package_name in self.installed_packages:
                return  # Package already installed

            try:
                importlib.import_module(package_name)
                self.installed_packages.add(package_name)
                logger.info(f"Package '{package_name}' is already installed.")
            except ImportError:
                logger.info(f"Package '{package_name}' not found. Installing...")
                try:
                    subprocess.check_call(["pip", "install", package_name])
                    importlib.import_module(package_name)
                    self.installed_packages.add(package_name)
                    logger.info(f"Package '{package_name}' installed successfully.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install package '{package_name}': {e}")
                    raise RuntimeError(f"Could not install package '{package_name}'") from e
                except ImportError as e:
                    logger.error(f"Package '{package_name}' was installed but could not be imported: {e}")
                    raise RuntimeError(f"Could not import package '{package_name}' after installation") from e
                
    def _get_model_class(self, framework_type, model_path=None):
        """
        Helper method to return the appropriate model class based on the framework type.

        Parameters:
        - framework_type: The type of the machine learning framework (e.g., 'tensorflow', 'sklearn', etc.).
        - model_path: The path to the model file that needs to be loaded.

        Returns: An instance of the corresponding model class.

        Raises:
        - ValueError: If the framework type is not supported.
        - RuntimeError: If the package installation fails.
        """
        framework_mapping = {
            'tensorflow': ('tensorflow', 'ml_models.tf_model', 'TensorFlowModel'),
            'sklearn': ('scikit-learn', 'ml_models.sklearn_model', 'SklearnModel'),
            'xgboost': ('xgboost', 'ml_models.xgboost_model', 'XGBoostModel'),
            'lightgbm': ('lightgbm', 'ml_models.lightgbm_model', 'LightGBMModel'),
            'pytorch': ('torch', 'ml_models.pytorch_model', 'PyTorchModel'),
            'catboost': ('catboost', 'ml_models.catboost_model', 'CatBoostModel'),
            'onnx': ('onnx', 'ml_models.onnx_model', 'ONNXModel'),
            'tflite': ('tflite', 'ml_models.tflite_model', 'TFLiteModel'),
        }

        if framework_type not in framework_mapping:
            raise ValueError(f"Unsupported framework: {framework_type}")

        package_name, module_path, class_name = framework_mapping[framework_type]
        self._install_if_missing(package_name)

        try:
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
            return ModelClass(model_path)
        except ImportError as e:
            logger.error(f"Could not import module '{module_path}': {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class '{class_name}' not found in module '{module_path}': {e}")
            raise

    def load_model(self, model_name: str, model_path: str, framework_type: str) -> bool:
        """
        Loads a machine learning model into the framework.

        Parameters:
        - model_name (str): The name to assign to the model in the framework.
        - model_path (str): The path to the model file that needs to be loaded.
        - framework_type (str): The type of the machine learning framework (e.g., 'tensorflow', 'sklearn', etc.).

        Logs a deployment event upon successful loading of the model.

        Returns:
        - bool: True if the model is loaded successfully, False otherwise.

        Raises:
        - ValueError: If the framework type is not supported.
        """
        try:
            model = self._get_model_class(framework_type, model_path)
            self.framework.models[model_name] = model
            log_deployment_event(f"Model {model_name} successfully loaded.")
            return True
        except ValueError as ve:
            log_deployment_event(f"Failed to load model {model_name}: {str(ve)}",log_level="error")
            raise
        except Exception as e:
            log_deployment_event(f"Unexpected error while loading model {model_name}: {str(e)}", log_level="error")
            return False
        
    def register_model(self, model_name: str, metadata: dict):
        """
        Registers a model with the given name and metadata.

        This method saves the provided metadata for the specified model name
        using the framework's save_metadata method. If the operation is successful,
        it logs an informational message and returns True. If an error occurs,
        it logs the exception and returns False.

        Args:
            model_name (str): The name of the model to register.
            metadata (dict): A dictionary containing metadata for the model.

        Returns:
            bool: True if the model was successfully registered, False otherwise.
        """
        try:
            self.framework.save_metadata(model_name, metadata)
            logger.info(f"Model '{model_name}' registered with metadata.")
            return True
        except Exception as e:
            logger.exception(f"Error registering model '{model_name}': {e}")
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
    
    