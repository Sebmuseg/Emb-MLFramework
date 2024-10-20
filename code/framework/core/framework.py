# core/framework.py
import os
import pickle
import logging
import threading
import json
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelNotFoundException(Exception):
    """Exception raised when a model is not found in the framework."""
    pass
class MLFramework:
    def __init__(self, metadata_dir: Optional[str] = None):
        self.models: Dict[str, Any] = {}
        self.metadata_dir = metadata_dir or './model_metadata'
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.lock = threading.Lock()
        logger.info(f"MLFramework initialized with metadata directory: {self.metadata_dir}")
    
    def load_model(self, model_name: str) -> None:
        """
        Load a machine learning model into the framework.

        Args:
            model_name (str): The name of the model to be loaded.

        Raises:
            FileNotFoundError: If the model metadata file is not found.
            KeyError: If the required keys are missing in the metadata.
        """
        with self.lock:
            if model_name in self.models:
                logger.info(f"Model '{model_name}' is already loaded.")
                return

            metadata = self.load_metadata(model_name)
            framework_type = metadata['framework_type']
            model_path = metadata['model_path']

            model_wrapper = self._create_model_wrapper(model_name, model_path, framework_type)
            self.models[model_name] = model_wrapper
            logger.info(f"Model '{model_name}' loaded using metadata.")
    
    def save_metadata(self, model_name: str, metadata: dict):
        """
        Save metadata for a given model to a JSON file.

        Args:
            model_name (str): The name of the model.
            metadata (dict): A dictionary containing the metadata to be saved.

        Returns:
            None

        Raises:
            OSError: If there is an issue writing to the file.

        Logs:
            Logs an info message indicating the metadata has been saved and the file path.
        """
        metadata_path = os.path.join(self.metadata_dir, f"{model_name}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata for model '{model_name}' saved to '{metadata_path}'.")
        
    def load_metadata(self, model_name: str) -> dict:
        """
        Load the metadata for a given model from a JSON file.

        Args:
            model_name (str): The name of the model whose metadata is to be loaded.

        Returns:
            dict: The metadata of the model.

        Raises:
            ModelNotFoundException: If the metadata file for the specified model is not found.

        Logs:
            - Info: When the metadata is successfully loaded.
            - Error: When the metadata file is not found.
        """
        metadata_path = os.path.join(self.metadata_dir, f"{model_name}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata for model '{model_name}' loaded from '{metadata_path}'.")
            return metadata
        else:
            logger.error(f"Metadata for model '{model_name}' not found.")
            raise ModelNotFoundException(f"Metadata for model '{model_name}' not found.")            
                    
    def save(self, model_name: str) -> None:
        """
        Save a ML model to disk.

        Args:
            model_name (str): The name of the model to be saved.

        Raises:
            ModelNotFoundException: If the model is not found in the framework.
        """
        if model_name in self.models:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            logger.info(f"Model {model_name} saved to {model_path}.")
        else:
            raise ModelNotFoundException(f"Model {model_name} not found!")
        
    def remove_model(self, model_name: str) -> None:
        """
        Remove a model from the framework.

        Args:
            model_name (str): The name of the model to be removed.

        Raises:
            ModelNotFoundException: If the model is not found in the framework.
        """
        with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                logger.info(f"Model '{model_name}' removed from memory.")

            metadata_path = os.path.join(self.metadata_dir, f"{model_name}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info(f"Metadata for model '{model_name}' deleted.")
            else:
                logger.warning(f"Metadata for model '{model_name}' not found.")
    
    def predict(self, model_name: str, input_data: Any) -> Any:
        """
        Execute a prediction with the loaded model.

        Args:
            model_name (str): The name of the model to use for prediction.
            input_data (Any): The input data for the prediction.

        Returns:
            Any: The prediction result from the model.

        Raises:
            ModelNotFoundException: If the model is not found in the framework.
        """
        model = self.models.get(model_name)
        if model:
            return model.predict(input_data)
        else:
            raise ModelNotFoundException(f"Model {model_name} not found!")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists all available models, including those currently loaded in memory 
        and those with metadata files in the specified directory.

        Returns:
            List[Dict[str, Any]]: A list of JSON objects representing the metadata of all available models.
        """
        with self.lock:
            # Models loaded in memory
            loaded_models = set(self.models.keys())

            # Models with metadata files
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]
            metadata_models = set(os.path.splitext(f)[0] for f in metadata_files)

            # Union of models in memory and models with metadata
            all_models = loaded_models.union(metadata_models)
            model_list = sorted(all_models)

            # Read and parse JSON metadata files
            json_list = []
            for model_name in model_list:
                metadata_file_path = os.path.join(self.metadata_dir, f"{model_name}.json")
                if os.path.exists(metadata_file_path):
                    with open(metadata_file_path, 'r') as file:
                        json_data = json.load(file)
                        json_list.append(json_data)
                else:
                    # If no metadata file exists, you can choose to append an empty dict or some default value
                    json_list.append({"model_name": model_name, "metadata": "No metadata available"})

            logger.info(f"Available models: {model_list}")
            return json_list

    