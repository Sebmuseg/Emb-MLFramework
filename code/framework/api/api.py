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
from optimizations.compression import reduce_features
from optimizations.pruning import apply_pruning
from optimizations.quantize import quantize_model
from conversion.onnx_conversion import convert_sklearn_model_to_onnx, convert_tf_model_to_onnx, convert_xgboost_model_to_onnx
import joblib  
from sklearn.metrics import accuracy_score, mean_squared_error



class FrameworkAPI:
    def __init__(self):
        self.framework = MLFramework()
    
    def load_model(self, model_name, model_path, framework_type):
            """
            Loads a model from disk based on the type of framework.
            
            Parameters:
            - model_name: The name you want to assign to the model in the framework.
            - model_path: The path to the model file.
            - framework_type: The framework type (e.g., 'tensorflow', 'sklearn', 'xgboost').
            """
            if framework_type == 'tensorflow':
                model = TensorFlowModel(model_path)
            elif framework_type == 'sklearn':
                model = SklearnModel(model_path)
            elif framework_type == 'xgboost':
                model = XGBoostModel(model_path)
            elif framework_type == 'lightgbm':
                model = LightGBMModel(model_path)
            elif framework_type == 'pytorch':
                model = PyTorchModel(model_path)
            elif framework_type == 'catboost':
                model = CatBoostModel(model_path)
            elif framework_type == 'onnx':
                model = ONNXModel(model_path)
            elif framework_type == 'tflite':
                model = TFLiteModel(model_path)
            else:
                raise ValueError(f"Unsupported framework: {framework_type}")
            
            # Store the loaded model in the framework
            self.framework.models[model_name] = model
            print(f"Model {model_name} successfully loaded.")
        
    def remove_model(self, model_name):
        #API call to remove a model using MLFramework
        self.framework.remove_model(model_name)

    def predict(self, model_name, input_data):
        #API call to predict using the MLFramework
        return self.framework.predict(model_name, input_data)

    def list_models(self):
        #API call to list all loaded models
        return self.framework.list_models()
    
    def save_model(self, model_name, file_path, framework_type):
        """
        Saves a model to disk based on the type of framework.
        
        Parameters:
        - model_name: The name of the model in the framework.
        - file_path: The output file path where the model will be saved.
        - framework_type: The framework type (e.g., 'tensorflow', 'sklearn', 'xgboost').
        """
        model = self.framework.models.get(model_name)
        
        if model:
            if framework_type == 'tensorflow':
                model.save(file_path)
            elif framework_type == 'sklearn':
                model.save(file_path)
            elif framework_type == 'xgboost':
                model.save(file_path)
            elif framework_type == 'lightgbm':
                model.save(file_path)
            elif framework_type == 'pytorch':
                model.save(file_path)
            elif framework_type == 'catboost':
                model.save(file_path)
            elif framework_type == 'onnx':
                model.save(file_path)
            elif framework_type == 'tflite':
                model.save(file_path)
            else:
                raise ValueError(f"Unsupported framework: {framework_type}")
        else:
            raise ValueError(f"Model {model_name} not found!")
        
        
    def quantize_loaded_model(self, model_name, output_dir):
        """Quantisiert ein bereits geladenes Modell und speichert es."""
        model = self.framework.models.get(model_name)
        if model:
            quantize_model(model.model_path, output_dir)
        else:
            raise ValueError(f"Model {model_name} not found!")
    
    def prune_loaded_model(self, model_name):
        """Prunt ein geladenes Modell."""
        model = self.framework.models.get(model_name)
        if model:
            pruned_model = apply_pruning(model.model)
            print(f"Model {model_name} has been pruned.")
        else:
            raise ValueError(f"Model {model_name} not found!")
        
    def compress_model(self, model_name, input_data):
        """Wendet Kompressionstechniken auf das geladene Modell an (Feature-Reduktion, Pruning)."""
        model = self.framework.models.get(model_name)
        if isinstance(model, SklearnModel):
            X_reduced = reduce_features(model, input_data)
            print(f"Features reduced for model {model_name}")
        elif isinstance(model, TensorFlowModel):
            pruned_model = apply_pruning(model.model)
            print(f"Model {model_name} pruned successfully.")
        else:
            raise ValueError(f"Unsupported model type for compression: {model_name}")
    
    def convert_model_to_onnx(self, model_name, input_shape, model_path):
        """
        Converts a loaded model to ONNX format based on the model type.
        
        Parameters:
        - model_name: The name of the model in the framework.
        - input_shape: Shape of the input (number of features).
        - model_path: Output path for saving the ONNX model.
        """
        model = self.framework.models.get(model_name)
        
        if isinstance(model, xgb.XGBModel):
            # XGBoost model
            convert_xgboost_model_to_onnx(model, input_shape, model_path)
        elif isinstance(model, (LogisticRegression, RandomForestClassifier)):
            # scikit-learn model (example with common classifiers)
            convert_sklearn_model_to_onnx(model, input_shape, model_path)
        elif isinstance(model, tf.keras.Model):
            # TensorFlow model (Keras-based)
            convert_tensorflow_model_to_onnx(model.saved_model_dir, model_path)
        else:
            raise ValueError(f"Model type for {model_name} is not supported for ONNX conversion.")



        
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
        