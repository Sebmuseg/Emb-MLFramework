
from ml_models.tf_model import TensorFlowModel
from ml_models.sklearn_model import SklearnModel
from ml_models.xgboost_model import XGBoostModel
from ml_models.lightgbm_model import LightGBMModel
from ml_models.pytorch_model import PyTorchModel
from ml_models.catboost_model import CatBoostModel
from ml_models.onnx_model import ONNXModel
from ml_models.tflite_model import TFLiteModel

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
            raise ValueError(f"Unsupported framework type: {framework_type}")
