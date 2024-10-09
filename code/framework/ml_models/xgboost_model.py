# ml_models/xgboost_model.py
import xgboost as xgb

class XGBoostModel:
    def __init__(self, model_path):
        self.model = xgb.Booster(model_file=model_path)
    
    def predict(self, input_data):
        dmatrix = xgb.DMatrix(input_data)
        return self.model.predict(dmatrix)