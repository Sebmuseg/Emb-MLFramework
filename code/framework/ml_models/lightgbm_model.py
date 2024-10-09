# ml_models/lightgbm_model.py
import lightgbm as lgb

class LightGBMModel:
    def __init__(self, model_path):
        self.model = lgb.Booster(model_file=model_path)
    
    def predict(self, input_data):
        return self.model.predict(input_data)