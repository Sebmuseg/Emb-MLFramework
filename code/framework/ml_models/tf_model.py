# ml_models/tf_model.py
import tensorflow as tf

class TensorFlowModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def predict(self, input_data):
        #Predicts with loaded model
        return self.model.predict(input_data)