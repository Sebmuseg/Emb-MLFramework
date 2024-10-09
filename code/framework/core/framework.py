# core/framework.py
class MLFramework:
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_name, model):
        #Load a ML-Model in the Framework.
        self.models[model_name] = model
        print(f"Model {model_name} successfully loaded.")
        
    def remove_model(self, model_name):
        #Removes a model from the framework
        if model_name in self.models:
            del self.models[model_name]
            print(f"Model {model_name} has been removed.")
        else:
            raise ValueError(f"Model {model_name} not found!")
    
    def predict(self, model_name, input_data):
        #Execute a prediction with the loaded model
        model = self.models.get(model_name)
        if model:
            return model.predict(input_data)
        else:
            raise ValueError(f"Model {model_name} not found!")

    def list_models(self):
        #Lists all the loaded models.
        return list(self.models.keys())

    