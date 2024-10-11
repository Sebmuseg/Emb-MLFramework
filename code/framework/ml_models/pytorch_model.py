# ml_models/pytorch_model.py
import torch

class PyTorchModel:
    def __init__(self, model_path=None, model=None):
        """
        Initializes the PyTorchModel. Loads the model if a path is provided.
        
        Parameters:
        - model_path: Optional, path to a saved model file.
        - model: Optional, a pre-built PyTorch model (e.g., a custom model instance).
        """
        if model_path:
            self.model = torch.load(model_path)  # Load the model from disk
        elif model:
            self.model = model  # Use the provided PyTorch model
        else:
            raise ValueError("Either model_path or model must be provided.")
        
        self.model.eval()  # Set the model to evaluation mode by default

    def predict(self, input_data):
        """
        Predict using the PyTorch model.

        Parameters:
        - input_data: Torch tensor or numpy array to be passed to the model.

        Returns:
        - Model predictions as a torch tensor.
        """
        self.model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            return self.model(input_data)

    def save(self, file_path):
        """
        Saves the current model to disk.
        
        Parameters:
        - file_path: Path to save the model.
        """
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")
    