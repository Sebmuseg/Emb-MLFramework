# ml_models/pytorch_model.py
import torch

class PyTorchModel:
    def __init__(self, model_path):
        """
        Load the PyTorch model from the given path.
        """
        # Load the full model or state_dict based on how it was saved
        self.model = torch.load(model_path)
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, input_data):
        """
        Predict using the PyTorch model.

        Parameters:
        - input_data: Torch tensor or numpy array to be passed to the model.

        Returns:
        - Model predictions.
        """
        with torch.no_grad():  # Disable gradient calculations for inference
            return self.model(input_data)