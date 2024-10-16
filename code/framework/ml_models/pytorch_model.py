# ml_models/pytorch_model.py
import torch
from pathlib import Path
from utils.logging_utils import log_deployment_event


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
        
        # Define the path to the `data` directory relative to this file's location
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

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

    def save(self, file_name):
        """
        Saves the current model to disk.
        
        Parameters:
        - file_name: The file name to save the model.
        """
        file_path = self.data_dir / file_name
        torch.save(self.model.state_dict(), file_path)
        log_deployment_event(f"Model saved to {file_path}")
        
    def train(self, train_data, model_params, num_epochs=10, batch_size=32, learning_rate=0.001):
        """
        Train the PyTorch model using the provided training data and model parameters.

        Parameters:
        - train_data: Training data (a PyTorch `Dataset` or raw input).
        - model_params: Dictionary of parameters for model training.
        - num_epochs: Number of training epochs (default: 10).
        - batch_size: Batch size for training (default: 32).
        - learning_rate: Learning rate for the optimizer (default: 0.001).

        Returns:
        - Trained PyTorch model.
        """
        try:
            # Create DataLoader for the training data
            if isinstance(train_data, Dataset):
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            else:
                raise ValueError("train_data must be a PyTorch Dataset.")

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # Log the training start
            self.logger.info(f"Starting PyTorch model training for {num_epochs} epochs with batch size {batch_size}.")

            # Training loop
            for epoch in range(num_epochs):
                running_loss = 0.0
                for inputs, labels in train_loader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Accumulate the running loss
                    running_loss += loss.item()

                # Log the average loss for the epoch
                avg_loss = running_loss / len(train_loader)
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Save the trained model
            model_file_path = self.data_dir / "trained_pytorch_model.pth"
            torch.save(self.model.state_dict(), model_file_path)
            self.logger.info(f"PyTorch model saved to {model_file_path}")

            return self.model

        except Exception as e:
            # Log the error and raise it
            self.logger.error(f"Error during PyTorch model training: {str(e)}")
            raise e
    