# ml_models/pytorch_model.py
import torch
import torch.optim as optim
import torch.nn as nn
import torch.jit

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
    
    def train(self, train_loader, epochs, loss_function, optimizer, device='cpu'):
        """
        Trains or fine-tunes the model on the given data.

        Parameters:
        - train_loader: DataLoader providing the training data.
        - epochs: Number of epochs to train for.
        - loss_function: Loss function to use for training (e.g., nn.CrossEntropyLoss()).
        - optimizer: Optimizer to use for training (e.g., optim.Adam(self.model.parameters())).
        - device: The device ('cpu' or 'cuda') on which to train the model.
        """
        self.model.train()  # Set model to training mode
        self.model.to(device)  # Move model to the correct device
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
    
    def evaluate(self, test_loader, loss_function, device='cpu'):
        """
        Evaluates the model on a test/validation set.

        Parameters:
        - test_loader: DataLoader providing the test/validation data.
        - loss_function: Loss function to compute performance (e.g., nn.CrossEntropyLoss()).
        - device: The device ('cpu' or 'cuda') on which to evaluate the model.

        Returns:
        - Average loss and accuracy over the test set.
        """
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculations for evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        print(f"Test Loss: {avg_loss}, Test Accuracy: {accuracy}%")
        return avg_loss, accuracy

    def update_parameters(self, new_parameters):
        """
        Update the model's parameters with a given state_dict (for fine-tuning).

        Parameters:
        - new_parameters: A dictionary containing the updated model parameters (state_dict).
        """
        self.model.load_state_dict(new_parameters)
        print("Model parameters updated.")

    def optimize_model(self, file_path):
        """
        Optimize the model for inference using TorchScript and save it to disk.
        
        Parameters:
        - file_path: Path to save the optimized model.
        """
        scripted_model = torch.jit.script(self.model)  # Convert to TorchScript
        scripted_model.save(file_path)
        print(f"Optimized TorchScript model saved to {file_path}")
    
    def get_model_info(self):
        """
        Returns basic information about the model, such as number of layers and parameters.
        
        Returns:
        - Dictionary containing model info (e.g., number of parameters).
        """
        num_parameters = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model.__class__.__name__,
            "num_parameters": num_parameters
        }