# ml_models/prune.py
import torch
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import prune_low_magnitude, pruning_schedule

def prune_model(model, framework_type, pruning_params=None):
    """
    Apply model pruning to reduce the size and complexity of the model.
    
    Parameters:
    - model: The model to be pruned.
    - framework_type: The type of model framework ('pytorch', 'tensorflow').
    - pruning_params: Optional parameters for pruning (e.g., sparsity level).
    
    Returns:
    - pruned_model: The pruned model.
    """
    if framework_type == 'pytorch':
        return prune_pytorch_model(model, pruning_params)
    
    elif framework_type == 'tensorflow':
        return prune_tensorflow_model(model, pruning_params)
    
    else:
        raise ValueError(f"Pruning not supported for the {framework_type} framework.")

def prune_pytorch_model(model, pruning_params=None):
    """
    Prune the PyTorch model using torch.nn.utils.prune.

    Parameters:
    - model: The PyTorch model to be pruned.
    - pruning_params: Parameters to control the level of pruning (e.g., sparsity).
    
    Returns:
    - pruned_model: The pruned PyTorch model.
    """
    # Set default pruning parameters if none are provided
    if pruning_params is None:
        pruning_params = {'amount': 0.5}  # Default to 50% pruning

    import torch.nn.utils.prune as prune

    # Apply global unstructured pruning (you can use other techniques if necessary)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_params['amount'])

    return model  # Return the pruned model

def prune_tensorflow_model(model, pruning_params=None):
    """
    Prune the TensorFlow model using TensorFlow Model Optimization Toolkit.

    Parameters:
    - model: The TensorFlow/Keras model to be pruned.
    - pruning_params: Parameters to control the level of pruning (e.g., sparsity).
    
    Returns:
    - pruned_model: The pruned TensorFlow model.
    """
    # Set default pruning parameters if none are provided
    if pruning_params is None:
        pruning_params = {
            'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0)  # 50% sparsity
        }

    # Apply pruning
    pruned_model = prune_low_magnitude(model, pruning_schedule=pruning_params['pruning_schedule'])

    # Compile the pruned model
    pruned_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)

    return pruned_model