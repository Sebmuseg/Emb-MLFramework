# ml_models/quantize.py
import torch
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate_layer, quantize_apply

def quantize_model(model, framework_type):
    """
    Apply model quantization to reduce model size and improve inference efficiency.
    
    Parameters:
    - model: The model to be quantized.
    - framework_type: The type of model framework ('pytorch', 'tensorflow').
    
    Returns:
    - quantized_model: The quantized model.
    """
    if framework_type == 'pytorch':
        return quantize_pytorch_model(model)
    
    elif framework_type == 'tensorflow':
        return quantize_tensorflow_model(model)
    
    else:
        raise ValueError(f"Quantization not supported for the {framework_type} framework.")

def quantize_pytorch_model(model):
    """
    Quantize a PyTorch model using dynamic quantization.
    
    Parameters:
    - model: The PyTorch model to be quantized.
    
    Returns:
    - quantized_model: The quantized PyTorch model.
    """
    # Apply dynamic quantization (default)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def quantize_tensorflow_model(model):
    """
    Quantize a TensorFlow model using TensorFlow Model Optimization Toolkit.
    
    Returns:
    - quantized_model: The quantized TensorFlow model.
    """
    # Annotate the layers to be quantized
    annotated_model = tf.keras.Sequential([
        quantize_annotate_layer(tf.keras.layers.Dense(10, activation='relu', input_shape=(100,))),
        quantize_annotate_layer(tf.keras.layers.Dense(2, activation='sigmoid'))
    ])

    # Apply quantization to the annotated model
    quantized_model = quantize_apply(annotated_model)

    return quantized_model