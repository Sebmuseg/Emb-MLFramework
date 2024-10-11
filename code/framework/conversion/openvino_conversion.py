# conversion/openvino_conversion.py
import subprocess

def convert_to_openvino(model_path, output_dir, framework, input_shape=None):
    """
    Converts a given model to OpenVINO's IR format using the OpenVINO Model Optimizer.

    Parameters:
    - model_path: The path to the model (TensorFlow .pb, ONNX .onnx, etc.).
    - output_dir: The directory where the converted IR files will be saved.
    - framework: The framework of the model ('tensorflow', 'onnx', 'pytorch', etc.).
    - input_shape: Optional input shape for the model. Required for some models (e.g., TensorFlow, PyTorch).

    Returns:
    - output_dir: Directory containing the IR (.xml and .bin) files.
    """
    # Check framework and set appropriate command flags for Model Optimizer
    mo_command = [
        "mo",  # Model Optimizer CLI tool (should be installed with OpenVINO)
        "--input_model", model_path,
        "--output_dir", output_dir,
    ]

    if framework == 'tensorflow':
        mo_command += ["--framework", "tf"]
        if input_shape:
            mo_command += ["--input_shape", input_shape]
    
    elif framework == 'onnx':
        mo_command += ["--framework", "onnx"]
    
    elif framework == 'pytorch':
        mo_command += ["--framework", "pytorch"]
        # For PyTorch models, you typically need to export them to ONNX first, then use ONNX conversion.
        raise ValueError("PyTorch models should first be converted to ONNX.")

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Run the Model Optimizer command
    try:
        print(f"Converting {framework} model to OpenVINO IR format...")
        result = subprocess.run(mo_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print(f"Model successfully converted to OpenVINO IR and saved in {output_dir}.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during OpenVINO conversion: {e.stderr.decode('utf-8')}")
        raise

    return output_dir