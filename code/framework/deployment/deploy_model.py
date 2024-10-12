# deployment/deploy_model.py
import os
import subprocess
from utils.model_utils import get_model_extension, get_inference_service_name
from transfer_model import transfer_model_to_device
from backup_replace import backup_existing_model, replace_model_on_device
from service_management import restart_inference_service
from utils.logging_utils import log_deployment_event


def deploy_model(model_name, model_path, model_format, device_ip, device_username, deployment_path, is_docker=False, docker_template=None):
    """
    Orchestrates the deployment of a model to a device. Supports both regular models and Docker-based models.
    
    Parameters:
    - model_name: The name of the model to be deployed.
    - model_path: Path to the model file.
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow', 'tflite').
    - device_ip: The IP address of the target device.
    - device_username: Username for SSH login to the device.
    - deployment_path: The path on the device where the model will be deployed.
    - is_docker: Boolean indicating whether the model should be packaged as a Docker container.
    - docker_template: Path to the Dockerfile template if the model is packaged in Docker.

    Returns: True if deployment was successful, False otherwise.
    """
    # Step 1: Validate the device environment
    environment_ready = validate_device_environment(device_ip, device_username)
    if not environment_ready:
        log_deployment_event(f"Environment validation failed for device {device_ip}", log_level='error')
        return False

    # Step 2: Package the model into Docker if necessary
    if is_docker:
        package_model_in_docker(model_name, model_path, docker_template, deployment_path)
        return True  # Docker-based deployments are handled separately

    # Step 3: Transfer the model to the device
    model_extension = get_model_extension(model_format)
    temp_model_name = f"temp_{model_name}{model_extension}"
    transfer_success = transfer_model_to_device(model_path, device_ip, device_username, deployment_path, temp_model_name)
    if not transfer_success:
        log_deployment_event(f"Failed to transfer model {model_name} to device {device_ip}", log_level='error')
        return False

    # Step 4: Backup the existing model and replace it with the new one
    backup_success = backup_existing_model(device_ip, device_username, deployment_path, model_name, model_extension)
    if not backup_success:
        log_deployment_event(f"Failed to backup existing model {model_name} on device {device_ip}", log_level='error')
        return False

    replace_success = replace_model_on_device(device_ip, device_username, deployment_path, temp_model_name, model_name, model_extension)
    if not replace_success:
        log_deployment_event(f"Failed to replace model {model_name} on device {device_ip}", log_level='error')
        return False

    # Step 5: Restart the inference service to activate the new model
    service_name = get_inference_service_name(model_format)
    restart_success = restart_inference_service(device_ip, device_username, service_name)
    if not restart_success:
        log_deployment_event(f"Failed to restart inference service for model {model_name} on device {device_ip}", log_level='error')
        return False

    log_deployment_event(f"Successfully deployed model {model_name} to device {device_ip}")
    return True
    

def package_model_in_docker(model_name, model_path, model_format, output_dir="/app/output"):
    """
    Packages a model into a Docker container based on its format.
    
    Parameters:
    - model_name: The name of the model.
    - model_path: Path to the model file.
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow', 'pytorch', 'catboost', 'lightgbm', 'sklearn', 'tflite', 'xgboost').
    - output_dir: Directory inside the container where the built Docker image and files will be stored.
    
    Returns: None
    """
    # Get the current file's directory (to find templates)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the docker_templates directory relative to the current file
    docker_template_dir = os.path.join(current_dir, "..", "docker_templates")
    
    try:
        # Ensure the output directory exists (inside the container)
        os.makedirs(output_dir, exist_ok=True)
        
        # Select Dockerfile template based on model format
        dockerfile_template = os.path.join(docker_template_dir, f"Dockerfile_{model_format}")
        if not os.path.exists(dockerfile_template):
            raise ValueError(f"Dockerfile template for {model_format} not found.")
        
        # Copy the Dockerfile and the model to the output directory (inside the container)
        subprocess.run(["cp", dockerfile_template, os.path.join(output_dir, "Dockerfile")], check=True)
        subprocess.run(["cp", model_path, os.path.join(output_dir, os.path.basename(model_path))], check=True)

        # Build the Docker image (from inside the container)
        build_command = ["docker", "build", "-t", f"{model_name}_docker_image", output_dir]
        subprocess.run(build_command, check=True)

        log_deployment_event(f"Model {model_name} successfully packaged into a Docker container.", log_level="info")
    except subprocess.CalledProcessError as e:
        log_deployment_event(f"Error building Docker image: {e}", log_level="error")
    except ValueError as ve:
        log_deployment_event(str(ve), log_level="error")

def deploy_docker_container(device_ip, device_username, image_name):
    """
    Deploy the Docker container on the edge device.

    Parameters:
    - device_ip: IP address of the target device.
    - device_username: SSH login username for the device.
    - image_name: Name of the Docker image to run.

    Returns: True if successful, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        run_command = f"docker run -d --name {image_name}_container {image_name}"
        ssh.exec_command(run_command)
        log_deployment_event(f"Docker container {image_name} deployed successfully on {device_ip}.", log_level="info")

        ssh.close()
        return True
    except Exception as e:
        log_deployment_event(f"Failed to deploy Docker container on {device_ip}: {e}", log_level="error")
        return False
        

def validate_device_environment(device_ip, device_username):
    """
    Validates that the target device is ready to receive a model for deployment.
    
    Parameters:
    - device_ip: IP address of the target device.
    - device_username: Username for SSH login.
    
    Returns: True if the environment is ready, False otherwise.
    """
    # Logic to check device status, available storage, etc.
    pass

        
def update_model(model_name, model_path, model_format, device_ip, device_username, deployment_path):
    """
    Orchestrates the update of a model on the target device, handling backup and replacement separately.
    
    Parameters:
    - model_name: The name of the model to update.
    - model_path: Path to the new model file.
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow', 'tflite').
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the device.
    - deployment_path: The file path where the model will be stored on the device.
    
    Returns: True if the update was successful, False otherwise.
    """
    # Step 1: Validate the environment
    environment_ready = validate_device_environment(device_ip, device_username)
    if not environment_ready:
        log_deployment_event(f"Environment validation failed for device {device_ip}", log_level='error')
        return False

    # Step 2: Transfer the new model to the device
    model_extension = get_model_extension(model_format)
    temp_model_name = f"temp_{model_name}{model_extension}"
    transfer_success = transfer_model_to_device(model_path, device_ip, device_username, deployment_path, temp_model_name)
    if not transfer_success:
        log_deployment_event(f"Failed to transfer model {model_name} to device {device_ip}", log_level='error')
        return False

    # Step 3: Backup the existing model
    backup_success = backup_existing_model(device_ip, device_username, deployment_path, model_name, model_extension)
    if not backup_success:
        log_deployment_event(f"Failed to backup model {model_name} on device {device_ip}", log_level='error')
        return False

    # Step 4: Replace the existing model with the new one
    replace_success = replace_model_on_device(device_ip, device_username, deployment_path, temp_model_name, model_name, model_extension)
    if not replace_success:
        log_deployment_event(f"Failed to replace model {model_name} on device {device_ip}", log_level='error')
        return False

    # Step 5: Restart the inference service to activate the new model
    service_name = get_inference_service_name(model_format)
    restart_success = restart_inference_service(device_ip, device_username, service_name)
    if not restart_success:
        log_deployment_event(f"Failed to restart inference service for model {model_name} on device {device_ip}", log_level='error')
        return False

    log_deployment_event(f"Successfully updated model {model_name} on device {device_ip}")
    return True