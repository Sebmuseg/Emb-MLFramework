# deployment/deploy_model.py
import os
import subprocess
from utils.model_utils import get_model_extension, get_inference_service_name
from transfer_model import transfer_model_to_device, transfer_docker_image_to_device
from backup_replace import backup_existing_model, replace_model_on_device, docker_container_exists, backup_existing_docker_container, replace_docker_container
from service_management import restart_inference_service, stop_existing_container, run_docker_container
from utils.logging_utils import log_deployment_event
from utils.device_utils import check_if_model_exists, check_disk_space, check_docker_installed, check_ssh_connection


def deploy_model(model_name, model_path, model_format, device_ip, device_username, deployment_path, is_docker=False, docker_template='../framework/docker_templates/'):
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
    environment_ready = validate_device_environment(device_ip, device_username, is_docker)
    if not environment_ready:
        log_deployment_event(f"Environment validation failed for device {device_ip}", log_level='error')
        return False

    # Step 2: Package the model into Docker if necessary
    if is_docker:
        # Step 2.1: Package the model into Docker
        docker_image_tag = package_model_in_docker(model_name, model_path, docker_template, deployment_path)
        if not docker_image_tag:
            log_deployment_event(f"Failed to package model {model_name} into Docker.", log_level="error")
            return False  # Exit if packaging fails

        # Step 2.2: Deploy the Docker image to the target device
        deploy_container = deploy_docker_container(model_name, device_ip, device_username, docker_image_tag, deployment_path)
        if not deploy_container:
            log_deployment_event(f"Failed to deploy service for {model_name} on device {device_ip}.", log_level="error")
            return False

        return True  # Docker-based deployment completed successfully

    # Step 3: Transfer the model to the device
    model_extension = get_model_extension(model_format)
    temp_model_name = f"temp_{model_name}{model_extension}"
    transfer_success = transfer_model_to_device(model_path, device_ip, device_username, deployment_path, temp_model_name)
    if not transfer_success:
        log_deployment_event(f"Failed to transfer model {model_name} to device {device_ip}", log_level='error')
        return False

    # Step 4: Optionally backup the existing model (if it exists)
    existing_model_found = check_if_model_exists(device_ip, device_username, deployment_path, model_name, model_extension)

    if existing_model_found:
        # Backup the existing model if it exists
        backup_success = backup_existing_model(device_ip, device_username, deployment_path, model_name, model_extension)
        if not backup_success:
            log_deployment_event(f"Failed to backup existing model {model_name} on device {device_ip}", log_level='error')
            return False
    else:
        log_deployment_event(f"No existing model found on {device_ip}. Skipping backup.", log_level='info')


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
        
        # Define the Docker image tag (e.g., model_name:latest)
        docker_image_tag = f"{model_name}_docker_image:latest"

        # Build the Docker image (from inside the container)
        build_command = ["docker", "build", "-t", docker_image_tag, output_dir]
        subprocess.run(build_command, check=True)

        log_deployment_event(f"Model {model_name} successfully packaged into a Docker container.", log_level="info")
        
        return docker_image_tag
    
    except subprocess.CalledProcessError as e:
        log_deployment_event(f"Error building Docker image: {e}", log_level="error")
    except ValueError as ve:
        log_deployment_event(str(ve), log_level="error")

def deploy_docker_container(model_name, device_ip, device_username, docker_image_tag, deployment_path="/app/deployment"):
    """
    Deploys a Docker container to a target device.

    Parameters:
    - model_name: The name of the model to be deployed.
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the device.
    - docker_image_tag: The Docker image tag to deploy (e.g., 'model_docker_image').
    - deployment_path: The path on the target device where the Docker container will be deployed.
    
    Returns: True if the deployment was successful, False otherwise.
    """
    try:
        # Step 1: Check if a Docker container already exists
        container_exists = docker_container_exists(device_ip, device_username, docker_image_tag)
        
        if container_exists:
            # Step 2: Backup the existing Docker container if it exists
            backup_success = backup_existing_docker_container(device_ip, device_username, docker_image_tag)
            if not backup_success:
                log_deployment_event(f"Failed to backup existing Docker container {docker_image_tag} on device {device_ip}", log_level='error')
                return False

            # Step 3: Stop the existing Docker container
            stop_success = stop_existing_container(device_ip, device_username, docker_image_tag)
            if not stop_success:
                log_deployment_event(f"Failed to stop existing Docker container {docker_image_tag} on device {device_ip}", log_level='error')
                return False
        else:
            log_deployment_event(f"No existing Docker container found for {docker_image_tag} on device {device_ip}. Proceeding with deployment.", log_level="info")
        
        # Step 4: Transfer Docker image to the target device
        transfer_success = transfer_docker_image_to_device(device_ip, device_username, docker_image_tag, deployment_path)
        if not transfer_success:
            log_deployment_event(f"Failed to transfer Docker image {docker_image_tag} to device {device_ip}", log_level='error')
            return False

        # Step 5: Run the new Docker container on the target device
        run_success = run_docker_container(device_ip, device_username, docker_image_tag, model_name, deployment_path)
        if not run_success:
            log_deployment_event(f"Failed to run Docker container {docker_image_tag} on device {device_ip}", log_level='error')
            return False

        # Step 6: Optionally restart services or log success
        # Uncomment the following line if a service restart is necessary for orchestration or monitoring
        # restart_service_on_device(device_ip, device_username, f"docker-{model_name}-service")
        log_deployment_event(f"Successfully deployed Docker container {docker_image_tag} for model {model_name} on device {device_ip}.", log_level='info')
        return True

    except Exception as e:
        log_deployment_event(f"Deployment failed for {model_name} on device {device_ip}: {e}", log_level="error")
        return False
        

def validate_device_environment(device_ip, device_username, is_docker=False):
    """
    Validates that the target device is ready to receive a model for deployment.
    
    Parameters:
    - device_ip: IP address of the target device.
    - device_username: Username for SSH login.
    - is_docker: Boolean. If deployment runs as a container.
    
    Returns: True if the environment is ready, False otherwise.
    """
    try:
        # Step 1: Check SSH connection to the device
        if not check_ssh_connection(device_ip, device_username):
            return False

        # Step 2: Check for Docker environment if applicable
        if is_docker and not check_docker_installed(device_ip, device_username):
            return False

        # Step 3: Check if sufficient disk space is available
        if not check_disk_space(device_ip, device_username):
            return False

        log_deployment_event(f"Device {device_ip} environment validated successfully.", log_level="info")
        return True

    except Exception as e:
        log_deployment_event(f"Failed to validate device environment on {device_ip}: {e}", log_level="error")
        return False

        
def update_model(model_name, model_path, model_format, device_ip, device_username, deployment_path, backup_existing=True, restart_service=True):
    """
    Orchestrates the update of a model on the target device, handling backup and replacement separately.
    
    Parameters:
    - model_name: The name of the model to update.
    - model_path: Path to the new model file.
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow', 'tflite').
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the device.
    - deployment_path: The file path where the model will be stored on the device.
    - backup_existing: Boolean indicating whether to backup the existing model (default is True).
    - restart_service: Boolean indicating whether to restart the inference service after deployment (default is True).
    
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

    # Step 3: Backup the existing model (if required and if it exists)
    if backup_existing:
        model_exists = check_if_model_exists(device_ip, deployment_path, model_name, model_extension)
        if model_exists:
            backup_success = backup_existing_model(device_ip, device_username, deployment_path, model_name, model_extension)
            if not backup_success:
                log_deployment_event(f"Failed to backup model {model_name} on device {device_ip}", log_level='error')
                return False
        else:
            log_deployment_event(f"No existing model found to backup on device {device_ip}. Proceeding with update.", log_level="info")

    # Step 4: Replace the existing model with the new one
    replace_success = replace_model_on_device(device_ip, device_username, deployment_path, temp_model_name, model_name, model_extension)
    if not replace_success:
        log_deployment_event(f"Failed to replace model {model_name} on device {device_ip}", log_level='error')
        return False

    # Step 5: Optionally restart the inference service to activate the new model
    if restart_service:
        service_name = get_inference_service_name(model_format)
        restart_success = restart_inference_service(device_ip, device_username, service_name)
        if not restart_success:
            log_deployment_event(f"Failed to restart inference service for model {model_name} on device {device_ip}", log_level='error')
            return False

    log_deployment_event(f"Successfully updated model {model_name} on device {device_ip}")
    return True