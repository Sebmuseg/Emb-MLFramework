# utils/device_utils.py
import paramiko
import os
from logging_utils import log_deployment_event


def check_if_model_exists(device_ip, device_username, deployment_path, model_name, model_extension):
    """
    Checks if a model already exists on the target device.
    
    Parameters:
    - device_ip: The IP address of the device.
    - device_username: The SSH login username for the device.
    - deployment_path: The directory on the device where the model is stored.
    - model_name: The name of the model.
    - model_extension: The file extension of the model (e.g., '.onnx').
    
    Returns: True if the model exists, False otherwise.
    """
    try:
        # Step 1: Set up SSH client to connect to the target device
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Construct the path of the existing model
        existing_model_path = os.path.join(deployment_path, f"{model_name}{model_extension}")

        # Step 3: Check if the file exists
        sftp = ssh.open_sftp()
        try:
            sftp.stat(existing_model_path)  # If this raises FileNotFoundError, the model doesn't exist
            sftp.close()
            ssh.close()
            return True  # Model exists
        except FileNotFoundError:
            sftp.close()
            ssh.close()
            return False  # Model doesn't exist

    except Exception as e:
        log_deployment_event(f"Failed to check if model exists on {device_ip}: {e}", log_level="error")
        return False