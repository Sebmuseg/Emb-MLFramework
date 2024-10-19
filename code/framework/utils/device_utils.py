# utils/device_utils.py
import paramiko
import os
from utils.logging_utils import log_deployment_event


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
    
    
def check_ssh_connection(device_ip, device_username):
    """
    Checks if the SSH connection to the device can be established.
    
    Returns: True if SSH connection is successful, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)
        ssh.close()
        return True
    except Exception as e:
        log_deployment_event(f"SSH connection failed to {device_ip}: {e}", log_level="error")
        return False
    
    
def check_docker_installed(device_ip, device_username):
    """
    Checks if Docker is installed on the target device.
    
    Returns: True if Docker is installed, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)
        
        stdin, stdout, stderr = ssh.exec_command("docker --version")
        output = stdout.read().decode()
        ssh.close()

        if "Docker version" in output:
            return True
        else:
            log_deployment_event(f"Docker not found on {device_ip}.", log_level="error")
            return False
    except Exception as e:
        log_deployment_event(f"Failed to check Docker installation on {device_ip}: {e}", log_level="error")
        return False
    
    
def check_disk_space(device_ip, device_username, required_space_gb=5):
    """
    Checks if the device has sufficient disk space for deployment.
    
    Returns: True if sufficient disk space is available, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)
        
        stdin, stdout, stderr = ssh.exec_command("df -h /")
        output = stdout.read().decode()
        ssh.close()

        # Parse the output and check available space (this is just an example, adjust parsing as necessary)
        available_space_gb = float(output.split()[10][:-1])  # Simplified example
        if available_space_gb >= required_space_gb:
            return True
        else:
            log_deployment_event(f"Insufficient disk space on {device_ip}: {available_space_gb} GB available.", log_level="error")
            return False
    except Exception as e:
        log_deployment_event(f"Failed to check disk space on {device_ip}: {e}", log_level="error")
        return False