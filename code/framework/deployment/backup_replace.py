# deployment/backup_replace.py
import paramiko
import os
from utils.logging_utils import log_deployment_event
from service_management import stop_existing_container


def backup_existing_model(device_ip, device_username, deployment_path, model_name, model_extension):
    """
    Backs up the existing model on the target device by renaming it with a '_backup' suffix or a timestamp.

    Parameters:
    - device_ip: The IP address of the target device.
    - device_username: The SSH login username for the device.
    - deployment_path: The path on the device where the model is stored.
    - model_name: The name of the model to back up.
    - model_extension: The file extension of the model (e.g., '.onnx').

    Returns: True if the backup is successful, False otherwise.
    """
    try:
        # Step 1: Set up SSH connection to the device
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Construct paths for the existing model and its backup
        existing_model_path = os.path.join(deployment_path, f"{model_name}{model_extension}")
        backup_model_path = os.path.join(deployment_path, f"{model_name}_backup{model_extension}")

        # Step 3: Open an SFTP session to interact with files on the device
        sftp = ssh.open_sftp()

        # Step 4: Check if the existing model exists, and back it up if it does
        try:
            sftp.stat(existing_model_path)  # Check if the file exists
            sftp.rename(existing_model_path, backup_model_path)  # Rename to backup
            log_deployment_event(f"Existing model {model_name} backed up to {backup_model_path}.")
        except FileNotFoundError:
            log_deployment_event(f"Model {model_name} not found for backup on device {device_ip}.", log_level='warning')
            return False

        # Step 5: Close SFTP and SSH connections
        sftp.close()
        ssh.close()

        return True
    except Exception as e:
        log_deployment_event(f"Failed to back up model {model_name} on device {device_ip}: {e}", log_level='error')
        return False
    
    
def replace_model_on_device(device_ip, device_username, deployment_path, temp_model_name, model_name, model_extension):
    """
    Replaces the existing model on the device with the new model after a successful backup.

    Parameters:
    - device_ip: The IP address of the device.
    - device_username: The SSH login username for the device.
    - deployment_path: The directory on the device where the model is stored.
    - temp_model_name: The temporary name for the new model on the device.
    - model_name: The name of the existing model to be replaced.
    - model_extension: The file extension of the model (e.g., '.onnx', '.pb').

    Returns: True if the replacement is successful, False otherwise.
    """
    try:
        # Step 1: Establish an SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Construct paths for the temporary model and the existing model
        temp_model_path = os.path.join(deployment_path, temp_model_name)
        existing_model_path = os.path.join(deployment_path, f"{model_name}{model_extension}")

        # Step 3: Open an SFTP session for file operations
        sftp = ssh.open_sftp()

        # Step 4: Replace the existing model with the new model (temp model)
        sftp.rename(temp_model_path, existing_model_path)
        log_deployment_event(f"Model {model_name} replaced with new version from {temp_model_name}.", log_level='info')

        # Step 5: Close the SFTP and SSH connections
        sftp.close()
        ssh.close()

        return True
    except Exception as e:
        log_deployment_event(f"Failed to replace model {model_name} on device {device_ip}: {e}", log_level='error')
        return False
    

def docker_container_exists(device_ip, device_username, container_name):
    """
    Checks if a Docker container exists on the edge device.

    Parameters:
    - device_ip: IP address of the device.
    - device_username: Username for SSH login.
    - container_name: The name of the Docker container to check.

    Returns: True if the container exists, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Run the docker ps -a command to check if the container exists
        check_command = f"docker ps -a --filter name={container_name} --format '{{{{.Names}}}}'"
        stdin, stdout, stderr = ssh.exec_command(check_command)
        output = stdout.read().decode().strip()

        ssh.close()

        # Check if the output contains the container name
        return output == container_name
    except Exception as e:
        log_deployment_event(f"Error checking if Docker container {container_name} exists: {e}", log_level="error")
        return False
    
    
def backup_existing_docker_container(device_ip, device_username, container_name):
    """
    Backs up the existing Docker container on the device by renaming it if it exists.
    
    Parameters:
    - device_ip: The IP address of the device.
    - device_username: The SSH login username.
    - container_name: The name of the Docker container to back up.
    
    Returns: True if successful, False otherwise.
    """
    try:
        if not docker_container_exists(device_ip, device_username, container_name):
            log_deployment_event(f"No existing Docker container {container_name} found on device {device_ip}. Skipping backup.", log_level="info")
            return True  # Nothing to back up if container doesn't exist

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Rename the existing Docker container
        backup_command = f"docker rename {container_name} {container_name}_backup"
        ssh.exec_command(backup_command)
        log_deployment_event(f"Existing Docker container {container_name} backed up successfully.", log_level="info")
        ssh.close()

        return True
    except Exception as e:
        log_deployment_event(f"Failed to backup Docker container {container_name}: {e}", log_level="error")
        return False
    
    
def replace_docker_container(device_ip, device_username, container_name):
    """
    Replaces the existing Docker container with the new Docker image, if it exists.
    
    Parameters:
    - device_ip: The IP address of the device.
    - device_username: The SSH login username.
    - container_name: The name of the Docker container to replace.
    
    Returns: True if successful, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Check if the container exists before trying to stop or remove it
        if docker_container_exists(device_ip, device_username, container_name):
            stop_command = f"docker stop {container_name}"
            remove_command = f"docker rm {container_name}"
            ssh.exec_command(stop_command)
            ssh.exec_command(remove_command)
            log_deployment_event(f"Stopped and removed existing Docker container {container_name}.", log_level="info")
        else:
            log_deployment_event(f"No existing Docker container {container_name} found on device {device_ip}. Proceeding with new deployment.", log_level="info")

        # Run the new container
        run_command = f"docker run -d --name {container_name} {container_name}_docker_image"
        ssh.exec_command(run_command)
        log_deployment_event(f"Replaced Docker container {container_name} with new image.", log_level="info")

        ssh.close()
        return True
    except Exception as e:
        log_deployment_event(f"Failed to replace Docker container {container_name}: {e}", log_level="error")
        return False
    
    
def check_backup_exists(device_ip, user, backup_path):
    """
    Checks if the backup model exists on the target device.
    
    Parameters:
    - device_ip: The IP address of the target device.
    - backup_path: The path to the backup model on the target device.

    Returns: True if the backup exists, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=user)  

        # Check if the backup file exists on the device
        sftp = ssh.open_sftp()
        try:
            sftp.stat(backup_path)  # Check if the file exists
            return True
        except FileNotFoundError:
            return False
        finally:
            sftp.close()
            ssh.close()

    except Exception as e:
        log_deployment_event(f"Error checking backup on device {device_ip}: {e}", log_level="error")
        return False
    
    
def transfer_backup_model(device_ip, user, backup_path, deployment_path, model_format):
    """
    Transfers the backup model from the backup path to the deployment path on the target device.
    
    Parameters:
    - device_ip: The IP address of the target device.
    - backup_path: The path to the backup model.
    - deployment_path: The destination path for the deployment.
    - model_format: The format of the model (used for extensions, etc.).

    Returns: True if the transfer was successful, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=user)  # Replace with actual username
        
        # Transfer the backup model to the deployment path
        sftp = ssh.open_sftp()
        sftp.put(backup_path, deployment_path)
        sftp.close()
        ssh.close()
        
        return True
    except Exception as e:
        log_deployment_event(f"Error transferring backup model on device {device_ip}: {e}", log_level="error")
        return False
    
    
