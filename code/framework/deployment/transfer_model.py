# deployment/transfer_model.py
import paramiko
import os
import subprocess
from scp import SCPClient
from utils.logging_utils import log_deployment_event

def transfer_model_to_device(model_path, device_ip, device_username, deployment_path, temp_model_name):
    """
    Transfers the model to a device via SFTP.

    Parameters:
    - model_path: The path to the model file on the local system.
    - device_ip: The IP address of the device.
    - device_username: The SSH login username for the device.
    - deployment_path: The directory on the device where the model will be uploaded.
    - temp_model_name: The temporary name for the new model on the device.

    Returns: True if the model is successfully transferred, False otherwise.
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)
        sftp = ssh.open_sftp()

        temp_deploy_path = os.path.join(deployment_path, temp_model_name)
        log_deployment_event(f"Transferring model to {device_ip}:{temp_deploy_path}")
        sftp.put(model_path, temp_deploy_path)

        sftp.close()
        ssh.close()
        return True
    except Exception as e:
        log_deployment_event(f"Error during model transfer: {e}", log_level= 'error')
        return False
    
    
def transfer_docker_image_to_device(device_ip, device_username, docker_image_tag, remote_path):
    """
    Transfers a Docker image to the target device using SCP.

    Parameters:
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the device.
    - docker_image_tag: The Docker image tag (e.g., 'model_docker_image') to be transferred.
    - remote_path: The path on the target device where the image will be stored.
    
    Returns: True if the transfer was successful, False otherwise.
    """
    try:
        # Step 1: Set up SSH client to connect to the target device
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Save the Docker image to a .tar file locally
        local_tar_file = f"{docker_image_tag}.tar"
        save_image_command = f"docker save -o {local_tar_file} {docker_image_tag}"
        subprocess.run(save_image_command.split(), check=True)
        
        log_deployment_event(f"Docker image {docker_image_tag} saved as {local_tar_file} locally.", log_level="info")

        # Step 3: Use SCP to transfer the .tar Docker image file to the target device
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_tar_file, remote_path)
            log_deployment_event(f"Successfully transferred Docker image {docker_image_tag} to {device_ip}:{remote_path}", log_level="info")

        # Close SSH connection
        ssh.close()

        # Cleanup: Remove the local .tar file after transfer
        os.remove(local_tar_file)

        return True

    except Exception as e:
        log_deployment_event(f"Failed to transfer Docker image {docker_image_tag} to device {device_ip}: {e}", log_level="error")
        return False