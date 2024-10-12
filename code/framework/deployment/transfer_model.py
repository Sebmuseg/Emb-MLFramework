# deployment/transfer_model.py
import paramiko
import os

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
        print(f"Transferring model to {device_ip}:{temp_deploy_path}")
        sftp.put(model_path, temp_deploy_path)

        sftp.close()
        ssh.close()
        return True
    except Exception as e:
        print(f"Error during model transfer: {e}")
        return False