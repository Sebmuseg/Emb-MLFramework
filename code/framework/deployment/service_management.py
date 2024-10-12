# deployment/service_management.py
import paramiko
from utils.logging_utils import log_deployment_event

    
def restart_inference_service(device_ip, device_username, service_name):
    """
    Restarts the inference service on the target device via SSH.

    Parameters:
    - device_ip: The IP address of the target device.
    - device_username: The SSH login username for the device.
    - service_name: The name of the inference service to restart (e.g., 'onnx-inference-service').

    Returns: True if the service restart was successful, False otherwise.
    """
    try:
        # Step 1: Establish an SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Execute the service restart command
        restart_command = f"sudo systemctl restart {service_name}"
        stdin, stdout, stderr = ssh.exec_command(restart_command)

        # Step 3: Read the output and check for errors
        stdout_output = stdout.read().decode().strip()
        stderr_output = stderr.read().decode().strip()

        if stderr_output:
            log_deployment_event(f"Failed to restart {service_name} on device {device_ip}: {stderr_output}", log_level='error')
            return False
        else:
            log_deployment_event(f"Successfully restarted {service_name} on device {device_ip}.", log_level='info')
            return True
    except Exception as e:
        log_deployment_event(f"Error during restart of {service_name} on device {device_ip}: {e}", log_level='error')
        return False
    finally:
        # Step 4: Close the SSH connection
        ssh.close()