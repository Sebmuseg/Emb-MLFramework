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
        
        
def stop_existing_container(device_ip, device_username, container_name):
    """
    Stops a running Docker container on a remote device via SSH if it's running.

    Parameters:
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the device.
    - container_name: The name of the Docker container to stop.

    Returns: True if the container was stopped successfully or not running, False if there was an error.
    """
    try:
        # Step 1: Set up SSH client to connect to the target device
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Check if the container is running using 'docker ps'
        check_command = f"docker ps --filter name={container_name} --format '{{{{.Names}}}}'"
        stdin, stdout, stderr = ssh.exec_command(check_command)

        # Step 3: Read the output
        output = stdout.read().decode().strip()
        error_output = stderr.read().decode()

        if error_output:
            log_deployment_event(f"Error checking container {container_name} status on device {device_ip}: {error_output}", log_level="error")
            return False

        # Step 4: If the container is not running, log the event and return
        if output != container_name:
            log_deployment_event(f"No running container found with name {container_name} on device {device_ip}.", log_level="info")
            ssh.close()
            return True  # No need to stop anything, the container is not running

        # Step 5: If the container is running, stop it
        stop_command = f"docker stop {container_name}"
        stdin, stdout, stderr = ssh.exec_command(stop_command)

        # Step 6: Log the output of the stop command and check for errors
        output = stdout.read().decode()
        error_output = stderr.read().decode()

        if error_output:
            log_deployment_event(f"Error stopping container {container_name} on device {device_ip}: {error_output}", log_level="error")
            return False
        else:
            log_deployment_event(f"Container {container_name} successfully stopped on device {device_ip}.", log_level="info")

        # Step 7: Close the SSH connection
        ssh.close()

        return True

    except Exception as e:
        log_deployment_event(f"Failed to stop container {container_name} on device {device_ip}: {e}", log_level="error")
        return False
    
    
def run_docker_container(device_ip, device_username, container_name, docker_image, ports=None, volumes=None, additional_args=""):
    """
    Runs a Docker container on a remote device via SSH.

    Parameters:
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the device.
    - container_name: The name of the Docker container to run.
    - docker_image: The name of the Docker image to run.
    - ports: A dictionary mapping host ports to container ports (e.g., {"8080": "80"}).
    - volumes: A dictionary mapping host directories to container directories (e.g., {"/host/path": "/container/path"}).
    - additional_args: Any additional arguments to pass to the Docker run command (optional).
    
    Returns: True if the container was started successfully, False otherwise.
    """
    try:
        # Step 1: Set up SSH client to connect to the target device
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Step 2: Construct the Docker run command
        run_command = f"docker run -d --name {container_name}"

        # Add port mappings, if provided
        if ports:
            for host_port, container_port in ports.items():
                run_command += f" -p {host_port}:{container_port}"

        # Add volume mappings, if provided
        if volumes:
            for host_dir, container_dir in volumes.items():
                run_command += f" -v {host_dir}:{container_dir}"

        # Add additional arguments if provided
        if additional_args:
            run_command += f" {additional_args}"

        # Add the Docker image
        run_command += f" {docker_image}"

        # Step 3: Execute the Docker run command on the target device
        stdin, stdout, stderr = ssh.exec_command(run_command)

        # Step 4: Log the output and check for errors
        output = stdout.read().decode()
        error_output = stderr.read().decode()

        if error_output:
            log_deployment_event(f"Error running container {container_name} on device {device_ip}: {error_output}", log_level="error")
            return False
        else:
            log_deployment_event(f"Container {container_name} successfully started on device {device_ip}.", log_level="info")
        
        # Step 5: Close the SSH connection
        ssh.close()

        return True

    except Exception as e:
        log_deployment_event(f"Failed to run container {container_name} on device {device_ip}: {e}", log_level="error")
        return False