# deployment/monitoring.py
from deployment.deploy_model import validate_device_environment
from utils.logging_utils import log_deployment_event
from utils.ressource_monitoring import monitor_docker_resources, monitor_system_resources
import time 

def monitor_device_resources(device_ip, device_username, resource_thresholds, is_docker=False, container_name=None):
    """
    Monitors the resource usage (CPU, memory, GPU) of the device running the model, 
    or the resource usage of a Docker container if specified.

    Parameters:
    - device_ip: The IP address of the target device.
    - device_username: The SSH login username for the device.
    - resource_thresholds: A dictionary of resource usage limits (e.g., {'cpu': 80, 'memory': 70}).
    - is_docker: Boolean indicating if the model is running in a Docker container.
    - container_name: The name of the Docker container (if applicable).
    
    Steps:
    - Continuously monitor CPU, memory, and GPU usage.
    - Log the resource usage at regular intervals.
    - Trigger alerts if usage exceeds the defined thresholds.
    
    Returns: None
    """
    try:
        # Validate device environment
        environment_ready = validate_device_environment(device_ip, device_username, is_docker)
        if not environment_ready:
            log_deployment_event(f"Environment validation failed for device {device_ip}. Cannot monitor resources.", log_level='error')
            return False
        
        while True:
            if is_docker and container_name:
                # Monitor Docker container resources
                resource_usage = monitor_docker_resources(container_name, resource_thresholds)
            else:
                # Monitor system resources (non-Docker)
                resource_usage = monitor_system_resources(resource_thresholds)
            
            if resource_usage:
                log_deployment_event(f"Resource usage on {device_ip}: {resource_usage}")
            else:
                log_deployment_event(f"Failed to retrieve resource usage on {device_ip}.", log_level='error')

            # Sleep for a defined interval before monitoring again
            time.sleep(10)  # Adjust the interval as needed

    except Exception as e:
        log_deployment_event(f"Error monitoring device resources on {device_ip}: {e}", log_level='error')
        return False