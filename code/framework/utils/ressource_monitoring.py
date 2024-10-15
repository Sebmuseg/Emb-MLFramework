# utils/ressource_monitoring.py
import paramiko
import psutil  # For non-Docker resource monitoring
import subprocess
import time
from utils.logging_utils import log_deployment_event  # Assuming you have this for logging

# Non-Docker Resource Monitoring
def monitor_system_resources(resource_thresholds):
    """
    Monitors CPU, memory, and disk usage for a non-Docker system.
    
    Parameters:
    - resource_thresholds: A dictionary of resource usage limits (e.g., {'cpu': 80, 'memory': 70}).
    
    Returns: Dictionary of resource usage
    """
    # Get CPU, Memory and Disk Usage
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    # Log the resource usage
    log_deployment_event(f"System CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%, Disk Usage: {disk_usage}%")

    # Check thresholds and alert if exceeded
    if cpu_usage > resource_thresholds.get('cpu', 100):
        log_deployment_event(f"CPU usage exceeded: {cpu_usage}%", log_level='warning')
    if memory_usage > resource_thresholds.get('memory', 100):
        log_deployment_event(f"Memory usage exceeded: {memory_usage}%", log_level='warning')
    if disk_usage > resource_thresholds.get('disk', 100):
        log_deployment_event(f"Disk usage exceeded: {disk_usage}%", log_level='warning')

    return {'cpu': cpu_usage, 'memory': memory_usage, 'disk': disk_usage}

# Docker Resource Monitoring
def monitor_docker_resources(container_name, resource_thresholds):
    """
    Monitors CPU, memory, and disk usage for a Docker container.
    
    Parameters:
    - container_name: Name of the Docker container to monitor.
    - resource_thresholds: A dictionary of resource usage limits (e.g., {'cpu': 80, 'memory': 70}).
    
    Returns: Dictionary of resource usage
    """
    try:
        # Get the resource stats for the Docker container
        docker_stats_command = f"docker stats {container_name} --no-stream --format '{{{{.CPUPerc}}}},{{{{.MemPerc}}}}'"
        result = subprocess.run(docker_stats_command, shell=True, capture_output=True, text=True)

        # Extract the CPU and memory percentages
        output = result.stdout.strip().split(',')
        cpu_usage = float(output[0].replace('%', ''))
        memory_usage = float(output[1].replace('%', ''))

        # Log the resource usage
        log_deployment_event(f"Docker Container '{container_name}' CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

        # Check thresholds and alert if exceeded
        if cpu_usage > resource_thresholds.get('cpu', 100):
            log_deployment_event(f"Container '{container_name}' CPU usage exceeded: {cpu_usage}%", log_level='warning')
        if memory_usage > resource_thresholds.get('memory', 100):
            log_deployment_event(f"Container '{container_name}' Memory usage exceeded: {memory_usage}%", log_level='warning')

        return {'cpu': cpu_usage, 'memory': memory_usage}

    except Exception as e:
        log_deployment_event(f"Error monitoring Docker container {container_name}: {e}", log_level='error')
        return None