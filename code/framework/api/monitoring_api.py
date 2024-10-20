from fastapi import APIRouter, HTTPException, BackgroundTasks
import psutil
from utils.resource_monitoring import monitor_system_resources

router = APIRouter()

# Variables to control the background task and its state
monitoring_task_running = False
collected_data = []
background_task = None

# Example: Monitor System Resources
@router.get("/monitor/system_resources")
def monitor_system_resources(device_ip: str, cpu_threshold: int = 80, memory_threshold: int = 70):
    """
    Monitor the system resources (CPU, memory) of a specified device.

    Parameters:
    - device_ip: The IP address of the target device.
    - cpu_threshold: The CPU usage threshold for alerting (default is 80%).
    - memory_threshold: The memory usage threshold for alerting (default is 70%).

    Returns:
    - JSON with current CPU and memory usage.
    """
    try:
        # In actual implementation, connect to the remote device
        # Using psutil for local monitoring for now
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()

        alert_status = {
            "cpu_alert": cpu_usage > cpu_threshold,
            "memory_alert": memory_info.percent > memory_threshold,
        }

        return {
            "status": "Monitoring",
            "cpu_usage": cpu_usage,
            "memory_usage": memory_info.percent,
            "alerts": alert_status,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example: Monitor Model Performance
@router.get("/monitor/model_performance")
def monitor_model_performance(model_name: str, device_ip: str):
    """
    Monitor the performance of a deployed model on a specified device.

    Parameters:
    - model_name: The name of the deployed model.
    - device_ip: The IP address of the target device.

    Returns:
    - JSON with model performance metrics (e.g., inference time, accuracy).
    """
    try:
        # Example logic: Replace with real monitoring tools (e.g., Prometheus)
        # Simulating performance metrics for now
        performance_metrics = {
            "inference_time_ms": 45.8,  # Example data
            "accuracy": 98.7,  # Example data
            "throughput": 1000  # Example data (inferences per second)
        }

        return {
            "status": "Monitoring",
            "model_name": model_name,
            "performance_metrics": performance_metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example: Get Device Metrics via Prometheus Node Exporter (Linux)
@router.get("/monitor/device_metrics")
def monitor_device_metrics(device_ip: str):
    """
    Retrieve metrics from Prometheus Node Exporter or similar monitoring tool on a device.

    Parameters:
    - device_ip: The IP address of the target device.

    Returns:
    - JSON with metrics collected by Prometheus or similar.
    """
    try:
        # Placeholder: Make request to Prometheus or Node Exporter for device metrics
        # Example logic: simulate the result
        metrics = {
            "cpu_usage": "75%",
            "memory_usage": "65%",
            "disk_usage": "40%",
            "gpu_usage": "70%"  # If applicable
        }

        return {
            "status": "Success",
            "device_ip": device_ip,
            "metrics": metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# API to start the resource monitoring task
@router.post("/start-monitoring/")
def start_monitoring(background_tasks: BackgroundTasks, cpu_threshold: int = 80, memory_threshold: int = 70, disk_threshold: int = 90):
    global monitoring_task_running, background_task
    if not monitoring_task_running:
        monitoring_task_running = True
        resource_thresholds = {'cpu': cpu_threshold, 'memory': memory_threshold, 'disk': disk_threshold}
        background_tasks.add_task(monitor_system_resources, resource_thresholds)
        return {"status": "Monitoring started"}
    else:
        return {"status": "Monitoring is already running"}

# API to stop the resource monitoring task
@router.post("/stop-monitoring/")
def stop_monitoring():
    global monitoring_task_running
    if monitoring_task_running:
        monitoring_task_running = False
        return {"status": "Monitoring stopped"}
    else:
        return {"status": "No monitoring task is running"}

# API to check the status of the monitoring task
@router.get("/monitoring-status/")
def get_monitoring_status():
    if monitoring_task_running:
        return {"status": "Monitoring is running", "data_collected": len(collected_data)}
    else:
        return {"status": "Monitoring is not running"}

# API to get the collected data
@router.get("/monitoring-data/")
def get_monitoring_data():
    return {"collected_data": collected_data}