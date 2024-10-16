from fastapi import APIRouter, HTTPException
import logging
from utils.logging_utils import log_deployment_event, log_monitoring_event

router = APIRouter()

# Configure the logging system
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example: Log Deployment Events
@router.post("/log/deployment")
def log_deployment(model_name: str, status: str, details: str):
    """
    Log events related to the deployment process.

    Parameters:
    - model_name: The name of the deployed model.
    - status: The current status of the deployment (e.g., success, failure).
    - details: Any additional details about the deployment event.

    Returns:
    - JSON response confirming that the event was logged.
    """
    try:
        log_message = f"Model: {model_name}, Status: {status}, Details: {details}"
        log_deployment_event(log_message)
        logger.info(f"Logged deployment event: {log_message}")

        return {"status": "Success", "message": "Deployment event logged successfully."}

    except Exception as e:
        logger.error(f"Error logging deployment event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Example: Log Monitoring Events
@router.post("/log/monitoring")
def log_monitoring(device_ip: str, resource_usage: dict):
    """
    Log events related to resource monitoring (e.g., CPU, memory, GPU).

    Parameters:
    - device_ip: The IP address of the monitored device.
    - resource_usage: Dictionary of resource usage data (e.g., {'cpu': 80, 'memory': 70}).

    Returns:
    - JSON response confirming that the event was logged.
    """
    try:
        log_message = f"Device: {device_ip}, Resource Usage: {resource_usage}"
        log_monitoring_event(log_message)
        logger.info(f"Logged monitoring event: {log_message}")

        return {"status": "Success", "message": "Monitoring event logged successfully."}

    except Exception as e:
        logger.error(f"Error logging monitoring event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Example: Query Logs from ELK (Elasticsearch)
@router.get("/log/query")
def query_logs(query: str):
    """
    Query logs stored in the ELK stack (Elasticsearch).

    Parameters:
    - query: Query string for searching the logs.

    Returns:
    - JSON response with the matched log entries.
    """
    try:
        # Example: Query Elasticsearch (this is a placeholder logic)
        # You would use an Elasticsearch client library like `elasticsearch-py`
        logs = [
            {"timestamp": "2024-10-15T10:15:00", "log": "Deployment success: Model X"},
            {"timestamp": "2024-10-15T10:16:00", "log": "Monitoring event: CPU usage high on device 192.168.1.5"},
        ]
        filtered_logs = [log for log in logs if query.lower() in log['log'].lower()]

        return {"status": "Success", "logs": filtered_logs}

    except Exception as e:
        logger.error(f"Error querying logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))