from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from utils.logging_utils import log_deployment_event, log_monitoring_event

router = APIRouter()

class CustomBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class LogEventRequest(CustomBaseModel):
    model_name: str
    status: str
    details: str

class MonitoringEventRequest(CustomBaseModel):
    device_ip: str
    resource_usage: dict

# Log Deployment Events
@router.post("/log/deployment")
def log_deployment(request: LogEventRequest):
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
        model_name = request.model_name
        status = request.status
        details = request.details
        log_deployment_event(f"Deployment event logged: Model {model_name}, Status: {status}, Details: {details}")
        
        log_message = f"Model: {model_name}, Status: {status}, Details: {details}"
        if status == "success":
            log_deployment_event(f"Logged deployment event: {log_message}")
            return {"status": "Success", "message": "Deployment event logged successfully."}
        else:
            log_deployment_event(f"Error logging deployment event: {log_message}", log_level="error")
        
    except Exception as e:
        log_deployment_event(f"Error logging deployment event: {str(e)}", log_level="error")
        raise HTTPException(status_code=500, detail=str(e))


# Log Monitoring Events
@router.post("/log/monitoring")
def log_monitoring(request: MonitoringEventRequest):
    """
    Log events related to resource monitoring (e.g., CPU, memory, GPU).

    Parameters:
    - device_ip: The IP address of the monitored device.
    - resource_usage: Dictionary of resource usage data (e.g., {'cpu': 80, 'memory': 70}).

    Returns:
    - JSON response confirming that the event was logged.
    """
    try:
        device_ip = request.device_ip
        resource_usage = request.resource_usage
        log_deployment_event(f"Monitoring event logged: Device {device_ip}, Resource Usage: {resource_usage}")
        
        log_message = f"Device: {device_ip}, Resource Usage: {resource_usage}"
        if resource_usage:
            log_monitoring_event(f"Logged monitoring event: {log_message}")
            return {"status": "Success", "message": "Monitoring event logged successfully."}
        log_monitoring_event(log_message)
        log_deployment_event(f"Logged monitoring event: {log_message}")

        return {"status": "Success", "message": "Monitoring event logged successfully."}

    except Exception as e:
        log_deployment_event(f"Error logging monitoring event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Query Logs from ELK (Elasticsearch)
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
        # Query Elasticsearch (this is a placeholder logic)
        # You would use an Elasticsearch client library like `elasticsearch-py`
        logs = [
            {"timestamp": "2024-10-15T10:15:00", "log": "Deployment success: Model X"},
            {"timestamp": "2024-10-15T10:16:00", "log": "Monitoring event: CPU usage high on device 192.168.1.5"},
        ]
        filtered_logs = [log for log in logs if query.lower() in log['log'].lower()]

        return {"status": "Success", "logs": filtered_logs}

    except Exception as e:
        log_deployment_event(f"Error querying logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))