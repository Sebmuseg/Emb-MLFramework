# api/deployment_api.py
from fastapi import APIRouter, HTTPException
from deployment.deploy_model import deploy_model
from deployment.backup_replace import backup_existing_model, transfer_backup_model, check_backup_exists
from deployment.transfer_model import transfer_model_to_device
from utils.logging_utils import log_deployment_event
from utils.model_utils import get_inference_service_name
from deployment.service_management import restart_inference_service
from deployment.deploy_model import package_model_in_docker, update_model
from core.framework_api import FrameworkAPI
from pydantic import BaseModel, ConfigDict


router = APIRouter()
framework = FrameworkAPI()

class CustomBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
class DeployModelRequest(CustomBaseModel):
    model_name: str
    device_ip: str
    model_format: str
    deployment_path: str
    device_username: str
    is_docker: bool = False

class BackupModelRequest(CustomBaseModel):
    model_name: str
    device_ip: str
    device_username: str
    deployment_path: str
    model_extension: str

class TransferModelRequest(CustomBaseModel):
    model_path: str 
    model_name: str 
    device_ip: str
    device_username: str 
    deployment_path: str
    temp_model_name: str
    
class DeployDockerModelRequest(CustomBaseModel):
    model_name: str
    model_path: str
    dockerfile_template: str
    output_dir: str
    
class UpdateModelRequest(CustomBaseModel):
    model_name: str
    device_ip: str
    username: str
    deployment_path: str
    model_format: str
    backup_existing: bool = True
    restart_service: bool = True
    
class RollbackModelRequest(CustomBaseModel):
    device_ip: str
    backup_path: str
    deployment_path: str
    model_format: str
    device_username: str
    


# Deployment Function
@router.post("/deploy")
async def deploy_model_endpoint(request: DeployModelRequest):
    """
    Deploy the specified model to the target device.

    Parameters:
    - model_name: Name of the model to deploy.
    - device_ip: IP address of the device where the model should be deployed.
    - model_format: The format of the model (e.g., 'onnx', 'tensorflow').
    - deployment_path: Path on the device where the model should be deployed.
    - device_username: Username for SSH login to the device.
    - is_docker: Whether the model should be deployed as a Docker container (default: False).

    Returns:
    - JSON response with deployment status and message.
    """
    try:
        # Extracting parameters from the request object
        model_name = request.model_name
        device_ip = request.device_ip
        model_format = request.model_format
        deployment_path = request.deployment_path
        device_username = request.device_username
        is_docker = request.is_docker
        log_deployment_event(f"Deploying model {model_name} to device {device_ip}...", log_level="info")
        
        model = framework.framework.models.get(model_name)
        if not model:
            log_deployment_event(f"Model {model_name} not found in the framework.", log_level="error")
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in the framework.")

        # Get the model path 
        model_path = model.get_model_path()
        log_deployment_event(f"Model path: {model_path}", log_level="info")
        
        # Call the deploy_model utility function
        deploy_success = deploy_model(model_name, model_path, model_format, device_ip, device_username, deployment_path, is_docker)

        if deploy_success:
            log_deployment_event(f"Model {model_name} deployed successfully to device {device_ip}.", log_level="info")
            return {"status": "Success", "message": f"Model {model_name} deployed to {device_ip}"}
        else:
            log_deployment_event(f"Failed to deploy model {model_name} to device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Model deployment failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Backup Function
@router.post("/backup")
async def backup_model(request: BackupModelRequest):
    """
    Backup the specified model on the target device before deployment.

    Parameters:
    - model_name: Name of the model to backup.
    - device_ip: IP address of the target device.
    - device_username: Username to log into the target device.
    - deployment_path: The file path where the model is stored on the device.
    - model_extension: The file extension of the model (e.g., '.onnx', '.tflite').

    Returns:
    - Success or failure message.
    """
    try:
        device_ip=request.device_ip, 
        device_username=request.device_username, 
        deployment_path=request.deployment_path, 
        model_name=request.model_name, 
        model_extension=request.model_extension
        log_deployment_event(f"Backing up model {model_name} on device {device_ip}...", log_level="info")
        
        # Logic for backing up the model
        backup_success = backup_existing_model(
            device_ip=device_ip, 
            device_username=device_username, 
            deployment_path=deployment_path, 
            model_name=model_name, 
            model_extension=model_extension
        )

        if backup_success:
            log_deployment_event(f"Model {model_name} backed up successfully on device {device_ip}.", log_level="info")
            return {"status": "Success", "message": f"Model {model_name} backed up successfully on device {device_ip}"}
        else:
            log_deployment_event(f"Failed to backup model {model_name} on device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Model backup failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Transfer Function
@router.post("/transfer")
async def transfer_model(request: TransferModelRequest):
    """
    Transfer the specified model to the target device.

    Parameters:
    - model_path: Path to the model file on the local system.
    - model_name: Name of the model to transfer.
    - device_ip: IP address of the target device.
    - device_username: Username to log into the target device.
    - deployment_path: The file path on the device where the model will be transferred.
    - temp_model_name: Temporary name for the model during the transfer.

    Returns:
    - Success or failure message.
    """
    try:
        model_path=request.model_path,
        model_name=request.model_name,
        device_ip=request.device_ip,
        device_username=request.device_username,
        deployment_path=request.deployment_path,
        temp_model_name=request.temp_model_name
        log_deployment_event(f"Transferring model {model_name} to device {device_ip}...", log_level="info")
        
        # Logic for transferring the model
        transfer_success = transfer_model_to_device(
            model_path=model_path, 
            device_ip=device_ip, 
            device_username=device_username, 
            deployment_path=deployment_path, 
            temp_model_name=temp_model_name
        )

        if transfer_success:
            log_deployment_event(f"Model {model_name} transferred successfully to device {device_ip}.", log_level="info")
            return {"status": "Success", "message": f"Model {model_name} successfully transferred to device {device_ip}"}
        else:
            log_deployment_event(f"Failed to transfer model {model_name} to device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Model transfer failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/deploy/docker")
async def deploy_model_in_docker(request: DeployDockerModelRequest):
    """
    API endpoint to package a model into a Docker container for deployment.

    Parameters (via request body):
    - model_name: The name of the model to package.
    - model_path: The path to the model file.
    - dockerfile_template: The path to the Dockerfile template.
    - output_dir: The directory to store the Docker image.

    Returns:
    - JSON response confirming the success or failure of the deployment.
    """
    # Extracting parameters from the request
    model_name = request.model_name
    model_path = request.model_path
    dockerfile_template = request.dockerfile_template
    output_dir = request.output_dir
    log_deployment_event(f"Packaging model {model_name} into Docker container...", log_level="info")

    # Input validation (already handled by Pydantic, but adding as extra precaution)
    if not model_name or not model_path or not dockerfile_template or not output_dir:
        log_deployment_event(f"Invalid input parameters for Docker packaging of model {model_name}", log_level="error")
        raise HTTPException(status_code=400, detail="Invalid input parameters")

    try:
        # Call the internal function to package the model into a Docker container
        success = package_model_in_docker(model_name, model_path, dockerfile_template, output_dir)
        
        if success:
            # Log and return success message
            log_deployment_event(f"Model {model_name} successfully packaged into Docker container.", log_level="info")
            return {"status": "Success", "message": f"Model {model_name} packaged into Docker container."}
        else:
            # Handle failure
            raise HTTPException(status_code=500, detail="Failed to package the model into Docker container.")

    except Exception as e:
        # Log and return the error message
        log_deployment_event(f"Error packaging model {model_name} into Docker container: {e}", log_level="error")
        raise HTTPException(status_code=500, detail=f"Error packaging model {model_name}: {str(e)}")
    
@router.post("/update")    
async def update_model_on_device(request: UpdateModelRequest):
        """
        API method to update an already deployed model on the target device with a new version.

        Parameters:
        - model_name: The name of the model to update.
        - device_ip: The IP address of the target device.
        - deployment_path: The file path where the model is stored on the device.
        - model_format: The format of the model (e.g., ONNX, TensorFlow Lite).
        - backup_existing: Boolean indicating whether to backup the existing model (default is True).
        - restart_service: Boolean indicating whether to restart the inference service after deployment (default is True).

        Returns:
        - True if the update was successful, False otherwise.
        """
        try:
            model_name=request.model_name,
            device_ip=request.device_ip,
            deployment_path=request.deployment_path,
            model_format=request.model_format,
            username=request.username,
            backup_existing=request.backup_existing,
            restart_service=request.restart_service
            log_deployment_event(f"Updating model {model_name} on device {device_ip}...", log_level="info")

            # Step 1: Check if the model exists in the framework
            model = framework.framework.models.get(model_name)
            if not model:
                log_deployment_event(f"Model {model_name} not found in the framework.", log_level="error")
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found!")   

            # Step 2: Get the model path from the model object
            model_path = model.get_model_path()  
            log_deployment_event(f"Model path: {model_path}")
        
            # Step 3: Call the deployment logic for updating the model
            success = update_model(
                model_name=model_name,
                model_path=model_path,
                model_format=model_format,
                device_ip=device_ip,
                device_username=username,  
                deployment_path=deployment_path,
                backup_existing=backup_existing,
                restart_service=restart_service
            )
            if success:
                log_deployment_event(f"Model {model_name} updated successfully on device {device_ip}.", log_level="info")
                return {"status": "Success", "message": f"Model {model_name} updated successfully on device {device_ip}."}
            else:
                log_deployment_event(f"Failed to update model {model_name} on device {device_ip}.", log_level="error")
                raise HTTPException(status_code=500, detail="Model update failed.")
            
        except Exception as e:
            log_deployment_event(f"Error updating model {model_name} on device {device_ip}: {e}", log_level="error")
            raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/rollback")
async def rollback_model_on_device(request: RollbackModelRequest):
    """
    API method to rollback a model to a previous version on the target device.

    Parameters:
    - device_ip: The IP address of the target device.
    - backup_path: The file path where the previous model version is stored.
    - deployment_path: The current deployment path of the model.
    - model_format: The format of the model (e.g., ONNX, OpenVINO, TensorFlow Lite).
    - username: The SSH login username for the target device.

    Returns:
    - True if rollback was successful, False otherwise.
    """
    try:
        device_ip=request.device_ip,
        backup_path=request.backup_path,
        deployment_path=request.deployment_path,
        model_format=request.model_format,
        username=request.device_username
        log_deployment_event(f"Rolling back model on device {device_ip}...", log_level="info")

        # Step 1: Validate if backup exists
        backup_exists = check_backup_exists(device_ip, username, backup_path)
        if not backup_exists:
            log_deployment_event(f"Backup not found for model on device {device_ip}.", log_level="error")
            raise HTTPException(status_code=404, detail="Backup not found for model.")

        # Step 2: Transfer the backup model to the deployment path
        transfer_success = transfer_backup_model(device_ip, username, backup_path, deployment_path, model_format)
        if not transfer_success:
            log_deployment_event(f"Failed to transfer backup model to device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Failed to transfer backup model.")

        # Step 3: Restart the inference service if necessary
        service_name = get_inference_service_name(model_format)
        if not service_name:
            log_deployment_event(f"Service name not found for model format {model_format}.", log_level="error")
            raise HTTPException(status_code=500, detail="Service name not found.")
        restart_success = restart_inference_service(device_ip, username, service_name)
        if not restart_success:
            log_deployment_event(f"Failed to restart service for model on device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Failed to restart service.")

        log_deployment_event(f"Successfully rolled back model on device {device_ip}.", log_level="info")
        return {"status": "Success", "message": f"Model rollback successful on device {device_ip}."}

    except Exception as e:
        log_deployment_event(f"Failed to rollback model on device {device_ip}: {e}", log_level="error")
        raise HTTPException(status_code=500, detail=str(e))
        