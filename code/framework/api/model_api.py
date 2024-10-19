from fastapi import APIRouter, HTTPException
from core.framework_api import FrameworkAPI
from pydantic import BaseModel
from utils.model_utils import evaluate_model, update_model_on_device, rollback_model_on_device
from utils.logging_utils import log_deployment_event

router = APIRouter()
framework_api = FrameworkAPI()

# Create a Pydantic model for validating input
class loadModelRequest(BaseModel):
    model_name: str
    model_path: str
    framework_type: str

class predictRequest(BaseModel):
    model_name: str
    input_data: dict

class removeModelRequest(BaseModel):
    model_name: str

class saveModelRequest(BaseModel):
    model_name: str
    file_path: str

class TrainModelRequest(BaseModel):
    model_name: str
    model_data_path: str
    model_params: dict

class EvaluateModelRequest(BaseModel):
    model_name: str
    test_data_path: str
    metrics: list
    
class UpdateModelRequest(BaseModel):
    model_name: str
    device_ip: str
    deployment_path: str
    model_format: str
    
class RollbackModelRequest(BaseModel):
    device_ip: str
    backup_path: str
    deployment_path: str
    model_format: str
    username: str

@router.post("/load_model")
async def load_model(request: loadModelRequest):
    try:
        model_name = request.model_name
        model_path = request.model_path
        framework_type = request.framework_type
        log_deployment_event(f"Loading model {model_name}...", log_level="info")
        
        success = framework_api.load_model(model_name, model_path, framework_type)
        if not success:
            log_deployment_event(f"Failed to load model {model_name}.", log_level="error")
            raise HTTPException(status_code=400, detail=f"Error loading model {model_name}")
        else:
            log_deployment_event(f"Model {model_name} loaded successfully.", log_level="info")
            return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        log_deployment_event(f"Error loading model {model_name}: {str(e)}", log_level="error")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_models")
async def list_models():
    try:
        models = framework_api.list_models()
        log_deployment_event(f"List of models: {models}", log_level="info")
        return {"models": models}
    except Exception as e:
        log_deployment_event(f"Error listing models: {str(e)}", log_level="error")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/predict")
async def predict(request: predictRequest):
    try:
        model_name = request.model_name
        input_data = request.input_data
        log_deployment_event(f"Predicting using model {model_name}...", log_level="info")
        
        result = framework_api.predict(model_name, input_data)
        if result is None:
            log_deployment_event(f"Failed to predict using model {model_name}.", log_level="error")
            raise HTTPException(status_code=400, detail=f"Error predicting using model {model_name}")
        else:
            log_deployment_event(f"Prediction using model {model_name}: {result}", log_level="info")
            return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/remove_model")
async def remove_model(request: removeModelRequest):
    try:
        model_name = request.model_name
        log_deployment_event(f"Removing model {model_name}...", log_level="info")
        
        success = framework_api.remove_model(model_name)
        if not success:
            log_deployment_event(f"Failed to remove model {model_name}.", log_level="error")
            raise HTTPException(status_code=400, detail=f"Error removing model {model_name}")
        else:
            log_deployment_event(f"Model {model_name} removed successfully.", log_level="info")
            return {"message": f"Model {model_name} removed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/save_model")
async def save_model(request: saveModelRequest):
    try:
        model_name = request.model_name
        file_path = request.file_path
        log_deployment_event(f"Saving model {model_name}...", log_level="info")
        
        if framework_api.save_model(model_name, file_path):
            log_deployment_event(f"Model {model_name} saved successfully.", log_level="info")
            return {"message": f"Model {model_name} saved successfully"}
        else:
            log_deployment_event(f"Failed to save model {model_name}.", log_level="error")
            raise HTTPException(status_code=500, detail=f"Error saving model {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Train a Model
@router.post("/model/train")
def train_model_endpoint(request: TrainModelRequest):
    """
    API endpoint to trigger the training of a model using the provided data and parameters.

    Parameters:
    - request: Contains model_name (str), model_data_path (str), model_params (dict).

    Returns:
    - JSON response confirming the training process started.
    """
    model_name = request.model_name
    model_data_path = request.model_data_path
    model_params = request.model_params  
    log_deployment_event(f"Training model {model_name}...", log_level="info")
    
    try:
        # Retrieve the model class from FrameworkAPI and trigger the training process
        model = framework_api.framework.models.get(model_name)

        if not model:
            log_deployment_event(f"Model {model_name} not found in the framework.", log_level="error")
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in the framework.")
        
        # Pass model_params and let each model handle its specific training needs
        training_result = model.train(model_data_path, **model_params)
        if "error" in training_result:
            log_deployment_event(f"Error training model {model_name}: {training_result['error']}", log_level="error")
            raise HTTPException(status_code=500, detail=f"Error training model {model_name}: {training_result['error']}")
        else:
            log_deployment_event(f"Model {model_name} trained successfully.", log_level="info")
            return {
                "status": "Success", 
                "message": f"Model {model_name} trained successfully.", 
                "result": training_result
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training for model {model_name}: {str(e)}")

# Evaluate a Model
@router.post("/model/evaluate")
def evaluate_model_endpoint(request: EvaluateModelRequest):
    """
    Evaluate a model on test data using given metrics.

    Parameters:
    - model_name: The name of the model to evaluate.
    - test_data_path: Path to the test data.
    - metrics: List of evaluation metrics to calculate (e.g., accuracy, precision, etc.).

    Returns:
    - JSON response with evaluation results.
    """
    try:
        model_name = request.model_name
        test_data_path = request.test_data_path
        metrics = request.metrics
        log_deployment_event(f"Evaluating model {model_name}...", log_level="info")
        
        result = evaluate_model(model_name, test_data_path, metrics)
        if "error" in result:
            log_deployment_event(f"Error evaluating model {model_name}: {result['error']}", log_level="error")
            raise HTTPException(status_code=500, detail=f"Error evaluating model {model_name}: {result['error']}")
        else:
            log_deployment_event(f"Model {model_name} evaluated successfully.", log_level="info")
            return {"status": "Success", "message": f"Model {model_name} evaluated successfully.", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model {model_name}: {str(e)}")


# Update a Model on a Device
@router.post("/model/update")
def update_model_endpoint(request: UpdateModelRequest):
    """
    Update an already deployed model on the target device.

    Parameters:
    - model_name: The name of the model to update.
    - device_ip: The IP address of the target device.
    - deployment_path: The path where the model is deployed.
    - model_format: The format of the model (e.g., ONNX, TensorFlow, etc.).

    Returns:
    - JSON response confirming that the model update was successful.
    """
    try:
        model_name = request.model_name
        device_ip = request.device_ip
        deployment_path = request.deployment_path
        model_format = request.model_format
        log_deployment_event(f"Updating model {model_name} on device {device_ip}...", log_level="info")
        
        success = update_model_on_device(model_name, device_ip, deployment_path, model_format)

        if success:
            log_deployment_event(f"Model {model_name} updated successfully on device {device_ip}.", log_level="info")
            return {"status": "Success", "message": f"Model {model_name} updated successfully on device {device_ip}."}
        else:
            log_deployment_event(f"Failed to update model {model_name} on device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Model update failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model {model_name} on device {device_ip}: {str(e)}")


# Example: Rollback a Model on a Device
@router.post("/model/rollback")
def rollback_model_endpoint(request: RollbackModelRequest):
    """
    Rollback a model to a previous version on the target device.

    Parameters:
    - device_ip: The IP address of the target device.
    - backup_path: Path to the backup model.
    - deployment_path: Path where the current model is deployed.
    - model_format: The format of the model (e.g., ONNX, TensorFlow, etc.).
    - username: The SSH login username for the target device.

    Returns:
    - JSON response confirming that the rollback was successful.
    """
    try:
        device_ip = request.device_ip
        backup_path = request.backup_path
        deployment_path = request.deployment_path
        model_format = request.model_format
        log_deployment_event(f"Rolling back model on device {device_ip}...", log_level="info")
        
        success = rollback_model_on_device(device_ip, backup_path, deployment_path, model_format)

        if success:
            log_deployment_event(f"Model rollback successful on device {device_ip}.", log_level="info")
            return {"status": "Success", "message": f"Model rollback successful on device {device_ip}."}
        else:
            log_deployment_event(f"Failed to rollback model on device {device_ip}.", log_level="error")
            raise HTTPException(status_code=500, detail="Model rollback failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model rollback on device {device_ip}: {str(e)}")