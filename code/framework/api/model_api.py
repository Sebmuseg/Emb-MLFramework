from fastapi import APIRouter, HTTPException
from core.framework_api import FrameworkAPI
from pydantic import BaseModel
from utils.model_utils import train_model, evaluate_model, update_model_on_device, rollback_model_on_device

router = APIRouter()
framework_api = FrameworkAPI()

# Create a Pydantic model for validating input
class TrainModelRequest(BaseModel):
    model_name: str
    model_data_path: str
    model_params: dict

@router.post("/load_model")
async def load_model(model_name: str, model_path: str, framework_type: str):
    try:
        framework_api.load_model(model_name, model_path, framework_type)
        return {"message": f"Model {model_name} loaded successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/list_models")
async def list_models():
    models = framework_api.list_models()
    return {"models": models}

@router.post("/predict")
async def predict(model_name: str, input_data: dict):
    try:
        result = framework_api.predict(model_name, input_data)
        return {"prediction": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/remove_model")
async def remove_model(model_name: str):
    try:
        framework_api.remove_model(model_name)
        return {"message": f"Model {model_name} removed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/save_model")
async def save_model(model_name: str, file_path: str):
    try:
        framework_api.save_model(model_name, file_path)
        return {"message": f"Model {model_name} saved successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Train a Model
@router.post("/model/train")
def train_model_endpoint(request: TrainModelRequest):
    """
    Endpoint to trigger the training of a model using the provided data and parameters.

    Parameters:
    - request: Contains model_name (str), model_data_path (str), model_params (dict).

    Returns:
    - JSON response confirming the training process started.
    """
    model_name = request.model_name
    model_data_path = request.model_data_path
    model_params = request.model_params
    
    try:
        # Retrieve the model class from FrameworkAPI and trigger the training process
        model = framework_api.framework.models.get(model_name)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in the framework.")
        
        # Assuming each model class has a `train` method that handles the training logic
        training_result = model.train(model_data_path, model_params)
        
        return {
            "status": "Success", 
            "message": f"Training started for model {model_name}.", 
            "result": training_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training for model {model_name}: {str(e)}")

# Example: Evaluate a Model
@router.post("/model/evaluate")
def evaluate_model_endpoint(model_name: str, test_data_path: str, metrics: list):
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
        result = evaluate_model(model_name, test_data_path, metrics)
        
        return {"status": "Success", "message": f"Model {model_name} evaluated successfully.", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating model {model_name}: {str(e)}")


# Example: Update a Model on a Device
@router.post("/model/update")
def update_model_endpoint(model_name: str, device_ip: str, deployment_path: str, model_format: str):
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
        success = update_model_on_device(model_name, device_ip, deployment_path, model_format)

        if success:
            return {"status": "Success", "message": f"Model {model_name} updated successfully on device {device_ip}."}
        else:
            raise HTTPException(status_code=500, detail="Model update failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating model {model_name} on device {device_ip}: {str(e)}")


# Example: Rollback a Model on a Device
@router.post("/model/rollback")
def rollback_model_endpoint(device_ip: str, backup_path: str, deployment_path: str, model_format: str):
    """
    Rollback a model to a previous version on the target device.

    Parameters:
    - device_ip: The IP address of the target device.
    - backup_path: Path to the backup model.
    - deployment_path: Path where the current model is deployed.
    - model_format: The format of the model (e.g., ONNX, TensorFlow, etc.).

    Returns:
    - JSON response confirming that the rollback was successful.
    """
    try:
        success = rollback_model_on_device(device_ip, backup_path, deployment_path, model_format)

        if success:
            return {"status": "Success", "message": f"Model rollback successful on device {device_ip}."}
        else:
            raise HTTPException(status_code=500, detail="Model rollback failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model rollback on device {device_ip}: {str(e)}")