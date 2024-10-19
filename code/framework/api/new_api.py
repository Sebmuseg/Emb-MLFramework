from fastapi import FastAPI
from .monitoring_api import router as monitoring_router
from .model_api import router as model_router
from .logging_api import router as logging_router
from .deployment_api import router as deployment_router

# Create an instance of FastAPI
app = FastAPI()

# Register routers
app.include_router(deployment_router, prefix="/deploy")
app.include_router(monitoring_router, prefix="/monitoring")
app.include_router(model_router, prefix="/models")
app.include_router(logging_router, prefix="/logging")

@app.get("/")
def read_root():
    return {"message": "Welcome to FeatherML API"}