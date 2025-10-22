# Save this code in a new file: fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional

# --- Configuration ---
MODEL_PATH = 'isro_launch_model_v2.pkl'

# --- Load Model (Done once on startup) ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file {MODEL_PATH} not found. Ensure you run main.py.")
    model = None
    
# --- Pydantic Schema for Input Data ---
# This defines the expected structure of the JSON payload for prediction
class LaunchFeatures(BaseModel):
    launch_vehicle: str
    orbit_type: str
    payload_weight_kg: float
    temperature_C: float
    wind_speed_kmh: float
    humidity_percent: float
    launch_window: str
    mission_type: str
    system_health_index: float
    vehicle_success_rate: float
    launch_site: str
    # New Advanced Features (Project 3)
    launch_month: int
    launch_quarter: int
    mission_complexity_score: float
    launch_window_risk_index: float

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ISRO Launch Risk API",
    description="Real-time prediction service for ISRO launch success probability using Stacking Ensemble Model.",
    version="1.0.0"
)

# --- Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "Model API is Online", "model": "isro_launch_model_v2.pkl"}

@app.post("/predict", tags=["Prediction"])
def predict_launch_success(features: LaunchFeatures):
    """
    Accepts launch parameters and returns the predicted success probability.
    """
    if model is None:
        return {"error": "Model failed to load. Please check model path."}

    # Convert the input Pydantic model to a pandas DataFrame for the pipeline
    input_data = features.model_dump()
    input_df = pd.DataFrame([input_data])
    
    # Prediction: model.predict_proba returns [[Prob_Failure, Prob_Success]]
    try:
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        
        return {
            "prediction_status": "Success",
            "launch_scenario": f"{input_data['launch_vehicle']} to {input_data['orbit_type']}",
            "predicted_success_probability": float(prediction_proba),
            "risk_level": "High" if prediction_proba < 0.8 else "Low"
        }
    except Exception as e:
        return {"prediction_status": "Failure", "detail": str(e)}

# Example of how to structure the input JSON (for documentation)
# NOTE: The actual complexity score and risk index must be pre-calculated
# {
#   "launch_vehicle": "GSLV Mk III",
#   "orbit_type": "GTO",
#   "payload_weight_kg": 4000.0,
#   "temperature_C": 30.0,
#   "wind_speed_kmh": 15.0,
#   "humidity_percent": 60.0,
#   "launch_window": "Evening",
#   "mission_type": "Communication",
#   "system_health_index": 70.0,
#   "vehicle_success_rate": 0.93,
#   "launch_site": "Satish Dhawan Space Centre",
#   "launch_month": 3,
#   "launch_quarter": 1,
#   "mission_complexity_score": 11.0, 
#   "launch_window_risk_index": 1.5
# }