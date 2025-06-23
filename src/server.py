import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.predict_water import predict_water_stress  # predict_water_stress fonksiyonu buradan geliyor

#uvicorn src.server:app --reload
app = FastAPI()

class PredictionRequest(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [lon_min, lat_min, lon_max, lat_max]")
    weather_forecast: Dict[str, Any]
    current_stress: Optional[float] = None
    days_ahead: Optional[int] = 7

@app.post("/predict")
def predict(request: PredictionRequest):
    result = predict_water_stress(
        bbox=request.bbox,
        weather_forecast=request.weather_forecast,
        current_stress=request.current_stress,
        days_ahead=request.days_ahead
    )
    if result is None:
        return {"error": "Prediction failed. Model may be missing or inputs are invalid."}
    return result
