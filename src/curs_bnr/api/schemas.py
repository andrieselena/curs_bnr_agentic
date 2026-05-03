from typing import Optional
from pydantic import BaseModel

class RateResponse(BaseModel):
    """Schema pentru un curs istoric."""
    id: int
    date: str
    value: float

class MetricResponse(BaseModel):
    """Schema pentru ultimele metrici de performanță ale modelului."""
    run_id: int
    model_name: str
    version: str
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    timestamp: str

class RunResponse(BaseModel):
    """Schema pentru o rulare de antrenament."""
    id: int
    timestamp: str
    model_name: str
    version: str
    hyperparameters: Optional[str]

class ForecastResponse(BaseModel):
    """Schema pentru o prognoză specifică."""
    id: int
    run_id: int
    forecast_date: str
    forecast_value: float

class MessageResponse(BaseModel):
    """Schema pentru mesaje text simple din partea API-ului."""
    message: str
