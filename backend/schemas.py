"""
Pydantic Validation Schemas for the FastAPI Backend
"""

from pydantic import BaseModel, Field

class FlightPredictionRequest(BaseModel):
    source: str = Field(..., example="DEL")
    destination: str = Field(..., example="BOM")
    airline: str = Field(..., example="IndiGo")
    duration_mins: int = Field(..., example=120)
    total_stops: int = Field(..., example=0)
    booking_window: int = Field(..., example=15)
    dep_time: str = Field(..., example="2026-09-10T08:00:00")
    travel_date: str = Field(..., example="2026-09-10")

class FlightPredictionResponse(BaseModel):
    estimated_fare: float
    fare_range_min: float
    fare_range_max: float
    confidence: str
    price_trend: str