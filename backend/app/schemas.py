from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class LocationOut(BaseModel):
    city: str
    airport: str
    iata: str
    state: str
    country: str
    latitude: float
    longitude: float


class FlightPredictionRequest(BaseModel):
    airline: str
    source_city: Optional[str] = None
    destination_city: Optional[str] = None
    source_iata: Optional[str] = None
    destination_iata: Optional[str] = None
    departure_time: str
    arrival_time: str
    stops: str
    class_type: str = Field(..., alias="class")
    duration: float
    days_left: int
    seat_type: Optional[str] = None

    model_config = {"populate_by_name": True}

    @field_validator("duration")
    @classmethod
    def duration_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Duration must be greater than 0.")
        if value > 48:
            raise ValueError("Please enter a realistic flight duration.")
        return value

    @field_validator("days_left")
    @classmethod
    def days_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Days left cannot be negative.")
        return value


class BatchPredictionRequest(BaseModel):
    items: list[FlightPredictionRequest]


class FlightPredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR"
    model: str
    confidence_note: str
    source: LocationOut
    destination: LocationOut
    distance_km: float
    fare_band: str
    out_of_training_distribution: bool
    reliability_note: Optional[str] = None
    summary: dict
    feature_importance: list[dict]


class MetricsResponse(BaseModel):
    best_model: str
    mae: float
    mse: float
    rmse: float
    r2: float
    comparison: list[dict] = []
