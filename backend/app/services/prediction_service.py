from __future__ import annotations

import pandas as pd
from fastapi import HTTPException

from backend.app.schemas import FlightPredictionRequest, LocationOut
from backend.app.services.model_service import load_importance, load_metadata, load_pipeline
from src.data.clean import AIRLINE_MAP, STOPS_MAP, TIME_MAP
from src.geo.locations import LocationNotFoundError, default_location_service
from src.models.train import CATEGORICAL, GEO_NUMERIC


def _normalize_airline(value: str) -> str:
    return AIRLINE_MAP.get(value.strip().casefold(), value.strip())


def _normalize_time(value: str) -> str:
    return TIME_MAP.get(value.strip().casefold().replace("-", " "), value.strip())


def _normalize_stops(value: str) -> str:
    key = value.strip().casefold().replace("_", " ")
    mapping = {
        **STOPS_MAP,
        "non-stop": "zero",
        "nonstop": "zero",
        "0": "zero",
        "1 stop": "one",
        "2+ stops": "two_or_more",
        "2 stops": "two_or_more",
    }
    return mapping.get(key, STOPS_MAP.get(value.strip().casefold(), value.strip()))


def _fare_band(price: float, terciles: dict) -> str:
    if price <= terciles.get("budget_max", 0):
        return "Budget"
    if price >= terciles.get("high_min", price + 1):
        return "High"
    return "Average"


def _resolve_location(iata: str | None, city: str | None):
    service = default_location_service()
    try:
        if iata:
            return service.get_by_iata(iata)
        if city:
            return service.resolve(city)
    except LocationNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"message": "Location not found.", "hint": "Try another city or airport.", "query": str(exc)},
        ) from exc
    raise HTTPException(status_code=400, detail="Provide a source and destination city or IATA code.")


def predict_fare(payload: FlightPredictionRequest) -> dict:
    origin = _resolve_location(payload.source_iata, payload.source_city)
    destination = _resolve_location(payload.destination_iata, payload.destination_city)
    if origin.iata == destination.iata:
        raise HTTPException(status_code=400, detail="Departure and destination cannot be the same.")

    metadata = load_metadata()
    allowed_airlines = {name.casefold(): name for name in metadata.get("airlines", [])}
    airline = _normalize_airline(payload.airline)
    if airline.casefold() not in allowed_airlines:
        raise HTTPException(status_code=400, detail=f"Unknown airline '{payload.airline}'.")
    airline = allowed_airlines[airline.casefold()]

    class_type = payload.class_type.strip().title()
    if class_type not in metadata.get("classes", [class_type]):
        raise HTTPException(status_code=400, detail=f"Unknown class '{payload.class_type}'.")

    departure_time = _normalize_time(payload.departure_time)
    arrival_time = _normalize_time(payload.arrival_time)
    if departure_time not in metadata.get("departure_times", []):
        raise HTTPException(status_code=400, detail=f"Unknown departure time '{payload.departure_time}'.")
    if arrival_time not in metadata.get("arrival_times", []):
        raise HTTPException(status_code=400, detail=f"Unknown arrival time '{payload.arrival_time}'.")

    stops = _normalize_stops(payload.stops)
    if stops not in metadata.get("stops", []):
        raise HTTPException(status_code=400, detail=f"Unknown stops value '{payload.stops}'.")

    service = default_location_service()
    distance_km = round(service.distance_between(origin, destination), 2)
    training_cities = {city.casefold() for city in metadata.get("training_cities", [])}
    ood = origin.city.casefold() not in training_cities or destination.city.casefold() not in training_cities

    row = {
        "airline": airline,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "stops": stops,
        "class": class_type,
        "source_city": origin.city,
        "destination_city": destination.city,
        "duration": float(payload.duration),
        "days_left": int(payload.days_left),
        "source_lat": origin.latitude,
        "source_lon": origin.longitude,
        "destination_lat": destination.latitude,
        "destination_lon": destination.longitude,
        "distance_km": distance_km,
    }
    frame = pd.DataFrame([row], columns=CATEGORICAL + GEO_NUMERIC)
    pipeline = load_pipeline()
    predicted = float(pipeline.predict(frame)[0])
    predicted = max(0.0, round(predicted, 2))

    terciles = metadata.get("price_terciles", {})
    importance = load_importance().get("features", [])
    reliability = None
    if ood:
        reliability = "Estimate may be less reliable for routes with limited historical training data."

    return {
        "predicted_price": predicted,
        "currency": "INR",
        "model": metadata.get("model_name", "unknown"),
        "confidence_note": "Estimated fare based on historical training data",
        "source": LocationOut(**origin.to_dict()),
        "destination": LocationOut(**destination.to_dict()),
        "distance_km": distance_km,
        "fare_band": _fare_band(predicted, terciles),
        "out_of_training_distribution": ood,
        "reliability_note": reliability,
        "summary": {
            "airline": airline,
            "class": class_type,
            "stops": stops,
            "duration": payload.duration,
            "days_left": payload.days_left,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
        },
        "feature_importance": importance,
    }
