from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.app.schemas import FlightPredictionRequest, LocationOut
from backend.app.services.live_fares import get_live_fares
from backend.app.services.model_service import (
    load_importance,
    load_market_calibrator,
    load_metadata,
    load_metrics,
    load_pipeline,
)
from src.data.clean import AIRLINE_MAP, STOPS_MAP, TIME_MAP
from src.evaluation.comparables import find_nearest_comparables
from src.features.feature_engineering import add_engineered_features
from src.geo.locations import LocationNotFoundError, default_location_service
from src.models.market_calibrator import DUAN_SMEARING_FACTOR

from src.models.train import (
    CATEGORICAL_CORE,
    CATEGORICAL_WITH_CLASS,
    NUMERICAL_CORE,
    AirfareModelRouter,  # Ensures class is available during unpickling
)


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
    if class_type not in metadata.get("classes", ["Economy", "Business"]):
        class_type = "Economy"

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

    # Reliability Tier determination
    if distance_km < 80.0 or distance_km > 3600.0:
        reliability_tier = "Out-of-distribution estimate"
        reliability_note = "Flight distance is outside typical domestic commercial boundaries."
    elif ood:
        reliability_tier = "Limited historical coverage"
        reliability_note = "Valid Indian airport, but this specific route has limited historical training observations."
    else:
        reliability_tier = "Good historical coverage"
        reliability_note = "Frequent scheduled route with high historical training density."

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

    raw_frame = pd.DataFrame([row])
    enriched_frame = add_engineered_features(raw_frame)

    pipeline = load_pipeline()
    raw_hist_price = float(pipeline.predict(enriched_frame)[0])
    # Apply Duan's smearing correction factor to reduce log-retransformation Jensen bias
    smearing_hist_price = raw_hist_price * DUAN_SMEARING_FACTOR
    hist_baseline = max(500.0, round(smearing_hist_price, 2))

    # Market Calibration Layer: continuous econometric transfer mapping 2022 baseline to 2026 market
    calibrator = load_market_calibrator()
    calibrated_market_price = calibrator.predict(
        historical_price=hist_baseline,
        distance_km=distance_km,
        days_left=int(payload.days_left),
        class_type=class_type,
    )
    predicted = max(500.0, round(calibrated_market_price, 2))

    # Calibration error uncertainty band — use LORO-validated MAE (no leakage)
    calib_mae = float(calibrator.metrics.get("mae_after", 1500.0))
    error_band = round(calib_mae, 2)
    validation_status = calibrator.validation_status if hasattr(calibrator, "validation_status") else "experimental"
    best_model_name = getattr(calibrator, "_best_model_name", "unknown")

    terciles = metadata.get("price_terciles", {})
    importance = load_importance().get("features", [])

    # Directional Historical Comparables diagnostic search
    comparables_result = find_nearest_comparables({
        "source_city": origin.city,
        "destination_city": destination.city,
        "class": class_type,
        "airline": airline,
        "stops": stops,
        "duration": float(payload.duration),
        "days_left": int(payload.days_left),
        "distance_km": distance_km,
    }, k=6)

    comp_median = comparables_result.get("median")
    diff_from_calibrated = round(predicted - comp_median, 2) if comp_median is not None else None
    diff_from_historical = round(hist_baseline - comp_median, 2) if comp_median is not None else None

    # Live Fares status query
    live_status = get_live_fares(origin.iata, destination.iata, cabin_class=class_type.upper())

    # Calibration label depends on validation status
    if validation_status == "validated_loro":
        calib_label = "2026 Market Calibrated (LORO-validated)"
        calib_note = "Calibrated against multi-route 2026 market evidence with leave-one-route-out cross-validation."
    else:
        calib_label = "Experimental market calibration"
        calib_note = "Experimental calibration — leave-one-route-out validation did not significantly beat uncalibrated baseline."

    return {
        "predicted_price": predicted,
        "currency": "INR",
        "model": metadata.get("model_name", "AirfareModelRouter") + " + 2026 Market Calibrator",
        "confidence_note": "2026 market-calibrated fare estimate based on cross-route fare evidence and historical ML baselines.",
        "uncertainty": {
            "typical_error_inr": error_band,
            "low": round(max(500.0, predicted - error_band), 2),
            "high": round(predicted + error_band, 2),
            "method": "loro_cross_validated_mae",
            "note": "Leave-one-route-out cross-validated MAE on 2026 market observations.",
        },
        "expected_price_range": {
            "low": round(max(500.0, predicted - error_band), 2),
            "high": round(predicted + error_band, 2),
            "method": "held_out_test_mae",
            "error_band": error_band,
            "note": "Range is the 2026 calibrated prediction ± LORO cross-validated MAE.",
        },
        "historical_baseline": {
            "raw_model_price": round(raw_hist_price, 2),
            "smearing_corrected_price": hist_baseline,
            "training_era": "2022 Historical Baseline",
            "note": "Raw ML inference trained on 2022 DGCA/Kaggle flight records.",
        },
        "market_calibration": {
            "calibrated_market_fare": predicted,
            "reference_era": "2026 Live Market",
            "macro_adjustment_inr": round(predicted - hist_baseline, 2),
            "inflation_multiplier": round(predicted / max(1.0, hist_baseline), 3),
            "method": f"LORO-validated {best_model_name} calibration",
            "validation_status": validation_status,
            "calibration_label": calib_label,
            "note": calib_note,
        },
        "source": LocationOut(**origin.to_dict()),
        "destination": LocationOut(**destination.to_dict()),
        "distance_km": distance_km,
        "fare_band": _fare_band(predicted, terciles),
        "out_of_training_distribution": ood,
        "reliability_tier": reliability_tier,
        "reliability_note": reliability_note,
        "historical_comparables": {
            "count": comparables_result.get("count", 0),
            "median": comp_median,
            "mean": comparables_result.get("mean"),
            "min": comparables_result.get("min"),
            "max": comparables_result.get("max"),
            "difference_from_median": diff_from_calibrated,
            "difference_from_historical_median": diff_from_historical,
            "route_match": comparables_result.get("route_match", "corridor_fallback"),
            "samples": comparables_result.get("comparables", []),
        },
        "live_fares": live_status,
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

