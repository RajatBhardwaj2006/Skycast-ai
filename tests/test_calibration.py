import pytest
from backend.app.schemas import FlightPredictionRequest
from backend.app.services.prediction_service import predict_fare
from src.evaluation.comparables import find_nearest_comparables
from src.models.market_calibrator import MarketCalibrator, DUAN_SMEARING_FACTOR


def test_directional_comparables_separation():
    # Query DEL -> BLR
    del_blr = find_nearest_comparables({
        "source_city": "Delhi",
        "destination_city": "Bangalore",
        "class": "Economy",
        "airline": "Indigo",
        "stops": "zero",
        "duration": 3.25,
        "days_left": 15,
        "distance_km": 1708.8,
    }, k=6)

    # Query BLR -> DEL
    blr_del = find_nearest_comparables({
        "source_city": "Bangalore",
        "destination_city": "Delhi",
        "class": "Economy",
        "airline": "Indigo",
        "stops": "zero",
        "duration": 2.75,
        "days_left": 15,
        "distance_km": 1708.8,
    }, k=6)

    assert del_blr["count"] > 0
    assert blr_del["count"] > 0
    assert del_blr["median"] == 4500.0
    assert blr_del["median"] == 7488.0
    assert del_blr["route_match"] == "exact_directional"
    assert blr_del["route_match"] == "exact_directional"
    # Ensure no BLR -> DEL samples in DEL -> BLR result
    for sample in del_blr["comparables"]:
        assert sample["source_city"] == "Delhi"
        assert sample["destination_city"] == "Bangalore"
        assert sample["match_type"] == "exact_route"


def test_duan_smearing_factor_validity():
    assert 1.01 <= DUAN_SMEARING_FACTOR <= 1.05


def test_market_calibrator_inference():
    calibrator = MarketCalibrator.load("models/market_calibrator.pkl")
    assert calibrator.is_fitted
    pred = calibrator.predict(
        historical_price=4888.0,
        distance_km=1708.8,
        days_left=15,
        class_type="Economy",
    )
    # Calibrated price should reflect current 2026 market range (~7,600 - 8,900)
    assert 7500.0 <= pred <= 9500.0


def test_predict_fare_dual_presentation():
    req = FlightPredictionRequest(
        source_iata="DEL",
        destination_iata="BLR",
        airline="Indigo",
        stops="zero",
        class_type="Economy",
        duration=3.25,
        days_left=15,
        departure_time="Morning",
        arrival_time="Evening",
    )
    res = predict_fare(req)
    assert res["predicted_price"] >= 7500.0
    assert "historical_baseline" in res
    assert res["historical_baseline"]["raw_model_price"] > 0
    assert "2022" in res["historical_baseline"]["training_era"]
    assert "market_calibration" in res
    assert "2026" in res["market_calibration"]["reference_era"]
    assert res["market_calibration"]["macro_adjustment_inr"] > 0
    assert res["historical_comparables"]["median"] == 4500.0
