from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_artifact_available" in response.json()


def test_model_metadata_and_artifact_endpoints():
    info = client.get("/model-info")
    metrics = client.get("/metrics")
    importance = client.get("/feature-importance")
    assert info.status_code == 200
    assert info.json()["model_name"] == "Random Forest (tuned)"
    assert metrics.status_code == 200
    assert metrics.json()["mae"] > 0
    assert importance.status_code == 200
    assert importance.json()["features"]
    validation = client.get("/validation")
    assert validation.status_code == 200
    assert validation.json()["route_group_audit"]["metrics"]["mae"] > 0


def test_location_search_leh():
    response = client.get("/locations/search", params={"q": "Le"})
    assert response.status_code == 200
    cities = [item["city"] for item in response.json()["results"]]
    assert any(name == "Leh" for name in cities)


def test_location_search_unknown():
    response = client.get("/locations/search", params={"q": "zzzznotacity"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["message"] == "Location not found."


def test_route_distance_leh_to_delhi():
    response = client.get("/route-distance", params={"source_iata": "IXL", "destination_iata": "DEL"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"]["city"] == "Leh"
    assert body["destination"]["iata"] == "DEL"
    assert 400 < body["distance_km"] < 900


def test_route_distance_rejects_same_airport():
    response = client.get("/route-distance", params={"source_iata": "DEL", "destination_iata": "DEL"})
    assert response.status_code == 400


def test_route_distance_is_symmetric_for_delhi_and_mumbai():
    forward = client.get("/route-distance", params={"source_iata": "DEL", "destination_iata": "BOM"})
    reverse = client.get("/route-distance", params={"source_iata": "BOM", "destination_iata": "DEL"})
    assert forward.status_code == 200
    assert reverse.status_code == 200
    assert forward.json()["distance_km"] == reverse.json()["distance_km"]


def test_route_distance_delhi_to_bangalore():
    response = client.get("/route-distance", params={"source_iata": "DEL", "destination_iata": "BLR"})
    assert response.status_code == 200
    assert response.json()["source"]["city"] == "Delhi"
    assert response.json()["destination"]["city"] == "Bangalore"
    assert response.json()["distance_km"] > 1000


def test_predict_same_city_rejected():
    payload = {
        "airline": "Air India",
        "source_city": "Delhi",
        "destination_city": "Delhi",
        "departure_time": "Morning",
        "arrival_time": "Evening",
        "stops": "zero",
        "class": "Economy",
        "duration": 2.0,
        "days_left": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "same" in response.json()["detail"].lower()


def test_negative_duration_rejected():
    payload = {
        "airline": "Air India",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "arrival_time": "Evening",
        "stops": "zero",
        "class": "Economy",
        "duration": -1,
        "days_left": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_days_left_rejected():
    payload = {
        "airline": "Air India",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "arrival_time": "Evening",
        "stops": "zero",
        "class": "Economy",
        "duration": 2.0,
        "days_left": -3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_flexible_route_when_model_present():
    from src.utils.paths import PROJECT_ROOT

    if not (PROJECT_ROOT / "models" / "airfare_pipeline.pkl").exists():
        return
    payload = {
        "airline": "Air India",
        "source_iata": "IXL",
        "destination_iata": "DEL",
        "departure_time": "Morning",
        "arrival_time": "Evening",
        "stops": "zero",
        "class": "Economy",
        "duration": 1.5,
        "days_left": 20,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price"] > 0
    assert body["distance_km"] > 0
    assert body["source"]["iata"] == "IXL"
    assert body["out_of_training_distribution"] is True
    assert body["expected_price_range"]["low"] <= body["predicted_price"]
    assert body["expected_price_range"]["high"] >= body["predicted_price"]
    assert body["expected_price_range"]["method"] == "held_out_test_mae"


def test_invalid_stops_rejected():
    payload = {
        "airline": "Air India", "source_city": "Delhi", "destination_city": "Mumbai",
        "departure_time": "Morning", "arrival_time": "Evening", "stops": "several",
        "class": "Economy", "duration": 2.0, "days_left": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "stops" in response.json()["detail"]


def test_booking_window_over_a_year_rejected():
    payload = {
        "airline": "Air India", "source_city": "Delhi", "destination_city": "Mumbai",
        "departure_time": "Morning", "arrival_time": "Evening", "stops": "zero",
        "class": "Economy", "duration": 2.0, "days_left": 366,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
