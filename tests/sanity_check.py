import pandas as pd
from backend.app.schemas import FlightPredictionRequest
from backend.app.services.prediction_service import predict_fare

def run_sanity():
    test_cases = [
        # Target calibration scenario: Amritsar -> Srinagar (253 km, Indigo, Economy, non-stop, 0.75h, 15d)
        {"orig": "ATQ", "dest": "SXR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 0.75, "days": 15, "note": "Target Calibrated Route"},
        {"orig": "ATQ", "dest": "SXR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 3.0, "days": 15, "note": "Unrealistic 3h Non-stop"},
        # Trunk routes
        {"orig": "DEL", "dest": "BOM", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 2.25, "days": 15, "note": "Delhi-Mumbai Trunk"},
        {"orig": "DEL", "dest": "BLR", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 2.75, "days": 15, "note": "Delhi-Bangalore Trunk"},
        {"orig": "BOM", "dest": "BLR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 1.75, "days": 15, "note": "Mumbai-Bangalore Trunk"},
        # Class sensitivity: Economy vs Business on DEL -> BOM
        {"orig": "DEL", "dest": "BOM", "airline": "Air India", "cls": "Business", "stops": "zero", "dur": 2.25, "days": 15, "note": "DEL-BOM Business"},
        # Regional / OOD generalization routes
        {"orig": "IXC", "dest": "BLR", "airline": "Indigo", "cls": "Economy", "stops": "one", "dur": 5.0, "days": 15, "note": "Chandigarh-Bangalore"},
        {"orig": "IXL", "dest": "DEL", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 1.25, "days": 15, "note": "Leh-Delhi Short"},
        {"orig": "DEL", "dest": "COK", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 3.25, "days": 15, "note": "Delhi-Kochi Long"},
        {"orig": "GAU", "dest": "DEL", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 2.5, "days": 15, "note": "Guwahati-Delhi East"},
        {"orig": "AMD", "dest": "COK", "airline": "Indigo", "cls": "Economy", "stops": "one", "dur": 4.5, "days": 15, "note": "Ahmedabad-Kochi"},
        # Booking window sensitivity
        {"orig": "DEL", "dest": "BOM", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 2.25, "days": 30, "note": "DEL-BOM 30d Early"},
        {"orig": "DEL", "dest": "BOM", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 2.25, "days": 2, "note": "DEL-BOM 2d Urgent"},
    ]

    print("\n| Route | Airline | Class | Stops | Dur | Days | Distance | Predicted | OOD? | Scenario |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tc in test_cases:
        req = FlightPredictionRequest(
            source_iata=tc["orig"],
            destination_iata=tc["dest"],
            airline=tc["airline"],
            class_type=tc["cls"],
            stops=tc["stops"],
            duration=tc["dur"],
            days_left=tc["days"],
            departure_time="Morning",
            arrival_time="Evening",
        )
        res = predict_fare(req)
        price = res["predicted_price"]
        dist = res["distance_km"]
        ood = "Yes" if res["out_of_training_distribution"] else "No"
        print(f"| {tc['orig']} -> {tc['dest']} | {tc['airline']} | {tc['cls']} | {tc['stops']} | {tc['dur']}h | {tc['days']}d | {dist} km | INR {price:,.2f} | {ood} | {tc['note']} |")

if __name__ == "__main__":
    run_sanity()
