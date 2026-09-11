from __future__ import annotations

import pandas as pd
from backend.app.schemas import FlightPredictionRequest
from backend.app.services.prediction_service import predict_fare


def run_sanity():
    test_cases = [
        # 1. ATQ -> SXR (Amritsar to Srinagar, ~253 km)
        {"orig": "ATQ", "dest": "SXR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 0.75, "days": 15},
        {"orig": "ATQ", "dest": "SXR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 0.75, "days": 2},
        {"orig": "ATQ", "dest": "SXR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 0.75, "days": 30},
        # 2. DEL -> BLR (Delhi to Bangalore, ~1,708 km)
        {"orig": "DEL", "dest": "BLR", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 2.75, "days": 15},
        {"orig": "DEL", "dest": "BLR", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 2.75, "days": 7},
        {"orig": "DEL", "dest": "BLR", "airline": "Air India", "cls": "Business", "stops": "zero", "dur": 2.75, "days": 15},
        # 3. DEL -> BOM (Delhi to Mumbai, ~1,137 km)
        {"orig": "DEL", "dest": "BOM", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 2.25, "days": 15},
        {"orig": "DEL", "dest": "BOM", "airline": "Air India", "cls": "Business", "stops": "zero", "dur": 2.25, "days": 15},
        # 4. BOM -> BLR (Mumbai to Bangalore, ~834 km)
        {"orig": "BOM", "dest": "BLR", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 1.75, "days": 15},
        # 5. IXC -> BLR (Chandigarh to Bangalore, ~1,945 km)
        {"orig": "IXC", "dest": "BLR", "airline": "Indigo", "cls": "Economy", "stops": "one", "dur": 5.0, "days": 15},
        # 6. IXL -> DEL (Leh to Delhi, ~621 km)
        {"orig": "IXL", "dest": "DEL", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 1.25, "days": 15},
        # 7. DEL -> COK (Delhi to Kochi, ~2,047 km)
        {"orig": "DEL", "dest": "COK", "airline": "Air India", "cls": "Economy", "stops": "zero", "dur": 3.25, "days": 15},
        # 8. GAU -> DEL (Guwahati to Delhi, ~1,455 km)
        {"orig": "GAU", "dest": "DEL", "airline": "Indigo", "cls": "Economy", "stops": "zero", "dur": 2.5, "days": 15},
        # 9. AMD -> COK (Ahmedabad to Kochi, ~1,491 km)
        {"orig": "AMD", "dest": "COK", "airline": "Indigo", "cls": "Economy", "stops": "one", "dur": 4.5, "days": 15},
    ]

    print("\n==========================================================================================================================")
    print("SKYCAST AUTOMATED DIAGNOSTIC EVALUATOR — PROBLEM ROUTES & NEAREST HISTORICAL COMPARABLES")
    print("==========================================================================================================================")
    header = f"{'Route':<11} | {'Dist':<8} | {'Airline':<10} | {'Class':<8} | {'Stops':<5} | {'Dur':<5} | {'Days':<4} | {'Prediction':<11} | {'Hist Median':<11} | {'Hist Mean':<10} | {'Hist Range':<15} | {'|Diff|':<8} | {'Reliability'}"
    print(header)
    print("-" * len(header))

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
        p = res["predicted_price"]
        dist = res["distance_km"]
        comp = res.get("historical_comparables", {})
        med = comp.get("median")
        mean_v = comp.get("mean")
        min_v = comp.get("min")
        max_v = comp.get("max")
        diff = abs(round(p - med, 2)) if med is not None else None

        route_str = f"{tc['orig']}->{tc['dest']}"
        dist_str = f"{dist:.0f}km"
        dur_str = f"{tc['dur']}h"
        days_str = f"{tc['days']}d"
        pred_str = f"INR {p:,.0f}"
        med_str = f"INR {med:,.0f}" if med else "N/A"
        mean_str = f"INR {mean_v:,.0f}" if mean_v else "N/A"
        range_str = f"{min_v:,.0f}-{max_v:,.0f}" if min_v and max_v else "N/A"
        diff_str = f"INR {diff:,.0f}" if diff is not None else "N/A"
        tier_str = res.get("reliability_tier", "Good")

        print(
            f"{route_str:<11} | {dist_str:<8} | {tc['airline']:<10} | {tc['cls']:<8} | {tc['stops']:<5} | "
            f"{dur_str:<5} | {days_str:<4} | {pred_str:<11} | {med_str:<11} | {mean_str:<10} | {range_str:<15} | "
            f"{diff_str:<8} | {tier_str}"
        )


if __name__ == "__main__":
    run_sanity()
