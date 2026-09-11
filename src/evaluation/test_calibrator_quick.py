"""Quick test of calibrator predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.market_calibrator import MarketCalibrator

c = MarketCalibrator.load("models/market_calibrator.pkl")
tests = [
    ("DEL-BLR Eco 15d", 4888.0, 1708.8, 15, "Economy"),
    ("BLR-DEL Eco 15d", 7488.0, 1708.8, 15, "Economy"),
    ("DEL-HYD Eco 15d", 2919.0, 1253.0, 15, "Economy"),
    ("DEL-BOM Eco 15d", 4200.0, 1148.2, 15, "Economy"),
    ("BOM-BLR Eco 15d", 5301.0, 842.1, 15, "Economy"),
    ("DEL-BLR Bus 15d", 16000.0, 1708.8, 15, "Business"),
    ("DEL-BLR Eco 7d", 6500.0, 1708.8, 7, "Economy"),
    ("DEL-BLR Eco 30d", 3800.0, 1708.8, 30, "Economy"),
]
print(f"Status: {c.validation_status}")
print(f"Best model: {c._best_model_name}")
print(f"LORO MAE: INR {c.metrics.get('mae_after', 'N/A')}")
print()
for label, hp, dist, days, cls in tests:
    pred = c.predict(hp, dist, days, cls)
    print(f"{label}: hist={hp:.0f} -> calibrated={pred:.0f}")
