"""Build, validate, and save the new MarketCalibrator with comprehensive LORO report."""
import json
import sys
from pathlib import Path

# Ensure project root on path
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.models.market_calibrator import MarketCalibrator, DUAN_SMEARING_FACTOR
from src.geo.locations import default_location_service

def main():
    from backend.app.services.model_service import load_pipeline

    val_path = PROJECT / "data" / "reference" / "market_validation_2026.json"
    obs = json.loads(val_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(obs)} market observations")

    router = load_pipeline()
    loc_service = default_location_service()

    calibrator = MarketCalibrator()
    calibrator.fit_from_dataset(obs, router, loc_service)

    # Save
    pkl_path = PROJECT / "models" / "market_calibrator.pkl"
    calibrator.save(pkl_path)
    print(f"Saved calibrator to {pkl_path}")

    # Print comprehensive report
    m = calibrator.metrics
    print("\n" + "="*70)
    print("MARKET CALIBRATOR — LORO CROSS-VALIDATED REPORT")
    print("="*70)
    print(f"Observations: {m['observation_count']}")
    print(f"Unique routes: {m['unique_routes']}")
    print(f"Unique corridors: {m['unique_corridors']}")
    print(f"Validation status: {m['validation_status']}")
    print(f"Best model: {m['best_model']}")
    print(f"Global inflation factor: {m['global_inflation_factor']:.3f}x")
    print()
    print("--- MAE Comparison ---")
    print(f"  No calibration:    INR {m['mae_no_calibration']:,.0f}")
    print(f"  Global inflation:  INR {m['mae_global_inflation']:,.0f}")
    print(f"  Best LORO MAE:     INR {m['mae_after']:,.0f}")
    print()

    print("--- Model Comparison (all LORO out-of-fold) ---")
    for name, info in m.get("model_comparison", {}).items():
        line = f"  {name:<20s} LORO-MAE: INR {info['loro_mae']:>7,.0f}"
        if "loro_r2" in info:
            line += f"  R²={info['loro_r2']:.4f}"
        if "loco_mae" in info:
            line += f"  LOCO-MAE: INR {info['loco_mae']:>7,.0f}"
        print(line)
    print()

    print("--- Per-Route Validation Table (out-of-fold predictions) ---")
    print(f"{'Route':<15s} {'Hist ML':>10s} {'Calib OOF':>10s} {'Market':>10s} {'Error%':>8s} {'OOF?':>5s}")
    print("-" * 60)
    for row in m.get("route_validation_table", []):
        print(f"{row['route']:<15s} {row['historical_ml']:>10,.0f} {row['calibrated_oof']:>10,.0f} {row['market_fare']:>10,.0f} {row['error_pct']:>7.1f}% {'Yes' if row['out_of_fold'] else 'No':>5s}")

    print()
    print(f"FINAL: validation_status = {calibrator.validation_status}")
    print(f"FINAL: best_model = {calibrator._best_model_name}")

    # Also save report as JSON artifact
    report_path = PROJECT / "models" / "calibration_report.json"
    report_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
