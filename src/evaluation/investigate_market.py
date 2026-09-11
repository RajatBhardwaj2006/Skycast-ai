import json
import numpy as np
import pandas as pd
from backend.app.services.model_service import load_pipeline
from src.features.feature_engineering import add_engineered_features
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def run_test():
    with open("data/reference/market_validation_2026.json", encoding="utf-8") as f:
        obs = json.load(f)

    from src.geo.locations import default_location_service
    loc_service = default_location_service()

    router = load_pipeline()
    records = []
    for item in obs:
        orig = loc_service.resolve(item["source_city"])
        dest = loc_service.resolve(item["destination_city"])
        dist = loc_service.distance_between(orig, dest)
        row = {
            "airline": item["airline"],
            "departure_time": "Morning",
            "arrival_time": "Evening",
            "stops": item["stops"],
            "class": item["cabin"],
            "source_city": orig.city,
            "destination_city": dest.city,
            "duration": item["duration"],
            "days_left": item["days_until_departure"],
            "distance_km": dist,
            "source_lat": orig.latitude,
            "source_lon": orig.longitude,
            "destination_lat": dest.latitude,
            "destination_lon": dest.longitude,
        }
        df = add_engineered_features(pd.DataFrame([row]))
        raw_p = router.predict(df)[0] * 1.01388 # with Duan smearing
        records.append({
            "route": item["route"],
            "airline": item["airline"],
            "cabin": item["cabin"],
            "days": item["days_until_departure"],
            "dist": dist,
            "hist_pred": raw_p,
            "market_fare": item["current_fare"],
        })

    mdf = pd.DataFrame(records)

    # Features for continuous calibration
    X = np.column_stack([
        np.log(mdf["hist_pred"]),
        mdf["dist"] / 1000.0,
        mdf["days"] / 15.0,
        (mdf["cabin"] == "Business").astype(float)
    ])
    y = np.log(mdf["market_fare"])

    calibrator = Ridge(alpha=0.5)
    calibrator.fit(X, y)
    pred_log = calibrator.predict(X)
    pred_cal = np.exp(pred_log)

    mdf["calibrated"] = np.round(pred_cal, 0)
    mdf["error"] = mdf["calibrated"] - mdf["market_fare"]

    print("\n=== CALIBRATOR EVALUATION ===")
    print(f"MAE before calibration: INR {mean_absolute_error(mdf['market_fare'], mdf['hist_pred']):.2f}")
    print(f"MAE after calibration:  INR {mean_absolute_error(mdf['market_fare'], mdf['calibrated']):.2f}")
    print(f"R2 after calibration:   {r2_score(mdf['market_fare'], mdf['calibrated']):.4f}")

    print("\n=== ROUTE PREDICTIONS AT 15 DAYS VS MARKET FARE ===")
    day15 = mdf[mdf["days"] == 15]
    for _, r in day15.iterrows():
        route = r['route']
        air = r['airline']
        cab = r['cabin']
        hp = r['hist_pred']
        cp = r['calibrated']
        mf = r['market_fare']
        err = r['error']
        print(f"{route:<10} | {air:<17} | {cab:<8} | Hist: INR {hp:<6.0f} | Calib: INR {cp:<6.0f} | Market: INR {mf:<6.0f} | Diff: INR {err:<5.0f}")

if __name__ == "__main__":
    run_test()
