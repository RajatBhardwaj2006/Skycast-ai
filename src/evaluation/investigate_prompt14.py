import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.evaluation.metrics import regression_metrics
from src.features.preprocess import build_preprocessor
from src.utils.config import load_config, resolve_path

def run_investigation():
    print("=== 1. LOADING PROCESSED DATASET ===")
    df = pd.read_csv("data/processed/processed_flights.csv", low_memory=False)
    print(f"Total records: {len(df):,}")
    
    # Check ATQ -> SXR comparable records in training data
    print("\n=== 4. SIMILAR HISTORICAL RECORDS FOR ATQ -> SXR (Distance ~253km) ===")
    # Comparable: short distance (< 500km), Indigo, Economy, non-stop (zero stops)
    comparables = df[
        (df["distance_km"] <= 550) & 
        (df["airline"] == "Indigo") & 
        (df["class"] == "Economy") & 
        (df["stops"] == "zero")
    ]
    print(f"Found {len(comparables)} comparable records (distance <= 550km, Indigo, Economy, zero stops):")
    print(comparables.groupby(["source_city", "destination_city"])[["distance_km", "duration", "price"]].agg(["count", "mean", "min", "max"]))
    
    # 15 days left specifically
    comp_15d = comparables[(comparables["days_left"] >= 12) & (comparables["days_left"] <= 18)]
    print(f"\n15-days booking window comparable records ({len(comp_15d)} records):")
    print(comp_15d[["source_city", "destination_city", "distance_km", "duration", "days_left", "price"]].head(15))
    print("Price statistics for 15-day window:", comp_15d["price"].describe())

    print("\n=== 2. CLASS FEATURE EXPERIMENT ===")
    # Test Model A (Current), Model B (No class), Model C (Current + interactions)
    random_state = 42
    test_size = 0.2
    
    CATEGORICAL = ["airline", "departure_time", "arrival_time", "stops", "class", "source_city", "destination_city"]
    GEO_NUMERIC = ["duration", "days_left", "distance_km", "source_lat", "source_lon", "destination_lat", "destination_lon"]
    
    # Model A: Current features (with class)
    cols_a = CATEGORICAL + GEO_NUMERIC
    cat_a = CATEGORICAL
    num_a = GEO_NUMERIC
    
    # Model B: Remove class
    cols_b = [c for c in cols_a if c != "class"]
    cat_b = [c for c in cat_a if c != "class"]
    num_b = num_a
    
    # Train/test split (random)
    X_tr, X_te, y_tr, y_te = train_test_split(df[cols_a], df["price"], test_size=test_size, random_state=random_state)
    
    print("\nTraining Model A (With Class) using HistGradientBoosting for fast fair comparison...")
    pipe_a = Pipeline([("prep", build_preprocessor(cat_a, num_a)), ("model", HistGradientBoostingRegressor(max_depth=10, random_state=random_state))])
    pipe_a.fit(X_tr[cols_a], y_tr)
    m_a = regression_metrics(y_te, pipe_a.predict(X_te[cols_a]))
    print(f"Model A (With Class) Random Holdout: MAE=INR {m_a['mae']:.2f}, RMSE=INR {m_a['rmse']:.2f}, R2={m_a['r2']:.4f}")
    
    print("Training Model B (Without Class)...")
    pipe_b = Pipeline([("prep", build_preprocessor(cat_b, num_b)), ("model", HistGradientBoostingRegressor(max_depth=10, random_state=random_state))])
    pipe_b.fit(X_tr[cols_b], y_tr)
    m_b = regression_metrics(y_te, pipe_b.predict(X_te[cols_b]))
    print(f"Model B (Without Class) Random Holdout: MAE=INR {m_b['mae']:.2f}, RMSE=INR {m_b['rmse']:.2f}, R2={m_b['r2']:.4f}")

    # Route-group holdout for Model A vs Model B
    groups = df["source_city"].astype(str) + " → " + df["destination_city"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(splitter.split(df, groups=groups))
    
    pipe_a.fit(df.iloc[tr_idx][cols_a], df.iloc[tr_idx]["price"])
    rg_a = regression_metrics(df.iloc[te_idx]["price"], pipe_a.predict(df.iloc[te_idx][cols_a]))
    print(f"Model A Route-Group Holdout: MAE=INR {rg_a['mae']:.2f}, RMSE=INR {rg_a['rmse']:.2f}, R2={rg_a['r2']:.4f}")
    
    pipe_b.fit(df.iloc[tr_idx][cols_b], df.iloc[tr_idx]["price"])
    rg_b = regression_metrics(df.iloc[te_idx]["price"], pipe_b.predict(df.iloc[te_idx][cols_b]))
    print(f"Model B Route-Group Holdout: MAE=INR {rg_b['mae']:.2f}, RMSE=INR {rg_b['rmse']:.2f}, R2={rg_b['r2']:.4f}")

    print("\n=== 3. DISTANCE BUCKET ERROR ANALYSIS ===")
    df_test = df.iloc[te_idx].copy()
    preds_a = pipe_a.predict(df_test[cols_a])
    df_test["pred"] = preds_a
    df_test["abs_err"] = (df_test["price"] - df_test["pred"]).abs()
    
    bins = [0, 300, 700, 1500, 2500, 10000]
    labels = ["0–300 km", "300–700 km", "700–1500 km", "1500–2500 km", "2500+ km"]
    df_test["dist_bucket"] = pd.cut(df_test["distance_km"], bins=bins, labels=labels, right=False)
    
    bucket_report = df_test.groupby("dist_bucket", observed=False).agg(
        records=("price", "count"),
        actual_mean_price=("price", "mean"),
        pred_mean_price=("pred", "mean"),
        mae=("abs_err", "mean"),
    )
    print(bucket_report)

if __name__ == "__main__":
    run_investigation()
