import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

def run():
    print("Loading data...")
    df = pd.read_csv("data/processed/processed_flights.csv", low_memory=False)
    
    # Filter to genuine observations where class is known
    df_clean = df[df["class"].isin(["Economy", "Business"]) & df["days_left"].notna()].copy()
    
    # Feature engineering
    df_clean["log_days_left"] = np.log1p(df_clean["days_left"])
    df_clean["duration_distance_ratio"] = df_clean["duration"] / df_clean["distance_km"].clip(lower=50)
    df_clean["speed_kmh"] = df_clean["distance_km"] / df_clean["duration"].clip(lower=0.5)
    
    cat_cols_base = ["airline", "departure_time", "arrival_time", "stops", "source_city", "destination_city"]
    num_cols = ["duration", "days_left", "log_days_left", "distance_km", "duration_distance_ratio", "source_lat", "source_lon", "destination_lat", "destination_lon"]
    
    def eval_m(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = float(1 - (np.sum((y_true - y_pred)**2) / max(1e-8, ss_tot)))
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

    print("\n=======================================================")
    print("1. EVALUATING ECONOMY-SPECIFIC MODEL (N = 206,666)")
    print("=======================================================")
    eco = df_clean[df_clean["class"] == "Economy"].copy()
    
    # Route Group Split
    groups = eco["source_city"] + "->" + eco["destination_city"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(eco, groups=groups))
    
    X_tr_eco, y_tr_eco = eco.iloc[tr_idx][cat_cols_base + num_cols], eco.iloc[tr_idx]["price"]
    X_te_eco, y_te_eco = eco.iloc[te_idx][cat_cols_base + num_cols], eco.iloc[te_idx]["price"]
    
    # Random Split
    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(eco[cat_cols_base + num_cols], eco["price"], test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_base),
        ("num", SimpleImputer(strategy="median"), num_cols)
    ])
    
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=16, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=60, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist"),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=10, learning_rate=0.08, max_iter=150, random_state=42),
    }
    
    print("\n--- Route-Group Validation (Unseen City Pairs) on Economy ---")
    for name, est in candidates.items():
        p_raw = Pipeline([("prep", preprocessor), ("model", est)])
        p_raw.fit(X_tr_eco, y_tr_eco)
        pred_raw = p_raw.predict(X_te_eco)
        m_raw = eval_m(y_te_eco, pred_raw)
        
        p_log = Pipeline([("prep", preprocessor), ("model", est)])
        p_log.fit(X_tr_eco, np.log1p(y_tr_eco))
        pred_log = np.expm1(p_log.predict(X_te_eco))
        m_log = eval_m(y_te_eco, pred_log)
        
        print(f"[{name}]")
        print(f"  Raw Target: MAE=INR {m_raw['mae']} | RMSE=INR {m_raw['rmse']} | R2={m_raw['r2']}")
        print(f"  Log Target: MAE=INR {m_log['mae']} | RMSE=INR {m_log['rmse']} | R2={m_log['r2']}")
        
    print("\n--- Random-Row Validation on Economy ---")
    for name, est in candidates.items():
        p_raw = Pipeline([("prep", preprocessor), ("model", est)])
        p_raw.fit(X_r_tr, y_r_tr)
        pred_raw = p_raw.predict(X_r_te)
        m_raw = eval_m(y_r_te, pred_raw)
        
        p_log = Pipeline([("prep", preprocessor), ("model", est)])
        p_log.fit(X_r_tr, np.log1p(y_r_tr))
        pred_log = np.expm1(p_log.predict(X_r_te))
        m_log = eval_m(y_r_te, pred_log)
        
        print(f"[{name}] Random MAE: Raw=INR {m_raw['mae']} (R2={m_raw['r2']}) | Log=INR {m_log['mae']} (R2={m_log['r2']})")

    # Feature Importance of Economy Model
    rf_model = Pipeline([("prep", preprocessor), ("model", candidates["Extra Trees"])])
    rf_model.fit(X_r_tr, y_r_tr)
    prep = rf_model.named_steps["prep"]
    mod = rf_model.named_steps["model"]
    fnames = prep.get_feature_names_out()
    imps = mod.feature_importances_
    grouped = {}
    for fn, val in zip(fnames, imps):
        base = fn.split("__")[-1]
        for c in cat_cols_base + num_cols:
            if base.startswith(c):
                base = c
                break
        grouped[base] = grouped.get(base, 0) + val
    print("\n--- Economy Model Feature Importances (Class is not a feature!) ---")
    for feat, imp in sorted(grouped.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<25}: {imp*100:.2f}%")

if __name__ == "__main__":
    run()
