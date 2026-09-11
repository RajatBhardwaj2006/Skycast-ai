import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from src.features.preprocess import build_preprocessor
from src.evaluation.metrics import regression_metrics

def test_ensemble():
    df = pd.read_csv("data/processed/processed_flights.csv", low_memory=False)

    CATEGORICAL = ["airline", "departure_time", "arrival_time", "stops", "class", "source_city", "destination_city"]
    GEO_NUMERIC = ["duration", "days_left", "distance_km", "source_lat", "source_lon", "destination_lat", "destination_lon"]
    feature_cols = CATEGORICAL + GEO_NUMERIC

    # Route-group split
    groups = df["source_city"].astype(str) + " -> " + df["destination_city"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(splitter.split(df, groups=groups))

    X_tr, y_tr = df.iloc[tr_idx][feature_cols], df.iloc[tr_idx]["price"]
    X_te, y_te = df.iloc[te_idx][feature_cols], df.iloc[te_idx]["price"]

    # Random split
    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(df[feature_cols], df["price"], test_size=0.2, random_state=42)

    et = ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")

    voting = VotingRegressor([("et", et), ("xgb", xgb)])

    pipe_rg = Pipeline([("prep", build_preprocessor(CATEGORICAL, GEO_NUMERIC)), ("model", voting)])
    pipe_rg.fit(X_tr, y_tr)
    m_rg = regression_metrics(y_te, pipe_rg.predict(X_te))
    print(f"Voting Ensemble Route-Group: MAE=INR {m_rg['mae']:.2f} | RMSE=INR {m_rg['rmse']:.2f} | R2={m_rg['r2']:.4f}")

    pipe_rand = Pipeline([("prep", build_preprocessor(CATEGORICAL, GEO_NUMERIC)), ("model", voting)])
    pipe_rand.fit(X_r_tr, y_r_tr)
    m_rand = regression_metrics(y_r_te, pipe_rand.predict(X_r_te))
    print(f"Voting Ensemble Random:      MAE=INR {m_rand['mae']:.2f} | RMSE=INR {m_rand['rmse']:.2f} | R2={m_rand['r2']:.4f}")

    test_row_3h = pd.DataFrame([{
        "airline": "Indigo", "departure_time": "Morning", "arrival_time": "Morning",
        "stops": "zero", "class": "Economy", "source_city": "Amritsar", "destination_city": "Srinagar",
        "duration": 3.0, "days_left": 15, "distance_km": 253.0,
        "source_lat": 31.7096, "source_lon": 74.7973, "destination_lat": 33.9871, "destination_lon": 74.7741
    }])
    test_row_1h = test_row_3h.copy()
    test_row_1h["duration"] = 1.0

    p3 = pipe_rand.predict(test_row_3h)[0]
    p1 = pipe_rand.predict(test_row_1h)[0]
    print(f"Voting ATQ-SXR (1h): INR {p1:,.2f} | ATQ-SXR (3h): INR {p3:,.2f}")

if __name__ == "__main__":
    test_ensemble()
