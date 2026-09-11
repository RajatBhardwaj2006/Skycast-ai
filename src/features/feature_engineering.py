"""
Legacy transformer kept for compatibility with earlier SkyCast stubs.
The production training path uses src.features.preprocess instead.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


import numpy as np
import pandas as pd


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive non-linear and physical flight features from raw inputs."""
    data = frame.copy()
    if "days_left" in data.columns:
        days = pd.to_numeric(data["days_left"], errors="coerce").fillna(15.0)
        data["log_days_left"] = np.log1p(days.clip(lower=0.0))
    else:
        data["log_days_left"] = np.log1p(15.0)

    dur = pd.to_numeric(data.get("duration", 2.0), errors="coerce").fillna(2.0).clip(lower=0.25)
    dist = pd.to_numeric(data.get("distance_km", 1000.0), errors="coerce").fillna(1000.0).clip(lower=50.0)
    data["duration_distance_ratio"] = dur / dist
    data["speed_kmh"] = dist / dur

    if "source_city" in data.columns and "destination_city" in data.columns:
        data["route_id"] = data["source_city"].astype(str) + "_" + data["destination_city"].astype(str)
    else:
        data["route_id"] = "unknown_route"

    if "airline" in data.columns:
        data["airline_route_id"] = data["airline"].astype(str) + "_" + data["route_id"]
    else:
        data["airline_route_id"] = "unknown_airline_route"

    return data


class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_engineered_features(X)


def build_preprocessing_pipeline(numerical_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numerical_features),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                categorical_features,
            ),
        ]
    )

