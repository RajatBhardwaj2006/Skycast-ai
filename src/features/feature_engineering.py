"""
Legacy transformer kept for compatibility with earlier SkyCast stubs.
The production training path uses src.features.preprocess instead.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "dep_time" in X.columns:
            X["dep_time"] = pd.to_datetime(X["dep_time"], errors="coerce")
            X["dep_hour"] = X["dep_time"].dt.hour
        if "travel_date" in X.columns:
            X["travel_date"] = pd.to_datetime(X["travel_date"], errors="coerce")
            X["day_of_week"] = X["travel_date"].dt.day_name()
        return X


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
