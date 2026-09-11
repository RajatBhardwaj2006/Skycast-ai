from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.features.feature_engineering import add_engineered_features


class AirfareModelRouter(BaseEstimator, RegressorMixin):
    """Class-Aware Routing Pipeline.
    
    Eliminates the 89% class dominance artifact and drastically improves
    unseen-route generalization by routing Economy and Business requests
    to dedicated estimators trained with log-transformed price targets.
    """

    def __init__(self, economy_model, business_model, global_model=None, metrics=None):
        self.economy_model = economy_model
        self.business_model = business_model
        self.global_model = global_model or economy_model
        self.metrics = metrics or {}

    @property
    def named_steps(self):
        # Compatibility property for feature importance inspections
        return self.economy_model.named_steps

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        if "log_days_left" not in df.columns or "duration_distance_ratio" not in df.columns:
            df = add_engineered_features(df)

        preds = np.zeros(len(df), dtype=float)
        classes = (
            df["class"].astype(str).str.strip().str.title()
            if "class" in df.columns
            else pd.Series(["Economy"] * len(df))
        )

        eco_mask = (classes == "Economy").values
        bus_mask = (classes == "Business").values
        other_mask = ~eco_mask & ~bus_mask

        if np.any(eco_mask):
            eco_sub = df.iloc[eco_mask]
            preds[eco_mask] = np.expm1(self.economy_model.predict(eco_sub))

        if np.any(bus_mask):
            bus_sub = df.iloc[bus_mask]
            preds[bus_mask] = np.expm1(self.business_model.predict(bus_sub))

        if np.any(other_mask):
            other_sub = df.iloc[other_mask]
            preds[other_mask] = np.expm1(self.global_model.predict(other_sub))

        # Fares cannot be negative; commercial floor ~INR 500
        return np.clip(preds, a_min=500.0, a_max=None)
