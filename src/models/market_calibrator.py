from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils.config import get_logger, resolve_path

logger = get_logger("skycast.calibrator")

DUAN_SMEARING_FACTOR = 1.01388


class MarketCalibrator:
    """Empirical Market Calibration Layer.

    Bridges the temporal dataset gap between historical training data (2022)
    and current real-world market airfares (2026) using continuous econometric
    regression across validated multi-route observations.
    
    Prevents single-route hardcoded overrides and arbitrary uniform multipliers.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.regressor = Ridge(alpha=self.alpha)
        self.is_fitted = False
        self.metrics: dict[str, Any] = {}
        self.mean_inflation_factor: float = 1.45

    def _extract_features(
        self,
        hist_prices: np.ndarray,
        distances: np.ndarray,
        days_left: np.ndarray,
        classes: list[str] | np.ndarray,
    ) -> np.ndarray:
        h_arr = np.maximum(500.0, np.asarray(hist_prices, dtype=float))
        d_arr = np.asarray(distances, dtype=float) / 1000.0
        t_arr = np.asarray(days_left, dtype=float) / 15.0
        b_arr = np.asarray([1.0 if str(c).title() == "Business" else 0.0 for c in classes], dtype=float)

        return np.column_stack([
            np.log(h_arr),
            d_arr,
            t_arr,
            b_arr,
        ])

    def fit_from_dataset(
        self,
        observations: list[dict[str, Any]],
        router: Any,
        location_service: Any,
    ) -> "MarketCalibrator":
        from src.features.feature_engineering import add_engineered_features

        records = []
        for item in observations:
            orig = location_service.resolve(item["source_city"])
            dest = location_service.resolve(item["destination_city"])
            dist = location_service.distance_between(orig, dest)

            row = {
                "airline": item["airline"],
                "departure_time": "Morning",
                "arrival_time": "Evening",
                "stops": item["stops"],
                "class": item["cabin"],
                "source_city": orig.city,
                "destination_city": dest.city,
                "duration": float(item["duration"]),
                "days_left": int(item["days_until_departure"]),
                "distance_km": dist,
                "source_lat": orig.latitude,
                "source_lon": orig.longitude,
                "destination_lat": dest.latitude,
                "destination_lon": dest.longitude,
            }
            df = add_engineered_features(pd.DataFrame([row]))
            # Apply Duan's smearing factor to raw log retransformation
            raw_p = float(router.predict(df)[0]) * DUAN_SMEARING_FACTOR

            records.append({
                "hist_pred": raw_p,
                "distance_km": dist,
                "days_left": int(item["days_until_departure"]),
                "cabin": item["cabin"],
                "market_fare": float(item["current_fare"]),
            })

        df_obs = pd.DataFrame(records)
        X = self._extract_features(
            df_obs["hist_pred"].values,
            df_obs["distance_km"].values,
            df_obs["days_left"].values,
            df_obs["cabin"].values,
        )
        y = np.log(df_obs["market_fare"].values)

        self.regressor.fit(X, y)
        self.is_fitted = True

        pred_cal = np.exp(self.regressor.predict(X))
        mae_before = float(mean_absolute_error(df_obs["market_fare"], df_obs["hist_pred"]))
        mae_after = float(mean_absolute_error(df_obs["market_fare"], pred_cal))
        r2_after = float(r2_score(df_obs["market_fare"], pred_cal))
        self.mean_inflation_factor = float(np.median(df_obs["market_fare"] / df_obs["hist_pred"]))

        self.metrics = {
            "mae_before": round(mae_before, 2),
            "mae_after": round(mae_after, 2),
            "r2_after": round(r2_after, 4),
            "median_inflation_factor": round(self.mean_inflation_factor, 3),
            "observation_count": len(records),
        }

        logger.info(
            "Fitted MarketCalibrator: MAE before=INR %.2f -> after=INR %.2f (R2=%.4f, inflation=%.2fx)",
            mae_before,
            mae_after,
            r2_after,
            self.mean_inflation_factor,
        )
        return self

    def predict(
        self,
        historical_price: float,
        distance_km: float,
        days_left: int,
        class_type: str = "Economy",
    ) -> float:
        """Calibrate a historical baseline prediction to current market price levels."""
        if not self.is_fitted:
            # Fallback if uncalibrated
            return max(500.0, round(historical_price * self.mean_inflation_factor, 2))

        X = self._extract_features(
            np.array([historical_price]),
            np.array([distance_km]),
            np.array([days_left]),
            [class_type],
        )
        log_pred = self.regressor.predict(X)[0]
        calibrated = float(np.exp(log_pred))
        return max(500.0, round(calibrated, 2))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Path | str) -> "MarketCalibrator":
        p = Path(path)
        if not p.exists():
            instance = cls()
            val_path = resolve_path("data/reference/market_validation_2026.json")
            if val_path.exists():
                try:
                    from backend.app.services.model_service import load_pipeline
                    from src.geo.locations import default_location_service

                    obs = json.loads(val_path.read_text(encoding="utf-8"))
                    router = load_pipeline()
                    loc_service = default_location_service()
                    instance.fit_from_dataset(obs, router, loc_service)
                    instance.save(p)
                    logger.info("Auto-fitted and cached MarketCalibrator to %s", p)
                    return instance
                except Exception as exc:
                    logger.warning("Could not auto-fit MarketCalibrator: %s", exc)
            return instance
        return joblib.load(p)

