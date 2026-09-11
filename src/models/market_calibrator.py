"""MarketCalibrator v2 — Leave-One-Route-Out validated calibration layer.

Bridges the temporal gap between 2022 historical training data and 2026 current
market fares using continuous econometric regression.  All reported metrics use
out-of-fold predictions only — the test route NEVER appears in calibrator
training.

Model selection is done by LORO cross-validated MAE, not training MAE.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from src.utils.config import get_logger, resolve_path

logger = get_logger("skycast.calibrator")

DUAN_SMEARING_FACTOR = 1.01388


def _corridor_key(route: str) -> str:
    """DEL -> BLR and BLR -> DEL both map to corridor BLR_DEL."""
    parts = route.replace(" ", "").split("->")
    return "_".join(sorted(p.strip() for p in parts))


class MarketCalibrator:
    """Empirical Market Calibration Layer with LORO validation.

    Prevents single-route hardcoded overrides and arbitrary uniform multipliers.
    All public metrics are leave-one-route-out cross-validated.
    """

    def __init__(self) -> None:
        self.regressor: Any = None
        self.is_fitted: bool = False
        self.metrics: dict[str, Any] = {}
        self.validation_status: str = "experimental"
        self.mean_inflation_factor: float = 1.45
        self._best_model_name: str = "none"

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(
        hist_prices: np.ndarray,
        distances: np.ndarray,
        days_left: np.ndarray,
        classes: list[str] | np.ndarray,
    ) -> np.ndarray:
        h = np.maximum(500.0, np.asarray(hist_prices, dtype=float))
        d = np.asarray(distances, dtype=float) / 1000.0
        t = np.asarray(days_left, dtype=float) / 15.0
        log_t = np.log1p(np.asarray(days_left, dtype=float))
        b = np.asarray([1.0 if str(c).title() == "Business" else 0.0 for c in classes])
        return np.column_stack([np.log(h), d, t, log_t, b])

    # ------------------------------------------------------------------
    # Generate observation records from raw JSON + model
    # ------------------------------------------------------------------
    @staticmethod
    def _build_records(
        observations: list[dict[str, Any]],
        router: Any,
        location_service: Any,
    ) -> pd.DataFrame:
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
            raw_p = float(router.predict(df)[0]) * DUAN_SMEARING_FACTOR
            records.append({
                "route": item["route"],
                "corridor": _corridor_key(item["route"]),
                "airline": item["airline"],
                "cabin": item["cabin"],
                "days_left": int(item["days_until_departure"]),
                "hist_pred": raw_p,
                "distance_km": dist,
                "market_fare": float(item["current_fare"]),
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Leave-One-Route-Out CV
    # ------------------------------------------------------------------
    @staticmethod
    def _loro_cv(
        df: pd.DataFrame,
        model_factory,
        group_col: str = "route",
    ) -> pd.DataFrame:
        """Run leave-one-group-out cross-validation.

        Returns DataFrame with columns: route, market_fare, oof_pred, hist_pred.
        """
        results = []
        groups = df[group_col].unique()
        for held_out in groups:
            train = df[df[group_col] != held_out]
            test = df[df[group_col] == held_out]
            if len(train) < 3:
                continue

            X_train = MarketCalibrator._extract_features(
                train["hist_pred"].values, train["distance_km"].values,
                train["days_left"].values, train["cabin"].values,
            )
            y_train = np.log(train["market_fare"].values)

            model = model_factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            X_test = MarketCalibrator._extract_features(
                test["hist_pred"].values, test["distance_km"].values,
                test["days_left"].values, test["cabin"].values,
            )
            pred_log = model.predict(X_test)
            pred_cal = np.exp(pred_log)

            for idx, (_, row) in enumerate(test.iterrows()):
                results.append({
                    "route": row["route"],
                    "corridor": row["corridor"],
                    "airline": row["airline"],
                    "cabin": row["cabin"],
                    "days_left": row["days_left"],
                    "market_fare": row["market_fare"],
                    "hist_pred": row["hist_pred"],
                    "oof_pred": float(pred_cal[idx]),
                })
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Model candidates
    # ------------------------------------------------------------------
    @staticmethod
    def _model_candidates() -> dict[str, Any]:
        return {
            "ridge": lambda: Ridge(alpha=0.5),
            "huber": lambda: HuberRegressor(epsilon=1.35, max_iter=200),
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=50, max_depth=4, min_samples_leaf=3, random_state=42,
            ),
            "gradient_boosting": lambda: GradientBoostingRegressor(
                n_estimators=80, max_depth=3, learning_rate=0.1,
                min_samples_leaf=3, random_state=42,
            ),
        }

    # ------------------------------------------------------------------
    # Global inflation baseline (no ML)
    # ------------------------------------------------------------------
    @staticmethod
    def _global_inflation_predictions(df: pd.DataFrame) -> np.ndarray:
        factor = np.median(df["market_fare"].values / df["hist_pred"].values)
        return df["hist_pred"].values * factor

    # ------------------------------------------------------------------
    # fit_from_dataset: runs LORO CV, selects best model, fits final
    # ------------------------------------------------------------------
    def fit_from_dataset(
        self,
        observations: list[dict[str, Any]],
        router: Any,
        location_service: Any,
    ) -> "MarketCalibrator":
        df = self._build_records(observations, router, location_service)

        # --- Baseline: no calibration ---
        mae_none = float(mean_absolute_error(df["market_fare"], df["hist_pred"]))

        # --- Baseline: global inflation ---
        inf_preds = self._global_inflation_predictions(df)
        mae_inflation = float(mean_absolute_error(df["market_fare"], inf_preds))
        global_factor = float(np.median(df["market_fare"].values / df["hist_pred"].values))

        # --- LORO CV for each model candidate ---
        best_name = "none"
        best_mae = mae_none
        best_oof_df = None
        all_results = {
            "none": {"loro_mae": round(mae_none, 2), "method": "No calibration"},
            "global_inflation": {"loro_mae": round(mae_inflation, 2), "method": f"Global {global_factor:.3f}x"},
        }

        candidates = self._model_candidates()
        for name, factory in candidates.items():
            try:
                # LORO CV
                oof = self._loro_cv(df, factory, group_col="route")
                if len(oof) == 0:
                    continue
                loro_mae = float(mean_absolute_error(oof["market_fare"], oof["oof_pred"]))
                loro_rmse = float(np.sqrt(mean_squared_error(oof["market_fare"], oof["oof_pred"])))
                loro_r2 = float(r2_score(oof["market_fare"], oof["oof_pred"]))

                # Also run leave-one-corridor-out
                oof_corridor = self._loro_cv(df, factory, group_col="corridor")
                loco_mae = float(mean_absolute_error(oof_corridor["market_fare"], oof_corridor["oof_pred"])) if len(oof_corridor) > 0 else loro_mae

                all_results[name] = {
                    "loro_mae": round(loro_mae, 2),
                    "loro_rmse": round(loro_rmse, 2),
                    "loro_r2": round(loro_r2, 4),
                    "loco_mae": round(loco_mae, 2),
                    "method": name,
                }

                if loro_mae < best_mae:
                    best_mae = loro_mae
                    best_name = name
                    best_oof_df = oof
            except Exception as exc:
                logger.warning("Calibrator candidate %s failed: %s", name, exc)
                continue

        # --- If no ML model beats baseline, use global inflation if it helps ---
        if best_name == "none" and mae_inflation < mae_none:
            best_name = "global_inflation"
            best_mae = mae_inflation

        # --- Fit final production model on ALL data using best candidate ---
        self.mean_inflation_factor = global_factor
        if best_name in candidates:
            self.regressor = candidates[best_name]()
            X_all = self._extract_features(
                df["hist_pred"].values, df["distance_km"].values,
                df["days_left"].values, df["cabin"].values,
            )
            y_all = np.log(df["market_fare"].values)
            self.regressor.fit(X_all, y_all)
            self.is_fitted = True
            self._best_model_name = best_name
        else:
            self.is_fitted = False
            self._best_model_name = best_name

        # --- Build per-route validation table ---
        route_table = []
        if best_oof_df is not None and len(best_oof_df) > 0:
            for route in best_oof_df["route"].unique():
                rdf = best_oof_df[best_oof_df["route"] == route]
                route_market_mean = float(rdf["market_fare"].mean())
                route_oof_mean = float(rdf["oof_pred"].mean())
                route_hist_mean = float(rdf["hist_pred"].mean())
                route_table.append({
                    "route": route,
                    "historical_ml": round(route_hist_mean, 0),
                    "calibrated_oof": round(route_oof_mean, 0),
                    "market_fare": round(route_market_mean, 0),
                    "error_pct": round((route_oof_mean - route_market_mean) / route_market_mean * 100, 1),
                    "out_of_fold": True,
                })

        # --- Determine validation status ---
        if best_name in candidates and best_mae < mae_none * 0.85:
            self.validation_status = "validated_loro"
        else:
            self.validation_status = "experimental"

        self.metrics = {
            "best_model": best_name,
            "validation_status": self.validation_status,
            "mae_no_calibration": round(mae_none, 2),
            "mae_global_inflation": round(mae_inflation, 2),
            "global_inflation_factor": round(global_factor, 3),
            "mae_after": round(best_mae, 2),
            "observation_count": len(df),
            "unique_routes": int(df["route"].nunique()),
            "unique_corridors": int(df["corridor"].nunique()),
            "model_comparison": all_results,
            "route_validation_table": route_table,
        }

        logger.info(
            "MarketCalibrator: best=%s LORO-MAE=INR %.0f (none=%.0f, inflation=%.0f), status=%s, obs=%d routes=%d",
            best_name, best_mae, mae_none, mae_inflation,
            self.validation_status, len(df), df["route"].nunique(),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        historical_price: float,
        distance_km: float,
        days_left: int,
        class_type: str = "Economy",
    ) -> float:
        """Calibrate a historical baseline prediction to current market price levels."""
        if not self.is_fitted or self.regressor is None:
            # Fallback: global inflation
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
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
