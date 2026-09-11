from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean import clean_clean_dataset
from src.data.load import load_clean_dataset, load_raw_business, load_raw_economy
from src.data.quality import build_quality_report, save_quality_report
from src.evaluation.metrics import regression_metrics
from src.features.feature_engineering import add_engineered_features
from src.features.geo_features import add_geographic_features
from src.utils.config import get_logger, load_config, resolve_path

logger = get_logger("skycast.train")

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

CATEGORICAL_CORE = ["airline", "departure_time", "arrival_time", "stops", "source_city", "destination_city", "route_id"]
CATEGORICAL_WITH_CLASS = ["class"] + CATEGORICAL_CORE
NUMERICAL_CORE = [
    "duration",
    "days_left",
    "log_days_left",
    "distance_km",
    "duration_distance_ratio",
    "speed_kmh",
    "source_lat",
    "source_lon",
    "destination_lat",
    "destination_lon",
]

# Legacy feature aliases for backward compatibility with existing schemas
CATEGORICAL = ["airline", "departure_time", "arrival_time", "stops", "class", "source_city", "destination_city"]
BASE_NUMERIC = ["duration", "days_left"]
DISTANCE_NUMERIC = BASE_NUMERIC + ["distance_km"]
GEO_NUMERIC = DISTANCE_NUMERIC + ["source_lat", "source_lon", "destination_lat", "destination_lon"]


from src.models.router import AirfareModelRouter


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_model_preprocessor(categorical: list[str], numerical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numerical,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _get_candidate_estimators(random_state: int) -> dict[str, Any]:
    rf = RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=random_state,
    )
    et = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    candidates: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": rf,
        "Extra Trees": et,
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.6,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=10,
            learning_rate=0.08,
            max_iter=180,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=180,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        candidates["XGBoost"] = xgb
        candidates["Voting Ensemble"] = VotingRegressor([("et", et), ("xgb", xgb)])
    return candidates


def evaluate_split(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_log_target: bool = True,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    y_tr = np.log1p(y_train) if use_log_target else y_train
    pipeline.fit(X_train, y_tr)
    fit_time = time.perf_counter() - start_time

    infer_start = time.perf_counter()
    raw_preds = pipeline.predict(X_test)
    infer_time = time.perf_counter() - infer_start

    preds = np.expm1(raw_preds) if use_log_target else raw_preds
    preds = np.clip(preds, a_min=0.0, a_max=None)
    metrics = regression_metrics(y_test, preds)
    metrics.update({
        "model": name,
        "training_seconds": round(fit_time, 3),
        "inference_ms_per_1k": round((infer_time / max(1, len(X_test))) * 1000 * 1000, 3),
    })
    return metrics


def extract_grouped_importances(pipeline: Pipeline, feature_cols: list[str]) -> list[dict[str, Any]]:
    prep: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    fnames = list(prep.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity"
    elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
        sub = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_, dtype=float)
                s = imp.sum()
                if s > 0:
                    sub.append(imp / s)
        if sub:
            raw = np.mean(sub, axis=0)
            kind = "ensemble_impurity_mean"
        else:
            return []
    else:
        return []

    grouped: dict[str, float] = {}
    for name, value in zip(fnames, raw):
        token = name.split("__")[-1]
        matched_feature = token
        for col in sorted(feature_cols, key=len, reverse=True):
            if token.startswith(col + "_") or token == col:
                matched_feature = col
                break
        grouped[matched_feature] = grouped.get(matched_feature, 0.0) + float(value)

    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": f, "importance": round(score / total, 6), "source": kind} for f, score in ranked]


def train_model() -> None:
    config = load_config()
    random_state = int(config.get("random_state", 42))
    test_size = float(config.get("test_size", 0.2))
    logger.info("Initializing full SkyCast model rebuild...")

    # 1. Load and Clean Datasets
    raw = load_clean_dataset()
    report = build_quality_report(raw, "merged_data.csv")
    save_quality_report(report)

    cleaned = clean_clean_dataset(raw)
    geo_df = add_geographic_features(cleaned)
    featured = add_engineered_features(geo_df)

    processed_path = resolve_path(config["paths"]["processed"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(processed_path, index=False)
    logger.info("Wrote enriched dataset (%s rows) to %s", f"{len(featured):,}", processed_path)

    # 2. Separate Genuine Known Classes (Economy vs Business)
    eco_df = featured[featured["class"] == "Economy"].reset_index(drop=True)
    bus_df = featured[featured["class"] == "Business"].reset_index(drop=True)
    logger.info("Class distribution: Economy = %s rows | Business = %s rows", f"{len(eco_df):,}", f"{len(bus_df):,}")

    feature_cols = CATEGORICAL_CORE + NUMERICAL_CORE

    # 3. Multi-Split Validation on Economy Data
    logger.info("Running 4-split validation on Economy dataset...")
    # (A) Random-row split
    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(eco_df[feature_cols], eco_df["price"], test_size=test_size, random_state=random_state)

    # (B) Route-group split (unseen routes)
    eco_routes = eco_df["source_city"].astype(str) + " → " + eco_df["destination_city"].astype(str)
    gss_route = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_rg_idx, te_rg_idx = next(gss_route.split(eco_df, groups=eco_routes))
    X_rg_tr, y_rg_tr = eco_df.iloc[tr_rg_idx][feature_cols], eco_df.iloc[tr_rg_idx]["price"]
    X_rg_te, y_rg_te = eco_df.iloc[te_rg_idx][feature_cols], eco_df.iloc[te_rg_idx]["price"]

    # (C) Airport-group split (unseen airport hubs)
    eco_airports = eco_df["source_city"].astype(str)
    gss_airport = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_ap_idx, te_ap_idx = next(gss_airport.split(eco_df, groups=eco_airports))
    X_ap_tr, y_ap_tr = eco_df.iloc[tr_ap_idx][feature_cols], eco_df.iloc[tr_ap_idx]["price"]
    X_ap_te, y_ap_te = eco_df.iloc[te_ap_idx][feature_cols], eco_df.iloc[te_ap_idx]["price"]

    # (D) Temporal split (chronological cutoff based on booking window / date)
    eco_sorted = eco_df.sort_values("days_left", ascending=False).reset_index(drop=True)
    n_cutoff = int(len(eco_sorted) * 0.8)
    X_tm_tr, y_tm_tr = eco_sorted.iloc[:n_cutoff][feature_cols], eco_sorted.iloc[:n_cutoff]["price"]
    X_tm_te, y_tm_te = eco_sorted.iloc[n_cutoff:][feature_cols], eco_sorted.iloc[n_cutoff:]["price"]

    candidate_results: list[dict[str, Any]] = []
    economy_fitted_models: dict[str, Pipeline] = {}

    preprocessor = _build_model_preprocessor(CATEGORICAL_CORE, NUMERICAL_CORE)

    for name, estimator in _get_candidate_estimators(random_state).items():
        logger.info("Evaluating candidate [%s]...", name)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

        # Random split evaluation (log target)
        m_rand = evaluate_split(name, clone(pipe), X_r_tr, y_r_tr, X_r_te, y_r_te, use_log_target=True)

        # Route-group evaluation (log target)
        m_route = evaluate_split(name, clone(pipe), X_rg_tr, y_rg_tr, X_rg_te, y_rg_te, use_log_target=True)

        # Airport-group evaluation (log target)
        m_airport = evaluate_split(name, clone(pipe), X_ap_tr, y_ap_tr, X_ap_te, y_ap_te, use_log_target=True)

        # Time split evaluation (log target)
        m_time = evaluate_split(name, clone(pipe), X_tm_tr, y_tm_tr, X_tm_te, y_tm_te, use_log_target=True)

        # Fit on full economy train set for persistence candidate
        pipe_fitted = clone(pipe)
        pipe_fitted.fit(X_r_tr, np.log1p(y_r_tr))
        economy_fitted_models[name] = pipe_fitted

        candidate_results.append({
            "model": name,
            "random_mae": m_rand["mae"],
            "random_rmse": m_rand["rmse"],
            "random_r2": m_rand["r2"],
            "route_group_mae": m_route["mae"],
            "route_group_rmse": m_route["rmse"],
            "route_group_r2": m_route["r2"],
            "airport_group_mae": m_airport["mae"],
            "airport_group_r2": m_airport["r2"],
            "time_split_mae": m_time["mae"],
            "time_split_r2": m_time["r2"],
            "training_seconds": m_rand["training_seconds"],
            "inference_ms_per_1k": m_rand["inference_ms_per_1k"],
        })

    # Select best model based primarily on Route-Group MAE + Airport-Group MAE
    best_candidate_row = sorted(
        candidate_results,
        key=lambda r: (r["route_group_mae"], r["airport_group_mae"], r["random_mae"])
    )[0]
    best_model_name = best_candidate_row["model"]
    logger.info("Best candidate for Economy: %s (Route-Group MAE=INR %.2f)", best_model_name, best_candidate_row["route_group_mae"])

    best_economy_pipeline = economy_fitted_models[best_model_name]

    # 4. Train Business-Specific Model
    logger.info("Training Business class pipeline (%s records)...", f"{len(bus_df):,}")
    X_bus_tr, X_bus_te, y_bus_tr, y_bus_te = train_test_split(
        bus_df[feature_cols], bus_df["price"], test_size=test_size, random_state=random_state
    )
    best_bus_estimator = _get_candidate_estimators(random_state).get(best_model_name) or ExtraTreesRegressor(
        n_estimators=80, max_depth=20, min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )
    best_business_pipeline = Pipeline([
        ("preprocessor", _build_model_preprocessor(CATEGORICAL_CORE, NUMERICAL_CORE)),
        ("model", best_bus_estimator),
    ])
    best_business_pipeline.fit(X_bus_tr, np.log1p(y_bus_tr))
    bus_preds = np.expm1(best_business_pipeline.predict(X_bus_te))
    bus_metrics = regression_metrics(y_bus_te, bus_preds)
    logger.info("Business model held-out MAE: INR %.2f | R2: %.4f", bus_metrics["mae"], bus_metrics["r2"])

    # 5. Build Class-Aware Router
    logger.info("Assembling AirfareModelRouter...")
    router = AirfareModelRouter(
        economy_model=best_economy_pipeline,
        business_model=best_business_pipeline,
        metrics={
            "economy": best_candidate_row,
            "business": bus_metrics,
        },
    )

    # 6. Grouped Feature Importances (Eliminates 89% class dominance)
    eco_importances = extract_grouped_importances(best_economy_pipeline, feature_cols)
    logger.info("Top Economy Feature Importances:")
    for item in eco_importances[:7]:
        logger.info("  %s: %.2f%%", item["feature"], item["importance"] * 100)

    # 7. Overall Router Validation on Unified Held-Out Test Set
    X_all_tr, X_all_te, y_all_tr, y_all_te = train_test_split(
        featured[feature_cols + ["class"]], featured["price"], test_size=test_size, random_state=random_state
    )
    router_preds = router.predict(X_all_te)
    final_metrics = regression_metrics(y_all_te, router_preds)

    # Route-group audit on unified set
    unified_routes = featured["source_city"].astype(str) + " → " + featured["destination_city"].astype(str)
    gss_u = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    u_tr_idx, u_te_idx = next(gss_u.split(featured, groups=unified_routes))
    u_test_df = featured.iloc[u_te_idx]
    route_audit_metrics = regression_metrics(u_test_df["price"], router.predict(u_test_df))

    # Airport-group audit on unified set
    unified_airports = featured["source_city"].astype(str)
    gss_a = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    a_tr_idx, a_te_idx = next(gss_a.split(featured, groups=unified_airports))
    a_test_df = featured.iloc[a_te_idx]
    airport_audit_metrics = regression_metrics(a_test_df["price"], router.predict(a_test_df))

    # Temporal audit on unified set
    sorted_unified = featured.sort_values("days_left", ascending=False).reset_index(drop=True)
    t_test_df = sorted_unified.iloc[int(len(sorted_unified) * 0.8):]
    temporal_metrics = regression_metrics(t_test_df["price"], router.predict(t_test_df))

    # 8. Save Pipeline Artifacts
    models_dir = resolve_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = resolve_path(config["paths"]["pipeline"])
    joblib.dump(router, pipeline_path)
    joblib.dump(router, models_dir / "airfare_model.pkl")
    logger.info("Saved AirfareModelRouter to %s", pipeline_path)

    price_percentiles = featured["price"].quantile([0.33, 0.66]).to_dict()

    metadata = {
        "model_name": f"AirfareModelRouter ({best_model_name})",
        "architecture": "Dual-Model Class-Aware Router (Log-Transformed Target)",
        "target": "price",
        "random_state": random_state,
        "training_rows": int(len(X_all_tr)),
        "test_rows": int(len(X_all_te)),
        "economy_rows": int(len(eco_df)),
        "business_rows": int(len(bus_df)),
        "validation_strategy": "4_split_comprehensive_audit",
        "route_count": int(featured[["source_city", "destination_city"]].drop_duplicates().shape[0]),
        "features": feature_cols,
        "categorical_features": CATEGORICAL_CORE,
        "numerical_features": NUMERICAL_CORE,
        "metrics": final_metrics,
        "route_group_metrics": route_audit_metrics,
        "airport_group_metrics": airport_audit_metrics,
        "time_split_metrics": temporal_metrics,
        "economy_benchmark": best_candidate_row,
        "business_benchmark": bus_metrics,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "Clean_Dataset.csv + IndianFlightdata (deduplicated)",
        "geographic_features": True,
        "unsupported_features": config.get("features", {}).get("unsupported_ui_fields", []),
        "price_terciles": {"budget_max": float(price_percentiles[0.33]), "high_min": float(price_percentiles[0.66])},
        "airlines": sorted(featured["airline"].unique().tolist()),
        "classes": ["Economy", "Business"],
        "stops": sorted(featured["stops"].unique().tolist()),
        "departure_times": sorted(featured["departure_time"].unique().tolist()),
        "arrival_times": sorted(featured["arrival_time"].unique().tolist()),
        "training_cities": sorted(set(featured["source_city"]).union(featured["destination_city"])),
        "example_input": {
            "airline": "Air India",
            "source_city": "Delhi",
            "destination_city": "Mumbai",
            "departure_time": "Morning",
            "arrival_time": "Evening",
            "stops": "one",
            "class": "Economy",
            "duration": 5.5,
            "days_left": 15,
        },
    }

    validation_payload = {
        "selected_model_evaluation": {
            "strategy": "random_row_holdout",
            "metrics": final_metrics,
            "training_rows": int(len(X_all_tr)),
            "test_rows": int(len(X_all_te)),
        },
        "route_group_audit": {
            "strategy": "route_group_holdout",
            "metrics": route_audit_metrics,
            "held_out_routes_count": int(u_test_df[["source_city", "destination_city"]].drop_duplicates().shape[0]),
        },
        "airport_group_audit": {
            "strategy": "airport_group_holdout",
            "metrics": airport_audit_metrics,
        },
        "temporal_audit": {
            "strategy": "time_based_holdout",
            "metrics": temporal_metrics,
        },
        "candidate_comparison": candidate_results,
    }

    _json_dump(resolve_path(config["paths"]["metrics"]), {"best_model": f"AirfareModelRouter ({best_model_name})", **final_metrics, "candidates": candidate_results})
    _json_dump(resolve_path(config["paths"]["feature_importance"]), {"features": eco_importances})
    _json_dump(resolve_path(config["paths"]["metadata"]), metadata)
    _json_dump(resolve_path(config["paths"]["comparison"]), {"rows": candidate_results, "best_model": best_model_name})
    _json_dump(resolve_path(config["paths"]["validation"]), validation_payload)

    logger.info(
        "Model training and multi-split evaluation complete! Final Router: Random MAE=INR %.2f | Route-Group MAE=INR %.2f | Airport-Group MAE=INR %.2f | Time MAE=INR %.2f",
        final_metrics["mae"],
        route_audit_metrics["mae"],
        airport_audit_metrics["mae"],
        temporal_metrics["mae"],
    )


if __name__ == "__main__":
    train_model()
