from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data.clean import clean_clean_dataset
from src.data.load import load_clean_dataset, load_raw_business, load_raw_economy
from src.data.quality import build_quality_report, save_quality_report
from src.evaluation.importance import grouped_feature_importance
from src.evaluation.metrics import regression_metrics
from src.features.geo_features import add_geographic_features
from src.features.preprocess import build_preprocessor
from src.utils.config import get_logger, load_config, resolve_path

logger = get_logger("skycast.train")

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


CATEGORICAL = ["airline", "departure_time", "arrival_time", "stops", "class", "source_city", "destination_city"]
BASE_NUMERIC = ["duration", "days_left"]
DISTANCE_NUMERIC = BASE_NUMERIC + ["distance_km"]
GEO_NUMERIC = DISTANCE_NUMERIC + ["source_lat", "source_lon", "destination_lat", "destination_lon"]


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _document_dataset_decision(processed: pd.DataFrame) -> dict:
    business = load_raw_business()
    economy = load_raw_economy()
    decision = {
        "recommended_modeling_dataset": "Clean_Dataset.csv",
        "reason": (
            "Clean_Dataset.csv already standardizes airline, class, stops, duration (hours), "
            "days_left and price. business.csv and economy.csv contain travel dates and clock times "
            "but no booking timestamp, so days_left cannot be derived without inventing a scrape date. "
            "Those files are retained for documentation and schema comparison only."
        ),
        "clean_rows": int(len(processed)),
        "business_rows": int(len(business)),
        "economy_rows": int(len(economy)),
        "days_left_invented": False,
        "seat_type_invented": False,
    }
    resolve_path("data/processed").mkdir(parents=True, exist_ok=True)
    _json_dump(resolve_path("data/processed/dataset_decision.json"), decision)
    return decision


def _split(frame: pd.DataFrame, feature_cols: list[str], random_state: int, test_size: float):
    X = frame[feature_cols]
    y = frame["price"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _fit_eval(name: str, model, preprocessor, X_train, X_test, y_train, y_test) -> tuple[dict, Pipeline]:
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    preds = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    metrics.update({"model": name, "training_seconds": round(elapsed, 3)})
    logger.info(
        "%s — MAE: ₹%.2f | RMSE: ₹%.2f | R²: %.4f | %.1fs",
        name,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        elapsed,
    )
    return metrics, pipeline


def _candidate_models(random_state: int) -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=16,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=random_state,
        ),
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
        models["XGBoost"] = XGBRegressor(
            n_estimators=180,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    return models


def _select_best(rows: list[dict]) -> dict:
    return sorted(rows, key=lambda row: (row["mae"], row["rmse"], -row["r2"]))[0]


def train_model() -> None:
    config = load_config()
    random_state = int(config["random_state"])
    test_size = float(config["test_size"])
    logger.info("Initializing training pipeline...")

    raw = load_clean_dataset()
    report = build_quality_report(raw, "Clean_Dataset.csv")
    save_quality_report(report)

    cleaned = clean_clean_dataset(raw)
    featured = add_geographic_features(cleaned)
    processed_path = resolve_path(config["paths"]["processed"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(processed_path, index=False)
    logger.info("Wrote processed dataset to %s", processed_path)
    _document_dataset_decision(featured)

    logger.info("Feature engineering complete. Running geographic representation experiment...")
    experiments = {
        "A_categorical": CATEGORICAL + BASE_NUMERIC,
        "B_categorical_distance": CATEGORICAL + DISTANCE_NUMERIC,
        "C_categorical_geo_distance": CATEGORICAL + GEO_NUMERIC,
    }
    geo_results = []
    for name, cols in experiments.items():
        cat = [c for c in cols if c in CATEGORICAL]
        num = [c for c in cols if c not in CATEGORICAL]
        X_train, X_test, y_train, y_test = _split(featured, cols, random_state, test_size)
        estimator = HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.08, max_iter=120, random_state=random_state
        )
        metrics, _ = _fit_eval(
            name,
            estimator,
            build_preprocessor(cat, num),
            X_train,
            X_test,
            y_train,
            y_test,
        )
        geo_results.append({**metrics, "features": cols})

    geo_payload = {
        "estimator": "HistGradientBoostingRegressor",
        "note": "Same estimator and split used to isolate the effect of distance and coordinates.",
        "results": geo_results,
        "winner": _select_best(geo_results)["model"],
    }
    _json_dump(resolve_path(config["paths"]["geo_experiment"]), geo_payload)

    feature_cols = CATEGORICAL + GEO_NUMERIC
    X_train, X_test, y_train, y_test = _split(featured, feature_cols, random_state, test_size)
    logger.info("Training models on geographic feature set (%s train / %s test)", f"{len(X_train):,}", f"{len(X_test):,}")

    comparison_rows = []
    fitted = {}
    for name, estimator in _candidate_models(random_state).items():
        x_tr, y_tr = X_train, y_train
        x_te, y_te = X_test, y_test
        if name == "Gradient Boosting" and len(X_train) > 60000:
            sample = X_train.sample(n=60000, random_state=random_state)
            x_tr = sample
            y_tr = y_train.loc[sample.index]
        metrics, pipeline = _fit_eval(
            name,
            estimator,
            build_preprocessor(CATEGORICAL, GEO_NUMERIC),
            x_tr,
            x_te,
            y_tr,
            y_te,
        )
        if name == "Gradient Boosting" and len(X_train) > 60000:
            metrics["note"] = "Trained on a 60,000-row sample because full-data sklearn GradientBoosting is too slow."
        comparison_rows.append(metrics)
        fitted[name] = pipeline

    best_row = _select_best(comparison_rows)
    best_name = best_row["model"]
    best_pipeline = fitted[best_name]
    logger.info("Best model: %s", best_name)

    if config.get("training", {}).get("tune_best_model") and best_name in {"XGBoost", "HistGradientBoosting", "Random Forest"}:
        logger.info("Hyperparameter tuning on a subsample for %s", best_name)
        tune_n = min(int(config["training"].get("tune_sample_size", 40000)), len(X_train))
        sample = X_train.sample(n=tune_n, random_state=random_state)
        y_sample = y_train.loc[sample.index]
        if best_name == "XGBoost" and XGBRegressor is not None:
            search_model = XGBRegressor(random_state=random_state, n_jobs=-1, tree_method="hist")
            grid = {
                "model__n_estimators": [120, 180, 240],
                "model__max_depth": [6, 8, 10],
                "model__learning_rate": [0.05, 0.08, 0.12],
            }
        elif best_name == "Random Forest":
            search_model = RandomForestRegressor(n_jobs=-1, random_state=random_state)
            grid = {
                "model__n_estimators": [40, 70, 100],
                "model__max_depth": [12, 16, 20],
                "model__min_samples_leaf": [2, 4, 8],
            }
        else:
            search_model = HistGradientBoostingRegressor(random_state=random_state)
            grid = {
                "model__max_iter": [120, 180, 240],
                "model__max_depth": [6, 10, 14],
                "model__learning_rate": [0.05, 0.08, 0.12],
            }
        search = RandomizedSearchCV(
            Pipeline([("preprocessor", build_preprocessor(CATEGORICAL, GEO_NUMERIC)), ("model", search_model)]),
            grid,
            n_iter=int(config["training"].get("tune_iter", 8)),
            cv=int(config["training"].get("cv_folds", 3)),
            scoring="neg_mean_absolute_error",
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(sample, y_sample)
        logger.info("Best tuned params: %s", search.best_params_)
        tuned = search.best_estimator_
        tuned.fit(X_train, y_train)
        tuned_metrics = regression_metrics(y_test, tuned.predict(X_test))
        tuned_metrics.update({"model": f"{best_name} (tuned)", "training_seconds": None})
        comparison_rows.append(tuned_metrics)
        if tuned_metrics["mae"] <= best_row["mae"]:
            best_pipeline = tuned
            best_name = tuned_metrics["model"]
            best_row = {**tuned_metrics}

    importance = grouped_feature_importance(best_pipeline)
    y_pred = best_pipeline.predict(X_test)
    final_metrics = regression_metrics(y_test, y_pred)
    price_percentiles = featured["price"].quantile([0.33, 0.66]).to_dict()

    metadata = {
        "model_name": best_name,
        "target": "price",
        "random_state": random_state,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": feature_cols,
        "categorical_features": CATEGORICAL,
        "numerical_features": GEO_NUMERIC,
        "metrics": final_metrics,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "Clean_Dataset.csv",
        "unknown_category_handling": "OneHotEncoder(handle_unknown='ignore')",
        "geographic_features": True,
        "unsupported_features": config["features"]["unsupported_ui_fields"],
        "price_terciles": {"budget_max": float(price_percentiles[0.33]), "high_min": float(price_percentiles[0.66])},
        "airlines": sorted(featured["airline"].unique().tolist()),
        "classes": sorted(featured["class"].unique().tolist()),
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

    models_dir = resolve_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = resolve_path(config["paths"]["pipeline"])
    joblib.dump(best_pipeline, pipeline_path)
    joblib.dump(best_pipeline, models_dir / "airfare_model.pkl")
    logger.info("Saving model... %s", pipeline_path)

    _json_dump(resolve_path(config["paths"]["metrics"]), {"best_model": best_name, **final_metrics, "comparison": comparison_rows})
    _json_dump(resolve_path(config["paths"]["feature_importance"]), {"features": importance})
    _json_dump(resolve_path(config["paths"]["metadata"]), metadata)
    _json_dump(resolve_path(config["paths"]["comparison"]), {"rows": comparison_rows, "best_model": best_name})

    logger.info(
        "Evaluating models complete. Best model: %s MAE=₹%.2f R²=%.4f",
        best_name,
        final_metrics["mae"],
        final_metrics["r2"],
    )


if __name__ == "__main__":
    train_model()
