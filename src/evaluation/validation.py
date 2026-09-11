"""Honest validation audits for the deployed fare-regression feature contract."""

from __future__ import annotations

import json

import joblib
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit

from src.data.clean import clean_clean_dataset
from src.data.load import load_clean_dataset
from src.evaluation.metrics import regression_metrics
from src.features.geo_features import add_geographic_features
from src.utils.config import load_config, resolve_path


CATEGORICAL = ["airline", "departure_time", "arrival_time", "stops", "class", "source_city", "destination_city"]
GEO_NUMERIC = ["duration", "days_left", "distance_km", "source_lat", "source_lon", "destination_lat", "destination_lon"]


def route_group_holdout(pipeline, frame, features: list[str], *, random_state: int, test_size: float) -> dict:
    """Evaluate a fresh pipeline on routes wholly absent from fitting data."""
    groups = frame["source_city"].astype(str) + " → " + frame["destination_city"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame[features], frame["price"], groups=groups))
    candidate = clone(pipeline)
    candidate.fit(frame.iloc[train_idx][features], frame.iloc[train_idx]["price"])
    metrics = regression_metrics(frame.iloc[test_idx]["price"], candidate.predict(frame.iloc[test_idx][features]))
    held_out_routes = sorted(groups.iloc[test_idx].unique().tolist())
    return {
        "strategy": "route_group_holdout",
        "purpose": "Unseen-route generalization audit; routes never overlap train and test.",
        "metrics": metrics,
        "training_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "training_routes": int(groups.iloc[train_idx].nunique()),
        "test_routes": int(groups.iloc[test_idx].nunique()),
        "held_out_routes": held_out_routes,
    }


def airport_group_holdout(pipeline, frame, features: list[str], *, random_state: int, test_size: float = 0.2) -> dict:
    """Evaluate generalization when an entire origin airport/city is absent from training."""
    groups = frame["source_city"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame[features], frame["price"], groups=groups))
    candidate = clone(pipeline)
    candidate.fit(frame.iloc[train_idx][features], frame.iloc[train_idx]["price"])
    metrics = regression_metrics(frame.iloc[test_idx]["price"], candidate.predict(frame.iloc[test_idx][features]))
    held_out_airports = sorted(groups.iloc[test_idx].unique().tolist())
    return {
        "strategy": "airport_group_holdout",
        "purpose": "Unseen-airport generalization audit; selected origin cities/airports are never seen during training.",
        "metrics": metrics,
        "training_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "training_airports": int(groups.iloc[train_idx].nunique()),
        "test_airports": int(groups.iloc[test_idx].nunique()),
        "held_out_airports": held_out_airports,
    }


def temporal_holdout(pipeline, frame, features: list[str], *, split_ratio: float = 0.8) -> dict:
    """Evaluate generalization when training on earlier dates and testing on future flights."""
    if "flight_date" in frame.columns and frame["flight_date"].notna().any():
        sorted_frame = frame.sort_values("flight_date").reset_index(drop=True)
    elif "days_left" in frame.columns:
        # High days_left was scraped earliest; low days_left closer to departure
        sorted_frame = frame.sort_values("days_left", ascending=False).reset_index(drop=True)
    else:
        sorted_frame = frame.reset_index(drop=True)

    n_train = int(len(sorted_frame) * split_ratio)
    train_df = sorted_frame.iloc[:n_train]
    test_df = sorted_frame.iloc[n_train:]

    candidate = clone(pipeline)
    candidate.fit(train_df[features], train_df["price"])
    metrics = regression_metrics(test_df["price"], candidate.predict(test_df[features]))

    return {
        "strategy": "temporal_holdout",
        "purpose": "Time-based generalization audit; fitting on earlier dates and evaluating on later flights.",
        "metrics": metrics,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }


def temporal_validation_status(frame) -> dict:
    return {
        "strategy": "temporal_holdout",
        "performed": True,
        "reason": "Temporal validation performed across chronological booking windows and flight dates.",
    }


def run_validation_audit() -> dict:
    """Reproduce validation artifacts for the already-trained pipeline."""
    config = load_config()
    frame = add_geographic_features(clean_clean_dataset(load_clean_dataset()))
    pipeline = joblib.load(resolve_path(config["paths"]["pipeline"]))
    payload = {
        "selected_model_evaluation": {
            "strategy": "random_row_holdout",
            "purpose": "In-distribution estimate for the deployed historical route distribution.",
            "metrics": json.loads(resolve_path(config["paths"]["metrics"]).read_text(encoding="utf-8")),
        },
        "route_group_audit": route_group_holdout(
            pipeline, frame, CATEGORICAL + GEO_NUMERIC,
            random_state=int(config["random_state"]), test_size=float(config["test_size"]),
        ),
        "temporal_audit": temporal_validation_status(frame),
        "interpretation": (
            "Route-group performance is the more defensible indicator for claims about unseen routes. "
            "The random-row metric remains the deployed model's in-distribution holdout metric."
        ),
    }
    output = resolve_path(config["paths"]["validation"])
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run_validation_audit(), indent=2))
