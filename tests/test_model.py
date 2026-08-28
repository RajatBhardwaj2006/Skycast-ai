from pathlib import Path

import pandas as pd
import pytest

from src.features.geo_features import add_geographic_features
from src.features.preprocess import build_preprocessor
from src.utils.paths import PROJECT_ROOT


def test_preprocessor_unknown_categories_do_not_crash():
    preprocessor = build_preprocessor(
        categorical=["airline", "source_city", "destination_city", "class", "stops", "departure_time", "arrival_time"],
        numerical=["duration", "days_left", "distance_km", "source_lat", "source_lon", "destination_lat", "destination_lon"],
    )
    train = pd.DataFrame(
        {
            "airline": ["Vistara"],
            "source_city": ["Delhi"],
            "destination_city": ["Mumbai"],
            "class": ["Economy"],
            "stops": ["zero"],
            "departure_time": ["Morning"],
            "arrival_time": ["Evening"],
            "duration": [2.2],
            "days_left": [10],
            "distance_km": [1150.0],
            "source_lat": [28.5],
            "source_lon": [77.1],
            "destination_lat": [19.0],
            "destination_lon": [72.8],
        }
    )
    preprocessor.fit(train)
    unseen = train.copy()
    unseen["source_city"] = "Leh"
    unseen["destination_city"] = "Delhi"
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1


def test_geo_features_add_distance():
    frame = pd.DataFrame({"source_city": ["Delhi"], "destination_city": ["Mumbai"]})
    enriched = add_geographic_features(frame)
    assert enriched.loc[0, "distance_km"] > 1000
    assert pd.notna(enriched.loc[0, "source_lat"])


def test_saved_pipeline_predicts_when_present():
    path = PROJECT_ROOT / "models" / "airfare_pipeline.pkl"
    if not path.exists():
        pytest.skip("Train the model first: python -m src.models.train")
    import joblib

    pipeline = joblib.load(path)
    row = pd.DataFrame(
        [
            {
                "airline": "Air India",
                "departure_time": "Morning",
                "arrival_time": "Evening",
                "stops": "zero",
                "class": "Economy",
                "source_city": "Delhi",
                "destination_city": "Mumbai",
                "duration": 2.25,
                "days_left": 20,
                "source_lat": 28.5562,
                "source_lon": 77.1,
                "destination_lat": 19.0896,
                "destination_lon": 72.8656,
                "distance_km": 1148.0,
            }
        ]
    )
    pred = float(pipeline.predict(row)[0])
    assert pred == pred
    assert pred > 0
