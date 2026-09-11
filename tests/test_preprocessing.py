import pandas as pd
import numpy as np
from src.features.preprocess import build_preprocessor

def test_build_preprocessor_structure():
    preprocessor = build_preprocessor(
        categorical=["airline", "class", "stops"],
        numerical=["duration", "distance_km"]
    )
    assert hasattr(preprocessor, "transformers")
    assert len(preprocessor.transformers) == 2

def test_build_preprocessor_handles_missing_values():
    preprocessor = build_preprocessor(
        categorical=["airline", "class"],
        numerical=["duration", "days_left"]
    )
    train_df = pd.DataFrame({
        "airline": ["IndiGo", "Air India", "SpiceJet"],
        "class": ["Economy", "Business", np.nan],
        "duration": [2.5, np.nan, 4.0],
        "days_left": [10, 20, np.nan],
    })
    transformed = preprocessor.fit_transform(train_df)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()

def test_build_preprocessor_handles_unseen_categories():
    preprocessor = build_preprocessor(
        categorical=["airline"],
        numerical=["duration"]
    )
    train_df = pd.DataFrame({
        "airline": ["IndiGo", "Air India"],
        "duration": [2.5, 3.0],
    })
    test_df = pd.DataFrame({
        "airline": ["UnseenAir"],
        "duration": [2.0],
    })
    preprocessor.fit(train_df)
    out = preprocessor.transform(test_df)
    assert out.shape[0] == 1
