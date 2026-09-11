from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import HTTPException

from backend.app.config import (
    COMPARISON_PATH,
    DECISION_PATH,
    GEO_EXPERIMENT_PATH,
    IMPORTANCE_PATH,
    MARKET_CALIBRATOR_PATH,
    MARKET_VALIDATION_PATH,
    METADATA_PATH,
    METRICS_PATH,
    PIPELINE_PATH,
    QUALITY_PATH,
    VALIDATION_PATH,
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Artifact missing: {path.name}. Train the model first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Run `python -m src.models.train` from the SkyCast root.",
        )
    try:
        return joblib.load(PIPELINE_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load the saved model: {exc}") from exc


def load_metrics() -> dict:
    return _read_json(METRICS_PATH)


def load_importance() -> dict:
    return _read_json(IMPORTANCE_PATH)


def load_metadata() -> dict:
    return _read_json(METADATA_PATH)


def load_geo_experiment() -> dict:
    return _read_json(GEO_EXPERIMENT_PATH)


def load_comparison() -> dict:
    return _read_json(COMPARISON_PATH)


def load_quality() -> dict:
    return _read_json(QUALITY_PATH)


def load_decision() -> dict:
    if not DECISION_PATH.exists():
        return {}
    return json.loads(DECISION_PATH.read_text(encoding="utf-8"))


def load_validation() -> dict:
    return _read_json(VALIDATION_PATH)


@lru_cache(maxsize=1)
def load_market_calibrator():
    from src.models.market_calibrator import MarketCalibrator
    return MarketCalibrator.load(MARKET_CALIBRATOR_PATH)


def load_market_validation() -> list[dict]:
    if not MARKET_VALIDATION_PATH.exists():
        return []
    return json.loads(MARKET_VALIDATION_PATH.read_text(encoding="utf-8"))
