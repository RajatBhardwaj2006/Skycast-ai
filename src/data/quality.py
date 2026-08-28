from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_logger, load_config, resolve_path

logger = get_logger("skycast.quality")


def build_quality_report(frame: pd.DataFrame, name: str = "dataset") -> dict:
    numeric = frame.select_dtypes(include=[np.number])
    report = {
        "name": name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": list(map(str, frame.columns)),
        "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        "missing_values": {str(col): int(val) for col, val in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_values": {str(col): int(frame[col].nunique(dropna=True)) for col in frame.columns},
        "numerical_statistics": {},
        "target_statistics": {},
        "potential_outliers": {},
    }
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        report["numerical_statistics"][str(col)] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }
        report["potential_outliers"][str(col)] = int(outlier_mask.sum())

    if "price" in frame.columns:
        price = pd.to_numeric(frame["price"], errors="coerce").dropna()
        report["target_statistics"] = {
            "min": float(price.min()),
            "max": float(price.max()),
            "mean": float(price.mean()),
            "median": float(price.median()),
            "std": float(price.std()),
        }
    return report


def save_quality_report(report: dict, path: Path | None = None) -> Path:
    out = path or resolve_path(load_config()["paths"]["quality_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved data quality report to %s", out)
    return out
