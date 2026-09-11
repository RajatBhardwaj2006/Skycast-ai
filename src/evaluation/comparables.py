from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path


@functools.lru_cache(maxsize=1)
def load_comparables_data() -> pd.DataFrame:
    config = load_config()
    processed_path = resolve_path(config["paths"]["processed"])
    if not processed_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(processed_path, low_memory=False)
    # Filter to genuine records where price and distance are valid
    valid = df[df["price"].notna() & (df["price"] > 0) & df["distance_km"].notna()].copy()
    return valid


def find_nearest_comparables(
    query: dict[str, Any],
    k: int = 8,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Find genuine historical observations that are physically and commercially similar to the query.
    
    This is a diagnostic explanation layer, NOT a price override.
    """
    df = dataset if dataset is not None else load_comparables_data()
    if df.empty:
        return {
            "comparables": [],
            "count": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "note": "Historical flight dataset not available for comparables.",
        }

    target_class = str(query.get("class", "Economy")).strip().title()
    target_airline = str(query.get("airline", "")).strip()
    target_stops = str(query.get("stops", "zero")).strip()
    target_dist = float(query.get("distance_km", 1000.0))
    target_dur = float(query.get("duration", 2.0))
    target_days = float(query.get("days_left", 15.0))

    # 1. Filter by class if known
    sub = df[df["class"].astype(str).str.title() == target_class].copy()
    if sub.empty:
        sub = df.copy()

    # 2. Prefer same stops if enough records exist
    if target_stops in sub["stops"].values:
        same_stops = sub[sub["stops"] == target_stops]
        if len(same_stops) >= k:
            sub = same_stops

    # 3. Calculate feature distance in normalized space
    d_dist = (sub["distance_km"] - target_dist) / 400.0
    d_dur = (sub["duration"] - target_dur) / 2.0
    d_days = (sub["days_left"].fillna(26.0) - target_days) / 10.0
    same_airline = (sub["airline"].astype(str).str.casefold() == target_airline.casefold()).astype(float)

    # Combined similarity score (lower is closer; same airline gets distance bonus)
    sub["similarity_score"] = (d_dist**2) + (d_dur**2) + (d_days**2) - (same_airline * 0.4)

    top = sub.sort_values("similarity_score").head(k)
    records = []
    for _, row in top.iterrows():
        records.append({
            "source_city": str(row["source_city"]),
            "destination_city": str(row["destination_city"]),
            "airline": str(row["airline"]),
            "stops": str(row["stops"]),
            "duration": round(float(row["duration"]), 2),
            "days_left": int(row["days_left"]) if pd.notna(row["days_left"]) else None,
            "distance_km": round(float(row["distance_km"]), 1),
            "price": float(row["price"]),
        })

    prices = [r["price"] for r in records]
    if not prices:
        return {"comparables": [], "count": 0, "median": None, "mean": None, "min": None, "max": None}

    med = round(float(np.median(prices)), 2)
    mean_val = round(float(np.mean(prices)), 2)
    min_val = round(float(np.min(prices)), 2)
    max_val = round(float(np.max(prices)), 2)

    return {
        "comparables": records,
        "count": len(records),
        "median": med,
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "target_class": target_class,
        "note": f"Found {len(records)} closest historical flight tickets across distance, duration, stops, and booking window.",
    }
