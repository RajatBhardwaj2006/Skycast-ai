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
    
    Strictly prioritizes exact directional route matches (e.g. DEL -> BLR) to prevent
    directional cross-contamination with reverse routes (e.g. BLR -> DEL).
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
    target_source = str(query.get("source_city", "")).strip().title()
    target_destination = str(query.get("destination_city", "")).strip().title()

    # 1. Filter by class if known
    sub = df[df["class"].astype(str).str.title() == target_class].copy()
    if sub.empty:
        sub = df.copy()

    # 2. Check directional and reverse route availability
    is_exact = (
        (sub["source_city"].astype(str).str.title() == target_source)
        & (sub["destination_city"].astype(str).str.title() == target_destination)
    ) if target_source and target_destination else pd.Series(False, index=sub.index)

    is_reverse = (
        (sub["source_city"].astype(str).str.title() == target_destination)
        & (sub["destination_city"].astype(str).str.title() == target_source)
    ) if target_source and target_destination else pd.Series(False, index=sub.index)

    same_airline = (sub["airline"].astype(str).str.casefold() == target_airline.casefold())
    same_stops = (sub["stops"].astype(str).str.casefold() == target_stops.casefold())

    # 3. Calculate feature distance in normalized space
    d_dist = (sub["distance_km"] - target_dist) / 300.0
    d_dur = (sub["duration"] - target_dur) / 2.0
    d_days = (sub["days_left"].fillna(26.0) - target_days) / 8.0

    # Distance scoring: exact route gets massive priority (-10.0), reverse route gets modest bonus (-2.0)
    sub["similarity_score"] = (
        (d_dist**2)
        + (d_dur**2)
        + (d_days**2)
        - (is_exact.astype(float) * 10.0)
        - (is_reverse.astype(float) * 2.0)
        - (same_airline.astype(float) * 0.8)
        - (same_stops.astype(float) * 0.5)
    )

    top = sub.sort_values("similarity_score").head(k)
    records = []
    exact_count = 0
    for _, row in top.iterrows():
        s_city = str(row["source_city"]).title()
        d_city = str(row["destination_city"]).title()
        if target_source and target_destination and s_city == target_source and d_city == target_destination:
            match_type = "exact_route"
            exact_count += 1
        elif target_source and target_destination and s_city == target_destination and d_city == target_source:
            match_type = "reverse_route"
        else:
            match_type = "similar_corridor"

        records.append({
            "source_city": s_city,
            "destination_city": d_city,
            "airline": str(row["airline"]),
            "stops": str(row["stops"]),
            "duration": round(float(row["duration"]), 2),
            "days_left": int(row["days_left"]) if pd.notna(row["days_left"]) else None,
            "distance_km": round(float(row["distance_km"]), 1),
            "price": float(row["price"]),
            "match_type": match_type,
        })

    prices = [r["price"] for r in records]
    if not prices:
        return {"comparables": [], "count": 0, "median": None, "mean": None, "min": None, "max": None}

    med = round(float(np.median(prices)), 2)
    mean_val = round(float(np.mean(prices)), 2)
    min_val = round(float(np.min(prices)), 2)
    max_val = round(float(np.max(prices)), 2)

    corridor_note = (
        f"Found {len(records)} directional historical flights for {target_source} → {target_destination}."
        if exact_count == len(records) and target_source and target_destination
        else f"Found {len(records)} closest historical flight tickets across distance, duration, stops, and booking window."
    )

    return {
        "comparables": records,
        "count": len(records),
        "median": med,
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "target_class": target_class,
        "route_match": "exact_directional" if exact_count == len(records) else "corridor_fallback",
        "note": corridor_note,
    }
