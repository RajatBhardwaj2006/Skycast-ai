from __future__ import annotations

import pandas as pd

from src.geo.haversine import haversine_km
from src.geo.locations import LocationNotFoundError, LocationService, default_location_service
from src.utils.config import get_logger

logger = get_logger("skycast.geo")


def add_geographic_features(
    frame: pd.DataFrame,
    source_col: str = "source_city",
    dest_col: str = "destination_city",
    locations: LocationService | None = None,
) -> pd.DataFrame:
    service = locations or default_location_service()
    data = frame.copy()

    cache: dict[str, tuple[float, float]] = {}

    def coords(city: str) -> tuple[float, float]:
        key = str(city)
        if key not in cache:
            loc = service.resolve(key)
            cache[key] = (loc.latitude, loc.longitude)
        return cache[key]

    source_coords = data[source_col].map(coords)
    dest_coords = data[dest_col].map(coords)
    data["source_lat"] = [pair[0] for pair in source_coords]
    data["source_lon"] = [pair[1] for pair in source_coords]
    data["destination_lat"] = [pair[0] for pair in dest_coords]
    data["destination_lon"] = [pair[1] for pair in dest_coords]
    data["distance_km"] = [
        haversine_km(slat, slon, dlat, dlon)
        for slat, slon, dlat, dlon in zip(
            data["source_lat"], data["source_lon"], data["destination_lat"], data["destination_lon"]
        )
    ]
    if data[["source_lat", "source_lon", "destination_lat", "destination_lon", "distance_km"]].isna().any().any():
        raise LocationNotFoundError("One or more training cities could not be geocoded.")
    logger.info("Added geographic features including distance_km")
    return data
