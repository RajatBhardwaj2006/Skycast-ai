from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two WGS84 points."""
    if any(value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))) for value in (lat1, lon1, lat2, lon2)):
        raise ValueError("Coordinates must be finite numbers.")

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    distance = EARTH_RADIUS_KM * c
    if distance < 0 or math.isnan(distance) or math.isinf(distance):
        raise ValueError("Invalid distance computation.")
    return distance
