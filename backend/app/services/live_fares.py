from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
import requests

from src.utils.config import get_logger

logger = get_logger("skycast.live_fares")

# Token cache for Amadeus OAuth2
_AMADEUS_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0}


def get_live_fares(
    origin_iata: str,
    destination_iata: str,
    departure_date: str | None = None,
    cabin_class: str = "ECONOMY",
    adults: int = 1,
) -> dict[str, Any]:
    """Search live airline flight offers via Amadeus or Duffel API if credentials are configured.
    
    If credentials are not present in the environment, returns a clean unconfigured status.
    DO NOT return fabricated/fake live prices.
    """
    amadeus_client_id = os.environ.get("AMADEUS_CLIENT_ID")
    amadeus_client_secret = os.environ.get("AMADEUS_CLIENT_SECRET")
    duffel_token = os.environ.get("DUFFEL_API_TOKEN")

    if not departure_date:
        # Default to 15 days from now for live fare comparison
        departure_date = (datetime.now(timezone.utc) + timedelta(days=15)).strftime("%Y-%m-%d")

    # If neither Amadeus nor Duffel credentials are configured:
    if not (amadeus_client_id and amadeus_client_secret) and not duffel_token:
        return {
            "configured": False,
            "provider": None,
            "status": "live_offers_unconfigured",
            "message": (
                "Real-time live flight search is unconfigured. Set AMADEUS_CLIENT_ID and "
                "AMADEUS_CLIENT_SECRET or DUFFEL_API_TOKEN in the environment to query live carrier quotes."
            ),
            "query": {
                "origin": origin_iata,
                "destination": destination_iata,
                "date": departure_date,
                "cabin": cabin_class,
            },
            "live_offers_count": 0,
            "lowest_fare": None,
            "median_fare": None,
            "highest_fare": None,
            "currency": "INR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 1. Try Amadeus API
    if amadeus_client_id and amadeus_client_secret:
        try:
            return _query_amadeus(
                origin_iata, destination_iata, departure_date, cabin_class, adults,
                amadeus_client_id, amadeus_client_secret
            )
        except Exception as exc:
            logger.warning("Amadeus live flight search failed: %s", exc)

    # 2. Try Duffel API
    if duffel_token:
        try:
            return _query_duffel(
                origin_iata, destination_iata, departure_date, cabin_class, adults, duffel_token
            )
        except Exception as exc:
            logger.warning("Duffel live flight search failed: %s", exc)

    return {
        "configured": True,
        "provider": "Amadeus/Duffel",
        "status": "live_search_unavailable",
        "message": "Live fare search provider returned an error or no flight offers were available for this route.",
        "query": {
            "origin": origin_iata,
            "destination": destination_iata,
            "date": departure_date,
            "cabin": cabin_class,
        },
        "live_offers_count": 0,
        "lowest_fare": None,
        "median_fare": None,
        "highest_fare": None,
        "currency": "INR",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_amadeus_token(client_id: str, client_secret: str) -> str:
    now = time.time()
    if _AMADEUS_TOKEN_CACHE["token"] and _AMADEUS_TOKEN_CACHE["expires_at"] > now + 60:
        return _AMADEUS_TOKEN_CACHE["token"]

    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    res = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=10,
    )
    res.raise_for_status()
    payload = res.json()
    _AMADEUS_TOKEN_CACHE["token"] = payload["access_token"]
    _AMADEUS_TOKEN_CACHE["expires_at"] = now + payload.get("expires_in", 1799)
    return _AMADEUS_TOKEN_CACHE["token"]


def _query_amadeus(
    origin: str,
    destination: str,
    date: str,
    cabin: str,
    adults: int,
    client_id: str,
    client_secret: str,
) -> dict[str, Any]:
    token = _get_amadeus_token(client_id, client_secret)
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": adults,
        "travelClass": cabin.upper(),
        "currencyCode": "INR",
        "max": 10,
    }
    res = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=12)
    res.raise_for_status()
    data = res.json()
    offers = data.get("data", [])
    if not offers:
        return {
            "configured": True,
            "provider": "Amadeus Flight Offers",
            "status": "no_offers_found",
            "message": f"No live scheduled flight offers found for {origin} -> {destination} on {date}.",
            "live_offers_count": 0,
            "lowest_fare": None,
            "median_fare": None,
            "highest_fare": None,
            "currency": "INR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    prices = []
    carriers = set()
    for off in offers:
        try:
            p = float(off["price"]["grandTotal"])
            prices.append(p)
            for it in off.get("itineraries", []):
                for seg in it.get("segments", []):
                    carriers.add(seg.get("carrierCode", ""))
        except (KeyError, ValueError):
            continue

    prices.sort()
    return {
        "configured": True,
        "provider": "Amadeus Flight Offers (Live)",
        "status": "live_offers_retrieved",
        "message": f"Retrieved {len(prices)} current flight offers from airlines.",
        "live_offers_count": len(prices),
        "lowest_fare": prices[0] if prices else None,
        "median_fare": prices[len(prices) // 2] if prices else None,
        "highest_fare": prices[-1] if prices else None,
        "currency": "INR",
        "carriers": sorted(c for c in carriers if c),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _query_duffel(
    origin: str,
    destination: str,
    date: str,
    cabin: str,
    adults: int,
    token: str,
) -> dict[str, Any]:
    url = "https://api.duffel.com/air/offer_requests"
    headers = {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": "v2",
        "Content-Type": "application/json",
    }
    cabin_map = {"ECONOMY": "economy", "BUSINESS": "business", "PREMIUM_ECONOMY": "premium_economy"}
    payload = {
        "data": {
            "slices": [{"origin": origin, "destination": destination, "departure_date": date}],
            "passengers": [{"type": "adult"} for _ in range(adults)],
            "cabin_class": cabin_map.get(cabin.upper(), "economy"),
        }
    }
    res = requests.post(url, json=payload, headers=headers, timeout=12)
    res.raise_for_status()
    data = res.json().get("data", {})
    offers = data.get("offers", [])
    prices = []
    for off in offers:
        try:
            amt = float(off.get("total_amount", 0))
            if amt > 0:
                prices.append(amt)
        except (ValueError, TypeError):
            continue

    prices.sort()
    return {
        "configured": True,
        "provider": "Duffel API (Live)",
        "status": "live_offers_retrieved",
        "message": f"Retrieved {len(prices)} current live flight offers via Duffel.",
        "live_offers_count": len(prices),
        "lowest_fare": prices[0] if prices else None,
        "median_fare": prices[len(prices) // 2] if prices else None,
        "highest_fare": prices[-1] if prices else None,
        "currency": "INR",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
