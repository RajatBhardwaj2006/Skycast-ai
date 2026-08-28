from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.geo.haversine import haversine_km
from src.utils.config import load_config, resolve_path


@dataclass(frozen=True)
class Location:
    city: str
    airport: str
    iata: str
    state: str
    country: str
    latitude: float
    longitude: float

    def to_dict(self) -> dict:
        return asdict(self)


class LocationNotFoundError(LookupError):
    pass


class LocationService:
    def __init__(self, csv_path: Path | None = None):
        config = load_config()
        self.csv_path = csv_path or resolve_path(config["paths"]["airports"])
        self._frame = self._load()
        self._by_iata: dict[str, Location] = {}
        self._by_city: dict[str, Location] = {}
        self._search_index: list[tuple[Location, str]] = []
        self._build_indexes()

    def _load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Airport reference dataset not found: {self.csv_path}")
        frame = pd.read_csv(self.csv_path)
        required = {"city", "airport", "iata", "state", "country", "latitude", "longitude"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Airport dataset missing columns: {sorted(missing)}")
        frame["iata"] = frame["iata"].astype(str).str.strip().str.upper()
        frame["city"] = frame["city"].astype(str).str.strip()
        frame["aliases"] = frame.get("aliases", "").fillna("").astype(str)
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame = frame.dropna(subset=["latitude", "longitude", "iata"])
        # Keep the first canonical row per IATA (reference file may include aliases as extra rows).
        frame = frame.drop_duplicates(subset=["iata"], keep="first").reset_index(drop=True)
        return frame

    def _row_to_location(self, row: pd.Series) -> Location:
        return Location(
            city=str(row["city"]),
            airport=str(row["airport"]),
            iata=str(row["iata"]).upper(),
            state=str(row["state"]),
            country=str(row["country"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )

    def _build_indexes(self) -> None:
        alias_lookup: dict[str, str] = {}
        for _, row in self._frame.iterrows():
            location = self._row_to_location(row)
            self._by_iata[location.iata] = location
            self._by_city[location.city.casefold()] = location
            tokens = {
                location.city,
                location.airport,
                location.iata,
                location.state,
            }
            for alias in str(row["aliases"]).split("|"):
                alias = alias.strip()
                if alias:
                    tokens.add(alias)
                    alias_lookup[alias.casefold()] = location.iata
            blob = " ".join(tokens).casefold()
            self._search_index.append((location, blob))
        # Map common city names from extra CSV rows that were dropped during IATA dedupe.
        raw = pd.read_csv(self.csv_path)
        raw["iata"] = raw["iata"].astype(str).str.strip().str.upper()
        for _, row in raw.iterrows():
            iata = row["iata"]
            if iata in self._by_iata:
                self._by_city[str(row["city"]).strip().casefold()] = self._by_iata[iata]
                for alias in str(row.get("aliases", "")).split("|"):
                    alias = alias.strip()
                    if alias:
                        self._by_city[alias.casefold()] = self._by_iata[iata]

    def all_locations(self) -> list[Location]:
        return list(self._by_iata.values())

    def get_by_iata(self, iata: str) -> Location:
        key = (iata or "").strip().upper()
        if key not in self._by_iata:
            raise LocationNotFoundError(key)
        return self._by_iata[key]

    def get_by_city(self, city: str) -> Location:
        key = (city or "").strip().casefold()
        if key not in self._by_city:
            raise LocationNotFoundError(city)
        return self._by_city[key]

    def resolve(self, query: str) -> Location:
        text = (query or "").strip()
        if not text:
            raise LocationNotFoundError(query)
        upper = text.upper()
        if len(upper) == 3 and upper in self._by_iata:
            return self._by_iata[upper]
        key = text.casefold()
        if key in self._by_city:
            return self._by_city[key]
        matches = self.search(text, limit=1)
        if matches:
            return matches[0]
        raise LocationNotFoundError(query)

    def search(self, query: str, limit: int = 8) -> list[Location]:
        q = (query or "").strip().casefold()
        if len(q) < 1:
            return []
        scored: list[tuple[int, str, Location]] = []
        seen: set[str] = set()
        for location, blob in self._search_index:
            if location.iata in seen:
                continue
            city = location.city.casefold()
            iata = location.iata.casefold()
            airport = location.airport.casefold()
            if q == iata:
                score = 0
            elif city.startswith(q) or iata.startswith(q):
                score = 1
            elif q in city or q in iata:
                score = 2
            elif airport.startswith(q) or q in airport or q in blob:
                score = 3
            else:
                continue
            seen.add(location.iata)
            scored.append((score, location.city, location))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored[:limit]]

    def distance_between(self, origin: Location, destination: Location) -> float:
        return haversine_km(origin.latitude, origin.longitude, destination.latitude, destination.longitude)

    def known_training_city(self, city: str) -> bool:
        known = {name.casefold() for name in load_config().get("known_training_cities", [])}
        return city.strip().casefold() in known


@lru_cache(maxsize=1)
def default_location_service() -> LocationService:
    return LocationService()
