from __future__ import annotations

import pandas as pd

from src.utils.config import get_logger

logger = get_logger("skycast.clean")

AIRLINE_MAP = {
    "air_india": "Air India",
    "air india": "Air India",
    "air india express": "Air India",
    "air_india_express": "Air India",
    "akasa air": "SpiceJet",
    "akasa": "SpiceJet",
    "go_first": "GO FIRST",
    "go first": "GO FIRST",
    "goair": "GO FIRST",
    "indigo": "Indigo",
    "airasia": "AirAsia",
    "air asia": "AirAsia",
    "spicejet": "SpiceJet",
    "vistara": "Vistara",
    "jet_airways": "Jet Airways",
    "jet airways": "Jet Airways",
    "multiple_carriers": "Multiple carriers",
    "multiple carriers": "Multiple carriers",
    "trujet": "Trujet",

}

TIME_MAP = {
    "early_morning": "Early Morning",
    "late_night": "Late Night",
    "morning": "Morning",
    "afternoon": "Afternoon",
    "evening": "Evening",
    "night": "Night",
}

STOPS_MAP = {
    "zero": "zero",
    "0": "zero",
    "non-stop": "zero",
    "nonstop": "zero",
    "non stop": "zero",
    "one": "one",
    "1": "one",
    "1-stop": "one",
    "1 stop": "one",
    "two_or_more": "two_or_more",
    "two or more": "two_or_more",
    "2+-stop": "two_or_more",
    "2+": "two_or_more",
    "2": "two_or_more",
    "2 stops": "two_or_more",
    "3": "two_or_more",
    "3 stops": "two_or_more",
    "4": "two_or_more",
    "4 stops": "two_or_more",
}


def _normalize_time_bin(value: object) -> str:
    text = str(value).strip()
    norm = text.casefold().replace("-", " ").replace("_", " ")
    for k, v in TIME_MAP.items():
        if norm == k or norm == v.casefold():
            return v
    parts = text.split()
    time_str = parts[0] if ":" in parts[0] else (parts[1] if len(parts) > 1 and ":" in parts[1] else "")
    if ":" in time_str:
        try:
            hour = int(time_str.split(":")[0])
            if 0 <= hour < 4:
                return "Late Night"
            elif 4 <= hour < 8:
                return "Early Morning"
            elif 8 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 16:
                return "Afternoon"
            elif 16 <= hour < 20:
                return "Evening"
            else:
                return "Night"
        except Exception:
            pass
    return TIME_MAP.get(norm, text.replace("_", " ").title())


def _normalize_label(value: object, mapping: dict[str, str]) -> str:
    text = str(value).strip()
    return mapping.get(text.casefold().replace("-", " ").replace("__", "_"), mapping.get(text.casefold(), text))


def clean_clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    data = frame.copy()
    unnamed = [col for col in data.columns if str(col).startswith("Unnamed")]
    if unnamed:
        data = data.drop(columns=unnamed)

    before = len(data)
    data = data.drop_duplicates()
    logger.info("Removed %s duplicate rows", before - len(data))

    data["airline"] = data["airline"].map(lambda v: AIRLINE_MAP.get(str(v).strip().casefold(), str(v).replace("_", " ")))
    data["departure_time"] = data["departure_time"].map(_normalize_time_bin)
    data["arrival_time"] = data["arrival_time"].map(_normalize_time_bin)
    data["stops"] = data["stops"].map(lambda v: STOPS_MAP.get(str(v).strip().casefold(), str(v)))
    data["class"] = data["class"].fillna("Unknown").astype(str).str.strip().str.title()
    data["class"] = data["class"].replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
    city_map = {'Banglore': 'Bangalore', 'New Delhi': 'Delhi', 'Cochin': 'Kochi'}
    data["source_city"] = data["source_city"].astype(str).str.strip().replace(city_map)
    data["destination_city"] = data["destination_city"].astype(str).str.strip().replace(city_map)

    data["duration"] = pd.to_numeric(data["duration"], errors="coerce")
    data["days_left"] = pd.to_numeric(data["days_left"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    invalid = (
        data["duration"].isna()
        | (data["duration"] <= 0)
        | data["price"].isna()
        | (data["price"] <= 0)
        | data["source_city"].eq(data["destination_city"])
    )
    dropped = int(invalid.sum())
    if dropped:
        logger.info("Dropping %s invalid rows (duration/price/same-city)", dropped)
        data = data.loc[~invalid].copy()

    data["route"] = data["source_city"] + "_" + data["destination_city"]
    logger.info("Cleaned dataset: %s rows", f"{len(data):,}")
    return data.reset_index(drop=True)
