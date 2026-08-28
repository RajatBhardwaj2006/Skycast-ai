from __future__ import annotations

import pandas as pd

from src.utils.config import get_logger

logger = get_logger("skycast.clean")

AIRLINE_MAP = {
    "air_india": "Air India",
    "air india": "Air India",
    "go_first": "GO FIRST",
    "go first": "GO FIRST",
    "indigo": "Indigo",
    "indiGo": "Indigo",
    "airasia": "AirAsia",
    "spicejet": "SpiceJet",
    "vistara": "Vistara",
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
    "non-stop": "zero",
    "nonstop": "zero",
    "non stop": "zero",
    "one": "one",
    "1": "one",
    "1-stop": "one",
    "two_or_more": "two_or_more",
    "two or more": "two_or_more",
    "2+-stop": "two_or_more",
    "2+": "two_or_more",
}


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
    data["departure_time"] = data["departure_time"].map(lambda v: TIME_MAP.get(str(v).strip().casefold(), str(v).replace("_", " ")))
    data["arrival_time"] = data["arrival_time"].map(lambda v: TIME_MAP.get(str(v).strip().casefold(), str(v).replace("_", " ")))
    data["stops"] = data["stops"].map(lambda v: STOPS_MAP.get(str(v).strip().casefold(), str(v)))
    data["class"] = data["class"].astype(str).str.strip().str.title()
    data["source_city"] = data["source_city"].astype(str).str.strip()
    data["destination_city"] = data["destination_city"].astype(str).str.strip()

    data["duration"] = pd.to_numeric(data["duration"], errors="coerce")
    data["days_left"] = pd.to_numeric(data["days_left"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    invalid = (
        data["duration"].isna()
        | (data["duration"] <= 0)
        | data["days_left"].isna()
        | (data["days_left"] < 0)
        | data["price"].isna()
        | (data["price"] <= 0)
        | data["source_city"].eq(data["destination_city"])
    )
    dropped = int(invalid.sum())
    if dropped:
        logger.info("Dropping %s invalid rows (duration/price/days_left/same-city)", dropped)
        data = data.loc[~invalid].copy()

    data["route"] = data["source_city"] + "_" + data["destination_city"]
    logger.info("Cleaned dataset: %s rows", f"{len(data):,}")
    return data.reset_index(drop=True)
