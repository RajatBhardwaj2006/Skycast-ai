from __future__ import annotations

import pandas as pd

from src.utils.config import get_logger, load_config, resolve_path

logger = get_logger("skycast.data")


def load_clean_dataset() -> pd.DataFrame:
    path = resolve_path(load_config()["paths"]["raw_clean"])
    logger.info("Loading dataset... %s", path)
    frame = pd.read_csv(path, low_memory=False)
    logger.info("Dataset loaded: %s rows, %s columns", f"{len(frame):,}", frame.shape[1])
    return frame


def load_raw_business() -> pd.DataFrame:
    return pd.read_csv(resolve_path(load_config()["paths"]["raw_business"]))


def load_raw_economy() -> pd.DataFrame:
    return pd.read_csv(resolve_path(load_config()["paths"]["raw_economy"]))
