from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import load_config


def feature_groups() -> tuple[list[str], list[str]]:
    config = load_config()
    return list(config["features"]["categorical"]), list(config["features"]["numerical"])


def build_preprocessor(categorical: list[str] | None = None, numerical: list[str] | None = None) -> ColumnTransformer:
    cat, num = feature_groups()
    categorical = categorical if categorical is not None else cat
    numerical = numerical if numerical is not None else num
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numerical),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        )
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_feature_frame(frame, columns: list[str]):
    return frame.loc[:, columns].copy()
