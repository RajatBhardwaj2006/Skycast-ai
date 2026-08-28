from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer


def grouped_feature_importance(pipeline) -> list[dict]:
    """Aggregate one-hot dummy importances back to original input columns."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity"
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        kind = "absolute_coefficient"
    else:
        return []

    grouped: dict[str, float] = {}
    for name, value in zip(feature_names, raw):
        original = name
        if "__" in name:
            original = name.split("__", 1)[1]
        if "_" in original and original.split("_")[0] in {"num", "cat"}:
            original = original.split("_", 1)[1]
        # ColumnTransformer names: num__duration / cat__airline_Vistara
        if name.startswith("num__"):
            original = name[len("num__") :]
        elif name.startswith("cat__"):
            rest = name[len("cat__") :]
            original = rest.rsplit("_", 1)[0] if "_" in rest else rest
        grouped[original] = grouped.get(original, 0.0) + float(value)

    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "importance": round(score / total, 6), "source": kind}
        for feature, score in ranked
    ]
