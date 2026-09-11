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
    elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
        sub_importances = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_, dtype=float)
                s = imp.sum()
                if s > 0:
                    sub_importances.append(imp / s)
        if sub_importances:
            raw = np.mean(sub_importances, axis=0)
            kind = "ensemble_impurity_mean"
        else:
            return []
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        kind = "absolute_coefficient"
    else:
        return []

    grouped: dict[str, float] = {}
    known_cols = ["class", "airline", "departure_time", "arrival_time", "stops", "source_city", "destination_city",
                  "duration", "days_left", "distance_km", "source_lat", "source_lon", "destination_lat", "destination_lon"]
    for name, value in zip(feature_names, raw):
        original = name
        if name.startswith("num__"):
            original = name[len("num__") :]
        elif name.startswith("cat__"):
            rest = name[len("cat__") :]
            matched = False
            for col in sorted(known_cols, key=len, reverse=True):
                if rest.startswith(col + "_") or rest == col:
                    original = col
                    matched = True
                    break
            if not matched:
                original = rest.rsplit("_", 1)[0] if "_" in rest else rest
        grouped[original] = grouped.get(original, 0.0) + float(value)

    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "importance": round(score / total, 6), "source": kind}
        for feature, score in ranked
    ]
