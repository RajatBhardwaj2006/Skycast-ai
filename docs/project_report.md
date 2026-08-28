# SkyCast technical report

## Scope

SkyCast provides historical airfare estimates through a FastAPI prediction service and Streamlit dashboard. The deployed artifact is the existing `models/airfare_pipeline.pkl`; requests never retrain or apply UI multipliers.

## Data and preprocessing

`Clean_Dataset.csv` contains 300,153 rows and is the modelling source. The cleaning pipeline standardizes category values, validates duration and booking-window fields, and removes invalid same-city records. Airport coordinates are joined from the project reference data, then haversine distance and origin/destination latitude/longitude are added.

Business and economy source extracts are deliberately not folded into the training target: without a booking timestamp, `days_left` cannot be derived honestly. This decision is recorded in `data/processed/dataset_decision.json`.

## Model

The feature contract is airline, departure time, arrival time, stops, class, origin city, destination city, duration, days left, source/destination coordinates and distance. Categorical values are one-hot encoded with unknown-category handling; numerical values are passed to the estimator. The production pipeline is a tuned Random Forest with `n_estimators=100`, `max_depth=16`, and `min_samples_leaf=2`.

Recorded held-out results: MAE ₹1,508.89, RMSE ₹3,052.77 and R² 0.9819. The comparison artifact includes linear, ridge, random forest, gradient boosting, histogram gradient boosting and XGBoost candidates. A separate geographic experiment found the categorical + coordinates + distance representation outperformed categorical-only variants under its fixed estimator/split.

## API and UI contract

The API resolves cities/IATA codes to reference coordinates, normalizes permitted values, validates duration and booking window, and produces the exact feature frame expected by the saved pipeline. The response exposes an `expected_price_range` computed as estimate ± the recorded held-out MAE. The response calls it an error band, not confidence.

The Streamlit client sends the API schema directly. Its travel date converts to `days_left`; clock time converts to the training time bands. It intentionally excludes aircraft type and seat modifiers, which are unsupported by the model. Dashboard charts use the saved JSON artifacts rather than fabricated values.

## Methodological caution

The high R² should not be interpreted as a guarantee. The current evaluation uses a random row split, so similar route/airline/class combinations can occur on both sides of the split. Before claims about out-of-route or future-time performance, evaluate grouped route and temporal splits, inspect near-duplicates, and consider prediction intervals built from residual calibration. SkyCast does not provide real-time airline prices or market intelligence.
