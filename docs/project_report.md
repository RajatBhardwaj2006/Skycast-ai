# SkyCast technical report

## Introduction and problem statement

SkyCast estimates the `price` target for a consumer airfare-estimation workflow. It is a supervised regression system, not a live airline inventory or booking service. The deployed artifact is `models/airfare_pipeline.pkl`; requests never retrain or apply UI multipliers. The primary user interface is a React/Vite dashboard, with the Streamlit client retained as a lightweight alternative.

## Objectives

The system validates historical data, engineers route geography, compares regression algorithms, persists the selected preprocessing-and-model pipeline, and serves real estimates through FastAPI. The web client supports searchable cities and airports, displays dynamically calculated route distance, and communicates uncertainty and out-of-distribution limitations plainly.

## Data and preprocessing

`Clean_Dataset.csv` contains 300,153 rows and is the modelling source. The cleaning pipeline standardizes category values, validates duration and booking-window fields, and removes invalid same-city records. Airport coordinates are joined from the project reference data, then haversine distance and origin/destination latitude/longitude are added.

Business and economy source extracts are deliberately not folded into the training target: without a booking timestamp, `days_left` cannot be derived honestly. This decision is recorded in `data/processed/dataset_decision.json`.

## Feature engineering and geography

The production feature contract is airline, departure/arrival period, stops, class, source/destination city, duration, days left, source/destination coordinates and haversine distance. The geographic experiment compares categorical/base features, then distance, then coordinates plus distance under the same HistGradientBoosting estimator and split. The recorded MAEs are ₹2,359.50, ₹2,273.02 and ₹2,272.16 respectively, so the full geographic representation was selected for the production comparison.

Airport lookup comes from `data/reference/airports.csv`. FastAPI resolves IATA/city selections and derives the distance itself; clients do not provide a trusted distance. This safely supports locations such as Leh (IXL), though a city absent from historical training categories is flagged as outside the training distribution.

## ML algorithms, tuning and evaluation

Categorical values are one-hot encoded with unknown-category handling; numerical values are passed to the estimator in the saved sklearn pipeline. Linear Regression, Ridge, Random Forest, Gradient Boosting, HistGradientBoosting and XGBoost were evaluated. The production pipeline is a tuned Random Forest with `n_estimators=100`, `max_depth=16`, and `min_samples_leaf=2`.

The in-distribution random-row holdout recorded MAE ₹1,508.89, RMSE ₹3,052.77 and R² 0.9819. A stricter route-group holdout, with six complete routes absent from training, recorded MAE ₹3,470.47, RMSE ₹6,321.44 and R² 0.9165. The route-group result is the scientifically defensible metric for unseen-route claims; the random-row result remains useful only for the deployed historical-route distribution. Temporal validation was not performed because `Clean_Dataset.csv` has no legitimate travel or booking date. The comparison artifact includes linear, ridge, random forest, gradient boosting, histogram gradient boosting and XGBoost candidates. A separate geographic experiment found the categorical + coordinates + distance representation outperformed categorical-only variants under its fixed estimator/split.

## Feature importance

The saved grouped impurity importance report is exposed through the API and dashboard. Class is the dominant reported feature (0.892748), followed by duration (0.051737), distance (0.015060), days left (0.013490), and airline (0.010992). These describe the fitted forest, not causal price effects for an individual flight.

## System architecture, backend and frontend

The API resolves cities/IATA codes to reference coordinates, normalizes permitted values, validates duration and booking window, and produces the exact feature frame expected by the saved pipeline. The response exposes an `expected_price_range` computed as estimate ± the recorded held-out MAE. The response calls it an error band, not confidence.

The React dashboard calls `/locations/search`, `/route-distance`, artifact endpoints and `/predict`. It has Dashboard, Predict Fare, Insights, Model Performance, Dataset, History and About views, responsive sidebar navigation, keyboard-capable airport autocomplete, route swap, current-session history and loading/error/empty states. The travel UI labels the MAE range as an approximate prediction error band. Seat type, luggage, lounge access and aircraft type are visibly disabled as future data features and never affect an estimate. Dashboard charts use saved JSON artifacts rather than fabricated values.

## Testing

The repository contains 25+ behavior-focused test functions covering data loading/cleaning, preprocessing, geographic calculations, model loading, location search, API validation, prediction and route distance. The browser build is checked with `npm run build`. Runtime verification includes all artifact endpoints, searchable Leh and Bangalore lookups, dynamic Leh-to-Delhi and Delhi-to-Bangalore distances, valid predictions and same-airport rejection.

## Limitations, conclusion and future scope

The random-row R² should not be interpreted as a guarantee. The final validation artifact records the substantially more conservative route-group audit for unseen-route interpretation. Chronological testing must wait for authentic booking or travel dates. Future work can add authentic booking-time, seat, baggage, aircraft and live-inventory sources. SkyCast does not provide real-time airline prices or market intelligence.
