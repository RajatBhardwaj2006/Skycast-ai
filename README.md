# SkyCast — AI Airfare Intelligence

SkyCast is a local AI airfare-estimation application. A FastAPI service applies the saved, tuned Random Forest model and a Streamlit dashboard collects only supported model inputs and presents the result with model-backed context.

## Features

- Cached model loading; no training happens during a request.
- Searchable airport/location API and geographic distance calculation.
- Validated single and batch prediction endpoints.
- Streamlit prediction flow for origin, destination, airline, class, travel date, time bands, duration and stops.
- Actual metrics, model comparison, feature importance and geographic experiment results from `models/`.
- A transparent error band of prediction ± held-out test MAE. It is not a confidence interval or live price quote.

## Architecture

`Clean_Dataset.csv → cleaning → geographic features → sklearn preprocessing pipeline → tuned Random Forest → FastAPI → Streamlit`

The model uses airline, departure/arrival time bands, stops, class, source/destination city, duration, days left, coordinates and haversine distance. It does not use aircraft type, baggage, seat selection or other ancillary-product fields.

## Model and data

The training dataset is `Clean_Dataset.csv` (300,153 rows). `business.csv` and `economy.csv` are retained for source/schema reference, not merged into training because a booking timestamp needed to derive `days_left` is unavailable. Geographic features are calculated from `data/reference/airports.csv`.

The final saved model is **Random Forest (tuned)** (`n_estimators=100`, `max_depth=16`, `min_samples_leaf=2`). On the recorded held-out random split it achieved MAE ₹1,508.89, RMSE ₹3,052.77 and R² 0.9819. Full model comparisons are in `models/model_comparison.json`.

## Setup and run

Install dependencies in a Python 3.10+ environment:

```bash
pip install -r requirements.txt
```

Start the API from the repository root:

```bash
uvicorn backend.app.main:app --reload
```

In another terminal, start the dashboard:

```bash
streamlit run frontend/app.py
```

The API defaults to `http://127.0.0.1:8000`; set `SKYCAST_API_URL` for a different dashboard target.

## API

- `GET /health`, `/model-info`, `/metrics`, `/feature-importance`, `/geo-experiment`, `/dataset-info`
- `GET /catalog` and `GET /locations/search?q=Delhi`
- `POST /predict` and `POST /batch-predict`

Example request:

```json
{"source_city":"Delhi","destination_city":"Mumbai","airline":"Air India","class":"Economy","departure_time":"Morning","arrival_time":"Evening","stops":"zero","duration":2.25,"days_left":20}
```

## Training and testing

Use the existing model for normal application use. Retraining is optional and overwrites artifacts:

```bash
python -m src.models.train
python -m pytest -q
python -m compileall backend src frontend
```

## Limitations

SkyCast is historical-model inference, not a live airline inventory or pricing feed. Its training routes are centred on Delhi, Mumbai, Bangalore, Kolkata, Hyderabad and Chennai; other routes can be accepted using the airport reference data but are flagged as outside the training-city distribution. The strong R² is from a random row split, which can allow similar routes to appear in both train and test partitions; a route- or time-aware validation split would be a valuable next evaluation.
