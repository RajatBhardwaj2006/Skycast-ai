# SkyCast

AI airfare price estimation: historical Indian domestic fares → a saved scikit-learn pipeline → FastAPI → a searchable React dashboard.

**Tagline:** Predict your flight fare before you book.

SkyCast is **not** a live airline feed, booking engine, or ticket shop. Every rupee shown is a **model estimate** from historical training data.

## Problem statement

Estimate airfare from route, airline, cabin class, stops, duration, booking window, and time-of-day features for a consumer estimation tool.

## Audit snapshot (do not guess)

| Dataset | Rows | Columns | Missing | Duplicates | Target |
|---|---:|---:|---:|---:|---|
| `Clean_Dataset.csv` | 300,153 | 12 | 0 | 0 | `price` (INR, int) |
| `business.csv` | 93,487 | 11 | 0 | 0 | `price` (string with commas) |
| `economy.csv` | 206,774 | 11 | 0 | 2 | `price` (string with commas) |

**Clean_Dataset columns:** airline, flight, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left, price (+ unnamed index).

**Training cities in the CSV (six only):** Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai.

**Modeling dataset:** `Clean_Dataset.csv`. `business.csv` / `economy.csv` have travel dates and clock times but **no booking timestamp**, so `days_left` cannot be derived without inventing a scrape date. They are not used as extra training rows.

**UI-only / not in data:** seat type, baggage, lounge, meal, aircraft type, refundability, seat selection. Shown as “coming soon”; **never sent to the model**.

**Flexible routes:** the UI searches `data/reference/airports.csv` (city, airport, IATA, lat/lon). Distance is Haversine `distance_km`. Unknown cities return “Location not found.” Same origin/destination is rejected. The model uses coordinates + distance so new cities do not crash (`OneHotEncoder(handle_unknown="ignore")`). Estimates off the six-city training set show a reliability note.

## Geographic experiment (measured)

HistGradientBoosting, same split:

| Variant | MAE | RMSE | R² |
|---|---:|---:|---:|
| A categorical | 2359.50 | 4056.55 | 0.9681 |
| B + distance | 2273.02 | 3963.61 | 0.9695 |
| C + coordinates + distance | **2272.16** | **3952.31** | **0.9697** |

Distance and coordinates **slightly improved** this estimator. The production pipeline uses **C** (geo + distance) with a **tuned Random Forest** on the full feature set.

**Hold-out (saved run):** MAE ≈ ₹1,509 · RMSE ≈ ₹3,053 · R² ≈ 0.982 · model: Random Forest (tuned). Retrain to refresh these files; do not treat them as promised forever.

## Install

From `X:\VScode\Artificial_intelligence_n_Machine_learning\SkyCast`:

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python -m src.models.train
```

Writes `models/airfare_pipeline.pkl`, `metrics.json`, `feature_importance.json`, `model_metadata.json`, `geo_experiment.json`, and `data/processed/*`.

## Backend

Run from the **SkyCast root** (so `src` and `backend` import cleanly):

```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /health`
- `GET /locations/search?q=Leh`
- `GET /catalog` · `/model-info` · `/metrics` · `/feature-importance` · `/geo-experiment` · `/dataset-info`
- `POST /predict` · `POST /batch-predict`

Example:

```json
{
  "airline": "Air India",
  "source_iata": "IXL",
  "destination_iata": "DEL",
  "departure_time": "Morning",
  "arrival_time": "Evening",
  "stops": "zero",
  "class": "Economy",
  "duration": 1.5,
  "days_left": 20
}
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens Vite on port 5173. Set `frontend/.env` `VITE_API_URL=http://127.0.0.1:8000`.

## Tests

```bash
pytest
```

Includes Haversine symmetry (Delhi↔Mumbai), zero self-distance, Leh→Delhi &gt; 0, API health, same-city rejection, negative duration / days_left, and location 404.

## Architecture

```
User → React (searchable FROM/TO) → FastAPI
  → resolve IATA/city in airports.csv
  → Haversine distance_km + lat/lon
  → saved Pipeline (preprocess + regressor)
  → estimated INR + model feature importance
```

Details: `docs/architecture.md`, `docs/api.md`, `docs/model_card.md`, `docs/data_dictionary.md`.

## Limitations

- Six cities dominate **labels** in training; other airports rely on geography.
- No live prices, no booking, no fabricated seat-price effects.
- Class dominates impurity importance in this dataset (Economy vs Business).

## Screenshots

Add product captures under `docs/screenshots/` when you record a demo.

## Future work

Activate seat/baggage/lounge only after those columns exist in real training data. Optional time-aware split if a true booking timestamp appears.
