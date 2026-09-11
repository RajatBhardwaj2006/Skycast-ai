# SkyCast — AI Airfare Intelligence Platform

AI airfare price estimation across India: historical Indian domestic flight data → scikit-learn & XGBoost regression pipelines → FastAPI REST API → searchable React/Vite dashboard.

> **Important:** SkyCast is a machine learning consumer estimation tool based on historical flight distributions, **not** a live airline ticketing engine or live fare scraper. Every price presented is a model prediction.

---

## 1. System Architecture

```
DATA SOURCES (Clean_Dataset.csv + IndianFlightdata)
  ↓
SCHEMA HARMONIZATION & DEDUPLICATION (No fabricated class / days_left)
  ↓
GEOSPATIAL ENGINEERING (Airport coordinates & dynamic Haversine distance)
  ↓
PREPROCESSING PIPELINE (ColumnTransformer, SimpleImputer, OneHotEncoder, StandardScaler)
  ↓
MODEL SELECTION & TUNING (Linear, Ridge, GradientBoosting, HistGradientBoosting, XGBoost, RandomForest)
  ↓
RIGOROUS VALIDATION (Random-row Holdout + Route-group Audit + Airport-group Audit)
  ↓
MODEL PERSISTENCE (Joblib serialized pipeline)
  ↓
FASTAPI BACKEND (REST endpoints, catalog, dynamic distance, reliability warnings)
  ↓
REACT / VITE FRONTEND (Searchable 248-airport catalog, dynamic distance, fare bands, model performance)
```

---

## 2. Dataset Expansion & Quality (Prompt 1.4)

| Dataset Source | Records | Columns | Role in Training |
|---|---|---|---|
| `Clean_Dataset.csv` | 300,153 | 11 | Primary historical dataset (Feb–Mar 2022) with full cabin class and booking window. |
| `IndianFlightdata - Sheet1.csv` | 10,463 (deduped) | 11 | Additional historical dataset (2019) introducing extra route coverage (including Kochi). |
| `Flight Data.xlsx` | 10,683 | 11 | Excluded as exact duplicate of CSV dataset to prevent train/test contamination. |
| **Merged Training Dataset** | **310,522** | **14** | Unified clean dataset (`data/processed/merged_data.csv`). Zero fabricated features. |

### Data Integrity Standards
- **No Class Fabrication**: Rows lacking explicit class from the new dataset are marked as `Unknown` rather than fabricating Economy or Business based on price heuristics.
- **No Days-Left Fabrication**: Days until departure is kept as missing when booking timestamp is absent, handled via `SimpleImputer(strategy='median')`.
- **Deduplication**: 220 internal duplicate rows in the secondary dataset were pruned before merging.

---

## 3. Retrained Model Performance & Audits

Evaluated on 310,522 rows (80/20 train/test split, 248,417 train / 62,105 test):

| Evaluation Strategy | Metric Purpose | MAE | RMSE | R² Score |
|---|---|---|---|---|
| **In-Distribution Random Holdout** | Test error on historical route distribution | **₹1,352.68** | **₹2,782.56** | **0.9846** |
| **Route-Group Holdout Audit** | Generalization to routes **never seen** during training (7 held-out routes) | **₹3,360.59** | **₹6,067.96** | **0.9292** |
| **Airport-Group Holdout Audit** | Generalization to origin airports **never seen** during training (Bangalore & Chennai held out) | **₹3,685.74** | **₹6,200.42** | **0.9282** |

### Candidate Model Comparison (Random-row Holdout)

| Model | MAE (₹) | RMSE (₹) | R² Score | Training Time | Inference (ms / 1k) |
|---|---|---|---|---|---|
| **Random Forest (tuned)** | **1,352.68** | **2,782.56** | **0.9846** | Tuned (CV=3) | 2.4 ms |
| Random Forest (baseline) | 1,536.81 | 3,018.64 | 0.9819 | 6.8s | 2.4 ms |
| XGBoost | 1,723.19 | 3,091.98 | 0.9810 | 3.6s | 2.4 ms |
| HistGradientBoosting | 2,130.10 | 3,647.36 | 0.9736 | 2.5s | 2.8 ms |
| Gradient Boosting | 2,953.47 | 4,900.75 | 0.9523 | 3.0s (subsample) | 2.3 ms |
| Linear Regression | 4,486.46 | 6,657.42 | 0.9120 | 1.3s | 1.7 ms |
| Ridge Regression | 4,486.58 | 6,657.88 | 0.9120 | 1.0s | 1.7 ms |

### Dataset Strategy Benchmarks
- **Model A (Clean_Dataset only)**: MAE ₹2,196.56 | RMSE ₹3,815.59 | R² 0.9718
- **Model B (New Indian Flight Data only)**: MAE ₹1,720.18 | RMSE ₹2,689.78 | R² 0.6451
- **Model C (Unified Merged Dataset)**: MAE ₹2,177.91 | RMSE ₹3,716.16 | R² 0.9726

---

## 4. All-India Airport Support & Geospatial Features

- **Airport Catalog**: 248 airports across all Indian states and Union Territories in `data/reference/airports.csv`.
- **Dynamic Haversine Distance**: Server computes great-circle distance dynamically from latitude and longitude coordinates. Distance is never hard-coded.
- **Unseen Route Generalization**: When users query airport pairs not in the historical training set (e.g. Chandigarh → Bangalore), the model uses geographic coordinates, distance, stops, duration, airline, class, and timing features. A clear UI badge indicates:
  > *⚠️ Limited historical training coverage — The airport is supported geographically, but this exact route has limited historical observations in the training data.*

---

## 5. Quickstart & Verification

### Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Retrain Model Pipeline
```bash
python -m src.models.train
```

### Run Python Automated Tests
```bash
python -m pytest -q
python -m compileall backend src
```

### Start FastAPI Backend
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start React Frontend
```bash
cd frontend
npm install
npm run build
npm run dev
```
Open `http://localhost:5173` to explore the dashboard.

---

## 6. Project Directory Layout

```
SkyCast/
├── backend/                  # FastAPI web server and prediction services
│   ├── app/
│   │   ├── main.py           # REST endpoints
│   │   ├── schemas.py        # Pydantic request/response validation schemas
│   │   └── services/         # Prediction and model loading logic
├── data/
│   ├── processed/            # merged_data.csv, data_quality_report.json
│   └── reference/            # airports.csv (248 Indian airports catalog)
├── docs/                     # data_expansion.md, model cards, architecture reports
├── frontend/                 # React 18 + Vite frontend
│   └── src/
│       ├── components/       # Searchable LocationSearch, Layout, Sidebar, TopBar
│       └── pages/            # Predict, ModelPerformance, Dataset, Dashboard
├── models/                   # Serialized airfare_pipeline.pkl, metrics.json, validation.json
├── notebooks/                # 01_eda.ipynb, 02_preprocessing.ipynb, 03_modeling.ipynb
├── src/                      # Core ML engineering package
│   ├── data/                 # clean.py, load.py, merge_datasets.py, quality.py
│   ├── evaluation/           # importance.py, metrics.py, validation.py
│   ├── features/             # geo_features.py, preprocess.py
│   ├── geo/                  # haversine.py, locations.py
│   └── models/               # train.py
└── tests/                    # 29 unit and integration tests
```
