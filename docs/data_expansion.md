# SkyCast — Data Expansion & Integration Report

## 1. Overview of Evaluated Datasets

| Dataset | File Path / Source | Rows | Columns | Key Features | Date Coverage | Class Availability | Booking Window (Days Left) |
|---|---|---|---|---|---|---|---|
| **Baseline (Clean Dataset)** | `Clean_Dataset.csv` | 300,153 | 11 | `airline, flight, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left, price` | Scraped over 50 days (Feb-Mar 2022) | Full (`Economy`, `Business`) | Full (`1` to `49` days) |
| **New Dataset 1** | `Flight-Price-Prediction-using-Machine-Learning-Random-Forest-XGBoost--main/IndianFlightdata - Sheet1.csv` | 10,683 (220 duplicates -> 10,463 unique) | 11 | `Airline, Date_of_Journey, Source, Destination, Route, Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info, Price` | March - June 2019 | Not explicitly in a dedicated column (predominantly economy; 6 rows named "Jet Airways Business") | Unavailable (no booking timestamp) |
| **New Dataset 2** | `Indian_Flight_Data-main/Flight Data.xlsx` | 10,683 | 11 | Identical schema and values to Dataset 1 | March - June 2019 | Identical to Dataset 1 | Unavailable |

---

## 2. Dataset Integration Decisions

### A. Deduplication & Redundancy Prevention
- **Dataset 2 (`Flight Data.xlsx`) is an exact duplicate** of Dataset 1 (`IndianFlightdata - Sheet1.csv`) in XLSX format. Merging both would artificially duplicate records, violating data integrity rules. Consequently, Dataset 2 was inspected, verified, and excluded from duplicate concatenation.
- **Dataset 1 contained 220 internal duplicate rows**, which were identified and removed, leaving 10,463 genuine flight records.

### B. Intelligent Schema Normalization (No Fabricated Features)
In accordance with Prompt 1.4 requirements:
- **No Class Fabrication**: Class was **not** invented or assigned via price heuristics. For Dataset 1 records, `class` was retained as `Unknown`. The pipeline's categorical imputer and one-hot encoder natively represent this category without corrupting the historical business/economy pricing structure of `Clean_Dataset.csv`.
- **No Days-Left Fabrication**: `days_left` was not fabricated. It was retained as `NaN` and handled via `SimpleImputer(strategy='median')` in the preprocessing pipeline.
- **Temporal Binning**: Raw clock times (`Dep_Time` like `22:20` and `Arrival_Time` like `01:10 22 Mar`) were mapped into the 6 canonical flight departure/arrival bins (`Early Morning`, `Morning`, `Afternoon`, `Evening`, `Night`, `Late Night`).
- **City & Airport Harmonization**: Origin/destination names (`Banglore` -> `Bangalore`, `New Delhi` -> `Delhi`, `Cochin` -> `Kochi`) were unified to resolve with `data/reference/airports.csv`.
- **Stops Normalization**: `Total_Stops` strings (`non-stop`, `1 stop`, `2 stops`, `3 stops`, `4 stops`) were mapped to `zero`, `one`, and `two_or_more`.
- **Duration Parsing**: String representations (`2h 50m`, `19h`, etc.) were parsed into float hours.

---

## 3. Dataset Strategy Comparison (Prompt 1.4 Section 8)

| Strategy | Description | Training Rows | Holdout MAE | Holdout RMSE | Holdout R² | Key Finding |
|---|---|---|---|---|---|---|
| **Model A** | `Clean_Dataset.csv` alone | 300,153 | ₹2,196.56 | ₹3,815.59 | 0.9718 | Strong baseline but limited to 6 tier-1 metro cities. |
| **Model B** | `IndianFlightdata` alone | 10,369 | ₹1,720.18 | ₹2,689.78 | 0.6451 | Low variance due to economy concentration, but lacks cabin class separation and small sample size yields lower R². |
| **Model C (Selected)** | Unified Merged Dataset (`merged_data.csv`) | 310,522 | ₹2,177.91 | ₹3,716.16 | 0.9726 | **Strongest overall performance**. Outperforms Model A on both MAE and RMSE while broadening destination coverage (including Kochi) and training routes. |

---

## 4. Final Merged Dataset Summary
- **Total Clean Rows**: 310,522
- **Origin / Destination Cities**: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai, Kochi
- **Airports Catalog Supported**: 248 airports across all Indian states and union territories (`data/reference/airports.csv`)
- **Distance Calculation**: Dynamic Haversine distance in kilometres computed between origin and destination coordinates.
