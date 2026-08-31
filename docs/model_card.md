# Model card

- **Task:** Regression, target `price` (INR).
- **Data:** `Clean_Dataset.csv` after dropping unnamed index, invalid duration/price, and same-city rows.
- **Split:** 80/20 `random_state=42`. Dates on business/economy files are not a booking calendar, so a time-based split was not used for the production model.
- **Features:** airline, departure/arrival period, stops, class, source/destination city (unknown ignored), duration, days_left, source/dest lat/lon, distance_km.
- **Not used:** seat type, baggage, lounge, meal, aircraft, refundability.
- **Selected model:** see `models/metrics.json` (Random Forest tuned on the saved run).
- **Geo experiment:** `models/geo_experiment.json` — variant C won on MAE/RMSE/R² for HistGradientBoosting.
- **Ethics / product:** estimates only; not live fares.
