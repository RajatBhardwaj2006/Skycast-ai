# Architecture

SkyCast separates **location search** from **fare regression**.

1. `data/reference/airports.csv` is independent of `Clean_Dataset.csv`.
2. `src/geo` resolves cities/IATA and computes Haversine kilometres.
3. `src/features/preprocess.py` builds a `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")`.
4. `python -m src.models.train` persists one sklearn `Pipeline`.
5. FastAPI loads that artifact only; it never retrains on startup.
6. React searches `/locations/search` and posts IATA codes to `/predict`.

Unsupported UI fields are not in the request body.
