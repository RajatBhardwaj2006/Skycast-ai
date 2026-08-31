# API

Base URL: `http://127.0.0.1:8000`

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/locations/search?q=` | 404 `{message, hint}` if no match |
| GET | `/catalog` | Airlines, classes, example input |
| GET | `/model-info` | Metadata from training |
| GET | `/metrics` | MAE, RMSE, R², comparison |
| GET | `/feature-importance` | Grouped importances |
| GET | `/geo-experiment` | Models A/B/C |
| GET | `/dataset-info` | Quality report + dataset decision |
| POST | `/predict` | Single estimate |
| POST | `/batch-predict` | `{ "items": [ ... ] }` |

CORS origins are listed in `config.yaml` (`localhost:5173` by default).
