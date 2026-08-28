from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path

config = load_config()
API_CORS_ORIGINS = list(config.get("api", {}).get("cors_origins", ["http://localhost:5173"]))
PIPELINE_PATH = resolve_path(config["paths"]["pipeline"])
METRICS_PATH = resolve_path(config["paths"]["metrics"])
IMPORTANCE_PATH = resolve_path(config["paths"]["feature_importance"])
METADATA_PATH = resolve_path(config["paths"]["metadata"])
GEO_EXPERIMENT_PATH = resolve_path(config["paths"]["geo_experiment"])
COMPARISON_PATH = resolve_path(config["paths"]["comparison"])
QUALITY_PATH = resolve_path(config["paths"]["quality_report"])
DECISION_PATH = resolve_path("data/processed/dataset_decision.json")
