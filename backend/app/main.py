from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import API_CORS_ORIGINS
from backend.app.schemas import BatchPredictionRequest, FlightPredictionRequest
from backend.app.services import model_service, prediction_service
from src.geo.locations import default_location_service

app = FastAPI(title="SkyCast Airfare Estimation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "skycast"}


@app.get("/")
def root():
    return {"message": "SkyCast API. See /docs for endpoints."}


@app.get("/locations/search")
def search_locations(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=20)):
    matches = default_location_service().search(q, limit=limit)
    if not matches:
        raise HTTPException(
            status_code=404,
            detail={"message": "Location not found.", "hint": "Try another city or airport."},
        )
    return {"results": [item.to_dict() for item in matches]}


@app.get("/catalog")
def catalog():
    meta = model_service.load_metadata()
    return {
        "airlines": meta.get("airlines", []),
        "classes": meta.get("classes", []),
        "stops": meta.get("stops", []),
        "departure_times": meta.get("departure_times", []),
        "arrival_times": meta.get("arrival_times", []),
        "training_cities": meta.get("training_cities", []),
        "unsupported_features": meta.get("unsupported_features", []),
        "example_input": meta.get("example_input", {}),
    }


@app.get("/model-info")
def model_info():
    return model_service.load_metadata()


@app.get("/metrics")
def metrics():
    return model_service.load_metrics()


@app.get("/feature-importance")
def feature_importance():
    return model_service.load_importance()


@app.get("/geo-experiment")
def geo_experiment():
    return model_service.load_geo_experiment()


@app.get("/dataset-info")
def dataset_info():
    return {
        "quality": model_service.load_quality(),
        "decision": model_service.load_decision(),
        "metadata": model_service.load_metadata(),
    }


@app.post("/predict")
def predict(payload: FlightPredictionRequest):
    return prediction_service.predict_fare(payload)


@app.post("/batch-predict")
def batch_predict(payload: BatchPredictionRequest):
    return {"predictions": [prediction_service.predict_fare(item) for item in payload.items]}
