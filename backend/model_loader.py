"""
Model Loader Utility for FastAPI
"""

import os
import joblib

class ModelLoader:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            model_path = os.path.join("models", "best_model.pkl")
            if os.path.exists(model_path):
                cls._model = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")
        return cls._model