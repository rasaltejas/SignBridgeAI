from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency guard
    FastAPI = None
    BaseModel = object

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gesture_model.joblib"

class PredictRequest:
    def __init__(self, landmarks: Optional[List[float]] = None, label: Optional[str] = None) -> None:
        self.landmarks = landmarks or []
        self.label = label


class _FallbackApp:
    def route(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func
        return decorator


app = _FallbackApp()


def _predict_response(landmarks: Optional[List[float]], label: Optional[str]) -> Dict[str, Any]:
    features = landmarks or []
    if not features:
        return {"error": "No landmarks were provided for prediction."}
    if len(features) % 3 != 0:
        return {"error": "Landmark payload must contain values in groups of three (x, y, z)."}
    return {
        "prediction": label or "unknown",
        "confidence": 0.0,
        "landmark_count": len(features) // 3,
        "model_available": MODEL_PATH.exists(),
    }


if FastAPI is not None:
    try:
        class _RequestModel(BaseModel):
            landmarks: Optional[List[float]] = None
            label: Optional[str] = None

        app = FastAPI(title="SignBridgeAI", version="0.1.0")

        @app.get("/")
        async def root() -> Dict[str, Any]:
            return {"message": "SignBridgeAI backend is running.", "model_path": str(MODEL_PATH)}

        @app.get("/health")
        async def health() -> Dict[str, Any]:
            return {"status": "ok", "model_available": MODEL_PATH.exists()}

        @app.post("/predict")
        async def predict(payload: _RequestModel) -> Dict[str, Any]:
            return _predict_response(payload.landmarks, payload.label)
    except Exception:
        app = _FallbackApp()


def root() -> Dict[str, Any]:
    return {"message": "SignBridgeAI backend is running.", "model_path": str(MODEL_PATH)}


def health() -> Dict[str, Any]:
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


def predict(payload: PredictRequest) -> Dict[str, Any]:
    return _predict_response(payload.landmarks, payload.label)


if __name__ == "__main__":
    try:
        import uvicorn
    except Exception:
        print("Install uvicorn to run the backend server: pip install uvicorn")
    else:
        uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
