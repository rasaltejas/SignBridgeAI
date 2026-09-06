from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
except Exception:  # pragma: no cover - depends on environment
    cv2 = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - depends on environment
    mp = None

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.joblib"


def _get_hands() -> Optional[Any]:
    if mp is None:
        return None
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def extract_landmarks(frame: "Any") -> Optional[list]:
    if frame is None or cv2 is None or mp is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = _get_hands()
    if hands is None:
        return None
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    flattened: list = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            flattened.extend([landmark.x, landmark.y, landmark.z])
    return flattened


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    if joblib is None:
        raise RuntimeError("joblib is required to load a trained model.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def predict_gesture(frame: "Any", model: Optional[Any] = None, model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    landmarks = extract_landmarks(frame)
    if landmarks is None:
        return {"prediction": "no_hand", "confidence": 0.0}

    if model is None:
        try:
            model = load_model(model_path)
        except Exception:
            return {"prediction": "model_missing", "confidence": 0.0, "message": "Train a model before running realtime prediction."}

    try:
        prediction = model.predict([landmarks])[0]
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([landmarks])[0]
            confidence = float(max(probabilities))
        return {"prediction": str(prediction), "confidence": confidence}
    except Exception as exc:  # pragma: no cover - runtime guard
        return {"prediction": "error", "confidence": 0.0, "message": str(exc)}


def run_realtime_prediction(source: int = 0, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to run the realtime prediction pipeline.")

    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open webcam source {source}.")

    model = None
    if model_path.exists() and joblib is not None:
        try:
            model = load_model(model_path)
        except Exception:
            model = None

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = predict_gesture(frame, model=model, model_path=model_path)
        label = result.get("prediction", "unknown")
        confidence = result.get("confidence", 0.0)

        if result.get("prediction") == "model_missing":
            cv2.putText(frame, "Train model first", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"{label}: {confidence:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("SignBridgeAI Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime SignBridgeAI prediction")
    parser.add_argument("--source", type=int, default=0, help="Webcam source index")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model")
    args = parser.parse_args()
    run_realtime_prediction(source=args.source, model_path=args.model)


if __name__ == "__main__":
    main()
