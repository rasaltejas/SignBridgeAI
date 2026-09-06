from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:
    import cv2
except Exception:  # pragma: no cover - depends on environment
    cv2 = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - depends on environment
    mp = None


def extract_landmarks(frame):
    if cv2 is None or mp is None or frame is None:
        raise RuntimeError("OpenCV and MediaPipe are required to extract landmark features.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return []

    flattened: List[float] = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            flattened.extend([landmark.x, landmark.y, landmark.z])
    return flattened


def extract_landmarks_from_image(image_path: str | Path) -> List[float]:
    image_path = Path(image_path)
    if cv2 is None:
        raise RuntimeError("OpenCV is required to read images.")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return extract_landmarks(frame)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract hand landmarks from an image")
    parser.add_argument("image", type=str, help="Path to an image file")
    args = parser.parse_args()

    landmarks = extract_landmarks_from_image(args.image)
    print(landmarks)
