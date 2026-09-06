from __future__ import annotations

import argparse
import csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"


def _get_hands():
    if mp is None:
        return None
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def _flatten_landmarks(hand_landmarks) -> List[float]:
    flattened: List[float] = []
    for landmark in hand_landmarks.landmark:
        flattened.extend([landmark.x, landmark.y, landmark.z])
    return flattened


def collect_samples(label: str, samples: int = 30, source: int = 0, output_path: Optional[Path] = None) -> Path:
    if cv2 is None or mp is None:
        raise RuntimeError("OpenCV and MediaPipe are required to collect landmark samples.")

    output_path = output_path or DATASET_DIR / f"{label.lower()}_landmarks.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["label"] + [f"landmark_{index}_{axis}" for index in range(21) for axis in ("x", "y", "z")]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        camera = cv2.VideoCapture(source)
        if not camera.isOpened():
            raise RuntimeError(f"Unable to open webcam source {source}.")

        hands = _get_hands()
        collected = 0

        while collected < samples:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    flat = _flatten_landmarks(hand_landmarks)
                    writer.writerow([label] + flat)
                    collected += 1
                    cv2.putText(frame, f"Collected {collected}/{samples}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if collected >= samples:
                        break

            cv2.putText(frame, f"Press 'q' to quit, 'space' to save sample ({label})", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Collect SignBridge data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand landmark data for a sign label")
    parser.add_argument("--label", required=True, help="Gesture label to collect")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to capture")
    parser.add_argument("--source", type=int, default=0, help="Webcam source index")
    parser.add_argument("--output", type=Path, default=None, help="Destination CSV file")
    args = parser.parse_args()
    path = collect_samples(args.label, samples=args.samples, source=args.source, output_path=args.output)
    print(f"Saved {args.label} landmark data to {path}")


if __name__ == "__main__":
    main()
