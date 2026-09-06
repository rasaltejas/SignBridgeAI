from __future__ import annotations

import argparse
from pathlib import Path

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - optional dependency
    RandomForestClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"


def load_dataset(dataset_dir: Path = DATASET_DIR):
    if pd is None:
        raise RuntimeError("pandas is required to train the model.")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset files were found in {dataset_dir}. Run collect_data.py first.")

    frames = [pd.read_csv(path) for path in csv_files]
    dataset = pd.concat(frames, ignore_index=True)
    features = dataset.drop(columns=["label"])
    labels = dataset["label"]
    return features, labels


def train_model(dataset_dir: Path = DATASET_DIR, output_path: Path = MODELS_DIR / "gesture_model.joblib") -> Path:
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required to train the model.")
    if joblib is None:
        raise RuntimeError("joblib is required to save the model.")

    features, labels = load_dataset(dataset_dir)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(features, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hand gesture classifier")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR, help="Directory containing landmark CSV files")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "gesture_model.joblib", help="Path for the trained model")
    args = parser.parse_args()

    model_path = train_model(dataset_dir=args.dataset, output_path=args.output)
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
