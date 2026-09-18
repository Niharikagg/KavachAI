import csv
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "attribute_count_normalized",
    "average_specificity",
    "maximum_specificity",
    "high_specificity_ratio",
    "sensitive_category_count_normalized",
    "uniqueness_risk",
]
TARGET_COLUMN = "risk_score"
DATASET_PATH = Path("data/synthetic/risk_ml_dataset.csv")
MODEL_PATH = Path("models/risk_model.joblib")


class RiskModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: RandomForestRegressor | None = None

    def load(self) -> None:
        self.model = joblib.load(self.model_path)

    def predict(self, features: dict[str, float]) -> float:
        if self.model is None:
            self.load()
        values = [[features[column] for column in FEATURE_COLUMNS]]
        return float(self.model.predict(values)[0])


def load_dataset(dataset_path: Path = DATASET_PATH) -> tuple[list[list[float]], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    with dataset_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
            targets.append(float(row[TARGET_COLUMN]))
    return features, targets


def train_model(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    features, targets = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"R2: {r2_score(y_test, predictions):.4f}")
    print("Actual vs predicted:")
    for actual, predicted in list(zip(y_test, predictions))[:5]:
        print(f"  actual={actual:.4f}, predicted={predicted:.4f}")

    return {
        "model": model,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "training_samples": len(x_train),
        "test_samples": len(x_test),
    }


if __name__ == "__main__":
    train_model()
